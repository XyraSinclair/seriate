//! The logprob reality map.
//!
//! OpenRouter logprob support is provider-dependent and quietly broken in
//! places. This module measures, per model, what is actually there: are
//! logprobs returned at all, how deep is the top-k, can the answer position
//! be found, how much probability mass is visible — and does the logprob
//! PMF agree with what the model actually samples (JSD between the two).
//! The resulting reports are the receipt that decides, per model, whether
//! logprob mode is real or the system must degrade loudly to sampling.

use crate::evidence::{evidence_from_resamples, jsd, AnswerEvidence};
use crate::gateway::{ChatSpec, Gateway, GatewayError};
use crate::instrument::ratio_letter::RatioLetterInstrument;
use crate::instrument::Instrument;
use crate::ontology::{Attribute, Entity};
use crate::record::AcquisitionMode;
use serde::{Deserialize, Serialize};

/// One model's measured logprob capability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbeReport {
    /// Model slug probed.
    pub model: String,
    /// Whether the provider returned any logprobs at all.
    pub logprobs_returned: bool,
    /// Maximum top-k depth observed across positions (0 when none).
    pub top_k_depth: usize,
    /// Whether the answer token position could be located and parsed.
    pub answer_position_found: bool,
    /// Probability mass visible at the answer position (None when no
    /// logprob evidence was obtained).
    pub visible_mass: Option<f64>,
    /// Jensen–Shannon divergence (base 2) between the logprob PMF and the
    /// pooled empirical PMF over sampled runs. Low = the logprobs mean what
    /// they claim; None when either side is missing.
    pub sampled_agreement_jsd: Option<f64>,
    /// Number of sampled answers pooled into the empirical side.
    pub samples: u32,
    /// Total provider cost across all probe calls, nanodollars.
    pub cost_nanodollars: i64,
    /// True when any call's cost was estimated rather than provider-reported.
    pub cost_is_estimate: bool,
}

/// Probe one model: one logprob-mode call plus `samples` sampled calls.
pub async fn probe_model(
    gateway: &Gateway,
    model: &str,
    attribute: &Attribute,
    slot_a: &Entity,
    slot_b: &Entity,
    samples: u8,
) -> Result<ProbeReport, GatewayError> {
    let instrument = RatioLetterInstrument;
    let rendered = instrument.render(attribute, slot_a, slot_b);

    let mut cost: i64 = 0;
    let mut cost_is_estimate = false;

    // Logprob-mode call.
    let spec = ChatSpec {
        model: model.to_string(),
        system: rendered.system.clone(),
        user: rendered.user.clone(),
        temperature: 0.0,
        max_tokens: 8,
        top_logprobs: Some(20),
        response_format_json: false,
    };
    let outcome = gateway.chat(&spec).await?;
    cost += outcome.usage.cost_nanodollars;
    cost_is_estimate |= outcome.usage.cost_is_estimate;

    let logprobs_returned = outcome.answer_logprobs.is_some();
    let top_k_depth = outcome
        .answer_logprobs
        .as_deref()
        .map(|positions| positions.iter().map(|p| p.top.len()).max().unwrap_or(0))
        .unwrap_or(0);

    let mut answer_position_found = false;
    let mut visible_mass = None;
    let mut logprob_evidence: Option<AnswerEvidence> = None;
    if let Ok(parsed) = instrument.parse(&outcome.content, outcome.answer_logprobs.as_deref()) {
        if parsed.mode == AcquisitionMode::Logprob {
            answer_position_found = true;
            visible_mass = Some(parsed.health.visible_mass);
            logprob_evidence = Some(parsed.evidence);
        }
    }

    // Sampled calls at temperature 1.0, logprobs off.
    let mut atoms = Vec::new();
    for _ in 0..samples {
        let spec = ChatSpec {
            model: model.to_string(),
            system: rendered.system.clone(),
            user: rendered.user.clone(),
            temperature: 1.0,
            max_tokens: 8,
            top_logprobs: None,
            response_format_json: false,
        };
        let sample = gateway.chat(&spec).await?;
        cost += sample.usage.cost_nanodollars;
        cost_is_estimate |= sample.usage.cost_is_estimate;
        if let Ok(parsed) = instrument.parse(&sample.content, None) {
            // Sampled evidence is a point mass; harvest its atom.
            for point in parsed.evidence.support() {
                if point.p > 0.5 {
                    atoms.push(point.atom);
                }
            }
        }
    }

    let sampled_agreement_jsd = match (&logprob_evidence, atoms.is_empty()) {
        (Some(lp), false) => evidence_from_resamples(&atoms)
            .ok()
            .map(|emp| jsd(lp, &emp)),
        _ => None,
    };

    Ok(ProbeReport {
        model: model.to_string(),
        logprobs_returned,
        top_k_depth,
        answer_position_found,
        visible_mass,
        sampled_agreement_jsd,
        samples: atoms.len() as u32,
        cost_nanodollars: cost,
        cost_is_estimate,
    })
}

/// Write reports as JSONL.
pub fn write_reports_jsonl(
    path: impl AsRef<std::path::Path>,
    reports: &[ProbeReport],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::fs::File::create(path)?;
    for report in reports {
        let line = serde_json::to_string(report).map_err(std::io::Error::other)?;
        writeln!(out, "{line}")?;
    }
    Ok(())
}

/// Read reports from JSONL.
pub fn read_reports_jsonl(path: impl AsRef<std::path::Path>) -> std::io::Result<Vec<ProbeReport>> {
    let raw = std::fs::read_to_string(path)?;
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(std::io::Error::other))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    /// Fixture: deterministic 'D' answer; logprobs included when `rich`.
    #[derive(Clone, Copy)]
    struct Fixture {
        rich: bool,
    }

    impl Respond for Fixture {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            let mut response = json!({
                "choices": [{
                    "message": { "content": "D" },
                    "finish_reason": "stop"
                }],
                "usage": { "prompt_tokens": 30, "completion_tokens": 1 }
            });
            if self.rich {
                response["choices"][0]["logprobs"] = json!({
                    "content": [{
                        "token": "D",
                        "logprob": -0.105_360_5, // 0.9
                        "top_logprobs": [
                            { "token": "D", "logprob": -0.105_360_5 },
                            { "token": "A", "logprob": -2.995_732_3 }, // 0.05
                        ]
                    }]
                });
            }
            ResponseTemplate::new(200).set_body_json(response)
        }
    }

    async fn probe_fixture(rich: bool) -> ProbeReport {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(Fixture { rich })
            .mount(&server)
            .await;
        let gateway = Gateway::new("sk-test", format!("{}/", server.uri()));
        let attribute = Attribute::new("brightness", "which is brighter");
        let a = Entity::new("the sun");
        let b = Entity::new("a candle");
        probe_model(&gateway, "test/model", &attribute, &a, &b, 3)
            .await
            .expect("probe")
    }

    #[tokio::test]
    async fn rich_fixture_reports_logprob_capability_and_agreement() {
        let report = probe_fixture(true).await;
        assert!(report.logprobs_returned);
        assert_eq!(report.top_k_depth, 2);
        assert!(report.answer_position_found);
        let mass = report.visible_mass.expect("visible mass");
        assert!(mass > 0.9 && mass <= 1.0, "{mass}");
        // Sampled runs also answer 'D': divergence must be small — the
        // logprob PMF has 0.9 on the sampled point.
        let d = report.sampled_agreement_jsd.expect("jsd");
        assert!(d < 0.3, "{d}");
        assert_eq!(report.samples, 3);
    }

    #[tokio::test]
    async fn poor_fixture_reports_degradation() {
        let report = probe_fixture(false).await;
        assert!(!report.logprobs_returned);
        assert_eq!(report.top_k_depth, 0);
        assert!(!report.answer_position_found);
        assert_eq!(report.visible_mass, None);
        assert_eq!(report.sampled_agreement_jsd, None);
    }

    #[tokio::test]
    async fn reports_round_trip_jsonl() {
        let report = probe_fixture(true).await;
        let dir = std::env::temp_dir().join(format!("seriate-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("reports.jsonl");
        write_reports_jsonl(&path, std::slice::from_ref(&report)).unwrap();
        let back = read_reports_jsonl(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].model, report.model);
        assert_eq!(back[0].visible_mass, report.visible_mass);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
