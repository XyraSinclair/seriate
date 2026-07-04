//! Adversarial integration battery over the assembled system:
//! gateway → instrument → evidence log → compiler, with a deterministic
//! wiremock provider. Every assertion is falsifiable by a plausible bug.

use serde_json::json;
use seriate::instrument::ratio_letter::RatioLetterInstrument;
use seriate::instrument::Instrument;
use seriate::{
    compile, evidence_from_resamples, jsd, AcquisitionMode, AnswerAtom, Attribute, ChatSpec, Cost,
    DecodeConfig, Entity, EvidenceLog, Gateway, JudgementRecord, PairKey, Presentation, Tolerances,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

// =========================================================================
// Fixture provider: single-letter answers with synthetic top-k logprobs.
// =========================================================================

/// Content-driven judge: scores entities by the count of '*' in their body;
/// answers with the correct letter for the PRESENTED order, plus a small
/// synthetic top-k (junk tokens included). Position-consistent by
/// construction (reads the actual slots from the prompt).
#[derive(Clone, Copy)]
struct StarJudge {
    /// When true, ignore content and always favor presented slot A —
    /// pure position bias, for the counterbalance-conflict test.
    position_biased: bool,
    /// When true, respond without any logprobs field.
    no_logprobs: bool,
}

fn extract_between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let i = s.find(start)? + start.len();
    let rest = &s[i..];
    let j = rest.find(end)?;
    Some(&rest[..j])
}

fn stars(s: &str) -> i64 {
    s.chars().filter(|&c| c == '*').count() as i64
}

impl Respond for StarJudge {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap_or_default();
        let user = body["messages"]
            .as_array()
            .and_then(|m| {
                m.iter()
                    .find(|x| x["role"] == "user")
                    .and_then(|x| x["content"].as_str())
            })
            .unwrap_or("")
            .to_string();

        // The ratio-letter prompt wraps slot bodies in <entity_A>/<entity_B>.
        let a = extract_between(&user, "<entity_A>", "</entity_A>").unwrap_or("");
        let b = extract_between(&user, "<entity_B>", "</entity_B>").unwrap_or("");
        let (delta, letter) = if self.position_biased {
            (1, 'D')
        } else {
            let d = stars(a) - stars(b);
            let letter = match d {
                0 => 'A',
                1..=2 => 'D',   // A wins, bucket 3
                3.. => 'H',     // A wins, bucket 7
                -2..=-1 => 'd', // B wins, bucket 3
                _ => 'h',       // B wins, bucket 7
            };
            (d, letter)
        };
        let _ = delta;

        let logprobs = json!({
            "content": [{
                "token": letter.to_string(),
                "logprob": -0.223_143_5, // p = 0.8
                "top_logprobs": [
                    { "token": letter.to_string(), "logprob": -0.223_143_5 },  // 0.8
                    { "token": "A", "logprob": -2.207_274_9 },                  // 0.11 parity
                    { "token": "The", "logprob": -2.995_732_3 },                // 0.05 junk
                ]
            }]
        });

        let mut response = json!({
            "choices": [{
                "message": { "content": letter.to_string() },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 40, "completion_tokens": 1 }
        });
        if !self.no_logprobs {
            response["choices"][0]["logprobs"] = logprobs;
        }
        ResponseTemplate::new(200).set_body_json(response)
    }
}

async fn start_judge(judge: StarJudge) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(judge)
        .mount(&server)
        .await;
    server
}

fn gateway_for(server: &MockServer) -> Gateway {
    Gateway::new("sk-test", format!("{}/", server.uri()))
}

/// Run one full annotate step for a presented pair through the real
/// gateway + instrument + log path; returns the stored judgement id.
async fn annotate_pair(
    gateway: &Gateway,
    log: &EvidenceLog,
    attribute: &Attribute,
    slot_a: &Entity,
    slot_b: &Entity,
) -> seriate::JudgementId {
    let instrument = RatioLetterInstrument;
    let rendered = instrument.render(attribute, slot_a, slot_b);
    let spec = ChatSpec {
        model: "test/judge".into(),
        system: rendered.system.clone(),
        user: rendered.user.clone(),
        temperature: 0.0,
        max_tokens: 4,
        top_logprobs: Some(20),
        response_format_json: false,
    };
    let outcome = gateway.chat(&spec).await.expect("chat");
    let parsed = instrument
        .parse(&outcome.content, outcome.answer_logprobs.as_deref())
        .expect("parse");
    log.insert_capture(&outcome.capture).expect("capture");
    let record = JudgementRecord::new(
        instrument.kind(),
        parsed.mode,
        attribute.id.clone(),
        Presentation {
            slot_a: slot_a.id.clone(),
            slot_b: slot_b.id.clone(),
        },
        rendered.template.clone(),
        instrument.parser_version(),
        "test/judge".into(),
        DecodeConfig {
            temperature: 0.0,
            max_tokens: 4,
            top_logprobs: Some(20),
        },
        outcome.capture.id.clone(),
        parsed.evidence,
        parsed.health,
        Cost {
            input_tokens: outcome.usage.input_tokens,
            output_tokens: outcome.usage.output_tokens,
            nanodollars: outcome.usage.cost_nanodollars,
            is_estimate: outcome.usage.cost_is_estimate,
        },
        1_700_000_000_000,
    );
    log.insert_judgement(&record).expect("judgement");
    record.id
}

fn roster() -> Vec<Entity> {
    vec![
        Entity::new("alpha ****"),
        Entity::new("bravo ***"),
        Entity::new("charlie **"),
        Entity::new("delta *"),
    ]
}

fn seed_log(log: &EvidenceLog, entities: &[Entity], attribute: &Attribute) {
    for e in entities {
        log.insert_entity(e).unwrap();
    }
    log.insert_attribute(attribute).unwrap();
}

// =========================================================================
// 1. End-to-end provenance: every number traceable to raw bytes.
// =========================================================================

#[tokio::test]
async fn provenance_chain_terminates_in_raw_bytes_for_every_judgement() {
    let server = start_judge(StarJudge {
        position_biased: false,
        no_logprobs: false,
    })
    .await;
    let gateway = gateway_for(&server);
    let log = EvidenceLog::open_in_memory().unwrap();
    let entities = roster();
    let attribute = Attribute::new("brightness", "count of stars, more is brighter");
    seed_log(&log, &entities, &attribute);

    let mut ids = Vec::new();
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            // Counterbalanced: both presentation orders.
            ids.push(annotate_pair(&gateway, &log, &attribute, &entities[i], &entities[j]).await);
            ids.push(annotate_pair(&gateway, &log, &attribute, &entities[j], &entities[i]).await);
        }
    }
    assert_eq!(ids.len(), 12, "6 pairs x 2 orders");

    for id in &ids {
        let chain = log.provenance(&id.0 .0).expect("provenance walk");
        assert!(chain.judgement.verify_id(), "record integrity");
        assert!(chain.capture.verify(), "capture bytes hash to capture id");
        assert_eq!(chain.judgement.capture, chain.capture.id);
        assert_eq!(chain.attribute.id, attribute.id);
        assert_eq!(chain.entities.len(), 2);
        // The chain's raw bytes really are provider bytes: parseable JSON
        // with the choices shape.
        let raw: serde_json::Value = serde_json::from_str(&chain.capture.raw).unwrap();
        assert!(raw["choices"].is_array());
    }
}

// =========================================================================
// 2. Logprob honesty: junk mass accounted, never dropped.
// =========================================================================

#[tokio::test]
async fn junk_topk_mass_is_split_between_off_alphabet_and_abstain() {
    let server = start_judge(StarJudge {
        position_biased: false,
        no_logprobs: false,
    })
    .await;
    let gateway = gateway_for(&server);
    let log = EvidenceLog::open_in_memory().unwrap();
    let entities = roster();
    let attribute = Attribute::new("brightness", "count of stars");
    seed_log(&log, &entities, &attribute);

    let id = annotate_pair(&gateway, &log, &attribute, &entities[0], &entities[3]).await;
    let chain = log.provenance(&id.0 .0).unwrap();
    let ev = &chain.judgement.evidence;

    // Fixture: 0.8 letter + 0.11 parity parsed; 0.05 junk ('The') visible;
    // 0.04 never shown. Informative = 0.91, OffAlphabet = 0.05, Abstain = 0.04.
    assert!((ev.informative_mass() - 0.91).abs() < 1e-3, "{ev:?}");
    assert!((ev.off_alphabet_mass() - 0.05).abs() < 1e-3);
    assert!((ev.abstain_mass() - 0.04).abs() < 1e-3);
    let total = ev.informative_mass() + ev.off_alphabet_mass() + ev.abstain_mass();
    assert!((total - 1.0).abs() < 1e-9, "every unit of mass accounted");
    assert_eq!(chain.judgement.mode, AcquisitionMode::Logprob);
}

// =========================================================================
// 3. Counterbalance exactness vs position bias.
// =========================================================================

#[tokio::test]
async fn counterbalanced_orders_reflect_for_honest_judge_and_conflict_for_biased() {
    // Honest judge: the two presentation orders yield evidences that are
    // exact reflections (content-driven answer).
    let honest = start_judge(StarJudge {
        position_biased: false,
        no_logprobs: false,
    })
    .await;
    let gateway = gateway_for(&honest);
    let log = EvidenceLog::open_in_memory().unwrap();
    let entities = roster();
    let attribute = Attribute::new("brightness", "count of stars");
    seed_log(&log, &entities, &attribute);

    let id_ab = annotate_pair(&gateway, &log, &attribute, &entities[0], &entities[2]).await;
    let id_ba = annotate_pair(&gateway, &log, &attribute, &entities[2], &entities[0]).await;
    let ev_ab = log.provenance(&id_ab.0 .0).unwrap().judgement.evidence;
    let ev_ba = log.provenance(&id_ba.0 .0).unwrap().judgement.evidence;
    assert_eq!(
        ev_ab.reflected().support(),
        ev_ba.support(),
        "honest judge: order swap = exact reflection"
    );

    // Biased judge: always favors presented slot A -> after
    // canonicalization the two orders CONFLICT, and the compiled posterior
    // is wider than the honest one.
    let biased = start_judge(StarJudge {
        position_biased: true,
        no_logprobs: false,
    })
    .await;
    let gateway_b = gateway_for(&biased);
    let log_b = EvidenceLog::open_in_memory().unwrap();
    seed_log(&log_b, &entities, &attribute);
    let idb_ab = annotate_pair(&gateway_b, &log_b, &attribute, &entities[0], &entities[2]).await;
    let idb_ba = annotate_pair(&gateway_b, &log_b, &attribute, &entities[2], &entities[0]).await;
    let evb_ab = log_b.provenance(&idb_ab.0 .0).unwrap().judgement.evidence;
    let evb_ba = log_b.provenance(&idb_ba.0 .0).unwrap().judgement.evidence;
    assert_ne!(
        evb_ab.reflected().support(),
        evb_ba.support(),
        "pure position bias: orders disagree after reflection"
    );

    // Compile both two-record sets over the same two entities; the biased
    // pair's mean log-ratio contributions cancel, honest ones reinforce.
    let two = vec![entities[0].clone(), entities[2].clone()];
    let honest_records = vec![
        log.provenance(&id_ab.0 .0).unwrap().judgement,
        log.provenance(&id_ba.0 .0).unwrap().judgement,
    ];
    let biased_records = vec![
        log_b.provenance(&idb_ab.0 .0).unwrap().judgement,
        log_b.provenance(&idb_ba.0 .0).unwrap().judgement,
    ];
    let tol = Tolerances::default();
    let honest_post = compile(&honest_records, &two, &tol).unwrap();
    let biased_post = compile(&biased_records, &two, &tol).unwrap();
    let gap = |p: &seriate::CompiledPosterior| {
        (p.entities[0].latent_mean - p.entities[1].latent_mean).abs()
    };
    assert!(
        gap(&honest_post) > 10.0 * gap(&biased_post),
        "position-biased evidence must cancel under counterbalancing: honest gap {} vs biased gap {}",
        gap(&honest_post),
        gap(&biased_post)
    );
}

// =========================================================================
// 4. Full pipeline planted-order recovery.
// =========================================================================

#[tokio::test]
async fn full_pipeline_recovers_planted_star_order() {
    let server = start_judge(StarJudge {
        position_biased: false,
        no_logprobs: false,
    })
    .await;
    let gateway = gateway_for(&server);
    let log = EvidenceLog::open_in_memory().unwrap();
    let entities = roster();
    let attribute = Attribute::new("brightness", "count of stars");
    seed_log(&log, &entities, &attribute);

    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            annotate_pair(&gateway, &log, &attribute, &entities[i], &entities[j]).await;
            annotate_pair(&gateway, &log, &attribute, &entities[j], &entities[i]).await;
        }
    }
    let records = log.judgements_for(&attribute.id).unwrap();
    assert_eq!(records.len(), 12);
    let posterior = compile(&records, &entities, &Tolerances::default()).unwrap();

    // Planted order: alpha(4) > bravo(3) > charlie(2) > delta(1).
    let means: Vec<f64> = posterior.entities.iter().map(|e| e.latent_mean).collect();
    assert!(
        means[0] > means[1] && means[1] > means[2] && means[2] > means[3],
        "monotone latents: {means:?}"
    );
    assert_eq!(posterior.components.len(), 1, "fully connected");
    assert_eq!(posterior.records_used, 12);
    let p = posterior.p_higher(0, 3).expect("same component");
    assert!(p > 0.9, "alpha over delta with confidence: {p}");
}

// =========================================================================
// 5. Tamper evidence: fail closed.
// =========================================================================

#[tokio::test]
async fn tampered_judgement_row_fails_provenance_and_export() {
    let server = start_judge(StarJudge {
        position_biased: false,
        no_logprobs: false,
    })
    .await;
    let gateway = gateway_for(&server);
    let dir = std::env::temp_dir().join(format!("seriate-tamper-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("tamper.sqlite");
    let _ = std::fs::remove_file(&db_path);
    let log = EvidenceLog::open(&db_path).unwrap();
    let entities = roster();
    let attribute = Attribute::new("brightness", "count of stars");
    seed_log(&log, &entities, &attribute);
    let id = annotate_pair(&gateway, &log, &attribute, &entities[0], &entities[1]).await;
    drop(log);

    // Mutate the stored judgement json directly (the attacker's move; the
    // log itself never updates).
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute(
        "UPDATE judgements SET json = replace(json, 'test/judge', 'evil/judge')",
        [],
    )
    .unwrap();
    drop(conn);

    let log = EvidenceLog::open(&db_path).unwrap();
    let walk = log.provenance(&id.0 .0);
    assert!(walk.is_err(), "tampered record must fail provenance");
    let export_path = dir.join("export.jsonl");
    // Export may write, but a fresh import of the tampered line must fail.
    if log.export_jsonl(&export_path).is_ok() {
        let fresh = EvidenceLog::open_in_memory().unwrap();
        assert!(
            fresh.import_jsonl(&export_path).is_err(),
            "tampered line must be rejected on import"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================================
// 6. Degradation loudness: logprobs absent -> Sampled mode, visible.
// =========================================================================

#[tokio::test]
async fn missing_logprobs_degrade_loudly_to_sampled_mode() {
    let server = start_judge(StarJudge {
        position_biased: false,
        no_logprobs: true,
    })
    .await;
    let gateway = gateway_for(&server);
    let log = EvidenceLog::open_in_memory().unwrap();
    let entities = roster();
    let attribute = Attribute::new("brightness", "count of stars");
    seed_log(&log, &entities, &attribute);

    let id = annotate_pair(&gateway, &log, &attribute, &entities[0], &entities[3]).await;
    let record = log.provenance(&id.0 .0).unwrap().judgement;
    assert_eq!(record.mode, AcquisitionMode::Sampled);
    // Sampled evidence is a point mass on the letter answered.
    assert_eq!(record.evidence.support().len(), 1);
    assert!(record.health.parsed_cleanly);
}

// =========================================================================
// 7. Determinism: identical runs, identical claims.
// =========================================================================

#[tokio::test]
async fn identical_runs_produce_identical_captures_and_evidence() {
    let server = start_judge(StarJudge {
        position_biased: false,
        no_logprobs: false,
    })
    .await;
    let gateway = gateway_for(&server);
    let entities = roster();
    let attribute = Attribute::new("brightness", "count of stars");

    let log1 = EvidenceLog::open_in_memory().unwrap();
    seed_log(&log1, &entities, &attribute);
    let id1 = annotate_pair(&gateway, &log1, &attribute, &entities[0], &entities[1]).await;
    let r1 = log1.provenance(&id1.0 .0).unwrap().judgement;

    let log2 = EvidenceLog::open_in_memory().unwrap();
    seed_log(&log2, &entities, &attribute);
    let id2 = annotate_pair(&gateway, &log2, &attribute, &entities[0], &entities[1]).await;
    let r2 = log2.provenance(&id2.0 .0).unwrap().judgement;

    // Same fixture, same request -> identical raw bytes and identical
    // evidence. Capture ids DIFFER by design: a capture is an event (its id
    // covers created_at_ms), so the same exchange re-observed at a new time
    // is a new capture — and therefore a new judgement record. What must be
    // identical is the CLAIM CONTENT, not the event identity.
    let c1 = log1.capture(&r1.capture).unwrap().unwrap();
    let c2 = log2.capture(&r2.capture).unwrap().unwrap();
    assert_eq!(c1.raw, c2.raw, "identical provider bytes");
    assert_eq!(c1.request_fingerprint, c2.request_fingerprint);
    assert_eq!(r1.evidence, r2.evidence, "identical parsed claim");
    assert_eq!(r1.template, r2.template);
    assert_eq!(r1.presentation, r2.presentation);
}

// =========================================================================
// 8. Sanity: canonical PairKey + jsd smoke over the fixture split.
// =========================================================================

#[tokio::test]
async fn fixture_pmf_agrees_with_its_own_sampled_point() {
    let server = start_judge(StarJudge {
        position_biased: false,
        no_logprobs: false,
    })
    .await;
    let gateway = gateway_for(&server);
    let log = EvidenceLog::open_in_memory().unwrap();
    let entities = roster();
    let attribute = Attribute::new("brightness", "count of stars");
    seed_log(&log, &entities, &attribute);

    let id = annotate_pair(&gateway, &log, &attribute, &entities[0], &entities[3]).await;
    let record = log.provenance(&id.0 .0).unwrap().judgement;
    // The sampled point (what the model actually said: 'H') must be the
    // argmax of the logprob PMF — divergence between the PMF and its own
    // sample should be far from maximal.
    let sampled = evidence_from_resamples(&[AnswerAtom::A(7)]).unwrap();
    let d = jsd(&record.evidence, &sampled);
    assert!(d < 0.5, "logprob PMF and its own sample roughly agree: {d}");
    // PairKey canonicality is preserved through storage.
    let pair = PairKey::new(&entities[0].id, &entities[3].id);
    assert_eq!(record.presentation.pair_key(), pair);
}
