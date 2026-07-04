//! Scalar control instrument: rate a SINGLE entity on an attribute with one
//! digit, `0`..`9`, single token, logprob-capable.
//!
//! This exists as a BASELINE/CONTROL, not a preferred elicitation path.
//! Absolute digit ratings don't carry the comparability guarantees a forced
//! comparison does: two different completions of "rate this 0-9" have no
//! shared anchor (what one model calls a 7 another may call a 4), and a
//! single entity's rating drifts with whatever else happened to be in
//! context. The pairwise instruments ([`super::ratio_letter`],
//! [`super::ordinal`]) and the k-wise instrument ([`super::kwise`]) are all
//! FORCED comparisons and are the paths that actually calibrate; scalar
//! exists so a run can report "how much does an absolute rating agree with
//! the comparative signal" as a receipt, not to replace the comparisons.
//!
//! Does not implement the pairwise [`super::Instrument`] trait: scalar has
//! one presented slot, not two.

use crate::gateway::TokenLogprob;
use crate::ontology::{Attribute, Entity};
use crate::record::{EvidenceHealth, InstrumentKind, ParserVersion};

use super::{
    candidate_mass, find_answer_position, merge_candidates, template_hash, trim_token,
    InstrumentError, RenderedPrompt,
};

/// Version string stamped on every record this parser produces.
pub const PARSER_VERSION: &str = "scalar/1";

/// The refusal token the prompt asks the model to use when it cannot rate
/// the entity at all.
pub const REFUSAL_TOKEN: &str = "!";

const SLOT_PLACEHOLDER: &str = "\u{2603}SLOT\u{2603}";

/// Single-entity absolute digit-rating control instrument.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarInstrument;

/// The result of parsing a scalar completion: a probability distribution
/// over which digit (0..=9) was meant, plus health. Like
/// [`super::kwise::KWiseOutcome`], there is no off-alphabet/abstain atom
/// here — `digit_pmf` is renormalized over recognized digits only, and is
/// legitimately empty when nothing recognizable was said.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarOutcome {
    /// `(digit, probability)`, digits in `0..=9`, summing to 1.0 when
    /// non-empty.
    pub digit_pmf: Vec<(u8, f64)>,
    /// Health facts about how the outcome was obtained.
    pub health: EvidenceHealth,
}

/// The ten tokens whose logprobs constitute the answer PMF.
pub fn scalar_alphabet() -> Vec<String> {
    (0u8..=9).map(|d| d.to_string()).collect()
}

fn digit(trimmed: &str) -> Option<u8> {
    let mut chars = trimmed.chars();
    match (chars.next(), chars.next()) {
        (Some(c @ '0'..='9'), None) => Some(c as u8 - b'0'),
        _ => None,
    }
}

fn is_refusal_token(trimmed: &str) -> bool {
    trimmed == REFUSAL_TOKEN
}

fn system_prompt() -> String {
    format!(
        "You are rating one entity on exactly one attribute. Use the attribute text as the only \
         criterion.\n\n\
         Rate how much of the attribute the entity has, on a scale from 0 (least) to 9 (most). \
         Answer with EXACTLY ONE character: the digit. That character must be the FIRST thing you \
         output — no punctuation, no explanation, nothing before or after it.\n\n\
         If the attribute is not applicable or the entity cannot be rated, answer with a single \
         '{REFUSAL_TOKEN}' instead of a digit.\n"
    )
}

fn render_user(attribute: &Attribute, body: &str) -> String {
    format!(
        "<attribute>\n<name>{}</name>\n<text>\n{}\n</text>\n</attribute>\n\n<entity>\n{}\n</entity>",
        attribute.name, attribute.text, body
    )
}

impl ScalarInstrument {
    /// Which [`InstrumentKind`] this is, for `JudgementRecord`.
    pub fn kind(&self) -> InstrumentKind {
        InstrumentKind::ScalarControl
    }

    /// The parser version to stamp on records this instrument produces.
    pub fn parser_version(&self) -> ParserVersion {
        ParserVersion(PARSER_VERSION.to_string())
    }

    /// Render the "rate this entity 0-9" prompt for `entity`.
    pub fn render(&self, attribute: &Attribute, entity: &Entity) -> RenderedPrompt {
        let system = system_prompt();
        let user = render_user(attribute, &entity.body);
        let skeleton = render_user(attribute, SLOT_PLACEHOLDER);
        RenderedPrompt {
            template: template_hash(&system, &skeleton),
            system,
            user,
            response_format_json: false,
            answer_alphabet: scalar_alphabet(),
        }
    }

    /// Parse a scalar completion.
    pub fn parse(
        &self,
        content: &str,
        answer_logprobs: Option<&[TokenLogprob]>,
    ) -> Result<ScalarOutcome, InstrumentError> {
        match answer_logprobs {
            Some(positions) if !positions.is_empty() => parse_logprob(positions),
            _ => parse_sampled(content),
        }
    }
}

fn parse_logprob(positions: &[TokenLogprob]) -> Result<ScalarOutcome, InstrumentError> {
    let position = find_answer_position(positions, |t| digit(t).is_some() || is_refusal_token(t))
        .ok_or(InstrumentError::Unparseable)?;
    let chosen = trim_token(&position.token);
    let refused = is_refusal_token(chosen);
    let parsed_cleanly = digit(chosen).is_some();

    let candidates = merge_candidates(position);
    let visible_mass = candidate_mass(&candidates);

    let mut raw = [0.0f64; 10];
    for (tok, logprob) in &candidates {
        if let Some(d) = digit(trim_token(tok)) {
            raw[d as usize] += logprob.exp();
        }
    }
    let recognized: f64 = raw.iter().sum();
    let digit_pmf = if recognized > 0.0 {
        raw.iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .map(|(d, p)| (d as u8, p / recognized))
            .collect()
    } else {
        Vec::new()
    };

    Ok(ScalarOutcome {
        digit_pmf,
        health: EvidenceHealth {
            visible_mass,
            parsed_cleanly,
            refused,
        },
    })
}

fn parse_sampled(content: &str) -> Result<ScalarOutcome, InstrumentError> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err(InstrumentError::EmptyContent);
    }
    let refused = is_refusal_token(trimmed);
    let first_token = trimmed.split_whitespace().next().unwrap_or(trimmed);
    let digit_pmf = match digit(first_token) {
        Some(d) => vec![(d, 1.0)],
        None => Vec::new(),
    };
    let parsed_cleanly = !digit_pmf.is_empty();
    Ok(ScalarOutcome {
        digit_pmf,
        health: EvidenceHealth {
            visible_mass: 1.0,
            parsed_cleanly,
            refused,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attr() -> Attribute {
        Attribute::new("intensity", "how intense the entity is")
    }

    fn tl(token: &str, p: f64, alts: &[(&str, f64)]) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob: p.ln(),
            top: alts.iter().map(|(t, p)| (t.to_string(), p.ln())).collect(),
        }
    }

    #[test]
    fn render_is_deterministic_and_hides_the_slot_from_the_template_hash() {
        let inst = ScalarInstrument;
        let e1 = Entity::new("first entity body");
        let e2 = Entity::new("a totally different body");
        let r1 = inst.render(&attr(), &e1);
        assert_eq!(r1, inst.render(&attr(), &e1));
        let r2 = inst.render(&attr(), &e2);
        assert_eq!(
            r1.template, r2.template,
            "different entity, same attribute -> same template"
        );
        assert_ne!(r1.user, r2.user);
        assert_eq!(r1.answer_alphabet.len(), 10);
    }

    #[test]
    fn different_attribute_changes_the_template_hash() {
        let inst = ScalarInstrument;
        let e = Entity::new("x");
        let r1 = inst.render(&attr(), &e);
        let other = Attribute::new("clarity", "how clear it is");
        let r2 = inst.render(&other, &e);
        assert_ne!(r1.template, r2.template);
    }

    #[test]
    fn logprob_parse_builds_a_normalized_digit_pmf() {
        let position = tl("7", 0.6, &[("7", 0.6), ("8", 0.2), ("###", 0.1)]);
        let outcome = ScalarInstrument.parse("7", Some(&[position])).unwrap();
        let total: f64 = outcome.digit_pmf.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-9);
        let p = |d: u8| {
            outcome
                .digit_pmf
                .iter()
                .find(|(x, _)| *x == d)
                .map(|(_, p)| *p)
        };
        assert!((p(7).unwrap() - 0.75).abs() < 1e-9);
        assert!((p(8).unwrap() - 0.25).abs() < 1e-9);
        assert!((outcome.health.visible_mass - 0.9).abs() < 1e-9);
        assert!(outcome.health.parsed_cleanly);
    }

    #[test]
    fn logprob_parse_refusal_with_no_recognized_digits_yields_empty_pmf() {
        let position = tl("!", 1.0, &[("!", 1.0)]);
        let outcome = ScalarInstrument.parse("!", Some(&[position])).unwrap();
        assert!(outcome.digit_pmf.is_empty());
        assert!(outcome.health.refused);
        assert!(!outcome.health.parsed_cleanly);
    }

    #[test]
    fn logprob_parse_errors_when_no_position_matches_at_all() {
        let position = tl("hello", 0.9, &[]);
        let err = ScalarInstrument
            .parse("hello", Some(&[position]))
            .unwrap_err();
        assert_eq!(err, InstrumentError::Unparseable);
    }

    #[test]
    fn sampled_fallback_reads_first_token_as_a_digit() {
        let outcome = ScalarInstrument.parse(" 3 trailing junk", None).unwrap();
        assert_eq!(outcome.digit_pmf, vec![(3, 1.0)]);
        assert!(outcome.health.parsed_cleanly);
    }

    #[test]
    fn sampled_fallback_empty_pmf_on_junk_or_refusal() {
        let junk = ScalarInstrument.parse("banana", None).unwrap();
        assert!(junk.digit_pmf.is_empty());
        assert!(!junk.health.parsed_cleanly);

        let refusal = ScalarInstrument.parse("!", None).unwrap();
        assert!(refusal.health.refused);
        assert!(refusal.digit_pmf.is_empty());
    }

    #[test]
    fn sampled_fallback_errors_on_empty_content() {
        assert_eq!(
            ScalarInstrument.parse("  ", None).unwrap_err(),
            InstrumentError::EmptyContent
        );
    }
}
