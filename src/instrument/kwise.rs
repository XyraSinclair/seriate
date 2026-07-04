//! K-wise instrument: "which of these k entities has the MOST of the
//! attribute", single-token answer over item letters `A..T` (k in 2..=20).
//! One call replaces up to `C(20,2) = 190` pairwise calls at the cost of a
//! coarser observation: it names a winner, not a full ranking, and it
//! shares one presentation context across all k entities (positional and
//! anchoring effects are a live risk the caller must counterbalance by
//! permuting slot order across repeated draws).
//!
//! K-wise does NOT implement the pairwise [`super::Instrument`] trait —
//! forcing a k-ary question through a two-slot interface would either lose
//! entities or fake a two-slot signature that never matches its render
//! arity. It exposes its own small surface instead, plus
//! [`lower_to_pairwise`] to project a k-wise outcome down into the same
//! `AnswerEvidence` currency the pairwise instruments produce, for callers
//! that want to feed everything through one compiler.

use crate::atom::AnswerAtom;
use crate::evidence::{AnswerEvidence, AtomProb, EvidenceError, PmfCompleteness};
use crate::gateway::TokenLogprob;
use crate::ontology::{Attribute, Entity};
use crate::record::{EvidenceHealth, InstrumentKind, ParserVersion};

use super::{
    candidate_mass, find_answer_position, merge_candidates, template_hash, trim_token,
    InstrumentError, RenderedPrompt,
};

/// Version string stamped on every record this parser produces.
pub const PARSER_VERSION: &str = "kwise/1";

/// The refusal token the prompt asks the model to use when it cannot pick a
/// winner at all.
pub const REFUSAL_TOKEN: &str = "!";

/// Minimum presented entities (below this, "most" is meaningless).
pub const MIN_K: usize = 2;
/// Maximum presented entities: `'A'..='T'` is 20 letters.
pub const MAX_K: usize = 20;

/// The fixed ordinal ladder bucket [`lower_to_pairwise`] pins winner-beats-
/// loser judgements to, matching [`super::ordinal::FIXED_BUCKET`]: a k-wise
/// winner is direction-only information, never a magnitude claim.
pub const FIXED_BUCKET: u8 = super::ordinal::FIXED_BUCKET;
/// Named tolerance below which a residual "loser" probability mass is
/// treated as exactly zero, so we don't emit `Truncated{unresolved: 1e-16}`
/// noise from float roundoff.
const RESIDUAL_MASS_EPS: f64 = 1e-12;

/// "Which of these k has the most" instrument.
#[derive(Clone, Copy, Debug, Default)]
pub struct KWiseInstrument;

/// The result of parsing a k-wise completion: a probability distribution
/// over WHICH presented index won, plus health. Unlike [`AnswerEvidence`],
/// there is no off-alphabet/abstain atom here — `winner_pmf` is renormalized
/// over recognized item letters only (see `health.visible_mass` for how
/// much of the raw mass that was), and is legitimately empty when nothing
/// recognizable was said.
#[derive(Clone, Debug, PartialEq)]
pub struct KWiseOutcome {
    /// `(presented index, probability)`, indices in `0..k`, summing to 1.0
    /// when non-empty.
    pub winner_pmf: Vec<(usize, f64)>,
    /// Health facts about how the outcome was obtained.
    pub health: EvidenceHealth,
}

fn item_letters(k: usize) -> Vec<String> {
    (0..k)
        .map(|i| ((b'A' + i as u8) as char).to_string())
        .collect()
}

fn item_index(trimmed: &str, k: usize) -> Option<usize> {
    let mut chars = trimmed.chars();
    match (chars.next(), chars.next()) {
        (Some(c @ 'A'..='T'), None) => {
            let i = (c as u8 - b'A') as usize;
            (i < k).then_some(i)
        }
        _ => None,
    }
}

fn is_refusal_token(trimmed: &str) -> bool {
    trimmed == REFUSAL_TOKEN
}

fn system_prompt(k: usize) -> String {
    let letters = item_letters(k).join(", ");
    format!(
        "You are ranking {k} entities, labeled {letters}, on exactly one attribute. Use the \
         attribute text as the only criterion.\n\n\
         Identify which ONE entity has the MOST of the attribute. Answer with EXACTLY ONE \
         character: the label of that entity. That character must be the FIRST thing you \
         output — no punctuation, no explanation, nothing before or after it.\n\n\
         If the attribute is not applicable or no single winner can be identified, answer with a \
         single '{REFUSAL_TOKEN}' instead of a letter.\n"
    )
}

fn render_user<'a>(attribute: &Attribute, bodies: impl Iterator<Item = &'a str>) -> String {
    let mut out = format!(
        "<attribute>\n<name>{}</name>\n<text>\n{}\n</text>\n</attribute>\n",
        attribute.name, attribute.text
    );
    for (i, body) in bodies.enumerate() {
        let letter = (b'A' + i as u8) as char;
        out.push_str(&format!("\n<entity_{letter}>\n{body}\n</entity_{letter}>"));
    }
    out
}

impl KWiseInstrument {
    /// Which [`InstrumentKind`] this is, for `JudgementRecord`.
    pub fn kind(&self) -> InstrumentKind {
        InstrumentKind::OrdinalKWise
    }

    /// The parser version to stamp on records this instrument produces.
    pub fn parser_version(&self) -> ParserVersion {
        ParserVersion(PARSER_VERSION.to_string())
    }

    /// Render the "which has the most" prompt for `presented`, labeled in
    /// slice order.
    ///
    /// # Panics
    /// Panics if `presented.len()` is outside `MIN_K..=MAX_K`. `k` is a
    /// harness-controlled knob, not foreign provider data, so this is a
    /// documented caller precondition rather than a `Result`.
    pub fn render(&self, attribute: &Attribute, presented: &[&Entity]) -> RenderedPrompt {
        let k = presented.len();
        assert!(
            (MIN_K..=MAX_K).contains(&k),
            "kwise requires {MIN_K}..={MAX_K} presented entities, got {k}"
        );
        let system = system_prompt(k);
        let user = render_user(attribute, presented.iter().map(|e| e.body.as_str()));
        let skeleton_bodies: Vec<String> = (0..k)
            .map(|i| format!("\u{2603}SLOT_{i}\u{2603}"))
            .collect();
        let skeleton = render_user(attribute, skeleton_bodies.iter().map(String::as_str));
        RenderedPrompt {
            template: template_hash(&system, &skeleton),
            system,
            user,
            response_format_json: false,
            answer_alphabet: item_letters(k),
        }
    }

    /// Parse a k-wise completion. `k` must match the `presented.len()` used
    /// to render the prompt this completion answers.
    pub fn parse(
        &self,
        k: usize,
        content: &str,
        answer_logprobs: Option<&[TokenLogprob]>,
    ) -> Result<KWiseOutcome, InstrumentError> {
        if !(MIN_K..=MAX_K).contains(&k) {
            return Err(InstrumentError::InvalidPresentationCount {
                min: MIN_K,
                max: MAX_K,
                got: k,
            });
        }
        match answer_logprobs {
            Some(positions) if !positions.is_empty() => parse_logprob(k, positions),
            _ => parse_sampled(k, content),
        }
    }
}

fn parse_logprob(k: usize, positions: &[TokenLogprob]) -> Result<KWiseOutcome, InstrumentError> {
    let position = find_answer_position(positions, |t| {
        item_index(t, k).is_some() || is_refusal_token(t)
    })
    .ok_or(InstrumentError::Unparseable)?;
    let chosen = trim_token(&position.token);
    let refused = is_refusal_token(chosen);
    let parsed_cleanly = item_index(chosen, k).is_some();

    let candidates = merge_candidates(position);
    let visible_mass = candidate_mass(&candidates);

    let mut raw = vec![0.0f64; k];
    for (tok, logprob) in &candidates {
        if let Some(i) = item_index(trim_token(tok), k) {
            raw[i] += logprob.exp();
        }
    }
    let recognized: f64 = raw.iter().sum();
    let winner_pmf = if recognized > 0.0 {
        raw.iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .map(|(i, p)| (i, p / recognized))
            .collect()
    } else {
        Vec::new()
    };

    Ok(KWiseOutcome {
        winner_pmf,
        health: EvidenceHealth {
            visible_mass,
            parsed_cleanly,
            refused,
        },
    })
}

fn parse_sampled(k: usize, content: &str) -> Result<KWiseOutcome, InstrumentError> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err(InstrumentError::EmptyContent);
    }
    let refused = is_refusal_token(trimmed);
    let first_token = trimmed.split_whitespace().next().unwrap_or(trimmed);
    let winner_pmf = match item_index(first_token, k) {
        Some(i) => vec![(i, 1.0)],
        None => Vec::new(),
    };
    Ok(KWiseOutcome {
        winner_pmf: winner_pmf.clone(),
        health: EvidenceHealth {
            visible_mass: 1.0,
            parsed_cleanly: !winner_pmf.is_empty(),
            refused,
        },
    })
}

/// Project a k-wise outcome down to pairwise evidence: for every winner
/// candidate `w` with `winner_pmf` probability `p_w > 0`, and every other
/// presented index `l`, emit `(w, l, evidence)` where `evidence` puts `p_w`
/// on "w beats l" at [`FIXED_BUCKET`] and the residual `1 - p_w` on
/// [`AnswerAtom::Abstain`].
///
/// This is a real approximation, not a derivation, and it is documented as
/// one on purpose:
/// - it discards ALL information about how the non-winning entities compare
///   to EACH OTHER (a k-wise draw says nothing about `l1` vs `l2`);
/// - it discards magnitude: "beats" is always the same modest fixed ratio,
///   never "beats by a lot";
/// - when `winner_pmf` has multiple nonzero entries (the model's logprobs
///   were split across candidates), each contributes its OWN weighted
///   pairwise observation for the same underlying pair — e.g. both `(0, 1,
///   ev_favoring_0)` and `(1, 0, ev_favoring_1)` can appear. This is
///   intentional: the compiler treats them as independent weighted
///   observations of the same latent pair rather than forcing a premature
///   point estimate here.
pub fn lower_to_pairwise(
    outcome: &KWiseOutcome,
    k: usize,
) -> Result<Vec<(usize, usize, AnswerEvidence)>, EvidenceError> {
    let mut out = Vec::new();
    for &(w, p_w) in &outcome.winner_pmf {
        if w >= k || p_w <= 0.0 {
            continue;
        }
        let p_w = p_w.clamp(0.0, 1.0);
        let residual = (1.0 - p_w).max(0.0);
        for l in 0..k {
            if l == w {
                continue;
            }
            let mut support = vec![AtomProb {
                atom: AnswerAtom::A(FIXED_BUCKET),
                p: p_w,
            }];
            let completeness = if residual > RESIDUAL_MASS_EPS {
                support.push(AtomProb {
                    atom: AnswerAtom::Abstain,
                    p: residual,
                });
                PmfCompleteness::Truncated {
                    shown_mass: p_w,
                    unresolved_mass: residual,
                }
            } else {
                PmfCompleteness::Complete
            };
            out.push((w, l, AnswerEvidence::new(support, completeness)?));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::PmfCompleteness;

    fn attr() -> Attribute {
        Attribute::new("charisma", "how charismatic the entity is")
    }

    fn entities(n: usize) -> Vec<Entity> {
        (0..n)
            .map(|i| Entity::new(format!("entity number {i}")))
            .collect()
    }

    fn tl(token: &str, p: f64, alts: &[(&str, f64)]) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob: p.ln(),
            top: alts.iter().map(|(t, p)| (t.to_string(), p.ln())).collect(),
        }
    }

    #[test]
    fn render_is_deterministic_and_hides_slots_from_the_template_hash() {
        let inst = KWiseInstrument;
        let e1 = entities(4);
        let refs1: Vec<&Entity> = e1.iter().collect();
        let r1 = inst.render(&attr(), &refs1);
        assert_eq!(r1, inst.render(&attr(), &refs1));

        let e2 = [
            Entity::new("totally different a"),
            Entity::new("totally different b"),
            Entity::new("totally different c"),
            Entity::new("totally different d"),
        ];
        let refs2: Vec<&Entity> = e2.iter().collect();
        let r2 = inst.render(&attr(), &refs2);
        assert_eq!(
            r1.template, r2.template,
            "same k, different entities -> same template"
        );
        assert_eq!(r1.answer_alphabet, vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn different_k_changes_the_template_hash() {
        let inst = KWiseInstrument;
        let e3 = entities(3);
        let e4 = entities(4);
        let refs3: Vec<&Entity> = e3.iter().collect();
        let refs4: Vec<&Entity> = e4.iter().collect();
        assert_ne!(
            inst.render(&attr(), &refs3).template,
            inst.render(&attr(), &refs4).template
        );
    }

    #[test]
    #[should_panic(expected = "kwise requires")]
    fn render_panics_below_min_k() {
        let e1 = entities(1);
        let refs1: Vec<&Entity> = e1.iter().collect();
        KWiseInstrument.render(&attr(), &refs1);
    }

    #[test]
    fn parse_rejects_invalid_k_structurally() {
        let err = KWiseInstrument.parse(1, "A", None).unwrap_err();
        assert_eq!(
            err,
            InstrumentError::InvalidPresentationCount {
                min: MIN_K,
                max: MAX_K,
                got: 1
            }
        );
    }

    #[test]
    fn logprob_parse_builds_a_normalized_winner_pmf_over_recognized_letters() {
        // 5 entities; chosen "C" at 60%, alt "A" at 20%, junk "###" at 10%.
        let position = tl("C", 0.6, &[("C", 0.6), ("A", 0.2), ("###", 0.1)]);
        let outcome = KWiseInstrument.parse(5, "C", Some(&[position])).unwrap();
        let total: f64 = outcome.winner_pmf.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "renormalized over recognized letters"
        );
        let p = |i: usize| {
            outcome
                .winner_pmf
                .iter()
                .find(|(idx, _)| *idx == i)
                .map(|(_, p)| *p)
        };
        assert!((p(2).unwrap() - 0.75).abs() < 1e-9, "C is index 2, 0.6/0.8");
        assert!((p(0).unwrap() - 0.25).abs() < 1e-9, "A is index 0, 0.2/0.8");
        assert!((outcome.health.visible_mass - 0.9).abs() < 1e-9);
        assert!(outcome.health.parsed_cleanly);
    }

    #[test]
    fn logprob_parse_ignores_letters_outside_the_presented_range() {
        // k=3 (A,B,C only); "T" is a valid kwise letter in general but out
        // of range here and must not be counted.
        let position = tl("A", 0.5, &[("A", 0.5), ("T", 0.5)]);
        let outcome = KWiseInstrument.parse(3, "A", Some(&[position])).unwrap();
        assert_eq!(outcome.winner_pmf, vec![(0, 1.0)]);
    }

    #[test]
    fn logprob_parse_refusal_with_no_recognized_letters_yields_empty_pmf() {
        let position = tl("!", 1.0, &[("!", 1.0)]);
        let outcome = KWiseInstrument.parse(4, "!", Some(&[position])).unwrap();
        assert!(outcome.winner_pmf.is_empty());
        assert!(outcome.health.refused);
        assert!(!outcome.health.parsed_cleanly);
    }

    #[test]
    fn logprob_parse_errors_when_no_position_matches_at_all() {
        let position = tl("hello", 0.9, &[]);
        let err = KWiseInstrument
            .parse(4, "hello", Some(&[position]))
            .unwrap_err();
        assert_eq!(err, InstrumentError::Unparseable);
    }

    #[test]
    fn sampled_fallback_reads_first_token_as_an_index() {
        let outcome = KWiseInstrument.parse(5, " D trailing", None).unwrap();
        assert_eq!(outcome.winner_pmf, vec![(3, 1.0)]);
        assert!(outcome.health.parsed_cleanly);
    }

    #[test]
    fn sampled_fallback_empty_pmf_on_junk_or_refusal() {
        let junk = KWiseInstrument.parse(5, "banana", None).unwrap();
        assert!(junk.winner_pmf.is_empty());
        assert!(!junk.health.parsed_cleanly);

        let refusal = KWiseInstrument.parse(5, "!", None).unwrap();
        assert!(refusal.health.refused);
        assert!(refusal.winner_pmf.is_empty());
    }

    #[test]
    fn sampled_fallback_errors_on_empty_content() {
        assert_eq!(
            KWiseInstrument.parse(5, "  ", None).unwrap_err(),
            InstrumentError::EmptyContent
        );
    }

    #[test]
    fn lowering_produces_winner_beats_every_loser_weighted_by_probability() {
        let outcome = KWiseOutcome {
            winner_pmf: vec![(1, 0.75), (0, 0.25)],
            health: EvidenceHealth {
                visible_mass: 1.0,
                parsed_cleanly: true,
                refused: false,
            },
        };
        let pairs = lower_to_pairwise(&outcome, 3).unwrap();
        // winner 1 beats {0, 2}; winner 0 beats {1, 2}: 4 pairs total.
        assert_eq!(pairs.len(), 4);
        let (w1_l2, _, ev) = pairs.iter().find(|(w, l, _)| *w == 1 && *l == 2).unwrap();
        assert_eq!(*w1_l2, 1);
        assert!((ev.p(AnswerAtom::A(FIXED_BUCKET)) - 0.75).abs() < 1e-9);
        assert!((ev.abstain_mass() - 0.25).abs() < 1e-9);
        match ev.completeness {
            PmfCompleteness::Truncated { .. } => {}
            other => panic!("expected Truncated, got {other:?}"),
        }
    }

    #[test]
    fn lowering_skips_zero_probability_and_out_of_range_winners() {
        let outcome = KWiseOutcome {
            winner_pmf: vec![(0, 1.0), (5, 0.3), (1, 0.0)],
            health: EvidenceHealth {
                visible_mass: 1.0,
                parsed_cleanly: true,
                refused: false,
            },
        };
        let pairs = lower_to_pairwise(&outcome, 3).unwrap();
        // Only winner index 0 is valid and nonzero: beats {1, 2}.
        assert_eq!(pairs.len(), 2);
        for (w, _, ev) in &pairs {
            assert_eq!(*w, 0);
            assert_eq!(ev.completeness, PmfCompleteness::Complete);
            assert_eq!(ev.p(AnswerAtom::A(FIXED_BUCKET)), 1.0);
        }
    }

    #[test]
    fn lowering_is_empty_for_an_empty_winner_pmf() {
        let outcome = KWiseOutcome {
            winner_pmf: vec![],
            health: EvidenceHealth {
                visible_mass: 1.0,
                parsed_cleanly: false,
                refused: false,
            },
        };
        assert!(lower_to_pairwise(&outcome, 4).unwrap().is_empty());
    }
}
