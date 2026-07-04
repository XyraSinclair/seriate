//! Instruments: the elicitation shapes that turn (attribute, entities) into
//! a rendered prompt, and a raw completion back into [`AnswerEvidence`].
//!
//! Four instruments, four honest trade-offs:
//! - [`ratio_letter`] — the flagship: one letter, 52-way alphabet, full
//!   ratio-ladder magnitude, logprob-native.
//! - [`ordinal`] — direction only, three-way alphabet, cheap and robust.
//! - [`kwise`] — one shot at k entities instead of `C(k,2)` pairs, lowered
//!   to pairwise evidence at a fixed approximate magnitude.
//! - [`scalar`] — single-entity absolute digit rating; a control, not a
//!   preferred path (absolute scales don't calibrate across models or
//!   entities the way a forced comparison does).
//!
//! ## `crate::gateway::TokenLogprob`
//!
//! `answer_logprobs` is one entry per completion **position**, in
//! generation order, carrying that position's chosen token/logprob plus the
//! top-k alternatives the provider considered there (`token`, `logprob`,
//! `top: Vec<(String, f64)>`) — mirroring the standard chat-completions
//! logprobs shape (`logprobs.content[i]`). This is what "find the answer
//! position" / "top-k entries at that position" below refer to. All gateway
//! field access is isolated to `find_answer_position` and
//! `merge_candidates`; no instrument's atom-parsing logic touches gateway
//! fields directly.
//!
//! Salvaged (redesigned) from the diamond2 quarry's answer-token position
//! disambiguation and tolerant-token parsing.

pub mod kwise;
pub mod ordinal;
pub mod ratio_letter;
pub mod scalar;

use crate::atom::AnswerAtom;
use crate::evidence::{AnswerEvidence, AtomProb, EvidenceError, PmfCompleteness};
use crate::gateway::TokenLogprob;
use crate::ontology::{Attribute, Entity, TemplateHash};
use crate::record::{AcquisitionMode, EvidenceHealth, InstrumentKind, ParserVersion};

/// A prompt ready to send to a provider, plus everything needed to bind the
/// eventual response back to what was asked.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderedPrompt {
    /// System/instruction text.
    pub system: String,
    /// User text with the actual entity bodies filled in.
    pub user: String,
    /// Content hash of `system + 0x1f + user-skeleton` where the user
    /// skeleton has entity-bodies (and only entity-bodies) blanked to fixed
    /// placeholders. Two renders with the SAME attribute and DIFFERENT
    /// entities always share this hash; it identifies the prompt's shape,
    /// not its content.
    pub template: TemplateHash,
    /// Whether the response is expected to be a JSON object (`false` for
    /// every instrument here: all of them ask for a bare single token).
    pub response_format_json: bool,
    /// The exact tokens whose logprobs constitute the answer PMF, in a
    /// stable order. Callers use this to size/validate `top_logprobs`
    /// requests to the provider.
    pub answer_alphabet: Vec<String>,
}

/// The result of parsing one raw completion into evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ParseOutcome {
    /// The parsed answer PMF.
    pub evidence: AnswerEvidence,
    /// Health facts about how the evidence was obtained.
    pub health: EvidenceHealth,
    /// Whether the evidence came from logprobs, sampling, or a fusion.
    pub mode: AcquisitionMode,
}

/// Honest failure taxonomy for rendering/parsing. These are STRUCTURAL
/// failures (nothing to work with at all) — a merely low-quality or
/// refusing answer is not an error, it is evidence with a health flag.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum InstrumentError {
    /// The raw completion had no content to inspect at all.
    #[error("completion content is empty")]
    EmptyContent,
    /// No answer position/token could be located in the completion or its
    /// logprobs — not even as an off-alphabet or refusal marker.
    #[error("no answer token could be located in the completion")]
    Unparseable,
    /// A k-wise/scalar call was made with a structurally invalid item count.
    #[error("expected {min}..={max} presented items, got {got}")]
    InvalidPresentationCount {
        /// Minimum valid item count, inclusive.
        min: usize,
        /// Maximum valid item count, inclusive.
        max: usize,
        /// The item count actually given.
        got: usize,
    },
    /// Evidence construction failed on otherwise-valid inputs.
    #[error("evidence construction failed: {0}")]
    Evidence(#[from] EvidenceError),
}

/// The pairwise elicitation contract shared by [`ratio_letter`] and
/// [`ordinal`]. K-wise and scalar instruments are NOT pairwise (they don't
/// have two presented slots) and intentionally do not implement this trait.
pub trait Instrument {
    /// Which [`InstrumentKind`] this is, for `JudgementRecord`.
    fn kind(&self) -> InstrumentKind;
    /// The parser version to stamp on records this instrument produces.
    fn parser_version(&self) -> ParserVersion;
    /// Render the comparison prompt for `slot_a` vs `slot_b` on `attribute`.
    fn render(&self, attribute: &Attribute, slot_a: &Entity, slot_b: &Entity) -> RenderedPrompt;
    /// Parse a raw completion (plus, if available, answer-position
    /// logprobs) into evidence. `content` is always the authoritative
    /// record of what the model said; `answer_logprobs`, when present,
    /// upgrades a single-sample point observation into a full PMF.
    fn parse(
        &self,
        content: &str,
        answer_logprobs: Option<&[TokenLogprob]>,
    ) -> Result<ParseOutcome, InstrumentError>;
}

/// Content hash over `system + 0x1f + user_skeleton`. `user_skeleton` must
/// have entity bodies replaced with fixed placeholders so the hash is
/// invariant to which entities were presented.
pub(crate) fn template_hash(system: &str, user_skeleton: &str) -> TemplateHash {
    let mut bytes = Vec::with_capacity(system.len() + 1 + user_skeleton.len());
    bytes.extend_from_slice(system.as_bytes());
    bytes.push(0x1f);
    bytes.extend_from_slice(user_skeleton.as_bytes());
    TemplateHash::derive(&bytes)
}

/// Strip whitespace and the punctuation a provider sometimes wraps a
/// single-character answer in (quotes, backticks, a trailing colon/comma).
pub(crate) fn trim_token(token: &str) -> &str {
    token
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | ':' | ','))
}

/// Find the first completion position whose CHOSEN token (trimmed) matches
/// `is_answer`. Models occasionally emit a filler/leading token before the
/// single-token answer, so the answer is not always position 0.
///
/// Salvaged (simplified) from diamond2's `provider_top_logprobs_from_chat_
/// completion_response` position-disambiguation loop; simplified here
/// because these instruments render single-token, non-JSON answers, so
/// there is no separate ground truth to cross-check positions against.
pub(crate) fn find_answer_position(
    positions: &[TokenLogprob],
    mut is_answer: impl FnMut(&str) -> bool,
) -> Option<&TokenLogprob> {
    positions.iter().find(|p| is_answer(trim_token(&p.token)))
}

/// The chosen token merged with its top-k alternatives, deduplicated by
/// trimmed token text (first occurrence wins) so a gateway that already
/// includes the chosen token inside `top` is never double-counted.
pub(crate) fn merge_candidates(position: &TokenLogprob) -> Vec<(String, f64)> {
    let mut seen = std::collections::HashSet::new();
    std::iter::once((position.token.clone(), position.logprob))
        .chain(position.top.iter().cloned())
        .filter(|(tok, _)| seen.insert(trim_token(tok).to_string()))
        .collect()
}

/// Total visible probability mass at a position: every top-k entry,
/// parseable or not, exactly as the brief specifies ("visible_mass = sum of
/// exp(logprob) over ALL top entries at that position").
pub(crate) fn candidate_mass(candidates: &[(String, f64)]) -> f64 {
    candidates.iter().map(|(_, lp)| lp.exp()).sum()
}

/// Evidence for "the provider showed us `visible_mass` worth of tokens at
/// the answer position, and none of them parsed to an informative atom" —
/// the zero-atom edge of [`crate::evidence::evidence_from_logprobs`], which
/// that function cannot express directly because it requires at least one
/// parsed atom. Used when an answer position is found (so we are not lost)
/// but every candidate there is refusal or junk.
pub(crate) fn evidence_all_off_alphabet(
    visible_mass: f64,
) -> Result<AnswerEvidence, EvidenceError> {
    let visible = visible_mass.clamp(0.0, 1.0);
    let mut support = vec![AtomProb {
        atom: AnswerAtom::OffAlphabet,
        p: visible,
    }];
    let unresolved = (1.0 - visible).max(0.0);
    let completeness = if unresolved > 0.0 {
        support.push(AtomProb {
            atom: AnswerAtom::Abstain,
            p: unresolved,
        });
        PmfCompleteness::Truncated {
            shown_mass: visible,
            unresolved_mass: unresolved,
        }
    } else {
        PmfCompleteness::Complete
    };
    AnswerEvidence::new(support, completeness)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(token: &str, logprob: f64, alts: &[(&str, f64)]) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob,
            top: alts.iter().map(|(t, p)| (t.to_string(), *p)).collect(),
        }
    }

    #[test]
    fn template_hash_is_deterministic_and_order_sensitive() {
        let a = template_hash("sys", "user-skeleton");
        let b = template_hash("sys", "user-skeleton");
        assert_eq!(a, b);
        let c = template_hash("sys", "different-skeleton");
        assert_ne!(a, c);
        let d = template_hash("different-sys", "user-skeleton");
        assert_ne!(a, d);
    }

    #[test]
    fn trim_token_strips_whitespace_and_wrapping_punctuation() {
        assert_eq!(trim_token("  B  "), "B");
        assert_eq!(trim_token("\"B\""), "B");
        assert_eq!(trim_token("`b`"), "b");
        assert_eq!(trim_token("B,"), "B");
        assert_eq!(trim_token("B:"), "B");
    }

    #[test]
    fn find_answer_position_skips_leading_filler_positions() {
        let positions = vec![tl(" ", -0.01, &[]), tl("B", -0.1, &[("b", -2.5)])];
        let found = find_answer_position(&positions, |t| t.len() == 1 && t != " ");
        assert_eq!(found.map(|p| p.token.as_str()), Some("B"));
    }

    #[test]
    fn find_answer_position_none_when_nothing_matches() {
        let positions = vec![tl("hello", -0.1, &[]), tl("world", -0.2, &[])];
        assert!(find_answer_position(&positions, |t| t == "B").is_none());
    }

    #[test]
    fn merge_candidates_dedupes_chosen_token_against_its_own_alternatives() {
        let p = tl("B", -0.05, &[("B", -0.05), ("c", -3.0)]);
        let merged = merge_candidates(&p);
        assert_eq!(
            merged.len(),
            2,
            "chosen token de-duplicated against alt list"
        );
        assert!(merged.iter().any(|(t, _)| t == "B"));
        assert!(merged.iter().any(|(t, _)| t == "c"));
    }

    #[test]
    fn candidate_mass_sums_every_entry_regardless_of_parseability() {
        let candidates = vec![
            ("B".to_string(), 0.5f64.ln()),
            ("###".to_string(), 0.25f64.ln()),
        ];
        assert!((candidate_mass(&candidates) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn evidence_all_off_alphabet_splits_visible_and_abstain() {
        let ev = evidence_all_off_alphabet(0.4).unwrap();
        assert!((ev.off_alphabet_mass() - 0.4).abs() < 1e-12);
        assert!((ev.abstain_mass() - 0.6).abs() < 1e-12);
        assert_eq!(ev.informative_mass(), 0.0);
        let complete = evidence_all_off_alphabet(1.0).unwrap();
        assert_eq!(complete.completeness, PmfCompleteness::Complete);
    }
}
