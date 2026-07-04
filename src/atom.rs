//! The answer alphabet: single-token encodings of pairwise judgements.
//!
//! A judgement between two presented entities is encoded as ONE letter so
//! that a single completion-token position carries the model's full prior
//! over the judgement space via top-k logprobs:
//!
//! - `A`/`a` — parity: the entities are equal on the attribute.
//! - `B`..`Z` — the entity in slot A has more, by the ladder magnitude
//!   indexed by the letter (B = smallest margin … Z = extreme).
//! - `b`..`z` — the entity in slot B has more, same magnitudes.
//!
//! Same letter, different case = same magnitude, opposite winner. This case
//! symmetry is what makes presentation counterbalancing exact: reflecting a
//! judgement is a case flip, not a re-elicitation.
//!
//! Salvaged (redesigned) from the diamond2 `cardinal-harness-v2` quarry.

use serde::{Deserialize, Serialize};

/// Ratio ladder for the 25 directional buckets (`B`..`Z` / `b`..`z`),
/// approximately geometric from a near-tie to three orders of magnitude.
pub const RATIO_LADDER: [f64; 25] = [
    1.06, 1.17, 1.33, 1.56, 1.85, 2.25, 2.78, 3.49, 4.45, 5.74, 7.51, 9.95, 13.3, 18.1, 24.8, 34.4,
    48.1, 68.1, 97.2, 140.0, 204.0, 299.0, 444.0, 663.0, 1000.0,
];

/// Which presented slot won a directional judgement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    A,
    B,
}

impl Side {
    pub fn flipped(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

/// One point in the judgement answer space.
///
/// `A(k)` / `B(k)` carry a 1-based bucket index into [`RATIO_LADDER`]
/// (k in 1..=25, from letter offsets 1..=25). Two escape atoms cover the
/// rest of the probability space: `OffAlphabet` (visible mass on tokens that
/// parse to nothing) and `Abstain` (mass the provider never showed us).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerAtom {
    /// Entities judged equal on the attribute.
    Parity,
    /// Probability mass never made visible by the provider.
    Abstain,
    /// Slot A has more; bucket 1..=25 into the ladder.
    A(u8),
    /// Slot B has more; bucket 1..=25 into the ladder.
    B(u8),
    /// Visible mass on tokens outside the answer alphabet.
    OffAlphabet,
}

impl AnswerAtom {
    /// Parse a single answer letter.
    pub fn from_letter(letter: char) -> Option<Self> {
        if !letter.is_ascii() {
            return None;
        }
        match letter as u8 {
            b'A' | b'a' => Some(Self::Parity),
            b @ b'B'..=b'Z' => Some(Self::A(b - b'A')),
            b @ b'b'..=b'z' => Some(Self::B(b - b'a')),
            _ => None,
        }
    }

    /// Render back to the canonical letter, when one exists.
    pub fn letter(self) -> Option<char> {
        match self {
            Self::Parity => Some('A'),
            Self::A(k) if (1..=RATIO_LADDER.len() as u8).contains(&k) => Some((b'A' + k) as char),
            Self::B(k) if (1..=RATIO_LADDER.len() as u8).contains(&k) => Some((b'a' + k) as char),
            _ => None,
        }
    }

    /// The same judgement with the presented slots exchanged.
    ///
    /// This is the exact counterbalance reflection: parity and the escape
    /// atoms are fixed points; directional atoms swap side, keep magnitude.
    pub fn reflected(self) -> Self {
        match self {
            Self::A(k) => Self::B(k),
            Self::B(k) => Self::A(k),
            other => other,
        }
    }

    /// Zero-based index into [`RATIO_LADDER`] for directional atoms.
    pub fn bucket_index(self) -> Option<usize> {
        match self {
            Self::A(k) | Self::B(k) => {
                let i = k.checked_sub(1)? as usize;
                (i < RATIO_LADDER.len()).then_some(i)
            }
            _ => None,
        }
    }

    /// Judged magnitude: 1.0 for parity, ladder value for directional atoms,
    /// `None` for the escape atoms.
    pub fn ratio(self) -> Option<f64> {
        match self {
            Self::Parity => Some(1.0),
            Self::A(_) | Self::B(_) => self.bucket_index().map(|i| RATIO_LADDER[i]),
            Self::Abstain | Self::OffAlphabet => None,
        }
    }

    /// Signed log-ratio: positive when slot A wins, negative when slot B
    /// wins, zero at parity. The additive quantity the compiler consumes.
    pub fn signed_log_ratio(self) -> Option<f64> {
        match self {
            Self::Parity => Some(0.0),
            Self::A(_) => self.ratio().map(f64::ln),
            Self::B(_) => self.ratio().map(|r| -r.ln()),
            Self::Abstain | Self::OffAlphabet => None,
        }
    }

    /// Which slot this atom favors, if directional.
    pub fn side(self) -> Option<Side> {
        match self {
            Self::A(_) => Some(Side::A),
            Self::B(_) => Some(Side::B),
            _ => None,
        }
    }

    /// True for atoms that carry judgement information (not escapes).
    pub fn is_informative(self) -> bool {
        !matches!(self, Self::Abstain | Self::OffAlphabet)
    }
}

/// Interpolate a continuous positive ratio onto the answer alphabet as a
/// one- or two-atom PMF whose expected signed log-ratio equals `ln(ratio)`
/// exactly (log-linear interpolation between adjacent rungs; saturates at
/// the ladder's ends). `side` states which slot the ratio favors; a ratio
/// of exactly 1.0 is parity regardless of side.
///
/// Salvaged invariant from the diamond2 quarry
/// (`ratio_interpolation_preserves_log_ratio_mean`).
pub fn interpolate_ratio(side: Side, ratio: f64) -> Option<Vec<(AnswerAtom, f64)>> {
    if !ratio.is_finite() || ratio <= 0.0 {
        return None;
    }
    // Normalize to "winner ratio >= 1" coordinates.
    let (side, r) = if ratio >= 1.0 {
        (side, ratio)
    } else {
        (side.flipped(), 1.0 / ratio)
    };
    let atom_at = |k: usize| match side {
        Side::A => AnswerAtom::A(k as u8),
        Side::B => AnswerAtom::B(k as u8),
    };
    let target = r.ln();
    // Rung log-values, with parity (0.0) as the virtual rung below index 1.
    let last = RATIO_LADDER.len();
    if r >= RATIO_LADDER[last - 1] {
        return Some(vec![(atom_at(last), 1.0)]);
    }
    let mut lower_log = 0.0; // parity
    let mut lower_atom = AnswerAtom::Parity;
    for (i, rung) in RATIO_LADDER.iter().enumerate() {
        let upper_log = rung.ln();
        if target <= upper_log {
            let upper_atom = atom_at(i + 1);
            let span = upper_log - lower_log;
            let w_upper = if span > 0.0 {
                (target - lower_log) / span
            } else {
                1.0
            };
            let w_lower = 1.0 - w_upper;
            let mut out = Vec::with_capacity(2);
            if w_lower > 0.0 {
                out.push((lower_atom, w_lower));
            }
            if w_upper > 0.0 {
                out.push((upper_atom, w_upper));
            }
            return Some(out);
        }
        lower_log = upper_log;
        lower_atom = atom_at(i + 1);
    }
    unreachable!("target below last rung is handled inside the loop");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letter_round_trip_covers_the_whole_alphabet() {
        for byte in b'A'..=b'Z' {
            let c = byte as char;
            let atom = AnswerAtom::from_letter(c).expect("uppercase parses");
            assert_eq!(atom.letter(), Some(c), "round trip {c}");
        }
        for byte in b'a'..=b'z' {
            let c = byte as char;
            let atom = AnswerAtom::from_letter(c).expect("lowercase parses");
            // 'a' canonicalizes to 'A' (parity has one canonical letter).
            let expected = if c == 'a' { 'A' } else { c };
            assert_eq!(atom.letter(), Some(expected), "round trip {c}");
        }
        assert_eq!(AnswerAtom::from_letter('0'), None);
        assert_eq!(AnswerAtom::from_letter('é'), None);
        assert_eq!(AnswerAtom::from_letter(' '), None);
    }

    #[test]
    fn case_flip_is_reflection() {
        for byte in b'B'..=b'Z' {
            let upper = AnswerAtom::from_letter(byte as char).unwrap();
            let lower = AnswerAtom::from_letter((byte + 32) as char).unwrap();
            assert_eq!(upper.reflected(), lower);
            assert_eq!(lower.reflected(), upper);
            assert_eq!(upper.ratio(), lower.ratio(), "magnitude survives case");
        }
        assert_eq!(AnswerAtom::Parity.reflected(), AnswerAtom::Parity);
        assert_eq!(AnswerAtom::Abstain.reflected(), AnswerAtom::Abstain);
    }

    #[test]
    fn signed_log_ratio_is_antisymmetric_under_reflection() {
        for k in 1..=25u8 {
            let a = AnswerAtom::A(k).signed_log_ratio().unwrap();
            let b = AnswerAtom::B(k).signed_log_ratio().unwrap();
            assert!((a + b).abs() < 1e-12, "antisymmetry at bucket {k}");
            assert!(a > 0.0);
        }
        assert_eq!(AnswerAtom::Parity.signed_log_ratio(), Some(0.0));
        assert_eq!(AnswerAtom::Abstain.signed_log_ratio(), None);
    }

    #[test]
    fn ladder_is_strictly_increasing_and_starts_past_parity() {
        let first = RATIO_LADDER[0];
        assert!(first > 1.0);
        for w in RATIO_LADDER.windows(2) {
            assert!(w[1] > w[0], "ladder strictly increasing");
        }
    }

    #[test]
    fn interpolation_preserves_log_ratio_mean_exactly() {
        for &r in &[1.0, 1.06, 1.4, 2.0, 3.0, 9.95, 500.0, 999.0] {
            let atoms = interpolate_ratio(Side::A, r).unwrap();
            let total: f64 = atoms.iter().map(|(_, w)| w).sum();
            assert!((total - 1.0).abs() < 1e-12, "weights normalized at {r}");
            let mean: f64 = atoms
                .iter()
                .map(|(a, w)| w * a.signed_log_ratio().unwrap())
                .sum();
            assert!(
                (mean - r.ln()).abs() < 1e-9,
                "mean preserved at {r}: {mean} vs {}",
                r.ln()
            );
        }
    }

    #[test]
    fn interpolation_saturates_and_flips() {
        // Beyond the top rung: full mass on Z.
        let atoms = interpolate_ratio(Side::A, 5000.0).unwrap();
        assert_eq!(atoms, vec![(AnswerAtom::A(25), 1.0)]);
        // Sub-unity ratios flip the side.
        let atoms = interpolate_ratio(Side::A, 0.5).unwrap();
        assert!(atoms
            .iter()
            .all(|(a, _)| matches!(a, AnswerAtom::B(_) | AnswerAtom::Parity)));
        let mean: f64 = atoms
            .iter()
            .map(|(a, w)| w * a.signed_log_ratio().unwrap())
            .sum();
        assert!((mean - 0.5f64.ln()).abs() < 1e-9);
        // Degenerate inputs are rejected.
        assert!(interpolate_ratio(Side::A, 0.0).is_none());
        assert!(interpolate_ratio(Side::A, f64::NAN).is_none());
    }

    #[test]
    fn out_of_range_buckets_are_inert() {
        assert_eq!(AnswerAtom::A(0).ratio(), None);
        assert_eq!(AnswerAtom::A(26).ratio(), None);
        assert_eq!(AnswerAtom::A(26).letter(), None);
        assert_eq!(AnswerAtom::B(0).signed_log_ratio(), None);
    }
}
