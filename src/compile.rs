//! Compile: evidence -> ordering posterior.
//!
//! Consumes [`JudgementRecord`]s for ONE attribute, plus an entity roster in
//! a fixed index space, and produces a per-entity latent-scale posterior: a
//! point estimate and a standard deviation, plus (within each connected
//! component of the comparison graph) a rank. This is a Bradley-Terry-style
//! fit, but on the raw log-ratio *moments* of each judgement's evidence PMF
//! rather than on bare win/loss counts, so a ratio-letter judgement moves
//! the fit exactly as far as its reported magnitude and confidence warrant,
//! and an under-informative judgement (heavy `Abstain`/`OffAlphabet` mass,
//! wide PMF) moves it correspondingly less.
//!
//! Pipeline:
//!
//! 1. **Canonicalize** (the direction-safety invariant) — every judgement's
//!    evidence is recorded in *presented* slot coordinates. Whenever a
//!    record's presentation is not the canonical `(pair.lo, pair.hi)` order,
//!    its evidence is [`reflected`](crate::evidence::AnswerEvidence::reflected)
//!    before use, so downstream a positive log-ratio always means
//!    `pair.lo` scored higher than `pair.hi`.
//! 2. **Weigh** — each record's evidence collapses to `(mean, var)` via
//!    [`log_ratio_moments`](crate::evidence::AnswerEvidence::log_ratio_moments).
//!    Refused records are skipped outright; records with no informative mass
//!    (so no moments) are skipped as uninformative. Surviving records get an
//!    observation weight of `informative_mass * clean_factor / (var +
//!    var_floor)`, where `clean_factor` is 1.0 for a cleanly parsed record
//!    and 0.25 otherwise, and `var_floor` guards against a division blowup
//!    on a (near-)zero-variance PMF.
//! 3. **Solve** — a weighted least squares fit of latent per-entity scores
//!    to the pairwise log-ratio observations, over the pair graph. This is
//!    exactly a weighted graph Laplacian system `(L + ridge*I) x = b`: `L`
//!    accumulates `weight` on the diagonal at both endpoints and `-weight`
//!    off-diagonal for every observed edge, `b` accumulates `weight * mean`
//!    signed by endpoint, and `ridge*I` is added so the (otherwise singular,
//!    since constant shifts are a Laplacian null direction) system is
//!    invertible. It is solved by hand-rolled Gauss-Jordan elimination with
//!    partial pivoting — O(n^3), which is fine at the "a few hundred
//!    entities" scale this compiler targets; nothing here is asymptotically
//!    clever on purpose.
//!
//!    Because the pre-ridge system is exactly consistent whenever the
//!    underlying data is (no residual achievable at all weight settings),
//!    adding `ridge*I` does not bias the *fitted differences* — it only
//!    selects, among the affine family of equally-good solutions that
//!    differ by a constant shift within a component, the minimum-norm one.
//!    That minimum-norm point is exactly the mean-zero point, so an explicit
//!    post-hoc re-centering of each connected component (subtracting its
//!    own mean) gives an *exact* mean-zero gauge, independent of `ridge`.
//! 4. **Report** — per-entity mean and standard deviation (the latter read
//!    off the diagonal of the solve's matrix inverse; this is a **diagonal
//!    approximation** — it ignores the covariance between entities, so it
//!    is exact only when treating each entity's marginal in isolation, and
//!    it also carries the (typically dominant, since `ridge` is small)
//!    uncertainty of the component's shared gauge-fixing constant, which is
//!    irrelevant post-centering but not subtracted back out. It is useful
//!    for relative comparisons — [`CompiledPosterior::p_higher`] and the
//!    monotonicity property "less informative evidence -> wider posterior"
//!    both hold for any `ridge > 0` — but should not be read as an
//!    absolute-scale credible interval.), rank within each connected
//!    component (entities with no comparative information at all — a
//!    singleton component — get `rank: None`, since "rank" against nobody
//!    is not a fact), and the component partition itself so callers can see
//!    exactly which entities the corpus never compared.

use crate::ontology::{Entity, EntityId};
use crate::record::JudgementRecord;
use std::borrow::Cow;
use std::collections::HashMap;

/// Named numerical tolerances for the compiler, so no epsilon is scattered
/// unnamed through the arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    /// Floor added to every observation's reported log-ratio variance before
    /// it is inverted into a weight. Without this, a PMF that happens to be
    /// a pure point mass (variance exactly 0.0, e.g. a single-atom letter
    /// answer with no logprob spread) would demand an infinite weight.
    pub var_floor: f64,
    /// Diagonal regularizer added to the weighted Laplacian before solving.
    /// Required for invertibility: a graph Laplacian is always singular on
    /// the constant-shift direction within each connected component (and
    /// trivially singular for any entity with zero observed comparisons).
    /// See the module doc for why this does not bias fitted differences.
    pub ridge: f64,
}

impl Default for Tolerances {
    /// `var_floor = 1e-3`, `ridge = 1e-6`.
    fn default() -> Self {
        Self {
            var_floor: 1e-3,
            ridge: 1e-6,
        }
    }
}

/// Errors compiling evidence into a posterior.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum CompileError {
    /// A record's presentation named an entity absent from the roster. The
    /// roster is a caller contract (it defines the index space); a record
    /// referencing an entity outside it is caller error, not something the
    /// compiler can silently route around.
    #[error(
        "judgement record presentation references entity {0:?}, which is not in the entity roster"
    )]
    UnknownEntity(EntityId),
}

/// One entity's compiled posterior over the latent attribute scale.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityPosterior {
    /// Identifies which roster entity this posterior is for.
    pub entity: EntityId,
    /// Fitted latent score. Only differences within a connected component
    /// are meaningful; the gauge is fixed so each component's scores sum to
    /// zero (see the module doc).
    pub latent_mean: f64,
    /// Standard deviation of the latent score, from the diagonal of the
    /// solve's matrix inverse (a diagonal approximation; see module doc).
    pub latent_std: f64,
    /// 1-based rank within this entity's connected component, descending by
    /// `latent_mean` (`Some(1)` is the top-scoring entity in the component).
    /// `None` in a singleton component: with no comparison touching the
    /// entity at all, there is nothing to rank it against.
    pub rank: Option<usize>,
}

/// The compiled posterior for one attribute over one entity roster.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledPosterior {
    /// One posterior per roster entity, in the same order as the roster
    /// passed to [`compile`].
    pub entities: Vec<EntityPosterior>,
    /// The comparison graph's connected components, as roster indices,
    /// each sorted ascending; components are ordered by their smallest
    /// index. An entity with zero surviving comparisons is its own
    /// singleton component.
    pub components: Vec<Vec<usize>>,
    /// Number of input records that contributed a weighted observation.
    pub records_used: usize,
    /// Number of input records skipped because `health.refused` was set.
    pub records_skipped_refused: usize,
    /// Full posterior covariance of the latents, `(L + ridge·I)⁻¹`, kept so
    /// pairwise difference variances can cancel the shared per-component
    /// gauge mode exactly (see `p_higher`). Row-major, `n × n`.
    covariance: Vec<Vec<f64>>,
    /// Number of (non-refused) input records skipped because their evidence
    /// carried no informative mass (so no log-ratio moments could be
    /// computed), or because the resulting weight was non-positive.
    pub records_skipped_uninformative: usize,
}

impl CompiledPosterior {
    /// `P(entity at roster index `i` scores higher on the attribute than the
    /// entity at roster index `j`)`, as a Gaussian tail probability over the
    /// two entities' fitted latent scores.
    ///
    /// This uses a **diagonal approximation** of `Var(s_i - s_j)`: it adds
    /// the two marginal variances and ignores their covariance, which for
    /// directly-compared entities is generally negative (shared evidence
    /// pins the difference tighter than either marginal alone), so this
    /// systematically *overstates* uncertainty for well-connected pairs. It
    /// is documented here rather than hidden behind a falsely-precise
    /// number.
    ///
    /// Returns `None` when either index is out of range, or when the two
    /// entities are in different connected components — no evidence in the
    /// corpus pins their relative scale at all, so no probability is
    /// defensible, not even an uninformative 0.5.
    pub fn p_higher(&self, i: usize, j: usize) -> Option<f64> {
        let a = self.entities.get(i)?;
        let b = self.entities.get(j)?;
        let same_component = self
            .components
            .iter()
            .any(|component| component.contains(&i) && component.contains(&j));
        if !same_component {
            return None;
        }
        let diff_mean = a.latent_mean - b.latent_mean;
        // Exact difference variance: Σᵢᵢ + Σⱼⱼ − 2Σᵢⱼ. The cross term is
        // what cancels the shared gauge mode (the ~1/ridge constant vector
        // every same-component entry carries); a diagonal-only
        // approximation here makes p_higher collapse toward 0.5 at small
        // ridge. Found by the integration battery.
        let diff_var = self.covariance[i][i] + self.covariance[j][j] - 2.0 * self.covariance[i][j];
        // Guard anyway rather than divide by exactly zero.
        let z = if diff_var > 0.0 {
            diff_mean / diff_var.sqrt()
        } else if diff_mean > 0.0 {
            f64::INFINITY
        } else if diff_mean < 0.0 {
            f64::NEG_INFINITY
        } else {
            0.0
        };
        Some(standard_normal_cdf(z))
    }
}

/// Compile judgement records for a single attribute into an ordering
/// posterior over `entities`.
///
/// `entities` defines the index space: `CompiledPosterior::entities[k]` is
/// the posterior for `entities[k]`. Every record's presentation must name
/// entities present in `entities`; a record naming any other entity is
/// reported as [`CompileError::UnknownEntity`] rather than silently ignored.
///
/// Callers are responsible for `records` all being about the same
/// attribute — this function does not read or check `record.attribute`.
pub fn compile(
    records: &[JudgementRecord],
    entities: &[Entity],
    tolerances: &Tolerances,
) -> Result<CompiledPosterior, CompileError> {
    let n = entities.len();
    let index_of: HashMap<EntityId, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.clone(), i))
        .collect();

    let mut laplacian = vec![vec![0.0_f64; n]; n];
    let mut rhs = vec![0.0_f64; n];
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];

    let mut records_used = 0usize;
    let mut records_skipped_refused = 0usize;
    let mut records_skipped_uninformative = 0usize;

    for record in records {
        if record.health.refused {
            records_skipped_refused += 1;
            continue;
        }

        // Step 1: canonicalize. `pair_key()` is independent of presented
        // order, so this is always the true (hash-order) canonical pair.
        let pair = record.presentation.pair_key();
        let canonical = record.presentation.is_canonical(&pair);
        let evidence = if canonical {
            Cow::Borrowed(&record.evidence)
        } else {
            Cow::Owned(record.evidence.reflected())
        };

        // Step 2: weigh.
        let Some((mean, var)) = evidence.log_ratio_moments() else {
            records_skipped_uninformative += 1;
            continue;
        };
        let clean_factor = if record.health.parsed_cleanly {
            1.0
        } else {
            0.25
        };
        let weight = evidence.informative_mass() * clean_factor / (var + tolerances.var_floor);
        if !weight.is_finite() || weight <= 0.0 {
            records_skipped_uninformative += 1;
            continue;
        }

        let lo = *index_of
            .get(&pair.lo)
            .ok_or_else(|| CompileError::UnknownEntity(pair.lo.clone()))?;
        let hi = *index_of
            .get(&pair.hi)
            .ok_or_else(|| CompileError::UnknownEntity(pair.hi.clone()))?;

        // mean == E[signed_log_ratio] in canonical coordinates: positive
        // means pair.lo (== lo here) scored higher, i.e. s_lo - s_hi.
        laplacian[lo][lo] += weight;
        laplacian[hi][hi] += weight;
        laplacian[lo][hi] -= weight;
        laplacian[hi][lo] -= weight;
        rhs[lo] += weight * mean;
        rhs[hi] -= weight * mean;
        adjacency[lo].push(hi);
        adjacency[hi].push(lo);

        records_used += 1;
    }

    for (i, row) in laplacian.iter_mut().enumerate() {
        row[i] += tolerances.ridge;
    }

    // Step 3: solve. `laplacian` is symmetric positive-definite by
    // construction (weighted Laplacian, which is positive-semidefinite, plus
    // ridge > 0 on the diagonal), so this never actually hits the singular
    // branch; it is checked rather than assumed because floating point is
    // floating point.
    let inverse = invert(&laplacian).unwrap_or_else(|| vec![vec![0.0; n]; n]);
    let mut latent = matvec(&inverse, &rhs);

    // Connected components over the surviving edges; an entity touched by no
    // surviving record is its own singleton component.
    let mut visited = vec![false; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(node) = stack.pop() {
            component.push(node);
            for &next in &adjacency[node] {
                if !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort_by_key(|component| component[0]);

    // Exact mean-zero gauge fix per component (see module doc: this is
    // exact, not an approximation, for the affine family the ridge-free
    // problem leaves undetermined).
    for component in &components {
        let mean: f64 = component.iter().map(|&i| latent[i]).sum::<f64>() / component.len() as f64;
        for &i in component {
            latent[i] -= mean;
        }
    }

    // Rank within each non-singleton component, descending by latent mean.
    let mut rank_of: Vec<Option<usize>> = vec![None; n];
    for component in &components {
        if component.len() < 2 {
            continue;
        }
        let mut order = component.clone();
        order.sort_by(|&a, &b| latent[b].total_cmp(&latent[a]).then(a.cmp(&b)));
        for (place, &idx) in order.iter().enumerate() {
            rank_of[idx] = Some(place + 1);
        }
    }

    // Per-entity variance in the CENTERED (mean-zero-per-component) gauge:
    // Var(sᵢ − mean_C(s)) = Σᵢᵢ − (2/|C|)·Σⱼ Σᵢⱼ + (1/|C|²)·Σⱼₖ Σⱼₖ over the
    // entity's component C. The gauge mode is a constant vector within a
    // component, so its (ridge-scale) contribution cancels exactly here —
    // latent_std is now a meaningful spread, not a gauge artifact.
    let mut variance = vec![0.0_f64; n];
    for component in &components {
        let m = component.len() as f64;
        let mut grand: f64 = 0.0;
        for &j in component {
            for &k in component {
                grand += inverse[j][k];
            }
        }
        for &i in component {
            let cross: f64 = component.iter().map(|&j| inverse[i][j]).sum();
            variance[i] = (inverse[i][i] - 2.0 * cross / m + grand / (m * m)).max(0.0);
        }
    }

    let entities_out = (0..n)
        .map(|i| EntityPosterior {
            entity: entities[i].id.clone(),
            latent_mean: latent[i],
            latent_std: variance[i].sqrt(),
            rank: rank_of[i],
        })
        .collect();

    Ok(CompiledPosterior {
        entities: entities_out,
        covariance: inverse,
        components,
        records_used,
        records_skipped_refused,
        records_skipped_uninformative,
    })
}

/// Below this pivot magnitude, [`invert`] gives up rather than divide by a
/// number floating-point rounding could have driven arbitrarily far from
/// the true (mathematically nonzero) pivot.
const SINGULARITY_THRESHOLD: f64 = 1e-12;

/// Gauss-Jordan matrix inversion with partial pivoting. `matrix` must be
/// square. Returns `None` only when the matrix is (numerically) singular.
///
/// O(n^3) time and space — appropriate for the "a few hundred entities"
/// scale this compiler targets; a sparse or per-component factorization
/// would start paying for itself well past that.
fn invert(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = matrix[i].clone();
            row.resize(2 * n, 0.0);
            row[n + i] = 1.0;
            row
        })
        .collect();

    for col in 0..n {
        let pivot_row =
            (col..n).max_by(|&a, &b| aug[a][col].abs().total_cmp(&aug[b][col].abs()))?;
        if aug[pivot_row][col].abs() < SINGULARITY_THRESHOLD {
            return None;
        }
        aug.swap(col, pivot_row);
        let pivot = aug[col][col];
        for v in aug[col].iter_mut() {
            *v /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            if factor == 0.0 {
                continue;
            }
            // Split so the pivot row (`col`) can be read while `row` is
            // written, without re-indexing `aug` per column.
            let (pivot, target) = if row < col {
                let (left, right) = aug.split_at_mut(col);
                (&right[0], &mut left[row])
            } else {
                let (left, right) = aug.split_at_mut(row);
                (&left[col], &mut right[0])
            };
            for (t, p) in target.iter_mut().zip(pivot.iter()) {
                *t -= factor * p;
            }
        }
    }

    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// Dense matrix-vector product.
fn matvec(matrix: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

/// Standard normal CDF via the Abramowitz & Stegun 7.1.26 rational
/// approximation to `erf` (absolute error `<= 1.5e-7`, ample for a
/// probability we already only trust to a diagonal approximation).
fn standard_normal_cdf(z: f64) -> f64 {
    if z.is_infinite() {
        return if z > 0.0 { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::{interpolate_ratio, AnswerAtom, Side};
    use crate::evidence::{AnswerEvidence, AtomProb, PmfCompleteness};
    use crate::ontology::{AttributeId, CaptureId, Presentation, TemplateHash};
    use crate::record::{
        AcquisitionMode, Cost, DecodeConfig, EvidenceHealth, InstrumentKind, ParserVersion,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn atom_evidence(atom: AnswerAtom) -> AnswerEvidence {
        AnswerEvidence::new([AtomProb { atom, p: 1.0 }], PmfCompleteness::Complete).unwrap()
    }

    fn mk_record(
        slot_a: &Entity,
        slot_b: &Entity,
        evidence: AnswerEvidence,
        parsed_cleanly: bool,
        refused: bool,
        nonce: u64,
    ) -> JudgementRecord {
        JudgementRecord::new(
            InstrumentKind::RatioLetterPairwise,
            AcquisitionMode::Logprob,
            AttributeId::derive(b"compile-test-attribute"),
            Presentation {
                slot_a: slot_a.id.clone(),
                slot_b: slot_b.id.clone(),
            },
            TemplateHash::derive(b"compile-test-template"),
            ParserVersion("compile-test/1".into()),
            "test/model".into(),
            DecodeConfig {
                temperature: 0.0,
                max_tokens: 8,
                top_logprobs: Some(20),
            },
            CaptureId::derive(&nonce.to_le_bytes()),
            evidence,
            EvidenceHealth {
                visible_mass: 1.0,
                parsed_cleanly,
                refused,
            },
            Cost::default(),
            1_700_000_000_000 + nonce,
        )
    }

    /// ORDINAL-FIRST: a chain of purely ordinal-style records (fixed
    /// magnitude bucket 7, both counterbalanced presentation orders for
    /// every adjacent pair) recovers the planted order exactly, with
    /// strictly monotone latents.
    #[test]
    fn ordinal_chain_recovers_planted_order() {
        let e: Vec<Entity> = ["alpha", "beta", "gamma", "delta"]
            .iter()
            .map(|s| Entity::new(*s))
            .collect();
        // Planted truth: e[0] < e[1] < e[2] < e[3].
        let mut records = Vec::new();
        let mut nonce = 0u64;
        for i in 0..e.len() - 1 {
            // Forward presentation: slot_a = lower, slot_b = higher.
            records.push(mk_record(
                &e[i],
                &e[i + 1],
                atom_evidence(AnswerAtom::B(7)),
                true,
                false,
                nonce,
            ));
            nonce += 1;
            // Counterbalanced presentation: slot_a = higher, slot_b = lower.
            records.push(mk_record(
                &e[i + 1],
                &e[i],
                atom_evidence(AnswerAtom::A(7)),
                true,
                false,
                nonce,
            ));
            nonce += 1;
        }

        let posterior = compile(&records, &e, &Tolerances::default()).unwrap();
        assert_eq!(posterior.records_used, records.len());
        assert_eq!(posterior.records_skipped_refused, 0);
        assert_eq!(posterior.records_skipped_uninformative, 0);
        assert_eq!(posterior.components, vec![vec![0, 1, 2, 3]]);

        for i in 0..e.len() - 1 {
            assert!(
                posterior.entities[i].latent_mean < posterior.entities[i + 1].latent_mean,
                "entity {i} should score below entity {}",
                i + 1
            );
        }
        for i in 0..e.len() {
            assert_eq!(posterior.entities[i].rank, Some(e.len() - i));
        }
    }

    /// Planted-truth recovery from ratio-letter-style PMF evidence, built
    /// via `interpolate_ratio` and `AnswerEvidence::new` for every pair.
    /// Because the pairwise data is exactly consistent with a single
    /// underlying score vector, the fit recovers planted differences to
    /// near machine precision (see module doc on why ridge does not bias
    /// this).
    #[test]
    fn ratio_letter_pmf_recovers_planted_scores() {
        let e: Vec<Entity> = ["p", "q", "r", "s"]
            .iter()
            .map(|s| Entity::new(*s))
            .collect();
        let scores = [0.0_f64, 1.0, 2.5, 4.0];

        let mut records = Vec::new();
        let mut nonce = 0u64;
        for i in 0..e.len() {
            for j in (i + 1)..e.len() {
                let ratio = (scores[j] - scores[i]).exp();
                let atoms = interpolate_ratio(Side::B, ratio).unwrap();
                let evidence = AnswerEvidence::new(
                    atoms.into_iter().map(|(atom, p)| AtomProb { atom, p }),
                    PmfCompleteness::Complete,
                )
                .unwrap();
                records.push(mk_record(&e[i], &e[j], evidence, true, false, nonce));
                nonce += 1;
            }
        }

        let posterior = compile(&records, &e, &Tolerances::default()).unwrap();
        assert_eq!(posterior.components, vec![vec![0, 1, 2, 3]]);

        for i in 0..e.len() {
            for j in (i + 1)..e.len() {
                let fitted_diff =
                    posterior.entities[j].latent_mean - posterior.entities[i].latent_mean;
                let planted_diff = scores[j] - scores[i];
                assert!(
                    (fitted_diff - planted_diff).abs() < 1e-6,
                    "pair ({i},{j}): fitted diff {fitted_diff} vs planted {planted_diff}"
                );
            }
        }
    }

    /// REFLECTION INVARIANCE: presenting records in random slot order, with
    /// evidence correctly reflected to match, yields the identical
    /// posterior to presenting everything canonically. The two rounds of
    /// reflection (test-side, then compile's own canonicalization) cancel
    /// exactly, so the compiled result must be bit-identical, not merely
    /// close.
    #[test]
    fn reflection_invariance_is_exact() {
        let e: Vec<Entity> = ["m", "n", "o", "p", "q"]
            .iter()
            .map(|s| Entity::new(*s))
            .collect();

        // Build every pair canonically (slot_a = pair.lo, slot_b = pair.hi)
        // with varied evidence so the test is not accidentally symmetric.
        let mut canonical_records = Vec::new();
        let mut nonce = 0u64;
        for i in 0..e.len() {
            for j in (i + 1)..e.len() {
                let pair = crate::ontology::PairKey::new(&e[i].id, &e[j].id);
                let (lo, hi) = if pair.lo == e[i].id {
                    (&e[i], &e[j])
                } else {
                    (&e[j], &e[i])
                };
                let ratio = 1.0 + 0.1 * ((i * 7 + j * 3) % 5) as f64;
                let atoms = interpolate_ratio(Side::B, ratio).unwrap();
                let evidence = AnswerEvidence::new(
                    atoms.into_iter().map(|(atom, p)| AtomProb { atom, p }),
                    PmfCompleteness::Complete,
                )
                .unwrap();
                canonical_records.push((lo.clone(), hi.clone(), evidence, nonce));
                nonce += 1;
            }
        }

        let all_canonical: Vec<JudgementRecord> = canonical_records
            .iter()
            .map(|(lo, hi, ev, n)| mk_record(lo, hi, ev.clone(), true, false, *n))
            .collect();

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let randomized: Vec<JudgementRecord> = canonical_records
            .iter()
            .map(|(lo, hi, ev, n)| {
                if rng.gen_bool(0.5) {
                    mk_record(hi, lo, ev.reflected(), true, false, *n)
                } else {
                    mk_record(lo, hi, ev.clone(), true, false, *n)
                }
            })
            .collect();

        let baseline = compile(&all_canonical, &e, &Tolerances::default()).unwrap();
        let shuffled = compile(&randomized, &e, &Tolerances::default()).unwrap();

        assert_eq!(baseline.components, shuffled.components);
        assert_eq!(baseline.records_used, shuffled.records_used);
        for (a, b) in baseline.entities.iter().zip(shuffled.entities.iter()) {
            assert_eq!(a.entity, b.entity);
            assert_eq!(
                a.latent_mean, b.latent_mean,
                "means differ for {:?}",
                a.entity
            );
            assert_eq!(a.latent_std, b.latent_std, "stds differ for {:?}", a.entity);
            assert_eq!(a.rank, b.rank);
        }
    }

    /// A comparison graph with two disjoint clusters compiles into two
    /// components, and `p_higher` refuses to compare across them.
    #[test]
    fn disconnected_graph_yields_two_components_and_no_cross_p_higher() {
        let e: Vec<Entity> = ["a0", "a1", "b0", "b1"]
            .iter()
            .map(|s| Entity::new(*s))
            .collect();
        let records = vec![
            mk_record(
                &e[0],
                &e[1],
                atom_evidence(AnswerAtom::B(3)),
                true,
                false,
                0,
            ),
            mk_record(
                &e[2],
                &e[3],
                atom_evidence(AnswerAtom::B(3)),
                true,
                false,
                1,
            ),
        ];

        let posterior = compile(&records, &e, &Tolerances::default()).unwrap();
        assert_eq!(posterior.components, vec![vec![0, 1], vec![2, 3]]);
        assert!(posterior.p_higher(0, 1).is_some());
        assert!(posterior.p_higher(2, 3).is_some());
        assert!(posterior.p_higher(0, 2).is_none());
        assert!(posterior.p_higher(0, 3).is_none());
        assert!(posterior.p_higher(1, 2).is_none());
        assert!(posterior.p_higher(1, 3).is_none());
    }

    /// Lower informative mass on an otherwise-identical observation widens
    /// the posterior standard deviation. Uses a larger-than-default ridge
    /// so the effect (real for any ridge > 0, per the module doc) is not
    /// swamped by the ridge-dominated gauge-fixing variance term at the
    /// default `ridge = 1e-6`.
    #[test]
    fn lower_informative_mass_widens_posterior_std() {
        let e: Vec<Entity> = ["x", "y"].iter().map(|s| Entity::new(*s)).collect();
        let tolerances = Tolerances {
            var_floor: 1e-3,
            ridge: 1.0,
        };

        let full_mass = crate::evidence::evidence_from_logprobs(
            &[crate::evidence::AtomLogprob {
                atom: AnswerAtom::B(5),
                logprob: 1.0_f64.ln(),
            }],
            Some(1.0),
        )
        .unwrap();
        let half_mass = crate::evidence::evidence_from_logprobs(
            &[crate::evidence::AtomLogprob {
                atom: AnswerAtom::B(5),
                logprob: 0.5_f64.ln(),
            }],
            Some(0.5),
        )
        .unwrap();
        assert!(half_mass.informative_mass() < full_mass.informative_mass());
        // Same point-mass shape informatively, so equal reported variance;
        // only the informative-mass factor in the weight should differ.
        assert_eq!(
            full_mass.log_ratio_moments().unwrap().1,
            half_mass.log_ratio_moments().unwrap().1
        );

        let confident = compile(
            &[mk_record(&e[0], &e[1], full_mass, true, false, 0)],
            &e,
            &tolerances,
        )
        .unwrap();
        let uncertain = compile(
            &[mk_record(&e[0], &e[1], half_mass, true, false, 1)],
            &e,
            &tolerances,
        )
        .unwrap();

        assert!(
            uncertain.entities[0].latent_std > confident.entities[0].latent_std,
            "uncertain std {} should exceed confident std {}",
            uncertain.entities[0].latent_std,
            confident.entities[0].latent_std
        );
        assert!(uncertain.entities[1].latent_std > confident.entities[1].latent_std);
    }

    /// Refused records are skipped and counted, and contribute no edge:
    /// two entities linked only by a refused record end up in separate
    /// singleton components with no rank and no cross `p_higher`.
    #[test]
    fn refused_records_are_skipped_and_counted() {
        let e: Vec<Entity> = ["r0", "r1"].iter().map(|s| Entity::new(*s)).collect();
        let records = vec![mk_record(
            &e[0],
            &e[1],
            atom_evidence(AnswerAtom::B(1)),
            true,
            true, // refused
            0,
        )];

        let posterior = compile(&records, &e, &Tolerances::default()).unwrap();
        assert_eq!(posterior.records_used, 0);
        assert_eq!(posterior.records_skipped_refused, 1);
        assert_eq!(posterior.records_skipped_uninformative, 0);
        assert_eq!(posterior.components, vec![vec![0], vec![1]]);
        assert_eq!(posterior.entities[0].rank, None);
        assert_eq!(posterior.entities[1].rank, None);
        assert!(posterior.p_higher(0, 1).is_none());
    }

    /// An unknown entity in a record's presentation is reported, not
    /// silently dropped.
    #[test]
    fn unknown_entity_is_an_error() {
        let e = vec![Entity::new("known")];
        let stranger = Entity::new("stranger");
        let records = vec![mk_record(
            &e[0],
            &stranger,
            atom_evidence(AnswerAtom::B(1)),
            true,
            false,
            0,
        )];
        let err = compile(&records, &e, &Tolerances::default()).unwrap_err();
        assert!(matches!(err, CompileError::UnknownEntity(id) if id == stranger.id));
    }
}
