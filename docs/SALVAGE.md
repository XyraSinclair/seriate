# Salvage map: the diamond2 quarry

seriate was built greenfield against a read-only audit of
`diamond2/crates/cardinal-harness-v2` (~230K-line `lib.rs`, never shipped).
Two independent auditors produced keep/redesign/discard verdicts; this is the
distilled record of what was taken and what was deliberately left.

## Taken verbatim (as ideas with fresh implementations)

- **The letter-case answer alphabet** (`A`/`a` parity, `B–Z` slot-A wins,
  `b–z` slot-B wins, 25-rung geometric ladder to 1000×): the crux jewel —
  a single completion token whose top-k logprobs ARE the judgement PMF.
  → `src/atom.rs`.
- **Three-way mass partition** (parsed atoms / OffAlphabet / Abstain):
  visible-but-unparseable mass and never-shown mass are separate, named
  quantities; nothing is silently dropped. → `evidence_from_logprobs`.
- **Continuous-ratio interpolation** preserving `E[log-ratio] = ln(r)`
  exactly (log-linear over adjacent rungs). → `interpolate_ratio`.
- **Provider-rounding clamp** (chosen-token logprob rounded to 0.0 pushing
  mass past 1.0): forgiven up to 1e-4, rejected beyond. → `evidence.rs`.
- **PairKey/Presentation canonicalization** with the direction-safety
  invariant (sign canonicalized against pair order, independent of
  presented slots). → `ontology.rs` + `compile.rs`.
- **Tolerant answer parsing + answer-token position disambiguation**
  (JSON-in-chatter recovery; picking the logprob position that is the
  answer, not a key token). → instruments.
- **Pure-function provenance boundary**: prompt rendering and response
  parsing are pure; raw provider bytes are the anchor. → `capture.rs`.

## Redesigned

- `PmfCompleteness::Bounded{-inf,+inf}` placeholder for fused evidence →
  honest `Fused { logprob_shown_mass, samples, weights }` variant.
- All-pub structs without validating constructors → content-addressed
  records built only through constructors that compute and verify ids.
- `EvidenceError` mixing evidence and matrix concerns → single-concern
  error enums per module.
- Five scattered float epsilons → named tolerances.
- 230K single-file monolith → focused modules.

## Discarded

- The entire `ontology.rs` "point ratio judgment" spine (PromptAction /
  Observation / RatioJudgment / EdgeEvidence …): zero call sites; it was
  superseded in-place by the PMF pipeline and never deleted. We ported the
  two ideas inside it (canonicalization, sign-flip-by-presentation) and
  none of the structs.
- `StopReport`/`StopConfig` stopping rule: aspirational, never wired.
- `ContrastKind::Padding`: decorative enum arm, no producer or consumer.
- Cache-padding plans: sound mechanism, gap-ridden implementation
  (char-count as token proxy, no under-pad postcondition); deferred until
  prompt-cache economics matter here.
- Self-reported-confidence fusion (`fused_ratio_json_evidence`): dead
  duplicate; the decision "is stated confidence a third evidence channel?"
  is still open and tracked, not silently inherited.

## Hazards the audit found (so nobody re-imports them)

- `shown_mass` vs `measurable_mass` were overlapping-but-different unit-mass
  partitions with similar names — the easiest porter trap in the quarry.
- Hand-built observations could silently misattribute missing mass
  (`visible_logprob_mass: None` defaulting) — seriate requires explicit
  visible mass whenever logprobs are present.
- `f64::INFINITY` as a "no data" sentinel — replaced with `Option`.
- Boundary-risk math consuming only the covariance diagonal while a
  full-covariance quality flag existed — documented in `compile.rs` as an
  explicit diagonal approximation.
