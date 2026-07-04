# Architecture

## The invariant

**Nothing in seriate fabricates a number without a judgement-record
ancestor.** Every latent score, every rank, every probability traces through
an unbroken id chain to raw provider bytes:

```
CompiledPosterior
  └─ JudgementRecord (content-addressed; id covers every field)
       ├─ AnswerEvidence (normalized PMF + PmfCompleteness)
       ├─ Presentation (which entity in which slot — auditable order)
       ├─ TemplateHash / ParserVersion / DecodeConfig / model
       └─ CaptureId ──► ProviderCapture (raw response bytes; id covers
                        request fingerprint, model, path, time, bytes)
```

`seriate provenance <judgement-id>` prints this chain, terminating in the
bytes. The chain is tamper-evident at every link: ids are blake3 over
domain-tagged canonical serializations; reads re-verify, not just writes.

## Layers

| Module | Owns | Refuses to own |
|---|---|---|
| `atom` | the 52-letter answer alphabet, ratio ladder, reflection algebra | prompt text |
| `evidence` | PMFs, completeness accounting, logprob/sample/fused construction, JSD | what the PMF is *about* |
| `ontology` | content-addressed entities, attributes, pairs, presentations | judgement storage |
| `record` | the immutable judgement record | acquisition |
| `capture` | raw provider bytes as events | parsing |
| `gateway` | one OpenRouter exchange, whole logprob array preserved | position disambiguation |
| `instrument` | render + parse contracts (ratio-letter, ordinal, k-wise, scalar) | networking, storage |
| `log` | append-only SQLite, provenance walks, JSONL export/import | interpretation |
| `compile` | evidence → ordering posterior (canonicalize, weigh, WLS, report) | elicitation |
| `probe` | the logprob reality map | judgement of models |

Boundary discipline: active pair selection, top-k stopping, and sorting UX
belong to [cardinal-harness](https://github.com/XyraSinclair/cardinal-harness),
which will consume seriate. Not here.

## Design decisions with teeth

1. **A capture is an event, not a claim.** Its id covers request
   fingerprint, model, path, timestamp, AND bytes — so byte-identical
   responses to different requests never collide, and an id collision can
   only mean identical content. (The first integration battery run caught
   the raw-bytes-only version of this id colliding across pairs.)
2. **Reads fail closed.** `verify_id()` runs on every judgement read, not
   only on insert — a row mutated behind the log's back never flows
   downstream. Import re-verifies every line and rejects the whole file on
   the first tampered record.
3. **`serde_json/float_roundtrip` is load-bearing.** Without it, JSON float
   parsing is off by 1 ULP and content ids do not survive storage. A
   provenance system that cannot round-trip its own records is lying;
   the battery caught this on day one.
4. **Gauge modes cancel exactly, not approximately.** The compiler keeps
   the full posterior covariance: per-entity spread is reported in the
   centered gauge and pairwise `p_higher` uses the cross-covariance term,
   so the ~1/ridge gauge artifact cancels algebraically. (Diagonal-only
   variance made `p_higher` collapse to 0.5 — battery catch #3.)
5. **Ordinal first.** The compiler produces defensible posteriors from
   direction-only evidence (fixed modest magnitude, PMF carries the
   uncertainty); ratio magnitudes upgrade precision but are never required.
6. **Escape mass is never dropped.** Visible-but-unparseable tokens are
   `OffAlphabet`; never-shown probability is `Abstain`; the split is
   computed, stored, and consulted by evidence weighting.
