# seriate

[![CI](https://github.com/XyraSinclair/seriate/actions/workflows/ci.yml/badge.svg)](https://github.com/XyraSinclair/seriate/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

*Seriation: the archaeological method of placing artifacts into ordinal series.*

`seriate` is provenanced attribute annotation and LLM prior elicitation over
entity sets. Its unit is the **structured, provenanced judgement**: an
immutable, content-addressed evidence record that never loses its origin —
model, template hash, parser version, decode config, presentation order, raw
provider capture, cost. The object elicited is the LLM's **prior over the
orderings of an entity set**: for entities E and attribute a, a posterior
over rankings with per-entity latents and uncertainty.

```console
$ seriate annotate --entities tweets.txt --attribute-name rawness \
    --attribute-text "how raw and unguarded the writing is"
$ seriate compile --attribute-name rawness
$ seriate provenance 260f130f          # any number → raw provider bytes
```

## Three invariants

1. **Evidence-log invariant.** Nothing fabricates a number without a
   judgement-record ancestor. `seriate provenance <id>` walks any output
   back to the raw bytes; every link is content-addressed and re-verified
   on read, not just on write. The log is append-only by construction (a
   test greps the source for UPDATE/DELETE).
2. **Ordinal-first.** The compiler produces defensible posteriors from
   purely direction-only evidence; ratio magnitudes upgrade precision but
   are never required.
3. **Logprobs when real, loud degradation when not.** The flagship
   instrument asks for ONE letter from a 52-token alphabet (case = which
   entity, letter = ratio-ladder magnitude), so a single completion
   position's top-k logprobs ARE the model's judgement PMF. Every record
   carries `PmfCompleteness` — how much probability mass was actually
   seen — and `AcquisitionMode` — logprob, sampled, or fused.

## The logprob reality map

Logprob support through OpenRouter is provider-dependent and quietly
broken; `seriate probe` measures it instead of assuming. First live sweep
([receipts](artifacts/live/logprob-reality-2026-07-04/)):

- **gpt-5.4-mini**: real logprobs, visible mass 0.983, JSD 0.128 against
  its own samples — trustworthy. But depth 5 despite requesting 20.
- **gpt-5.5**: `logprobs are not supported with reasoning models` — the
  reasoning class refuses structurally.
- **deepseek-v4-flash**: full 20-deep logprobs whose mass sits far from
  where the model actually samples (JSD **0.813**). Presence ≠ meaning;
  consuming these without the agreement check would poison evidence.
- **Anthropic, Gemini, Llama-4, Kimi, GLM**: no logprobs at all; Grok
  rejects the parameter.

Re-run it yourself; it costs a nickel:

```console
$ seriate probe --models openai/gpt-5.4-mini,anthropic/claude-sonnet-5 --samples 5
```

## Instruments

| Instrument | Question | Answer space | Logprob-native |
|---|---|---|---|
| `ratio-letter` | which has more of X, and how many times more? | 52 letters (case=side, letter=magnitude, `A`=parity, `!`=refuse) | yes — one token |
| `ordinal` | which has more of X? | `A` / `B` / `=` | yes |
| k-wise | which of these k has the most X? | item letters | yes; lowered to weighted pairwise |
| scalar | rate this entity 0–9 on X | digits | yes; control/baseline only |

All evidence lands as normalized PMFs over the answer alphabet with the
unparseable-but-visible mass (`OffAlphabet`) and never-shown mass
(`Abstain`) accounted separately — nothing is silently dropped.

## Compilation

Judgement records → ordering posterior: evidence is canonicalized against
pair order (presentation reflection is exact — a case flip in the
alphabet), weighted by informative mass, parser health, and PMF variance,
then fitted by weighted least squares on the pair graph (ridge-regularized
Laplacian, hand-rolled dense solve). The full posterior covariance is kept
so the gauge mode cancels algebraically: per-entity spreads are centered,
and `p_higher(i, j)` uses the exact difference variance. Disconnected
comparison graphs are reported as components — never silently compared.

## What lives elsewhere

Active pair selection, top-k stopping, and sorting UX belong to
[cardinal-harness](https://github.com/XyraSinclair/cardinal-harness), which
will consume seriate. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/SALVAGE.md](docs/SALVAGE.md) (what was taken from the diamond2 quarry
and what was deliberately left), and the adversarial integration battery in
`tests/` — which found three real bugs (capture-id event semantics, 1-ULP
float round-trip id drift, gauge-mode variance leakage) before any user
could.

## License

MIT
