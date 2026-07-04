# seriate

*Seriation: the archaeological method of placing artifacts into ordinal series.*

`seriate` is provenanced attribute annotation and LLM prior elicitation over
entity sets. Its unit is the **structured, provenanced judgement**: an
immutable, content-addressed evidence record that never loses its origin —
model, template hash, parser version, decode config, presentation order, raw
provider capture, cost. The object elicited is the LLM's **prior over the
orderings of an entity set**: for entities E and attribute a, a posterior
over rankings, with per-entity latents and uncertainty.

Three invariants:

1. **Evidence-log invariant**: nothing in the system fabricates a number
   without an evidence-record ancestor; `seriate provenance <id>` walks any
   output back to raw provider bytes.
2. **Ordinal-first**: the compiler produces defensible posteriors from purely
   ordinal evidence; ratio magnitudes are an upgrade, never a requirement.
3. **Logprobs when real, loud degradation when not**: instruments run in
   logprob mode (single-token answer encodings whose top-k logprobs yield a
   PMF over the whole judgement space) or sampled mode, and every judgement
   records how much probability mass was actually visible.

Status: v0.1 under construction. Sibling project:
[cardinal-harness](https://github.com/XyraSinclair/cardinal-harness) (active
pair selection, top-k stopping, sorting UX) will consume seriate; the
boundary is deliberate.

License: MIT
