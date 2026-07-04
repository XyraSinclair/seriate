# The logprob reality map — 2026-07-04

First live sweep of `seriate probe`: 13 model slugs, one fixture pair
(sun vs candle, "brightness"), one logprob-mode call (temperature 0,
`top_logprobs: 20`) plus 5 sampled calls (temperature 1) per model.
Total cost ≈ $0.05. Raw table in `table.txt`, structured reports in
`reports.jsonl`, errors in `stderr.txt`.

## Findings

| Model | Logprobs | Depth | Visible mass | JSD vs own samples | Verdict |
|---|---|---|---|---|---|
| openai/gpt-5.4-mini | yes | **5** (20 requested) | 0.983 | **0.128** | real and trustworthy; provider silently caps depth |
| openai/gpt-5.5 | hard 400 | — | — | — | "logprobs are not supported with reasoning models" — structural |
| deepseek/deepseek-v4-flash | yes | 20 | 0.986 | **0.813** | logprobs present but DISAGREE with the model's own sampling |
| anthropic/claude-sonnet-5, claude-haiku-4.5 | no | 0 | — | — | no logprobs via OpenRouter |
| google/gemini-3.1-pro-preview, gemini-3.5-flash | no | 0 | — | — | no logprobs via OpenRouter |
| meta-llama/llama-4-maverick | no | 0 | — | — | serving provider returns none |
| moonshotai/kimi-k2.6, z-ai/glm-5.2 | no | 0 | — | — | none |
| x-ai/grok-4.3 | rejected | — | — | — | provider 400s the logprobs parameter |
| qwen/qwen3.7-plus, mistralai/mistral-medium-3-5 | blocked | — | — | — | account data-policy guardrails, not a model property |

## What this means for elicitation

1. **Logprob mode is an OpenAI-non-reasoning-family privilege** (in this
   sweep). Everything else must run in sampled mode — which seriate
   degrades to loudly, per record, via `AcquisitionMode` and
   `PmfCompleteness`.
2. **Presence is not meaning.** DeepSeek returns a full 20-deep PMF whose
   mass sits far from where the model actually samples (JSD 0.813 vs
   gpt-5.4-mini's 0.128). Consuming logprobs without this agreement check
   would silently poison evidence.
3. **Requested depth is a suggestion.** gpt-5.4-mini returned 5 of 20
   requested alternatives; `PmfCompleteness::Truncated` carries the
   shortfall into every downstream weight.
4. Two provider quirks now encoded in the client: OpenAI's responses path
   requires `max_tokens ≥ 16`; slugs churn fast enough that the catalog
   must be consulted, not memorized.

## Honest caveats

One run, one fixture pair, one prompt shape. The logprob call runs at
temperature 0 while agreement samples run at temperature 1, so a small JSD
gap is expected even from an honest provider (temperature scaling); 0.813
is far beyond that. Depth caps and support may vary by provider routing.
This is a reality *map*, not a permanent verdict — re-run it; it costs a
nickel.
