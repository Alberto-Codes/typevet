# Gemma 4 on the JevBench public original split

Kind: explanation.

Gemma 4 answered 71 of 72 tasks correctly in one run on JevBench's public
original split. The run used typevet's native judgment path and a local
llama.cpp server. This is a baseline for one custom model pack and serving
configuration, not an official JevBench composite or a leaderboard comparison.
[Issue #420](https://github.com/Alberto-Codes/typevet/issues/420) records the run.

## What ran

The runner evaluated all 72 rows of `datasets/public/original.jsonl` at
[JevBench revision bb05a335](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc).
The dataset SHA-256 is `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180`.
typevet ran at revision `5e33f81c8d7420fed2714bc0c3c53448c4dc1793` on 2026-10-03.

The call path used `open_judgment` and `TypevetSystemOnePort`.
The pinned upstream `jevbench.cli summarize` computed the metrics.
The run made no tuning changes, retries for improvement or calibration fits.
The [runner reference](../reference/eval-jevbench-runner.md) gives commands and mappings.

The model was
[Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF](https://huggingface.co/Alberto-Codes/gemma-4-31B-it-fit24gib-GGUF/tree/43db4293a8fdf444c0325b0fe6063592ccd1522d),
revision `43db4293a8fdf444c0325b0fe6063592ccd1522d`.
This custom vramfit pack uses mixed precision: a Q2_K base with per-layer
overrides. The server's Q2_K header alone does not describe the pack.
The loaded decoder and projector hashes matched the published files exactly.
[The identity check](https://github.com/Alberto-Codes/typevet/issues/420#issuecomment-5970381747)
records the match.

| Serving property | Value |
|---|---|
| Alias | `gemma-4-31b-kv9-q4km-mm` |
| Context | 4,096 tokens |
| K/V cache | `q8_0` / `q8_0` |
| Parallel slots | 1 |
| Probed framing | `native_gemma4_turn` |

## Results

| Metric | Result |
|---|---:|
| Attempted / planned tasks | 72 / 72 |
| Correct tasks | 71 |
| Accuracy / family macro accuracy | 0.9861111111 |
| Valid / strict-valid distributions | 72 / 72 |
| Provider failures | 0 |
| Renormalized outputs | 0 |
| Upstream multiclass Brier mean | 0.0074286811 |
| Top-label ECE, 10 bins | 0.0079611596 |
| Score MAE, 12 ordinal rows | 0.0000006022334884 |
| Latency p50 / p95 | 1.178427 s / 1.345814 s |
| Measured monetary cost | `null` (unknown) |

| Family | Correct / attempted tasks |
|---|---:|
| Adequacy | 12 / 12 |
| Extraction | 12 / 12 |
| Intent | 12 / 12 |
| Ordinal | 12 / 12 |
| Policy | 11 / 12 |
| Routing | 12 / 12 |

The sole incorrect task was `original-policy-06-1`. Its gold label was `yes`.
The model predicted `no`, with P(no) = 0.5139805433. The run retained this
answer without a rerun or wording change.

The supervisor checked all 72 raw hashes, unique task coverage and request
field allowlists. Gold metadata did not enter the provider requests.
An independent calculation from the retained distributions matched upstream
multiclass Brier exactly.

## How to read the result

These numbers describe only the public original split. They do not cover
the easy or hard splits, a sealed evaluation, or the official composite.
They do not establish parity with named leaderboard models.
The small public set also cannot establish general calibration quality.

The exact custom pack and serving configuration define this baseline.
The model card's other serving measurements do not describe this run.
Unknown tariffs remain `null`. The ledger's 1.44 USD reservation charge is
not measured spend.

## Evidence

The [compact receipt](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/jevbench/receipts/original_gemma_fit24gib_llama_cpp.json)
holds full-precision metrics, pins, model file hashes, configuration and
source evidence hashes. Raw logs and per-task results remain local.
The compact receipt records the checked result but cannot support independent
score recomputation on its own.
