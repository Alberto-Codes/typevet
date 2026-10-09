# Run a small live judgment eval

Kind: how-to.

Run one **bounded** live check after the router and Gemma 4 weights work. Pick
**TPJEP eight** (in-repo pytest) or read the **frozen partner-6** receipt on
[#133](https://github.com/Alberto-Codes/typevet/issues/133). Both use
`ScoringJudgmentAdapter` and local candidate scoring.

## Prerequisites

1. Complete [Run Gemma 4 on llama.cpp](run-gemma4-llamacpp.md).
2. Set `TYPEVET_LLAMA__BASE_URL` if the router is not `http://127.0.0.1:8090`.
3. Pin model id **`gemma-4-31b-24gib-kv11-decoder`** unless your catalog differs.

Live pytest **skips** when the router is down. A skip is not a pass.

## Option A — TPJEP eight-task smoke

This path exercises eight vendored JevBench rows through `JudgmentPort`. Gold
`expected` values never enter model inputs.

Offline proof first:

```bash
uv run pytest evals/tests/unit/test_eval_tpjep_loader.py evals/tests/contract/test_tpjep_runner_offline.py -q
```

Live run (slow first load):

```bash
uv run pytest evals/tests/live/test_tpjep_smoke_live.py -m live -q
```

Artifacts land under `scratchpad/tpjep/` (gitignored). Details:
[TPJEP v0 eight-task runner](../reference/eval-tpjep-runner.md).

## Option B — frozen partner-6 receipt (read-only)

typevet does **not** ship a partner-6 pytest yet. The supervisor recorded a
**post-#152** live receipt on
[#133](https://github.com/Alberto-Codes/typevet/issues/133#issuecomment-5843131304).

| Field | Value |
|---|---|
| Code tip | [`3f25851`](https://github.com/Alberto-Codes/typevet/commit/3f25851) |
| Binding fix | [#152](https://github.com/Alberto-Codes/typevet/issues/152) [`249f168`](https://github.com/Alberto-Codes/typevet/commit/249f168) |
| Model | `gemma-4-31b-24gib-kv11-decoder` |
| Template | Degraded ChatML + Control→label mapping |
| Broad fraud agreement | **6/6** (was 3/6 pre-fix on the same six rows) |
| Semantic controls | **pos_unauthorized** and **neg_ordinary** both **passed** |
| Elapsed | 29.849 s for 18 workload fields + 6 control fields |

Interpretation stays honest: binding repair **changed** decisions away from
all-`duplicate_charge`. That evidence does **not** mean calibration or close
#133. Full table:
[Judgment live receipts](../reference/judgment-live-receipts.md).

## After the run

- Compare `n_answered` and `n_prob_valid` in the TPJEP summary JSON.
- Do not report ECE or “calibrated” from these smokes alone.
- File or extend an issue when you need a new pin or a larger sample.
