# PSAI vision Choice probability evidence

Kind: reference.

This page documents offline raw probability evidence for issue
[#180](https://github.com/Alberto-Codes/typevet/issues/180) revision 3. Parent
capability work lives under
[#109](https://github.com/Alberto-Codes/typevet/issues/109).

## Tracked fixtures

| Path | Role |
|---|---|
| `tests/fixtures/psai/vision_choice_evidence/source_matrix_v1.json` | Sanitized rev2 visual Choice matrix (16 scoring calls) |
| `tests/fixtures/psai/vision_choice_evidence/source_c10_repair_v1.json` | Two-call C10 opposite-gold donor repair |
| `tests/fixtures/psai/vision_choice_evidence/corrected_v1.json` | Versioned corrected artifact with SHA-256 source links |

Evaluation helpers live in the `typevet-evals` workspace member, in
[`typevet_evals.psai_vision_probability_evidence`](https://github.com/Alberto-Codes/typevet/blob/main/evals/src/typevet_evals/psai_vision_probability_evidence.py).

## Raw mass vs normalized confidence

| Term | Definition |
|---|---|
| **Raw mass** | `sum(exp(logprob))` over declared single-token candidates |
| **Legacy denominator** | `sum(exp(logprob - max(logprob)))` (scratchpad bug; ≈1 for two labels) |
| **Normalized confidence** | Softmax over declared candidates only (TypeLLM execute path) |

Repair recomputes raw mass from preserved `raw_logprobs`. It does not change
labels, predictions, or normalized confidence.

## Cumulative scoring calls

| Run | Calls | Status |
|---|---|---|
| Invalid Hub category matrix (protocol B) | 16 | Void for rev2; recorded separately |
| Rev2 visual Choice matrix | 16 | Source for `source_matrix_v1.json` |
| C10 swap donor repair | 2 | Source for `source_c10_repair_v1.json` |

The invalid Hub matrix and the rev2 matrix are different runs. The repair budget
is additive.

## Omitted arms and C10 history

Four **omitted** visibility arms (`C01`, `C03`, `C08`, `C10`) are expected
**non-credits**. They remain in the artifact for audit; they are not semantic
passes.

The initial rev2 matrix failed **C10-swapped** when the swap donor was
`cmcc8u6ym018l1p1yxhf18gc2` (neg/neg for the Shop button question). Repair
re-pinned the donor to `cmcc8u6yd00wr1p1yj7aot3ae` (freeze C04, opposite-gold).

## Completeness verdict

A corrected artifact is **complete** when source SHA-256 digests match the
pinned JSON files and every outcome carries recomputed raw mass. That verdict
does **not** claim a pristine 16/16 semantic pass: four omitted arms fail by
design, one swap failed before donor repair, and the earlier Hub matrix stays
invalid.

## Offline verification

```bash
uv run pytest evals/tests/unit/test_psai_vision_probability_evidence.py -q
```

No live inference is required to verify mass arithmetic.
