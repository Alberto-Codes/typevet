# Judgment live receipts

Kind: reference.

This page lists **measured** small live judgment runs. Each row links primary
evidence on GitHub. Do not copy numbers into product claims without the same
pins.

## finvet-derived six-row receipt (post-#152)

Source comment:
[#133 post-#152 frozen finvet-6 live receipt](https://github.com/Alberto-Codes/typevet/issues/133#issuecomment-5843131304).

| Pin | Value |
|---|---|
| Commit tip | [`3f25851`](https://github.com/Alberto-Codes/typevet/commit/3f25851) |
| Binding fix | [#152](https://github.com/Alberto-Codes/typevet/issues/152) [`249f168`](https://github.com/Alberto-Codes/typevet/commit/249f168) — Control→label in field instructions |
| Model | `gemma-4-31b-24gib-kv11-decoder` |
| Server build | `b11176-f805c57a2` (as recorded on #133) |
| Template | Degraded ChatML + Control→label mapping |
| Sample | 6 finvet-style rows + 2 predeclared semantic controls |
| Elapsed | 29.849 s (24 scored fields total) |

### Outcome summary

| Metric | Pre-fix baseline (#133 exploratory) | Post-#152 |
|---|---|---|
| Broad fraud agreement | 3/6 | **6/6** |
| All Choice = `duplicate_charge` | yes | **no** |
| Semantic controls | not reported on baseline row | **both passed** |

### Semantic controls (post-#152)

| Control | passed | Modal choice / noul (short) |
|---|---|---|
| pos_unauthorized | True | `unauthorized_transaction`, noul ≈ 0.993 |
| neg_ordinary | True | `not_fraud`, noul ≈ 0.010 |

Per-row Choice, unauthorized noul, urgency, and broad flags are in the #133
comment table. Full probability maps were stored in
`scratchpad/finvet6/post152_receipt.json` (gitignored).

### What this receipt does not prove

- Calibration, ECE, or production quality ([#133](https://github.com/Alberto-Codes/typevet/issues/133) remains open).
- Native Gemma chat template parity ([#129](https://github.com/Alberto-Codes/typevet/issues/129)).
- Generalization beyond the frozen six messages.

## TPJEP eight-task live smoke

| Pin | Value |
|---|---|
| Test | `tests/live/test_tpjep_smoke_live.py` |
| Fixture | `tests/fixtures/tpjep/eight_task_smoke.jsonl` |
| Default model | `gemma-4-31b-24gib-kv11-decoder` |
| Command | `uv run pytest tests/live/test_tpjep_smoke_live.py -m live -q` |
| Artifacts | `scratchpad/tpjep/live_summary.json`, `live_attempts.jsonl` |

Receipt rules live in `tests/fixtures/tpjep/live_acceptance.py`. See
[TPJEP v0 eight-task runner](eval-tpjep-runner.md).

## Related pages

- [Run a small live judgment eval](../how-to/run-a-small-live-judgment-eval.md)
- [Native typed judgments — limitations](../explanation/native-typed-judgments.md#limitations)
- [Verified evidence and inferred claims](../explanation/verification.md)
