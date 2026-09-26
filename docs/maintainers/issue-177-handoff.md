# Issue #177 handoff (consumer proof)

Kind: reference, for maintainers.

## Phase status

| Slice | Status |
|---|---|
| r2 Score EV + accounting | Accepted (contract tests green) |
| r3 harness exit wiring | Accepted after repair tests |
| r4 #184 CLI | Unchanged (0/1/2 contract tests) |
| r1 wheel-isolated harness | Landed: `scripts/psai_vision_consumer_wheel_proof.py` |
| r5 exclusive receipts + pins | `write_receipt_exclusive`; manifest/image pins on receipt |
| r6 live wheel matrix | **Wired** — `run_live_wheel_proof` + `psai_vision_consumer_live` (isolated wheel) |

## Freeze protocol (post before live rerun)

See [consumer live protocol rev 2](consumer-live-protocol-rev2.md). Post that table on
[#177](https://github.com/Alberto-Codes/typevet/issues/177) before any live rerun.

## Live status

- `TYPEVET_REQUIRE_LIVE=1` with router down: fail-closed (live gate).
- Router up + `TYPEVET_REQUIRE_LIVE=1`: offline wheel proof then isolated live matrix;
  receipts use `evidence_kind: isolated_wheel_live` and expected-outcome acceptance.

## Related

- [#191](https://github.com/Alberto-Codes/typevet/issues/191): `TYPEVET_REQUIRE_LIVE` strict live gate.
- Historical receipt: `tests/fixtures/consumer/live-receipt-v1-checkout.json` (preserved).
- Accounting sidecar: `tests/fixtures/consumer/live-receipt-v1-accounting-correction.json` (14 judgment / 16 scoring).

## Commands

```bash
uv run python scripts/run_psai_vision_consumer_proof.py
uv run python scripts/run_psai_vision_consumer_proof.py --dev --fixture-root tests/fixtures/psai/vision_smoke
TYPEVET_WHEEL_SHA256=$(sha256sum dist/typevet-*.whl | awk '{print $1}') \
  uv run python scripts/psai_vision_consumer_wheel_proof.py
```
