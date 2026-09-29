# Issue #177 handoff (consumer proof)

Kind: reference, for maintainers.

## Phase status

| Slice | Status |
|---|---|
| r2 Score EV + accounting | Accepted (contract tests green) |
| r3 harness exit wiring | Accepted after repair tests |
| r4 instruction-variant E2E | **Landed (slice 4)** — `scripts/consumer_instruction_variant_proof.py` + replay metrics |
| r1 wheel-isolated harness | Landed: `scripts/psai_vision_consumer_wheel_proof.py` |
| r5 exclusive receipts + pins | `write_receipt_exclusive`; manifest/image pins on receipt |
| r6 live wheel matrix | Wired — `run_live_wheel_proof` + `typevet_evals.psai_vision_consumer.live` |
| r7 fail-closed repair | **Landed** — `6cf48de` (Refs #177); acceptance [#5849008536](https://github.com/Alberto-Codes/typevet/issues/177#issuecomment-5849008536) |

## Freeze protocol (post before live rerun)

See [consumer live protocol rev 2](consumer-live-protocol-rev2.md). Post that table on
[#177](https://github.com/Alberto-Codes/typevet/issues/177) before any live rerun.

Accepted repair spec: [#5849004384](https://github.com/Alberto-Codes/typevet/issues/177#issuecomment-5849004384).

## Live status

- `TYPEVET_REQUIRE_LIVE=1` with router down: fail-closed (live gate).
- Router up + `TYPEVET_REQUIRE_LIVE=1`: offline wheel proof then isolated live matrix;
  receipts use `evidence_kind: isolated_wheel_live` and expected-outcome acceptance.

## Related

- [#191](https://github.com/Alberto-Codes/typevet/issues/191): `TYPEVET_REQUIRE_LIVE` strict live gate.
- [#132](https://github.com/Alberto-Codes/typevet/issues/132): offline replay metrics (post-repair).
- [#174](https://github.com/Alberto-Codes/typevet/issues/174): runtime factory (`open_gemma_native_vision_judgment`).
- Historical receipt: `tests/fixtures/consumer/live-receipt-v1-checkout.json` (preserved).
- Accounting sidecar: `tests/fixtures/consumer/live-receipt-v1-accounting-correction.json` (14 judgment / 16 scoring).

## Commands

```bash
uv run python scripts/run_psai_vision_consumer_proof.py
uv run python scripts/run_psai_vision_consumer_proof.py --dev --fixture-root tests/fixtures/psai/vision_smoke
TYPEVET_WHEEL_SHA256=$(sha256sum dist/typevet-*.whl | awk '{print $1}') \
  uv run python scripts/psai_vision_consumer_wheel_proof.py
uv run pytest -q evals/tests/unit/test_psai_vision_consumer_failclosed.py
uv run python scripts/consumer_public_api_demo.py
uv run python scripts/run_consumer_instruction_variant_proof.py
uv run python scripts/run_consumer_instruction_variant_proof.py --dev
TYPEVET_WHEEL_SHA256=$(sha256sum dist/typevet-*.whl | awk '{print $1}') \
  uv run python scripts/consumer_instruction_variant_proof.py
uv run pytest -q evals/tests/unit/test_instruction_variant_consumer_proof.py
uv run pytest -q evals/tests/contract/test_gemma_native_vision_wheel_consumer.py
uv run python scripts/gemma_native_vision_wheel_proof.py
```

Wheel harnesses write receipts under a temp ``work_dir`` unless ``--out-dir`` is set;
they do not default to ``tests/fixtures/consumer/``.

## Post scope-correction handoff (#177 / #191)

Executable local checks:

```bash
uv run pytest -q --cov=typevet --cov-report=term-missing
uv run pytest -q tests/contract/test_runtime_gemma_vision_factory.py
uv run pytest -q evals/tests/contract/test_gemma_native_vision_wheel_consumer.py
uv run pytest -q evals/tests/unit/test_outcome_replay_metrics.py
TYPEVET_REQUIRE_LIVE=1 uv run pytest -q evals/tests/unit/test_eval_runner_live_gate.py
uv run python scripts/gemma_native_vision_wheel_proof.py
```

Limitations:

- Runtime proof (factory smoke, live routers) is not quality proof; probabilities stay
  router-derived and uncalibrated.
- Isolated wheel factory smoke uses an offline httpx stub; live matrix rerun is optional
  when ``TYPEVET_REQUIRE_LIVE=1`` and a router are available.
- Historical consumer JSON fixtures keep legacy ``comparison_verdict`` fields; new runs
  emit ``replay_descriptive_label`` and descriptive v2 replay blocks only.

Freeze protocol for slice 4: [consumer instruction-variant protocol](consumer-instruction-variant-protocol.md) (post on #177 before live).

## Slice 4 acceptance spec (issue comment body)

**Accepted specification — instruction-variant consumer proof (slice 4)**

Ready when: groomed child or #177 carries this contract; freeze protocol doc committed.

Done when:

- Wheel-isolated runner `scripts/consumer_instruction_variant_proof.py` executes
  `proof_main` from installed `typevet` with public imports only.
- Two frozen PSAI present-image cases (C01, C03) run under caller-supplied seed vs
  candidate Noul instructions; offline uses `ScriptedScoringFake`; live uses
  `open_gemma_native_vision_judgment` when router open and
  `TYPEVET_REQUIRE_LIVE=1`.
- Call budget frozen before live (≤8 scoring hard cap; 4 scheduled for slice).
- Invalid model and unsupported-template probes retained with nonzero
  `failed_attempts`.
- `#132` `compare_matched_prompt_outcomes` descriptive v2 on saved distributions;
  receipts expose `replay_descriptive_label` (no promotion verdicts).
- Unit + contract tests green; handoff updated. Commit `Refs #177`.


## Limitations

- Isolated wheel live matrix not re-run in repair acceptance; rev2 committed receipt remains the offline verifier anchor.
- Public runtime factory does not replace evaluation harness receipts or wheel proof scripts.
- Offline replay metrics (#132) score saved distributions only; they do not calibrate Gemma.
