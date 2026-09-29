# Testing pyramid and markers

Kind: reference.

This page is the lookup for pytest markers, default commands, and what each
layer can verify. Law lives in [CLAUDE.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md) and
[AGENTS.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md). Narrative and fixture labeling sit in
[Verified evidence and inferred claims](../explanation/verification.md).

## Three layers

| Layer | Marker | Path | Default CI | Coverage counted |
|---|---|---|---|---|
| Unit | `unit` | `tests/unit/` | yes | yes |
| Contract | `contract` | `tests/contract/` | yes | yes |
| Live | `live` | `tests/live/` | no | no |

Default pytest excludes `live` (`-m "not live"` in `pyproject.toml`). The
default suite must keep **≥ 90** coverage (`tool.coverage.report.fail_under`).
A live pass does not replace unit or contract proof.

## What each layer proves

**Unit.** Pure domain rules, inbound wiring with fakes, adapter edge paths with
controlled inputs. No real network and no live model weights.

**Contract.** The offline fake generation adapter and
``LlamaCppGenerationAdapter`` (sync) or ``AsyncLlamaCppGenerationAdapter``
(async) behave the same on **shared fixtures** under
[tests/fixtures/generation_contract.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/fixtures/generation_contract.py).
HTTP is replayed with ``httpx.MockTransport``. Agreement verifies adapter
compatibility for those labeled cases only.

**Live.** One exercised call against the configured router and model. Opt in
with ``pytest -m live``. See [Run Gemma 4 on llama.cpp](../how-to/run-gemma4-llamacpp.md).

For release evidence, set ``TYPEVET_REQUIRE_LIVE=1`` so missing router or model
configuration **fails** live tests instead of skipping. See
[live eval runner](eval-live-runner.md).

The [live eval runner](eval-live-runner.md) adds an optional BoolQ/Banking77
slice with attempted / schema-valid / gold-match counters over
``GenerationPort``. That is task accuracy on gold for a tiny limit, not ECE.

Valid JSON shape for a run is not the same as correct judgment. Do not infer
calibration or task accuracy from pyramid passes alone unless the run used the
loader eval runner and you report its gold-match counter explicitly.

## Shared GenerationPort fixtures (judgevet shape)

Contract fixtures are **synthetic**: the test author defines the request, fake
value or failure, and mocked chat-completion body. Each fixture has a ``name``
and ``label`` field for scope reporting on issues.

Sync parity: [tests/contract/test_outbound.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_outbound.py).
Async parity: [tests/contract/test_async_outbound.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_async_outbound.py).

Add new port behaviour to the fixture list first, then extend fakes and the
llama.cpp adapter until both sides agree. Do not weaken the default coverage
floor or add a fourth pyramid layer to do it.

## Shared JudgmentPort fixtures (judgevet shape)

Offline judgment contract fixtures live in
[tests/fixtures/judgment_contract.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/fixtures/judgment_contract.py).
They exercise `ContractJudgmentFake` against labeled success and error cases.
There is no live llama.cpp judgment adapter in the default suite yet.

Suite: [tests/contract/test_judgment_port.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_judgment_port.py).

Scoring-backed judgment adapter fixtures live in
[tests/fixtures/judgment_scoring_contract.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/fixtures/judgment_scoring_contract.py).
Suite: [tests/contract/test_judgment_scoring_adapter.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_judgment_scoring_adapter.py).

## Shared CandidateScoringPort fixtures

Offline scoring contract fixtures live in
[tests/fixtures/scoring_contract.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/fixtures/scoring_contract.py).
`ContractScoringFake` there aliases public
[typevet.testing.ScriptedScoringFake](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/testing/__init__.py).
They exercise that fake against labeled success and fail-closed
error cases (missing candidates, non-finite logprobs, unsupported stage).
There is no live llama.cpp scoring adapter in the default suite yet.

Suite: [tests/contract/test_scoring_port.py](https://github.com/Alberto-Codes/typevet/blob/main/tests/contract/test_scoring_port.py).

## Commands

```bash
uv run pytest -m "unit or contract"
uv run pytest -m contract
uv run pytest -m live   # opt-in; not default CI
uv run pytest --cov=typevet --cov-report=term-missing
uv run mkdocs build --strict   # site build; also a unit test
```

Pre-commit runs unit and contract via the configured pytest hook. Import-linter
contracts in `pyproject.toml` are unrelated to pytest ``contract`` markers.

## Related pages

- [Glossary — testing pyramid](glossary.md#terms)
- [Worker runs](worker-runs.md) (launch evidence)
- Parent tracking: [issue #28](https://github.com/Alberto-Codes/typevet/issues/28),
  [issue #89](https://github.com/Alberto-Codes/typevet/issues/89)
