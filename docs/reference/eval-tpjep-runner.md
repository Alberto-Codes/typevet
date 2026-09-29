# TPJEP v0 eight-task runner

Kind: reference.

Parent: [#106](https://github.com/Alberto-Codes/typevet/issues/106),
[#131](https://github.com/Alberto-Codes/typevet/issues/131) (records),
[#130](https://github.com/Alberto-Codes/typevet/issues/130) (pins).

Loads eight vendored JevBench rows, maps them to native ``Noul`` / ``Choice`` /
``Score`` questions, and runs them through ``JudgmentPort``. Gold ``expected``
values never enter model inputs.

## Modules

| Piece | Location |
|---|---|
| Loader + pins | ``typevet.evaluation.tpjep.loader`` |
| Runner + receipt metadata | ``typevet.evaluation.tpjep.runner`` |
| Answer → record scoring | ``typevet.evaluation.tpjep.outcome`` |
| Eight-row fixture | ``tests/fixtures/tpjep/eight_task_smoke.jsonl`` |
| Provenance | ``tests/fixtures/tpjep/PROVENANCE.md`` |

Run metadata records **both** dataset hashes from #130: manifest
(``typellm_manifest_sha256``) on each attempt row, plus manifest and local
concat hashes on ``TpjepRunMetadata``.

## Commands

Offline:

```bash
uv run pytest tests/unit/test_eval_tpjep_loader.py tests/contract/test_tpjep_runner_offline.py -q
```

Live (requires local llama.cpp router; skips are not passes):

```bash
uv run pytest tests/live/test_tpjep_smoke_live.py -m live -q
```

Live JSONL and summary are written under ``scratchpad/tpjep/`` (gitignored).
