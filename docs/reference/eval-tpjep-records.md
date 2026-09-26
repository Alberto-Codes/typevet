# TPJEP v0 attempt result records

Kind: reference.

Parent: [#131](https://github.com/Alberto-Codes/typevet/issues/131),
[#130](https://github.com/Alberto-Codes/typevet/issues/130) (protocol pin),
[#106](https://github.com/Alberto-Codes/typevet/issues/106) (live smoke runner).

Offline JSONL holds **one frozen object per scheduled attempt**. Skips and
transport failures stay in ``n_scheduled``. Raw prompts and chain-of-thought
are **not** stored by default.

## Module

| Piece | Location |
|---|---|
| Record + summary types | ``typevet.eval_tpjep_records`` |
| Mixed smoke JSONL (5 rows) | ``tests/fixtures/tpjep/mixed_attempts_smoke.jsonl`` |
| Unit tests | ``tests/unit/test_eval_tpjep_records.py`` |
| Eight-task runner | [eval-tpjep-runner.md](eval-tpjep-runner.md) |

## Required fields

``task_id``, ``source_tier``, ``question_type``, ``model``,
``dataset_git_commit``, ``dataset_hash_recipe``, ``dataset_hash``,
``protocol`` (``TPJEP-v0``), ``outcome`` (``answered`` | ``prob_invalid`` |
``schema_invalid`` | ``transport_failed`` | ``skipped``), ``predicted``,
``expected`` (scoring only), ``prob_valid``, ``correct`` (nullable),
``duration_ms``. Optional: ``probabilities``, ``error_type`` /
``error_message`` (bounded), ``usage``, ``server_build``, ``template_class``.

## Summary from records

``summarize_tpjep_records`` returns counts only from the JSONL list:

| Counter | Meaning |
|---|---|
| ``n_scheduled`` | Lines in the attempt file |
| ``n_prob_valid`` | Rows with ``prob_valid`` true |
| ``n_correct`` | Rows with ``correct`` true |
| ``accuracy_on_prob_valid`` | ``n_correct / n_prob_valid`` when denominator > 0 |
| ``prob_valid_rate`` | ``n_prob_valid / (n_scheduled - n_skipped)`` |
| ``n_success`` | Same as ``n_correct``; skips never count as success |

Dataset pin fields (``dataset_git_commit``, ``dataset_hash_recipe``,
``dataset_hash``, ``protocol``, ``model``) must match across all rows in one
run file.

## Commands

```bash
uv run pytest tests/unit/test_eval_tpjep_records.py -m unit -q
```
