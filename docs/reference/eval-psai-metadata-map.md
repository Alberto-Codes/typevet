# PSAI metadata loader and Decision map

Kind: reference. MIT metadata-only System One slice for computer-use PSAI.
Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51);
design [#56](https://github.com/Alberto-Codes/typevet/issues/56); research
[#52](https://github.com/Alberto-Codes/typevet/issues/52); loader
[#71](https://github.com/Alberto-Codes/typevet/issues/71).

## Role in typevet

| Piece | Module / path |
|---|---|
| Metadata loader + export | `typevet.evaluation.datasets.psai` |
| Parquet streaming (column projection) | `typevet.evaluation.datasets.psai_download` |
| Dedupe / shuffle / JSONL | `typevet.evaluation.datasets.psai_stream` |
| Combined Decision JSON Schema | `evals/fixtures/psai_metadata_decisions_schema_v1.json` |
| Manifest | `evals/fixtures/psai_metadata_manifest_v1.yaml` |
| CI smoke JSONL (12 rows) | `tests/fixtures/psai/metadata_smoke.jsonl` |

Complementary metadata regression only. Do not claim GUI understanding, task
success, or vision metrics ([#57](https://github.com/Alberto-Codes/typevet/issues/57)).

## Hub → Decision field map (v1)

Eval unit is one **task** row (dedupe on `unique_data_id`). Sample flow: stream
metadata → dedupe → **shuffle** (seeded hash order) → optional limit.

| Hub field | Decision | Syntax | Labels / notes |
|---|---|---|---|
| `category` | `category` | Choice (2) | `BROWSER_TASK`, `COMPUTER_TASK` |
| `difficulty` | `difficulty` | Choice (3) | `EASY`, `MEDIUM`, `HARD` |
| `benchmark` | `benchmark` | Choice (3) | `Web Bench`, `Proprietary`, `Web Voyager` |
| `appType` | `appType` | Choice (2) | `SINGLE_APP`, `MULTI_APP` |
| `os` | `os` | Choice (≤4) | `CROSS_PLATFORM`, `WINDOWS`, `MAC`, `LINUX` |
| `requires_login` | `requires_login` | Noul (bool) | `""` → false; `yes` / `no` strings |
| `subCategory[0]` | `sub_category` | Choice + `OTHER` | 17 catalog labels + `OTHER` fallback |

Out of v1: event-count **Score**, screenshot/video/DOM decode, full corpus in CI.

## State shape

Exported `state` is an object with **`task_name` only** (natural-language task
description). Gold metadata labels live under `expected`, not in `state`. Banned
keys include `category`, `difficulty`, `screenshots`, `events`, and standard
leakage keys.

## Streaming access

Offline CI uses vendored JSONL. At eval time, operators may stream train
parquet shards from Hugging Face:

1. List shard URLs via the datasets-server `/parquet` API.
2. Read **metadata columns only** with pyarrow (never load `screenshots` or
   `events` columns).
3. Dedupe, shuffle, and map through `load_train_split`.

Pyarrow is required for parquet streaming; it is not a core typevet runtime
dependency.

## License

MIT ([HF dataset card](https://huggingface.co/datasets/anaisleila/computer-use-data-psai)).
Listed under deferred multimodal scope in
[Eval complementary manifest](eval-complementary-manifest.md) until vision port.

## Related pages

- [Eval partner data policy](eval-partner-data-policy.md)
- [BoolQ loader and answer Noul](eval-boolq-loader.md)
