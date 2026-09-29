# Hyperpartisan loader and hyperpartisan Noul fixture

Kind: reference. Public CC BY 4.0 corpus for article-level binary Noul.
Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51);
research [#68](https://github.com/Alberto-Codes/typevet/issues/68); loader
[#81](https://github.com/Alberto-Codes/typevet/issues/81).

## Role in typevet

| Piece | Module / path |
|---|---|
| Byarticle loader + export | `typevet_evals.datasets.hyperpartisan` |
| Primary Noul | `hyperpartisan` (boolean + `return_probabilities`) |
| Versioned JSON Schema | `evals/fixtures/hyperpartisan_hyperpartisan_noul_schema_v1.json` |
| Holdout manifest | `evals/fixtures/hyperpartisan_holdout_manifest_v1.yaml` |
| CI smoke JSONL (12 rows) | `tests/fixtures/hyperpartisan/byarticle_smoke.jsonl` |

Complementary Noul stress only. Do not claim SemEval official by-article test
scores.

## Corpus scope

| Split | In typevet v1 |
|---|---|
| **byarticle train** (645) | Yes — HTML cleanup, `{title, body}` state, stratified holdout |
| **bypublisher** | **No** — distant publisher labels excluded ([#68](https://github.com/Alberto-Codes/typevet/issues/68)) |

Official by-article test labels were not published. Eval uses a **seeded
stratified holdout** carved from the 645 train articles (`holdout_fraction`
0.2, default seed `0`).

## Labels and state

| Field | Noul `expected.hyperpartisan` |
|---|---|
| article-level bool | same boolean |

Exported tasks use object `state` `{title, body}` only. Banned keys include
`hyperpartisan`, `publisher`, and standard leakage keys.

HTML in source bodies is stripped to plain text; bodies are capped at 8000
characters after cleanup.

## Non-goals

- bypublisher split or publisher-name labels
- SemEval hidden-test replication
- Full 645-row corpus committed to git (offline operator fixtures only)

## License

CC BY 4.0. Rank 4 in [Eval complementary manifest](eval-complementary-manifest.md).

## Related pages

- [BoolQ loader and answer Noul](eval-boolq-loader.md)
- [Civil Comments loader and is_toxic fixture](eval-civil-comments-loader.md)
- [Eval partner data policy](eval-partner-data-policy.md)
