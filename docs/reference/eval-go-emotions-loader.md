# go_emotions loader and emotion Choice fixture

Kind: reference. Apache 2.0 simplified config for 24-way emotion Choice.
Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51); loader
[#79](https://github.com/Alberto-Codes/typevet/issues/79); policy
[#67](https://github.com/Alberto-Codes/typevet/issues/67) /
[#80](https://github.com/Alberto-Codes/typevet/issues/80).

## Role in typevet

| Piece | Module / path |
|---|---|
| ``simplified`` train loader | `typevet_evals.datasets.go_emotions` |
| HF datasets-server download | `typevet_evals.datasets.go_emotions_download` |
| Primary Choice | `emotion` (24-enum Hub order minus prune) |
| Versioned JSON Schema | `evals/fixtures/go_emotions_emotion_choice_schema_v1.json` |
| CI JSONL subset | `tests/fixtures/go_emotions/simplified_train_subset.jsonl` |

Conversion and prune rules live in
[go_emotions conversion and prune policy](go-emotions-conversion-and-prune.md).
v1 default is **strict exactly-one** gold (ambiguous multi-label rows drop).

## State shape (v1)

Eval ``state`` is an object:

```json
{"text": "..."}
```

Gold labels live on the example as ``choice_label`` (after strict conversion).
Banned in ``state``: ``labels``, ``label``, ``expected``, ``ground_truth``,
``answer_key``, ``emotion``.

## Splits and sampling

| HF config | Split | typevet use |
|---|---|---|
| ``simplified`` | ``train`` | complementary Choice eval (v1) |

Contract fixtures ship 12 single-label rows. Optional ``balanced=True`` on
``load_train_split`` balances across the 24-enum when ``limit`` allows.

## License

go_emotions is Apache 2.0. Listed in rank 6 of
[Complementary eval manifest](eval-complementary-manifest.md).

## Related pages

- [go_emotions conversion and prune policy](go-emotions-conversion-and-prune.md)
- [Eval complementary manifest](eval-complementary-manifest.md)
