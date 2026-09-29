# PubMedQA loader and answer Choice fixture

Kind: reference. Public MIT gold subset for three-way biomedical QA. Parent epic:
[#51](https://github.com/Alberto-Codes/typevet/issues/51); loader issue
[#78](https://github.com/Alberto-Codes/typevet/issues/78); design
[#75](https://github.com/Alberto-Codes/typevet/issues/75).

## Role in typevet

| Piece | Module / path |
|---|---|
| ``pqa_labeled`` loader | `typevet_evals.datasets.pubmedqa` |
| Primary Choice | `answer` (enum `yes`, `no`, `maybe`) |
| Versioned JSON Schema | `evals/fixtures/pubmedqa_answer_choice_schema_v1.json` |
| CI JSONL subset | `tests/fixtures/pubmedqa/pqa_labeled_subset.jsonl` |

Use **``pqa_labeled`` only**. Do not load ``pqa_artificial`` or ``pqa_unlabeled``
through this module.

## State shape (#75)

Eval ``state`` is an object:

```json
{
  "question": "...",
  "contexts": [
    {"label": "BACKGROUND", "text": "..."},
    {"label": "RESULTS", "text": "..."}
  ]
}
```

HF ``context.contexts`` and ``context.labels`` are zipped in Hub list order.
Banned in ``state``: ``answer``, ``expected``, ``label``, ``ground_truth``,
``answer_key``, ``final_decision``, ``long_answer``.

Gold labels live on the example as ``choice_label`` (from HF ``final_decision``).

## Splits and sampling

| HF config | Split | typevet use |
|---|---|---|
| ``pqa_labeled`` | ``train`` | complementary Choice eval |

Contract fixtures target 12–24 rows (`tests/fixtures/...` ships 12). Optional
``balanced=True`` on ``load_labeled_split`` yields equal counts per label when
``limit`` is divisible by three.

## License

PubMedQA is MIT. Listed in rank 3 of
[Complementary eval manifest](eval-complementary-manifest.md).

## Related pages

- [Eval complementary manifest](eval-complementary-manifest.md)
- [Banking77 proxy loader](banking77-proxy-and-metrics.md) — separate corpus pattern
