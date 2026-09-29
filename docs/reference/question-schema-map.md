# Question records → JSON Schema

Kind: reference. Pure mapping from loader JevBench-shaped ``questions`` entries
to object schemas that [``compile_json_schema``](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/domain/decision_compile.py)
accepts ([#102](https://github.com/Alberto-Codes/typevet/issues/102)).

Parent: [#95](https://github.com/Alberto-Codes/typevet/issues/95) (gap 6).

## Record shape (export)

Complementary loaders attach a ``questions`` list on each exported task. Each
element is a mapping with:

| Key | Required | Notes |
|---|---|---|
| ``name`` | yes | JSON Schema property name; matches ``expected`` keys |
| ``syntax`` | yes | ``Noul``, ``Choice``, or ``Score`` (System One primitive) |
| ``instructions`` | yes | Becomes the property ``instructions`` string |
| ``labels`` | Choice / Score always; Noul optional | See mapping table below |
| ``return_probabilities`` | no | Copied when present (boolean fields and enums) |
| ``depends_on`` | no | Copied to the property schema when present |
| ``permutations`` | no | Copied for enum fields when present |

Examples live in
[``typevet.evaluation.datasets.boolq``](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/evaluation/datasets/boolq.py)
(``questions_payload``),
[``typevet.evaluation.datasets.hyperpartisan``](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/evaluation/datasets/hyperpartisan.py),
and [``typevet.evaluation.datasets.psai``](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/evaluation/datasets/psai.py).

## Mapping rules

| ``syntax`` | ``labels`` | JSON Schema property |
|---|---|---|
| **Choice** | non-empty strings (≤24) | ``type: string``, ``enum: labels`` |
| **Noul** | omitted or empty | ``type: boolean`` (hyperpartisan, PSAI ``requires_login``, DIFrauD ``is_scam``) |
| **Noul** | exactly two non-empty strings | ``type: string``, ``enum: labels``, optional ``return_probabilities`` (BoolQ ``no``/``yes``, CLINC ``in_scope``) |
| **Score** | ≥2 values | Closed enum: ints as ``type: integer``; digit strings converted to ints; other non-empty strings stay ``type: string`` (JevBench hard-tier rubric indices) |

Root object:

- ``type: object``
- ``properties``: one entry per record, in list order
- ``required``: all ``name`` values, same order
- ``additionalProperties: false`` by default (matches versioned eval fixtures)

The mapper does **not** emit ``x-score`` or open numeric ranges; TypeLLM
compilation rejects those. Score maps to the same closed-enum path as Choice.

## API

Module: [``typevet.question_schema``](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/question_schema.py).

| Function | Role |
|---|---|
| ``question_record_to_property`` | One record → ``(name, property)`` |
| ``question_records_to_json_schema`` | Ordered list → root object schema |
| ``compile_question_records`` | List → ``compile_json_schema`` decisions |

## Verification

Unit tests in [``tests/unit/test_question_schema.py``](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_question_schema.py)
round-trip loader ``questions_payload()`` values against versioned fixture
schemas (BoolQ, Hyperpartisan, PSAI metadata) and sample Choice / Score rows.

## Non-goals

- Runtime ``system_one`` execution ([#22](https://github.com/Alberto-Codes/typevet/issues/22))
- Live eval runner ([#98](https://github.com/Alberto-Codes/typevet/issues/98))
- Translating native JevBench task files (top-level ``question.type`` /
  ``criteria``) — only the flattened export records above

## See also

- [TypeLLM, Jev and judgevet](../explanation/typellm-and-judgevet.md) — primitive
  vocabulary and compile subset
- [Complementary eval manifest](eval-complementary-manifest.md)
