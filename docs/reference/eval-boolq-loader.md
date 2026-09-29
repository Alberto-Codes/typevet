# BoolQ loader and answer Noul fixture

Kind: reference. Public CC BY-SA corpus for passage-grounded yes/no Noul.
Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51);
research [#64](https://github.com/Alberto-Codes/typevet/issues/64); design
[#75](https://github.com/Alberto-Codes/typevet/issues/75); SA judgment
[#76](https://github.com/Alberto-Codes/typevet/issues/76); loader
[#77](https://github.com/Alberto-Codes/typevet/issues/77).

## Role in typevet

| Piece | Module / path |
|---|---|
| Validation-split loader + export | `typevet.evaluation.datasets.boolq` |
| HF validation JSONL stream | `typevet.evaluation.datasets.boolq_download` |
| Primary Noul | `answer` (`no` / `yes` + `return_probabilities`) |
| Versioned JSON Schema | `evals/fixtures/boolq_answer_noul_schema_v1.json` |
| Seeded tier manifest | `evals/fixtures/boolq_tier_manifest_v1.yaml` |
| CI smoke JSONL (≤24 rows) | `tests/fixtures/boolq/validation_smoke.jsonl` |

BoolQ complements Banking77, DIFrauD, and Civil Comments. Do not ask fraud,
scam, or toxicity questions on these rows.

## Labels and state

Hub ``answer`` is boolean. typevet maps:

| Hub `answer` | Noul `expected.answer` |
|---|---|
| `true` | `yes` |
| `false` | `no` |

Exported tasks use object ``state`` ``{passage, question}`` only. Banned keys:
``answer``, ``expected``, ``label``, ``ground_truth``, ``answer_key``. String
``state`` fallback: ``serialize_boolq_state`` with ``--- passage ---`` and
``--- question ---`` markers (#75).

## Eval tiers (balanced validation split)

| Tier | Total rows | Per class | Use |
|---|---:|---:|---|
| A | 24 | 12 | Smoke / contract (in-repo cap #76) |
| B | 256 | 128 | Seeded regression |

``load_tier_a``, ``load_tier_b``, and ``load_tier`` apply hash-based seeded
ordering (not ``random``) and class balance before capping. Default seed: ``0``.

Live probability calibration (Brier/ECE) stays parked until #11; schemas may
still declare ``return_probabilities`` for structure.

## CC BY-SA and fixtures

BoolQ passages are **ShareAlike** when copied into artifacts ([#76](https://github.com/Alberto-Codes/typevet/issues/76)):

- **In-repo:** metadata plus a **tiny smoke** bundle (12–24 rows) with this
  attribution notice and license call-out.
- **Bulk:** do **not** commit full validation passage JSONL; stream from Hugging
  Face via ``download_validation_jsonl`` or a gitignored local cache.

**Attribution:** BoolQ dataset; Clark et al.; CC BY-SA 3.0. See the
[Google BoolQ card](https://huggingface.co/datasets/google/boolq).

## License

CC BY-SA 3.0. Listed as rank 1 in
[Eval complementary manifest](eval-complementary-manifest.md).

## Related pages

- [Banking77 proxy and metrics](banking77-proxy-and-metrics.md)
- [Civil Comments loader and is_toxic fixture](eval-civil-comments-loader.md)
- [DIFrauD loader and is_scam fixture](eval-difraud-loader.md)
- [PubMedQA loader and answer Choice fixture](eval-pubmedqa-loader.md)
- [Eval partner data policy](eval-partner-data-policy.md)
