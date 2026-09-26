# Civil Comments loader and is_toxic Noul fixture

Kind: reference. Public CC0 corpus for toxicity-threshold binary Noul on comment
text. Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51);
research [#65](https://github.com/Alberto-Codes/typevet/issues/65); loader
[#72](https://github.com/Alberto-Codes/typevet/issues/72).

## Role in typevet

| Piece | Module / path |
|---|---|
| Test-split loader | `typevet.eval_civil_comments` |
| Primary Noul | `is_toxic` (boolean + `return_probabilities`) |
| Versioned JSON Schema | `evals/fixtures/civil_comments_is_toxic_noul_schema_v1.json` |
| Seeded tier manifest | `evals/fixtures/civil_comments_tier_manifest_v1.yaml` |
| CI CSV subset | `tests/fixtures/civil_comments/test_subset.csv` |

Civil Comments complements Banking77 (banking proxy) and DIFrauD (scam/legit).
Do not ask fraud or scam questions on these rows.

## Threshold and labels

The Hugging Face card exposes continuous ``toxicity`` in ``[0, 1]``. typevet
adopts **τ = 0.5** (#65):

| Condition | typevet `label` |
|---|---|
| `toxicity >= 0.5` | `toxic` |
| `toxicity < 0.5` | `not_toxic` |

Other attribute columns (`insult`, `threat`, `identity_attack`, …) are **v1
non-goals**; the loader reads ``text`` and ``toxicity`` only.

## Eval tiers (balanced test split)

| Tier | Total rows | Per class | Use |
|---|---:|---:|---|
| A | 200 | 100 | Smoke / fast regression |
| B | 2000 | 1000 | Default complementary card |

``load_tier_a``, ``load_tier_b``, and ``load_tier`` apply hash-based seeded
ordering (not ``random``) and class balance before capping. Default seed: ``0``
(manifest ``default_seed``).

Live probability calibration (Brier/ECE) stays parked until #11 and partner
policy #50; schemas may still declare ``return_probabilities`` for structure.

## Sensitive content

Real Civil Comments text can include slurs and harassment. CI uses a **mild
synthetic subset** under ``tests/fixtures/`` only. Do not commit full-test text
bundles; large offline shards belong in gitignored local paths per #50.

## License

CC0 1.0. Listed as a complementary open set in
[Eval complementary manifest](eval-complementary-manifest.md).

## Related pages

- [DIFrauD loader and is_scam fixture](eval-difraud-loader.md)
- [Banking77 proxy and metrics](banking77-proxy-and-metrics.md)
- [Eval partner data policy](eval-partner-data-policy.md)
