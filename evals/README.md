# typevet eval manifests

Kind: reference (index).

This directory holds the **versioned eval inventory** for typevet: licenses,
Decision shapes, split notes and exclusions. Future loaders then do not repeat
the dataset research.

It is also the `typevet-evals` uv workspace member. Its import package is
`typevet_evals`, under `src/`, and its tests are under `tests/`. The member
holds the eval runner, the dataset loaders, the evaluation harnesses and the
wheel proof tools. It is never published, and the library wheel does not hold
it. The library never imports `typevet_evals`. See
[ADR 0002](../docs/adr/0002-package-layout.md) and
[supported imports](../docs/reference/supported-imports.md#not-in-the-wheel).

The manifests and `fixtures/` stay tracked, but they are not package data.

## Files

| File | Purpose |
|---|---|
| [complementary-manifest.yaml](complementary-manifest.yaml) | JevBench pointer + #54 ranked complementary text sets (#63) |
| [fixtures/difraud_is_scam_noul_schema_v1.json](fixtures/difraud_is_scam_noul_schema_v1.json) | Versioned `is_scam` Noul JSON Schema for DIFrauD (#59) |
| [fixtures/civil_comments_is_toxic_noul_schema_v1.json](fixtures/civil_comments_is_toxic_noul_schema_v1.json) | Versioned `is_toxic` Noul JSON Schema for Civil Comments (#72) |
| [fixtures/civil_comments_tier_manifest_v1.yaml](fixtures/civil_comments_tier_manifest_v1.yaml) | Seeded tier A/B limits and τ=0.5 policy (#72) |
| [fixtures/pubmedqa_answer_choice_schema_v1.json](fixtures/pubmedqa_answer_choice_schema_v1.json) | Versioned yes/no/maybe Choice schema for PubMedQA (#78) |
| [fixtures/boolq_answer_noul_schema_v1.json](fixtures/boolq_answer_noul_schema_v1.json) | Versioned `answer` Noul JSON Schema for BoolQ (#77) |
| [fixtures/boolq_tier_manifest_v1.yaml](fixtures/boolq_tier_manifest_v1.yaml) | Seeded tier A/B limits and CC BY-SA smoke policy (#77) |
| [fixtures/hyperpartisan_hyperpartisan_noul_schema_v1.json](fixtures/hyperpartisan_hyperpartisan_noul_schema_v1.json) | Versioned `hyperpartisan` Noul JSON Schema (#81) |
| [fixtures/hyperpartisan_holdout_manifest_v1.yaml](fixtures/hyperpartisan_holdout_manifest_v1.yaml) | Stratified holdout from byarticle train; excludes bypublisher (#81) |
| [fixtures/clinc_banking_intent_choice_schema_v1.json](fixtures/clinc_banking_intent_choice_schema_v1.json) | Versioned 15-intent banking Choice schema for CLINC (#73) |
| [fixtures/clinc_in_scope_noul_schema_v1.json](fixtures/clinc_in_scope_noul_schema_v1.json) | Optional OOS `in_scope` Noul schema for CLINC (#73) |
| [fixtures/go_emotions_emotion_choice_schema_v1.json](fixtures/go_emotions_emotion_choice_schema_v1.json) | Versioned 24-emotion Choice schema for go_emotions (#79) |
| [fixtures/psai_metadata_decisions_schema_v1.json](fixtures/psai_metadata_decisions_schema_v1.json) | Combined metadata Choice/Noul schema for PSAI (#71) |
| [fixtures/psai_metadata_manifest_v1.yaml](fixtures/psai_metadata_manifest_v1.yaml) | Metadata-only slice manifest; excludes vision (#71) |

Human-readable commentary and epic links:
[Eval complementary manifest](../docs/reference/eval-complementary-manifest.md).

Loader reference pages (Python modules under `src/typevet_evals/datasets/`):

| Corpus | Doc |
|---|---|
| BoolQ | [BoolQ loader and answer Noul](../docs/reference/eval-boolq-loader.md) |
| CLINC150 | [CLINC loader and domain Choice](../docs/reference/eval-clinc-loader.md) |
| Hyperpartisan | [Hyperpartisan loader and hyperpartisan Noul](../docs/reference/eval-hyperpartisan-loader.md) |
| PSAI computer-use (metadata) | [PSAI metadata Decision map](../docs/reference/eval-psai-metadata-map.md) |

## Rules

- Manifests record **metadata only**. No corpus download scripts here.
- Partner-only finvet data stays out; see
  [Eval partner data policy](../docs/reference/eval-partner-data-policy.md).
- Primary typed-decision bench remains **JevBench** ([#24](https://github.com/Alberto-Codes/typevet/issues/24));
  complementary sets are stress/regression only ([#54](https://github.com/Alberto-Codes/typevet/issues/54)).

Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51).
