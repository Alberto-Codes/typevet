# typevet eval manifests

Kind: reference (index).

This directory holds **versioned eval inventory** for typevet — licenses,
Decision shapes, split notes, and exclusions — so future loaders do not
re-litigate dataset research.

## Files

| File | Purpose |
|---|---|
| [complementary-manifest.yaml](complementary-manifest.yaml) | JevBench pointer + #54 ranked complementary text sets (#63) |
| [fixtures/difraud_is_scam_noul_schema_v1.json](fixtures/difraud_is_scam_noul_schema_v1.json) | Versioned `is_scam` Noul JSON Schema for DIFrauD (#59) |
| [fixtures/civil_comments_is_toxic_noul_schema_v1.json](fixtures/civil_comments_is_toxic_noul_schema_v1.json) | Versioned `is_toxic` Noul JSON Schema for Civil Comments (#72) |
| [fixtures/civil_comments_tier_manifest_v1.yaml](fixtures/civil_comments_tier_manifest_v1.yaml) | Seeded tier A/B limits and τ=0.5 policy (#72) |

Human-readable commentary and epic links:
[Eval complementary manifest](../docs/reference/eval-complementary-manifest.md).

## Rules

- Manifests record **metadata only**. No corpus download scripts here.
- Partner-only finvet data stays out; see
  [Eval partner data policy](../docs/reference/eval-partner-data-policy.md).
- Primary typed-decision bench remains **JevBench** ([#24](https://github.com/Alberto-Codes/typevet/issues/24));
  complementary sets are stress/regression only ([#54](https://github.com/Alberto-Codes/typevet/issues/54)).

Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51).
