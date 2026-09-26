# typevet eval manifests

Kind: reference (index).

This directory holds **versioned eval inventory** for typevet — licenses,
Decision shapes, split notes, and exclusions — so future loaders do not
re-litigate dataset research.

## Files

| File | Purpose |
|---|---|
| [complementary-manifest.yaml](complementary-manifest.yaml) | JevBench pointer + #54 ranked complementary text sets (#63) |

Human-readable commentary and epic links:
[Eval complementary manifest](../docs/reference/eval-complementary-manifest.md).

## Rules

- Manifests record **metadata only**. No corpus download scripts here.
- Partner-only finvet data stays out; see
  [Eval partner data policy](../docs/reference/eval-partner-data-policy.md).
- Primary typed-decision bench remains **JevBench** ([#24](https://github.com/Alberto-Codes/typevet/issues/24));
  complementary sets are stress/regression only ([#54](https://github.com/Alberto-Codes/typevet/issues/54)).

Parent epic: [#51](https://github.com/Alberto-Codes/typevet/issues/51).
