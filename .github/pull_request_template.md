## What changed and why

<!-- Problem first. Then the behaviour that replaces it. -->

## Issue

Closes #

<!-- Or Refs #N. Every PR names an issue. -->

## Validation

<!-- Name the commands you ran. Paste what they printed when useful. -->

- [ ] Acceptance check from the issue contract
- [ ] Gate table from `CLAUDE.md` (or named in the brief)

## Docs and prose

- [ ] Diátaxis kind is correct for any new page
- [ ] Writing system and glossary terms respected
- [ ] Conventional Commits subject matches this change

## Review

<!-- Independent acceptance when the contract requires it. -->

- [ ] Diff matches allowed paths
- [ ] No `Co-Authored-By` for a model (use `Generated-By` / `Specified-By` on commits)

---

Prefer one behaviour per PR. Title is a Conventional Commits subject
(`feat` | `fix` | `docs` | `refactor` | `test` | `chore` | `perf` | `build` | `ci`).
`gh pr create --body-file` skips this template. Cover the same headings.
See [commits](../docs/reference/commits.md) and
[groom worker issues](../docs/maintainers/groom-worker-issues.md).
