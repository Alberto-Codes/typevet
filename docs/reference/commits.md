# Commit messages

Kind: reference.

Every commit follows [Conventional Commits
1.0.0](https://www.conventionalcommits.org/en/v1.0.0/). The specification is
the law for the grammar. This page states the vocabulary this repo adds.

## The subject

```text
<type>[(scope)][!]: <description>
```

One type. An optional scope in parentheses. An optional `!` for a breaking
change. Then a colon, one space and the description. The description does not
end with a period.

## The types

| Type | Use it for |
|---|---|
| `feat` | A new capability in the product surface |
| `fix` | A defect in shipped behaviour |
| `docs` | A page under `docs/`, the README or `CLAUDE.md` |
| `test` | A test, a fake or a fixture |
| `refactor` | A change that keeps behaviour and moves code |
| `perf` | A change that makes the code faster |
| `style` | Formatting with no behaviour change |
| `build` | Packaging, lockfiles or build config |
| `ci` | A workflow under `.github/` |
| `chore` | Tooling and housekeeping that fits nothing above |
| `revert` | A commit that undoes another |

Do not invent types. Prefer a scope when it names a clear surface
(`docs`, `harness`, `domain`).

## Issues

A commit that finishes an issue closes it from the footer:

```text
Closes #N
```

A commit that touches an issue but does not finish it uses `Refs #N`. Do not
close issues by hand with `gh issue close` when a commit can close them.

## Worker trailers

See [worker runs](https://github.com/Alberto-Codes/typevet/blob/main/docs/reference/worker-runs.md). `Generated-By` and `Specified-By` are
evidence. Never use `Co-Authored-By` for a model.

## Gate

When hooks exist, `prepare-commit-msg` strips auto-appended harness
`Co-Authored-By` lines (for example Cursor's `cursoragent@cursor.com`) before
the `commit-msg` check runs. The check still refuses any forbidden co-author
trailer that remains, or a message that breaks the rules above. In CI,
`--range` warns on harness co-authors only on commits already on `main` at or
before the #85 cutover; newer commits must not carry them.

Install both hook stages:

```console
uv run pre-commit install -t prepare-commit-msg -t commit-msg
```
