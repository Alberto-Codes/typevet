# The writing system

Kind: reference.

This repo writes from [ASD-STE100 Issue
9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf). It takes the
sentence rules and the one-term-per-concept rule. It takes neither the
controlled dictionary nor the approved-verb list. The repo follows a **local
adaptation**, not the standard. It does not claim ASD-STE100 compliance.

Sister projects (automarket, judgevet, finvet, gepa-adk) use the same idea.
Short, active, glossary-bound prose helps every model, every harness and every
human reader.

[CLAUDE.md](../../CLAUDE.md) names the modes as law. This page states scope,
deviations and what the rules leave alone.

## Diátaxis is law

Documentation follows [Diátaxis](https://diataxis.fr/). Each page has one
kind and one job:

| Kind | Job | Location |
|---|---|---|
| Tutorial | Teach by doing | `docs/tutorials/` when present |
| How-to | Steps for a known task | `docs/how-to/` or `docs/maintainers/` |
| Explanation | Answer why / understanding | `docs/explanation/` when present |
| Reference | Lookup facts | `docs/reference/` |

Do not mix kinds on one page. Name the kind in the page header. The README is
the overview. Do not add a second summary file.

## Conventional Commits is law

Every commit follows [Conventional Commits
1.0.0](https://www.conventionalcommits.org/en/v1.0.0/). See
[commits](commits.md). The closed type vocabulary is the only allowed set of
types. A commit that finishes an issue closes it from the footer (`Closes #N`).

## Modes

| Text | Mode |
|---|---|
| Reference pages | Strict |
| How-to and maintainer guides | Strict |
| Tutorials | Strict in each numbered step. Flavored between steps |
| Explanation pages | Flavored |
| `README.md` and `CLAUDE.md` / `AGENTS.md` | Strict |
| Commit messages | Strict |
| Code comments and docstrings (when code exists) | Strict |
| Issue bodies and acceptance comments | Flavored |
| Pull request descriptions (if used) | Flavored |
| `scratchpad/` | Working text. Neither mode |

### Strict mode

- Write one instruction per sentence.
- Keep a sentence to 20 words.
- Use the active voice. Name the actor.
- Use a verb, not a noun built from a verb.
- Use one modal word or none. Do not stack them.
- Write no marketing adjective (`seamless`, `robust`, `powerful`,
  `blazing`, `cutting-edge`, and the like).
- State every number with its unit.
- Write no semicolon.
- Write one em-dash per paragraph, at most.
- Take every domain term from the [glossary](glossary.md).

### Flavored mode

Keep the glossary, the active voice and the units. A longer sentence, a range
and an analogy are allowed. A second em-dash in a paragraph is allowed. Write
no marketing adjective.

## Local deviations from ASD-STE100

1. **20-word limit in strict mode** (STE allows 25 for some descriptive
   sentences).
2. **No controlled dictionary.** The glossary approves terms.
3. **No approved-verb list.**
4. **No procedure-versus-description split.** Strict mode uses one limit.

## What the rules leave alone

- A code block, a command or command output
- An identifier, a file path or a table column name
- A quoted upstream source (cite it; do not rewrite it as house prose)
- YAML front matter and HTML comments

A rule binds the text a change writes or rewrites. Frozen text stays as it is
until that change owns it.

## Gates

When packaging and hooks exist, add:

- a commit-message check for Conventional Commits
- a plain-English / sentence-length check (sister: judgevet
  `check_plain_english`, automarket STE sentence report)
- a terminology check against the glossary (sister: judgevet
  `check_terminology`)

Until those gates exist, the rules still bind. Models and humans apply them by
eye. Do not wait for a script to write clear prose.
