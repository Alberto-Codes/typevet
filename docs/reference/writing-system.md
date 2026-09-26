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

### Architecture decision records (when ready)

Durable architecture choices live under `docs/adr/` (sisters: automarket,
gepa-adk). Prefer automarket’s shape: `NNNN-slug.md`, Context / Decision /
Consequences, status Proposed → Accepted with a human mark.

ADRs are not how-to pages and not the issue bus. File a write issue when a
judgment locks and should outlive chat. Do not invent ADRs ahead of a locked
decision. Tracker: GitHub epic “adopt ADR log”. See also
[groom worker issues](../maintainers/groom-worker-issues.md).

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

| Gate | Command |
|---|---|
| Owned prose | `uv run python scripts/check_plain_english.py` |
| Named files | `uv run python scripts/check_plain_english.py PATH ...` |
| Saved patch | `uv run python scripts/check_plain_english.py --diff PATCH` |

The prose gate enforces the 20-word sentence limit and five banned adjectives.
The banned words are `seamless`, `robust`, `powerful`, `blazing` and `cutting-edge`.
Matches ignore case and require whole words.
Other strict-mode rules still require manual review.
The gate is not yet a hook.

### Ownership

Without arguments, the gate compares tracked working-tree files against `HEAD`.
It also checks untracked Markdown files within the default scope.
That scope includes `README.md`, `CLAUDE.md`, `docs/reference/`, `docs/how-to/` and `docs/maintainers/`.
`AGENTS.md` aliases `CLAUDE.md`.
The gate excludes `scratchpad/` from default scope.

An added or modified line selects each complete sentence that overlaps that line, including unchanged wrapped lines.
Untouched sentences remain outside the check, even within the same paragraph.
Multiple sentences on one changed line are all selected.
Findings report the first source line of the selected sentence.
Explicit paths check complete Markdown files, including files outside the default scope.
The output reports selected file and paragraph counts.
A clean checkout reports zero owned paragraphs.

Saved patches select added lines within the default scope.
Each added line must match the current file at its recorded line number.
Use a zero-context Git patch for committed changes:

```bash
git diff --no-ext-diff --unified=0 BASE HEAD > /tmp/typevet-owned.diff
uv run python scripts/check_plain_english.py --diff /tmp/typevet-owned.diff
```

Record both revisions with the result.
Exit status `0` means no findings; `1` means findings; `2` means invalid input.
Missing files, directories, unsupported suffixes and unreadable text are invalid inputs.
Deleted files select no new prose.
Deletion-only edits do not select surviving paragraphs; review those edits manually.
Out-of-scope file contents are ignored, including binary, empty-file and mode-only changes.
Supported Markdown mode-only and empty-file changes select no prose.
Malformed owned patches, context lines, owned binary patches and owned renames are rejected.
Quoted Git paths are unsupported; use explicit paths for those files.
Git commands require `/usr/bin/git`; installations elsewhere are unsupported.
Explicit file checks do not require Git.

### Scanner limits

The scanner uses lightweight Markdown rules, not a full CommonMark parser.
Blank lines separate paragraphs; headings, list items and table cells start separate prose units.
Sentence boundaries are `.`, `?`, `!` and paragraph ends.
Abbreviations and decimal points can split sentences.
Words contain letters or digits; internal apostrophes and hyphens keep a word together.
Thus `don't` and `well-known` each count as one word.

The scanner excludes fenced code, indented code, inline code, URLs and file paths.
It retains visible link labels and table body prose, but excludes table headers.
It excludes YAML front matter and closed HTML comments.
An unclosed HTML comment is invalid input.
A contiguous blockquote is excluded when its quoted lines contain a URL or a reference-style source link.
Uncited blockquotes remain prose.
Nested Markdown, multiline inline literals and complex link destinations need manual review.
File-path detection covers slash paths and common source or configuration suffixes.

Tutorial steps, code comments, docstrings, commit prose and flavored pages remain manual for this checker.
Terminology checks remain manual until their separate gate exists.
The commit-message hook enforces Conventional Commits.
