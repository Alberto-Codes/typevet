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

[CLAUDE.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md) names the modes as law. This page states scope,
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

### Architecture decision records

Durable architecture choices live in the [ADR index](../adr/README.md).
Use `NNNN-slug.md` with a number and title, status, decision date, acceptance authority and evidence.
Include Context, Decision and Consequences sections. Mark each record as reference documentation.

New architecture decisions require human approval before a proposed record becomes Accepted.
For this initial record, the user authorized the supervisor to record an already accepted decision.
This authorization does not approve new architecture decisions.
Record the actual historical authority and evidence. Do not invent a human approval mark.

Record an already locked decision without expanding its scope.
ADRs preserve decisions. GitHub issues retain contracts, handoffs and acceptance evidence.
See [groom worker issues](../maintainers/groom-worker-issues.md) for the issue procedure.

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
| Terminology | `uv run python scripts/check_terminology.py` |

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
Other terminology rules remain manual outside the configured replacements.
The commit-message hook enforces Conventional Commits.

### Terminology gate

The terminology gate checks `README.md`, `CLAUDE.md` and all Markdown under `docs/` by default.
This includes flavored pages, tutorials and ADRs.
`AGENTS.md` remains an alias; `scratchpad/` remains outside default scope.
It uses the same sentence ownership, exclusions, line diagnostics and exit statuses as the prose gate.
It does not enforce sentence length on flavored pages.
Neither prose gate is installed as a hook yet.

The map in `scripts/terminology.toml` currently replaces only `issue-bus` with `issue bus`.
This narrow map does not infer synonyms or validate every glossary concept.
Matching ignores case and checks literal phrases within each visible sentence.
Letters, digits and underscores cannot touch a phrase boundary.
Punctuation can touch that boundary; inflected spellings are separate terms.
The shared scanner limits also apply here.

Each `[[terms]]` entry requires exactly two strings: `forbidden` and `preferred`.
Use single spaces without leading, trailing or control characters.
Preferred terms must match glossary headings or bold entries, ignoring case.
Empty maps, unknown fields, duplicate phrases and overlapping forbidden phrases are invalid.
A forbidden phrase cannot occur within a canonical glossary term.
Several distinct spellings can share one preferred term.

Named files and saved patches use the same CLI forms:

```bash
uv run python scripts/check_terminology.py PATH ...
uv run python scripts/check_terminology.py --diff /tmp/typevet-owned.diff
uv run python scripts/check_terminology.py --config PATH_TO_MAP PATH ...
```

The default map and glossary resolve relative to the script, independently of the current directory.
`--config` selects another map; its targets still require entries in this repository's glossary.
Invalid configuration returns status `2`, even when no prose is selected.
