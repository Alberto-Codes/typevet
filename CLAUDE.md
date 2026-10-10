# CLAUDE.md

Kind: reference and how-to, for agents. Guidance for coding agents. `AGENTS.md` is a symlink to this file.

## Start here (every fresh session)

1. Read this file once. Obey it over chat habit.
2. **GitHub issues are the bus.** Durable asks, triage, contracts, handoffs and
   acceptance live on issues. Chat is for the current turn only. Do not keep a
   multi-step plan only in chat when an issue should exist.
3. List open issues before inventing work:
   `gh issue list -R Alberto-Codes/typevet --state open`.
4. If a new goal, research open question, bug or coding discovery needs a
   durable ask and no issue captures it, **file or update an issue**. Standing
   permission: do not wait for chat approval when the ask is clear. Then
   triage, size and split. Only then delegate.
5. Use [groom worker issues](docs/maintainers/groom-worker-issues.md) for the
   select → classify → split → contract path. Use
   [delegate a bounded change](docs/maintainers/delegate-work.md) only after a
   contract exists on the issue. For behaviour under `src/`, run
   specifier (when needed) → builder → fresh acceptance-reviewer → supervisor
   commit; do not implement in the main session by default.
6. Write under [the writing system](docs/reference/writing-system.md): Diátaxis
   page kinds, Conventional Commits and the ASD-STE100 local prose profile.

## What this repo is

**typevet** evaluates and hardens type-safe generation around
[TypeLLM](https://github.com/TypeLLM/TypeLLM). The product surface is still
forming. Sister projects that share this worker pattern:
[judgevet](https://github.com/Alberto-Codes/judgevet),
[gepa-adk](https://github.com/Alberto-Codes/gepa-adk),
[automarket](https://github.com/Alberto-Codes/automarket) and
[docvet](https://github.com/Alberto-Codes/docvet).

## Supervised workers

The [shared roles](docs/reference/worker-runs.md#shared-roles) govern every harness.
Roles describe responsibility, not model brands. Record the actual supervisor,
worker, harness and model weight for each dispatch.

Three independent choices:

| Choice | Meaning |
|---|---|
| Role | supervisor, specifier, builder, acceptance-reviewer |
| Harness | which program runs the model (Cursor, Claude Code, Codex, Copilot, pi, …) |
| Weight | light, medium or heavy capacity for this job |

Do not collapse these. `worker-fit` means the contract is mechanical enough for
any verified worker harness. It does **not** mean “use pi” or “use a small
model”. Pick harness and weight separately after the issue is ready.

The supervisor harness is independent of the worker harness. Preserve any
user-selected supervisor. Choose the worker independently.

typevet has receipt-backed worker use through native Codex agents, pi, Claude Code sub agents and Cursor CLI.
The `delegate-to-harness` skill and `scripts/harness_build.sh` run a Cursor or Codex builder slice when its quota pool has headroom.
pi uses the `delegate-to-pi` skill. Claude uses the Agent tool and definitions in `.claude/agents/`.
Cursor uses print mode, guarded by `.cursor/cli.json`. Any other worker harness meets the same role, isolation
and evidence bar in
[the worker run contract](docs/reference/worker-runs.md).

The supervisor selects work, decides boundaries, accepts the result and
commits. A worker follows its brief, skips session bookkeeping, and never
commits, pushes or changes policy unless the brief assigns it. Each checkout
has one writer. Workers preserve unrelated changes.

### Delegation chain (behaviour changes)

Non-trivial **behaviour** changes — production code under `src/` and the tests
that prove it — follow one supervised path. **Do not implement them in the main
supervisor session** unless the accepted contract explicitly assigns that role
to the supervisor (rare; docs-only and policy slices only).

Required order (sister projects: judgevet, automarket, gepa-adk):

1. **Groomed issue** — classified, sized, parent linked when needed; see
   [groom worker issues](docs/maintainers/groom-worker-issues.md).
2. **Accepted specification** — a titled issue comment (specifier or
   supervisor); URL and text in every brief.
3. **Builder** — allowed paths only; red then green; gate table; no commit.
4. **Acceptance-reviewer** — **fresh** worker session; not the builder, not
   the supervisor verifying its own diff.
5. **Supervisor** — resolve findings, run remaining gates, **commit only after**
   acceptance review accepts the integrated result, including any repair.

Named supervisor exceptions still require fresh independent acceptance. Self-review is not independent evidence.

Research-only and docs-only deliverables still need the issue and contract.
The contract may omit the builder for an issue comment or named documentation paths.
Acceptance review applies when the contract requires it.

A chat turn that edits behaviour without a builder dispatch breaks this chain.
Standing permission to file and groom issues is not permission to bypass it.

**Fable is supervisor-only.** Never run a Claude Code sub agent (or any
worker dispatch) on Fable / `fable`. Pass an explicit worker model on every
Agent call. Do not `fork` from a Fable session. User-scope routing in
`~/.claude/CLAUDE.md` is authoritative for Claude Code; this repo repeats the
rule so every harness sees it.

**Copilot workers use `--model auto` only.** The Copilot CLI lists many
models (including Fable). Do not pick them for typevet work. `auto` (and
`--auto-tier` if needed) is the only allowed Copilot worker choice. Record
the resolved identity if the receipt shows one; otherwise `unknown`.

## Issues before agents

A research, port, design or implementation ask that will outlive one turn
belongs on a GitHub issue before a worker runs.

**Standing permission to open and groom.** Supervisors may file, label, split
and groom issues from research returns, bugs and where coding takes the work
without waiting for chat approval. List open and closed issues first to avoid
duplicates. Do not invent product scope that contradicts an epic non-goal.

- **Parent / epic** (`size-L`): the goal and combined done-when. Not one
  builder dispatch.
- **Child / task** (`size-S` or `size-M`): one behaviour or one research
  deliverable with mechanical acceptance.
- **Labels:** `judgment` vs `worker-fit`; `size-S` / `size-M` / `size-L`;
  `ready` when dispatchable; priority as needed.
- **Accepted contract** is a titled issue comment. The brief links that
  comment URL and pastes its text. Chat summaries are not the contract.

Workers with web access (any harness) are fine for research slices once the
issue names the question, sources to prefer, and the artifact to return
(comment body, not a private essay).

## Bounded execution

The supervisor owns the cost of the whole assignment, including workers and
repeated context. Explicit user scope and required gates still govern.

- **Define done first.** On the issue. Keep the contract under 150 words.
  Link existing specifications instead of rewriting them. For behaviour, the
  acceptance test and its expected red output are part of done.
- **Require a reason for each action.** Advance the decision, repair a
  demonstrated blocker, or satisfy a required gate. Skip actions that serve
  none of these.
- **Bound delegation.** Default to one implementation (or research) dispatch,
  one independent acceptance review and one repair dispatch per behaviour.
  Before exceeding these limits, report the unresolved assertion and why
  another dispatch could resolve it.
- **Red before green.** Behavioural dispatches prove the test fails for the
  missing behaviour before production edits. Preserve both outputs in the
  return.
- **Use one validation path.** The gate table and the hooks are that path. Do
  not add a second review pipeline. Reuse passing checks for unchanged
  revisions. Do not duplicate by hand what the commit or push hook runs.
- **Keep delivery small.** Use the existing issue and one commit. Create no
  extra report or dashboard unless the deliverable requires it.
- **Stop at the agreed outcome.** Report the result and remaining evidence
  gaps on the issue. Do not turn a worker outage or harness failure into
  another project. Report unavailable usage counters as unknown.

## Non-negotiables

- **uv is the ground floor.** Install, sync, run tools, tests and hooks through
  `uv` / `uv run`. Do not invent a parallel pip/poetry/conda path. When the
  package does not exist yet, still plan gates and scripts for `uv run`.
- **TDD is law for behavioural changes.** Write or locate the acceptance test
  first. Run it and keep the red output. Implement until green. Do not weaken,
  skip or delete a test to obtain green. A test that cannot fail is not
  evidence. Research-only slices may omit a code test; their accept-when is
  still named first.
- **Testing pyramid is law** (sisters: judgevet markers; gepa-adk ADR-005).
  Three layers only:

  | Layer | Marker / path | Proves | Default run |
  |---|---|---|---|
  | Unit | `unit` / `tests/unit/` | Pure domain and inbound with fakes; no network | yes |
  | Contract | `contract` / `tests/contract/` | Fake port and real adapter agree on shared fixtures | yes |
  | Live | `live` / `tests/live/` | One exercised local (or remote) call | no (`-m "not live"`) |

  Coverage floor **90** on the default (non-live) suite. A live pass does not
  replace unit or contract. Valid structure ≠ model quality. New behaviour
  names which layer owns the proof. Prefer shared fixtures under
  `tests/fixtures/` when contract suites grow (judgevet shape).
- **Static analysis is ruff and ty in `pyproject.toml`.** No SonarQube,
  SonarCloud or other Sonar product. Strict rule sets live in toml (judgevet
  ruff profile is the reference; see #7).
- **Diátaxis is law.** Every docs page is one kind: tutorial, how-to,
  explanation or reference. See
  [the writing system](docs/reference/writing-system.md).
- **Conventional Commits 1.0.0 is law.** Closed type vocabulary only. See
  [commits](docs/reference/commits.md).
- **ASD-STE100 Issue 9 informs prose; the local profile is law.** Strict and
  flavored modes, glossary one-term-per-concept, no marketing adjectives. The
  repo does not claim full ASD-STE100 compliance. See
  [the writing system](docs/reference/writing-system.md).
- **Never silence a gate.** Fix the cause. Do not add `per-file-ignores`,
  `# noqa`, `# type: ignore`, `--no-verify`, or a narrowed scope.
- **Fixing one gate must not break another.** Account for the whole table before completion.
  Use the [gate evidence rules](docs/reference/worker-runs.md#gate-evidence) for unchanged inputs and normal hooks.
- **No live-service claim without a call that exercised it.** Offline stand-ins
  say nothing about a live model or API.
- **Do not invent product claims.** Until a page moves past `sketch`, treat
  architecture and behaviour descriptions as provisional.
- **Do not replace the issue bus with chat.** A fresh supervisor must recover
  the plan from GitHub, not from a prior session transcript.

## Architecture

Hex layers are fixed for the MVP (`domain` / `ports` / `adapters` /
`runtime` / `testing`). Within a layer, prefer **flat modules**:

- One concern → one module file at that package level
  (`domain/models.py`, not `domain/models/request.py`).
- Do not add a nested package until several modules clearly share a
  sub-boundary and import-linter needs it.
- Every package `__init__.py` is a real surface: module docstring
  (Attributes / Examples / See Also as docvet requires), `__all__`, and
  re-exports of the public names. Do not leave empty or one-line inits.
- Public imports prefer the package path (`from typevet.domain import …`)
  so agents and callers discover the surface from the init.
  [Supported imports](docs/reference/supported-imports.md) lists each
  `__all__`; a contract test keeps the page and the code equal.
- Each serving backend is one outbound package
  (`adapters/outbound/llama_cpp/`, `adapters/outbound/vllm/`). Its
  `__init__` does not import the factory module.
- Evaluation code lives in the `evals/` workspace member (`typevet_evals`),
  not in the library wheel. `src/typevet` never imports `typevet_evals`;
  root tests may. See [ADR 0002](docs/adr/0002-package-layout.md).

Record lasting decisions in docs, ADRs or issue contracts, not only in chat.

## Build and gates

| gate | command |
|---|---|
| ruff | `uv run ruff check .` and `uv run ruff format --check .` |
| ty | `uv run ty check` |
| import-linter | `uv run lint-imports` |
| loc | `uv run python scripts/check_loc.py src evals/src examples` |
| suppressions | `uv run python scripts/check_suppressions.py` |
| owned prose | `uv run python scripts/check_plain_english.py` |
| terminology | `uv run python scripts/check_terminology.py` |
| docvet | `uv run docvet check --all` and `uv run --directory evals docvet check --all` |
| pytest | `uv run pytest -q` |
| coverage (push) | `uv run pytest -q --cov=typevet --cov=typevet_evals --cov-report=term-missing` |
| uv-secure (push) | `uv audit --locked --preview-features audit-command` |
| commit-msg | `uv run python scripts/check_commit_msg.py` (hook) |

`.pre-commit-config.yaml` is the complete hook inventory; the push-stage
`dependency-audit` hook runs the uv-secure row above.

Install hooks once per clone:

`uv run pre-commit install -t pre-commit -t pre-push -t commit-msg`

### Gates run themselves (Claude Code)

A `PostToolUse` hook in `.claude/settings.json` runs `scripts/vet_file.sh`
after every `Write`, `Edit` or `Bash` call once `pyproject.toml` exists. It
runs `uv run ruff format`, `uv run ruff check` and `uv run docvet` on each
changed Python file, plus `check_loc` under `src`, `evals/src` and `examples` when that
script exists.
Silence means those gates are green for the files touched; do not re-run them
by hand. `ty`, `lint-imports` and `pytest` stay in pre-commit and the gate
table. Until the package exists, the hook exits quietly. **uv is required** for
every gate command; never call ruff/ty/docvet/pytest outside `uv run`.

Cursor CLI and other harnesses do not load this Claude project hook. Their
briefs still require focused checks on edited files via `uv run`.
Account for the full gate table under the shared gate evidence rules.

**User-scope harness setup is allowed and not this repo's job.** If Claude,
Cursor, Codex, Copilot or pi is already configured in the operator's user
settings (hooks, permissions, models, skills), leave it. The repo ships
project floors (`.claude/settings.json`, `.cursor/cli.json`, `AGENTS.md`) for
checkouts that lack user config. Do not duplicate or fight a working
user-level setup. Do not require every harness's full config to live in-tree.

### A summary is not evidence

**Never report work complete while a gate is red.** Report what is red and say
why you think it should be accepted.

**A green gate table is not an audit either.** Gates cannot see a test that
passes whether or not the behaviour happens, a helper that raises where the
specification said return, or an unwrapped secret bound to a local.

**Prove a property with a command, not a sentence.** Break the precondition
and show the test go red. A test that cannot fail is not evidence.

## Commits

Conventional Commits 1.0.0 is law. Details live in
[docs/reference/commits.md](docs/reference/commits.md). The type comes from the
closed vocabulary (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`,
`perf`, `build`, `ci`, `style`, `revert`).

**A commit that finishes an issue closes it from the footer.** Write
`Closes #N`. Do not close issues by hand with `gh issue close`. Use `Refs #N`
for an issue the commit touches but does not finish.

Attribution is per-commit and factual. `Generated-By` and `Specified-By` are
worker evaluation evidence. They claim no authorship. Never use
`Co-Authored-By` for a model or harness.

A commit a local model wrote carries `Generated-By: <model> (local, via pi)`.
When a reasoning model wrote the specification and a coder implemented it, the
commit names both, as `Specified-By: <model>` and `Generated-By: <model>`.
Never merge the roles.

A commit a Claude sub agent wrote carries `Generated-By: <resolved model id>
(via Claude Code Agent tool, <agent name>)`. A specification it wrote carries
`Specified-By: <resolved model id> (via Claude Code Agent tool, specifier)`.
The resolved model ID comes from the agent's return, never from the requested
alias. If the agent does not report it, record `unknown`. A commit written by
hand, and the supervisor's own commits, carry no worker trailer.

A Cursor worker commit carries
`Generated-By: <requested model id> (via Cursor CLI <version>, print mode)`.

## Never destroy work you did not create

A delegated session does not run `git checkout`, `git restore`, `git reset`,
`git stash`, `git clean`, or `rm -rf` against any path, and does not modify a
file its task did not name. Unrelated modifications in the tree belong to
someone else; leave them and mention them in the summary.

Default branch is `main`. Prefer one commit per finished slice on `main` once
hooks exist. Never pass `--no-verify`.
