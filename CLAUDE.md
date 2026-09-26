# CLAUDE.md

Kind: reference and how-to, for agents. Guidance for coding agents. `AGENTS.md` is a symlink to this file.

## Start here (every fresh session)

1. Read this file once. Obey it over chat habit.
2. **GitHub issues are the bus.** Durable asks, triage, contracts, handoffs and
   acceptance live on issues. Chat is for the current turn only. Do not keep a
   multi-step plan only in chat when an issue should exist.
3. List open issues before inventing work:
   `gh issue list -R Alberto-Codes/typevet --state open`.
4. If the user states a new goal and no issue captures it, **file or update an
   issue first** (or draft the body and ask to file). Then triage, size and
   split. Only then delegate.
5. Use [groom worker issues](docs/maintainers/groom-worker-issues.md) for the
   select → classify → split → contract path. Use
   [delegate a bounded change](docs/maintainers/delegate-work.md) only after a
   contract exists on the issue.
6. Write under [the writing system](docs/reference/writing-system.md): Diátaxis
   page kinds, Conventional Commits and the ASD-STE100 local prose profile.

## What this repo is

**typevet** evaluates and hardens type-safe generation around
[TypeLLM](https://github.com/TypeLLM/TypeLLM). The product surface is still
forming. Sister projects that share this worker pattern:
[judgevet](https://github.com/Alberto-Codes/judgevet),
[finvet](https://github.com/Alberto-Codes/finvet),
[gepa-adk](https://github.com/Alberto-Codes/gepa-adk),
[automarket](https://github.com/Alberto-Codes/automarket) and
[docvet](https://github.com/Alberto-Codes/docvet).

## Supervised workers

Roles describe responsibility, not model brands. Record the actual supervisor,
worker, harness and model weight for each dispatch.

Three independent choices:

| Choice | Meaning |
|---|---|
| Role | supervisor, specifier, builder, reviewer |
| Harness | which program runs the model (Cursor, Claude Code, Codex, Copilot, pi, …) |
| Weight | light, medium or heavy capacity for this job |

Do not collapse these. `worker-fit` means the contract is mechanical enough for
any verified worker harness. It does **not** mean “use pi” or “use a small
model”. Pick harness and weight separately after the issue is ready.

The supervisor harness is independent of the worker harness. Preserve any
user-selected supervisor. Choose the worker independently.

typevet has three verified worker harnesses: pi, through the `delegate-to-pi`
skill; Claude Code sub agents, through the Agent tool and the definitions in
`.claude/agents/`; and the Cursor CLI in print mode, guarded by
`.cursor/cli.json`. Any other worker harness meets the same role, isolation
and evidence bar in
[the worker run contract](docs/reference/worker-runs.md).

The supervisor selects work, decides boundaries, accepts the result and
commits. A worker follows its brief, skips session bookkeeping, and never
commits, pushes or changes policy unless the brief assigns it. Each checkout
has one writer. Workers preserve unrelated changes.

## Issues before agents

A research, port, design or implementation ask that will outlive one turn
belongs on a GitHub issue before a worker runs.

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
  Link existing specifications instead of rewriting them.
- **Require a reason for each action.** Advance the decision, repair a
  demonstrated blocker, or satisfy a required gate. Skip actions that serve
  none of these.
- **Bound delegation.** Default to one implementation (or research) dispatch,
  one independent acceptance review and one repair dispatch per behaviour.
  Before exceeding these limits, report the unresolved assertion and why
  another dispatch could resolve it.
- **Use one validation path.** The gate table and the hooks are that path. Do
  not add a second review pipeline. Reuse passing checks for unchanged
  revisions. Do not duplicate by hand what the commit or push hook runs.
- **Keep delivery small.** Use the existing issue and one commit. Create no
  extra report or dashboard unless the deliverable requires it.
- **Stop at the agreed outcome.** Report the result and remaining evidence
  gaps on the issue. Do not turn a worker outage or harness failure into
  another project. Report unavailable usage counters as unknown.

## Non-negotiables

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
- **Fixing one gate must not break another.** Run the whole table before you
  report.
- **No live-service claim without a call that exercised it.** Offline stand-ins
  say nothing about a live model or API.
- **Do not invent product claims.** Until a page moves past `sketch`, treat
  architecture and behaviour descriptions as provisional.
- **Do not replace the issue bus with chat.** A fresh supervisor must recover
  the plan from GitHub, not from a prior session transcript.

## Architecture

The package layout is not fixed yet. Prefer small modules, typed public
surfaces and one inbound adapter for the CLI when that exists. Record lasting
decisions in docs or issue contracts, not only in chat.

## Build and gates

Gates are not installed yet. Until they are, the brief names every required
check. When a gate lands, add it to this table and to the hooks in the same
change.

| gate | command |
|---|---|
| _(none yet)_ | named in the brief |

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
