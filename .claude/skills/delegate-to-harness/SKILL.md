---
name: delegate-to-harness
description: Hand one mechanical, gate-checked builder slice to an external harness (Cursor, Codex or pi) when its usage pool has headroom. The slice needs an accepted contract on its issue and size S or M. The supervisor still decides, verifies, gets a fresh review and commits. The gates and the diff are the result, never the agent's summary.
---

# Delegate to a harness

Each harness bills a separate usage pool.
An external harness runs the build. The supervisor keeps every decision and every check.
The [worker run contract](../../../docs/reference/worker-runs.md#harness-boundary) holds the launch evidence and the trailers.

## Trigger

Read the usage of every pool with the `quota` skill before you route a slice.
Headroom means used percent below elapsed percent; ahead of its clock means used percent above elapsed percent.
Equal percents count as no headroom.
Route a mechanical slice only to a pool with headroom.
A pool ahead of its clock is not a target, for example used 25 of elapsed 22.
Record the pool ratios you read in the assignment receipt on the issue.

The `quota` skill is a user-scope skill. Its command is `quota-axi --full`.
It reports each pool as used percent against elapsed percent.
A session never sees `/usage`, the status line or a dashboard. Run the skill instead.

| Pool | Row in the reader | Fallback, for a person |
|---|---|---|
| Claude Code | `claude seven_day`, `claude five_hour`, `claude model:fable` | `/usage` in the terminal |
| Codex | `codex weekly` | `/status` in an interactive `codex` session |
| Cursor | `cursor included_usage`, `cursor auto_usage`, `cursor grok_bot` | The Cursor dashboard, which needs a login |
| pi | None. pi runs locally and costs time, not quota | None |

When Cursor and pi both qualify, route to Cursor first.
Route to pi only when every pool is ahead of its clock, or the slice is size S and offline-only.

## Harnesses

- **Cursor.** The `agent` CLI with the `auto` router. The router never names the routed model.
- **Codex.** The `codex exec` CLI with a named model. The CLI has no automatic routing.
- **pi.** The local harness. Follow the user-scope `delegate-to-pi` skill. Always pass `--model`.

| Harness | Model | Effort | Use it for |
|---|---|---|---|
| Cursor | `auto` | None | Mechanical, gate-checked work |
| Cursor | `cursor-grok-*` | None | When the Cursor plan pool is ahead of its clock |
| Codex | `gpt-6-luna` | `low` or `medium` | Mechanical, gate-checked work |
| Codex | `gpt-6.1-sol` | `medium` | Harder work with a clear contract |
| Codex | `gpt-6-astra` | `high` | Only on a stated supervisor reason |
| pi | Per the `delegate-to-pi` skill | Per that skill | Size-S mechanical work with a decided shape |

Always pass a model and an effort to the runner. Otherwise the installed Codex config decides.

Cursor `auto` and named frontier ids bill the Cursor plan pool at the routed model's list price.
The reader rows for that pool are `cursor included_usage` and `cursor auto_usage`.
`cursor-grok-*` ids bill a separate Cursor pool, the reader row `cursor grok_bot`.

`scripts/harness_build.sh` does not cover pi. Run pi by hand from the worktree.

## When not

These stay on Claude Code:

- the specifier,
- the acceptance reviewer,
- any judgment, ruling or verdict,
- a slice whose shape is undecided,
- a size-L slice.

A worker never runs on Fable.
Copilot is not covered by the runner. See the worker run contract.

## The brief

Write the brief to a file outside the checkout. The script passes the whole file as the prompt.
Give the same four parts as a Claude builder brief:

1. the goal,
2. the scope, with what is out of scope,
3. the context the agent lacks, with the contract text,
4. the return format.

The brief file must contain these two lines. The runner refuses a brief without them:

```text
Accepted contract: <issue comment URL>
Do not commit.
```

`AGENTS.md` is a symlink to `CLAUDE.md` here.
The Cursor CLI reads `AGENTS.md` and `CLAUDE.md` at the project root. Codex reads `AGENTS.md`.
The brief still carries every rule the agent needs, because the brief is the contract.
Add the exact gate commands from `CLAUDE.md`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run lint-imports
uv run python scripts/check_loc.py src evals/src examples
uv run python scripts/check_suppressions.py
uv run python scripts/check_plain_english.py
uv run python scripts/check_terminology.py
uv run docvet check --all
uv run --directory evals docvet check --all
uv run pytest -q
```

## Run

The worktree is an isolated `git worktree` under `.claude/worktrees/<name>`. Never use a sibling directory.
The script refuses the main checkout.

1. Snapshot the worktree with `git -C <worktree> diff HEAD > before.patch`.
   Also save `git -C <worktree> status --short`. Never use `git stash`.
2. Run the script from the repository root:

   ```bash
   scripts/harness_build.sh cursor <worktree> <brief-file>
   scripts/harness_build.sh codex <worktree> <brief-file> gpt-6-luna low
   ```

The script sets a refusing `pre-commit` hook through the environment.
On Cursor the worktree must carry the tracked `.cursor/cli.json` deny list.
It first prints `receipt:` and the receipt path.
The receipt is the raw agent output under `${XDG_STATE_HOME:-$HOME/.local/state}/typevet/harness/`.
The receipt name holds the UTC time, the process id and the harness.
For Codex the script appends one final line of type `harness_build.last_message` holding the last message.
Then it prints `result:`, `usage:`, the status, the diff stat against HEAD and the untracked files.
It warns when HEAD moved. It exits with the agent's code.
The script stops a run after 1800 seconds.

## Evidence

The gates and `git diff` are the result. The agent's summary is only an assertion.
Run each gate yourself on the worktree. Read the diff against the contract.
A fresh acceptance reviewer still accepts the slice before any commit.
Put the receipt file path and the usage in the run receipt on the issue.

## Trailers

`<version>` is the output of `agent --version` or `codex --version`.

- Cursor `auto`: `Generated-By: auto (via Cursor CLI <version>, print mode; routed model unnamed)`
- Cursor named model: `Generated-By: <requested model id> (via Cursor CLI <version>, print mode)`
- Codex: `Generated-By: <model> (via Codex CLI <version>, exec, effort <level>)`
- pi: `Generated-By: <model> (local, via pi)`

## Caveats

- Neither sandbox stops a write outside the worktree, for example to `/tmp`.
- A chained `cat x && git commit` got past the Cursor deny rule `Shell(git)` once.
- The script therefore sets a refusing `pre-commit` hook. `git commit --no-verify` skips that hook.
- The hook guards commits only. The script warns when HEAD moved.
- On Cursor, `.git` stays writable. A chained `git reset --hard` or `git checkout -- .` is not blocked.
- With `--force`, the Cursor agent asked to rerun a blocked command "with full permissions".
- The sandbox is therefore not a dependable layer.
- Codex can start MCP servers from its own config. One server wrote an untracked directory into a worktree.
- Check `git status --short` for each untracked file before acceptance.
- pi has no sandbox, no path confinement and no commit guard.
