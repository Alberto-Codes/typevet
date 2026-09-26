# Delegate a bounded change

Kind: how-to, for maintainers.

Use this procedure for supervised implementation or bounded research in
typevet, whatever the model or harness. **Groom the issue first**
([groom worker issues](groom-worker-issues.md)). The supervisor selects work,
decides boundaries, verifies behaviour and commits. The worker implements a
scoped change (or returns a research artifact) and returns evidence. Apply the
[bounded execution limits](../../AGENTS.md#bounded-execution) before dispatch.
Use one validation path. Do not add another harness to repeat required
repository checks.

## Worker harnesses and model weight

The supervisor harness is independent of the worker. Cursor, Claude Code,
Codex, Copilot and pi may each supervise. Preserve any user-selected
supervisor model.

typevet uses three verified worker harnesses. The
[worker run contract](../reference/worker-runs.md) records launch evidence.

- **pi** runs local models through the `delegate-to-pi` skill.
- **Claude sub agents** run through the Agent tool with definitions in
  `.claude/agents/`.
- **Cursor CLI** runs Cursor-pool models in print mode, guarded by
  `.cursor/cli.json`.

The Claude definitions are `builder.md`, `acceptance-reviewer.md` and
`specifier.md`. Other harnesses map to the same roles without those files.

Pick **weight** separately from harness. Brands below are examples, not locks.

| Job | Weight | Typical harness choices |
|---|---|---|
| Lookups and file search | light | any harness with a light model |
| Research, docs and web mapping | medium | any harness with a mid model; web when the issue allows |
| Implementation and gate repairs | heavy | `builder` / coder model on Claude, Cursor, pi or Codex |
| Acceptance review | heavy | fresh session; `acceptance-reviewer` or equivalent |
| Specification | heavy | `specifier` or a reasoning model; read-only |

`worker-fit` on an issue means the contract is mechanical enough for any
verified worker harness. It does not select pi or a light model.

The supervisor never spawns a sub agent to verify its own work.
The supervisor never asks a worker to double-check itself.
An acceptance review runs in a fresh agent, separate from the builder.

**Never run a Claude Code sub agent on Fable.** Fable (Claude Fable) is a
supervisor / main-conversation model only. On every Agent tool call, pass an
explicit worker `model` (`haiku`, `sonnet`, or `opus` as the job needs). Do not
use `fork` (it inherits Fable). Do not request Fable, `fable`, or any Fable
alias as a worker or Cursor Task model.

## 1. Make the task ready

Confirm the issue exists and is classified (`judgment` or `worker-fit`, size,
weight hint). State the decision, the acceptance check, the allowed paths (or
return artifact) and the stopping condition on the issue. The accepted
specification lives in an issue comment, never on disk. Pass the exact
comment URL and its text in the brief. Never substitute the latest comment.
Builders and Cursor workers may lack `gh`, so the brief carries the contract
text.

Request a read-only specification pass only when a concrete design question
remains. Review its answer before implementation. Skip that pass for a known
defect with a decided fix and regression.

For behavioural changes, require a regression that fails before the fix.
Check that the failure shows missing behaviour, not a broken fixture. Keep
the test and implementation in the same deliverable and the same dispatch.
**TDD is law:** red output first, then green. The brief names the exact
`uv run` command.

For research slices, require the comment sections or file path the child issue
names. Prefer returning findings as an issue comment the next supervisor can
read.

## 2. Size by behaviour

Start with one behaviour across two to four production files, plus tests and
documentation — or one research artifact with a named accept-when. Treat file
count as an initial tuning rule, not a measured worker capability. Split
independent behaviours even when they fit in one file. Keep an invariant
together when a split would leave an unsafe intermediate state.

| Task shape | Dispatch |
|---|---|
| Decided behaviour with a failing acceptance test | Implement directly |
| Several independent behaviours or acceptance commands | Split into named slices or child issues |
| Unresolved API, port or source decision | Resolve on the issue first (`judgment`) |
| Small mechanical correction with an obvious proof | Short repair brief (`worker-fit`, often light or medium) |
| Broad failure after repeated repairs | Reassess the contract and split again |
| `size-L` tracker | Do not dispatch whole; groom children first |

One GitHub issue need not equal one worker task. Name the parent issue and
slice in every dispatch.

## 3. Launch with a bounded brief

Use the template below. Replace every placeholder before dispatch. Keep task
context short. Link the issue and evidence instead of copying chat history.

Use one writer per checkout. Record the base revision and existing
modifications before launch. Keep temporary briefs and receipts under the
ignored `scratchpad/` directory, or outside the checkout.

For pi, inspect the installed harness help before choosing options. For a
Claude sub agent, pass `model` on every Agent call and name the agent
definition. For Cursor, paste the full contract into the brief, because `gh`
is denied to the worker. Planning and review stay read-only. Implementation
needs local edit and test permissions.

```text
Role: bounded worker. The supervisor owns acceptance and the commit.
Supervisor / worker / harness / weight: <actual assignments>
Issue and slice: <issue number, one behaviour or research artifact>
Base revision and existing modifications: <exact values>
Accepted specification: <exact issue comment URL and its text>
Decision and reason: <settled shape and why>
Read first: <targeted files and cited source URLs>
Allowed edits / returns: <paths, or issue comment only>
Out of scope: <adjacent work and tempting incorrect fixes>

Acceptance:
- <observable invariant, exact command or required comment sections>
- <regression or parent lines that must stay unchanged>
Run the acceptance check before implementation when it is a test. Preserve red output.
Do not weaken the acceptance check to obtain green output.
Run focused checks during edits, then the CLAUDE.md gate table.
Report required checks you did not run.

Skip session bookkeeping and backlog sweeps.
Do not edit CLAUDE.md, AGENTS.md or policy files.
Do not commit, push, pass --no-verify or add a gate suppression.
Preserve unrelated changes. Do not reset, clean, stash or restore them.
If required edits exceed the allowed scope, return the missing scope.

Return: changed paths or comment body, red/green evidence, unrun checks,
remaining gaps and your model identity. Leave the diff for review and stop.
```

## 4. Accept behaviour and finish

Read the diff (or research comment) against the agreed scope. Run an
independent probe of the defining behaviour when code changed. Confirm that
the regression can detect the original defect.

The gate inventory is the gate table in the
[repository rules](../../AGENTS.md#build-and-gates) and, when present,
`.pre-commit-config.yaml`. Never weaken a gate or report an unrun gate as
green.

Return a failed assertion and a narrow correction brief when review finds a
defect. After two unsuccessful repairs of the same defect, reassess the
contract. Preserve the useful diff and evidence during reassessment.

The supervisor publishes by committing. The commit carries factual trailers
from the worker's reported identity. Passing tests do not authorize a live
API call. Post the run receipt on the issue.

## 5. Record outcomes and tune

Record proof and remaining work on the issue. Keep one record per slice.
Link the commit instead of copying its evidence.

| Field | Evidence |
|---|---|
| Identity | Issue, slice, base revision, supervisor, worker model, harness, weight, session identifier |
| Input | Accepted specification, planning or implementation |
| Size | Behaviour count, production files, tests and documentation |
| Cost | Elapsed time, reported tokens if available, supervisor repair time |
| Outcome | Accepted unchanged, accepted after repair, rejected or externally blocked |
| Proof | Red and green output, independent probe, gate results, accepted commit |
| Lesson | Specification, implementation, test, harness, environment or external failure |

Compare accepted behaviour and supervisor repair time, not lines written or
worker confidence. Group measurements by role, harness, weight, task type and
size. Treat unavailable measurements as unknown. Do not combine unlike token
counters. Separate external blocking from implementation failure. Persist
recurring rules in docs. Keep task facts in the accepted contract.
