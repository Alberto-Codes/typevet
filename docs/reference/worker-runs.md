# Worker run contract

Kind: reference. This page is the contract for each worker run: roles,
harnesses, launch evidence, commit trailers and the run receipt.

This contract separates the project requirements from the model and the
harness that run them. The
[delegation procedure](../maintainers/delegate-work.md) holds the task
sequence and the acceptance rules. [The glossary](glossary.md) defines
"harness".

## Roles

| Role | Responsibility |
|---|---|
| Supervisor | Selects work, settles decisions, owns acceptance and commits |
| Specifier, when needed | Proposes a bounded decision and an executable contract. Reports uncertainty and evidence |
| Worker | Implements the assigned contract within its allowed paths. Returns evidence |
| Reviewer | Checks the actual diff. Exercises the defining behaviour independently |

The roles can use the same model family. A model name alone does not make a
review independent. The reviewer uses production evidence and independent
probes. The reviewer does not use the worker's claim that the work is
complete. The supervisor never spawns a sub agent to verify its own work. A
change of model never expands permissions. A change of model never authorizes
a live API call.

## Harness boundary

The supervisor harness is independent of the worker. Cursor, Claude Code,
Codex, Copilot and pi may each supervise. Preserve any user-selected
supervisor model. Choose **model weight** (light / medium / heavy)
independently of the harness.

typevet has three verified worker harnesses:

- pi runs local models through the `delegate-to-pi` skill.
- Claude Code sub agents run through the Agent tool, with definitions in
  `.claude/agents/`.
- The Cursor CLI runs Cursor-pool models in print mode.

Choose the worker independently from the supervisor. `worker-fit` on an issue
authorizes any of these harnesses; it does not select one.

**User-scope configuration** (global Claude/Cursor/Codex/Copilot/pi settings,
skills, hooks, model defaults) may already exist on the operator machine. That
is fine. Record the effective harness in the receipt. Do not treat missing
in-repo harness chrome as a blocker when the user harness is set up. Project
files here are the floor for a clean checkout, not the only allowed place to
configure a harness.

| Harness | Required launch evidence |
|---|---|
| pi | Installed version. Provider and model. Effective `--thinking` setting. Instruction loading. Tool permissions. Session identifier |
| Claude sub agent | Requested alias (`haiku`, `sonnet` or `opus` only — never Fable). Resolved model ID from the agent's return, or `unknown`. Effort. Allowed tools. Agent definition name |
| Cursor CLI | Installed version. Requested model ID. Identity that the worker reports, or `unknown`. `session_id` and usage from the JSON receipt. `.cursor/cli.json` deny list |
| Any other harness | The same role, context, isolation, observation and return requirements |

Also record the intended **weight** class for the job, even when the harness
only exposes a brand alias.

Read the installed help and configuration before you write a pi launch
command. Do not copy flags, approval modes or token limits between harnesses.
Keep planning and review read-only. Give implementation only the permissions
that its assigned edits and checks need.

Launch a Cursor worker from the checkout root, with the brief as the prompt.
Keep `brief.md` and `receipt.json` under `scratchpad/` or outside the
checkout:

```bash
cursor-agent -p --force --trust --output-format json \
  --model cursor-grok-4.6-medium "$(cat brief.md)" > receipt.json
```

`--force` applies edits without a prompt. Thus `.cursor/cli.json` is the only
guard. It denies `git`, `gh`, `rm`, the policy files and the credential files.
Pass only `cursor-grok-*`, `grok-*`, `composer-*` or `gemini-*` IDs. Other IDs
and `auto` bill the Anthropic and OpenAI pool instead. A `resource_exhausted`
error before any edit is a service failure. Retry once.

A requested alias such as `opus` is not a resolved model identity. Record
both the requested alias and the resolved identity that the agent reports. If
the harness does not show the identity, record `unknown`. Do not guess.

## Commit trailers

Trailers are evidence. This repository is partly an evaluation of its workers.
`Generated-By` and `Specified-By` record the worker. They claim no authorship.
Never use `Co-Authored-By` for a model or harness.

- A pi-written commit carries `Generated-By: <model> (local, via pi)`.
- A Claude sub agent commit carries
  `Generated-By: <resolved model id> (via Claude Code Agent tool, <agent name>)`.
- A Claude sub agent specification carries
  `Specified-By: <resolved model id> (via Claude Code Agent tool, specifier)`.
- A Cursor worker commit carries
  `Generated-By: <requested model id> (via Cursor CLI <version>, print mode)`.
- The supervisor's own commits carry no worker trailer.
- Never invent a resolved ID. The ID comes from the agent's return, never from
  the requested alias.

## Run receipt

Keep this information in the acceptance record on the issue. Do not add a
second ledger that duplicates it.

| Field | Meaning |
|---|---|
| Assignment | Issue, slice and accepted comment. Supervisor and worker role. Allowed paths |
| Baseline | Base revision. Existing modifications. Acceptance-test revision |
| Configuration | Harness and version. Requested and resolved model. Weight class. Effort or reasoning setting. Tool permissions |
| Observation | Session or agent identifier. Start and end times. Output location. Final exit state |
| Outcome | Acceptance disposition. Tested revision. Independent evidence. Repairs. Remaining blockers |
| Cost | Elapsed time. Supervisor repair time. Reported tokens with their source |

Record the factual contributions separately when different models specify and
implement a change. Do not give the worker credit or blame for supervisor
corrections. Store short, redacted evidence. Raw logs stay local until someone
checks them and authorizes their publication. Never put an API key or an
environment dump in a brief, a receipt or an issue comment.

## Preserve the baseline

Use one writer per checkout. Before dispatch, record HEAD and the staged and
unstaged changes. Record the untracked task files that the worker could
overwrite, including files under `scratchpad/`.

After the worker stops, compare the baseline with the returned diff. A clean
final tree does not prove safety. A worker can discard an earlier uncommitted
change and leave no diff. Investigate missing files and unexpectedly restored
files before you accept the run. Check for an unexpected HEAD change,
including a commit that the brief forbade. Do not restore files automatically
while another actor may be editing them.

## Observe completion and classify failure

A launch command, a live process, an idle log or a zero exit code alone does
not prove completed work. Confirm the final output, the changed files and the
acceptance evidence. Do not stop another session because its log is silent.
Do not launch a competing writer because a log is silent.

| Failure class | Supervisor action |
|---|---|
| Specification | Correct the accepted contract. Do not blame a faithful implementation |
| Implementation | Return the failing input and the expected behaviour as a bounded repair |
| Acceptance test | Repair the fixture or the oracle. Do not weaken the intended behaviour |
| Harness or configuration | Diagnose the launch, model resolution, permissions, reasoning setting and completion state |
| Environment | Name the unavailable dependency or service. Repair it within scope |
| External dependency | Record the blocking event. Continue the independent authorized work |

Do not shrink every task because of one harness fault. Do not call a tool
failure a model-quality failure. Do not call a missing metric a zero-cost run.
A regression needs a demonstrated failure on the defect, or a deliberate local
mutation.

## Change a model or harness with evidence

Start a replacement on one small slice with a known acceptance oracle. Check
that it loads instructions, uses permitted tools, preserves the baseline and
reports completion. Run the same acceptance contract and the same required
gates as for the current worker. Record the repairs and the configuration
before you expand its scope. One successful slice shows compatibility for that
slice. It does not show general reliability.

Reuse existing tests and real issue slices. Do not build a benchmark project
first. Update the launch recipe when a recurring failure is specific to the
harness. Update shared policy only when the lesson applies across tasks and
models. Keep machine-specific paths, credentials and process identifiers out
of shared policy.
