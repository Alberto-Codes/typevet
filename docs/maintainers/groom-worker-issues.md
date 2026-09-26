# Groom issues into worker tasks

Kind: how-to, for maintainers and supervisors.

GitHub issues are the durable bus across supervisor harnesses (Cursor, Claude
Code, Codex, Copilot, pi, …). Chat does not hand off. Use this procedure when
a goal is new, large, or still fuzzy. Sister projects follow the same shape
(see automarket's grooming how-to and judgevet's `judgment` / size labels).

This is not a full Agile process. It keeps grooming, sizing, definition of
ready and definition of done. Drop the ceremony that does not serve handoff.

## Why issue-first

A fresh session must recover the plan without the prior transcript. The issue
holds the ask, triage, accepted contract, child links, and acceptance
evidence. Workers receive a comment URL. Supervisors record receipts on the
same issue.

Break work down until the smallest agent can finish one accept-when. Verify
each leaf. Roll results up the parent. Do not flatten every thread into one
dispatch.

## 1. Capture the ask

Search open and closed issues before creating work:

```bash
gh issue list -R Alberto-Codes/typevet --state open
gh issue list -R Alberto-Codes/typevet --state all --search "<keywords>"
```

If nothing matches, open a parent or child issue. **Standing permission:**
supervisors file and groom from research open questions, bugs and coding
discoveries without waiting for chat confirmation when the ask is clear.
Prefer the GitHub issue forms under `.github/ISSUE_TEMPLATE/` (epic, bounded
task, bug). Record observed need, desired outcome, non-goals, and a
provisional done-when. Do not start research or implementation agents from
chat alone. Do not invent scope that contradicts an epic non-goal.

## 2. Classify

Keep state and priority labels. Add these on the body or accepted contract:

| Field | Values |
|---|---|
| Execution | `judgment` — a decision remains; `worker-fit` — the contract is decided and mechanical enough for any verified worker harness |
| Size | `size-S` one behaviour or one research artifact; `size-M` several separable behaviours; `size-L` tracker / epic — split before dispatch |
| Depends on | Named issue, revision, evidence or external event |
| Weight hint | `light` / `medium` / `heavy` for the next dispatch (orthogonal to harness) |

`worker-fit` is **not** “use pi”. pi is one harness among several. Light and
heavy models exist in every harness family.

`ready` means work can start now. A `ready` + `judgment` item starts with a
decision or specifier pass, not blind implementation.

### Definition of ready (DoR)

An issue is ready to dispatch when all of these hold:

- Labels include `ready` and either `worker-fit` or `judgment`.
- Size is set. `size-L` is never one builder dispatch.
- Parent link is set when this is a child.
- Accept-when is named (command, required comment sections, or artifact path).
- For behavioural `worker-fit` work, the acceptance test and expected red
  failure are named (TDD).
- Allowed return and non-goals are named.
- For `worker-fit`, an accepted contract comment exists (URL + text for the
  brief).
- For `judgment`, the next step is a specifier pass or a supervisor decision,
  not blind implementation.

### Definition of done (DoD)

A slice is done when all of these hold:

- Accept-when is green (or the research comment has every required section).
- A run receipt sits on the issue.
- Code slices have an independent acceptance review when the contract requires
  one.
- The commit uses `Closes #N` or `Refs #N` as appropriate.
- The parent has a status note when children remain open.

## 3. Split before dispatch

Turn `size-L` and most `size-M` work into child issues or named slices.

| Parent | Children (examples) |
|---|---|
| Rebuild type-safe generation under hex, non-SGLang | Map TypeLLM portable core vs SGLang glue; hex port sketch; backend options; Gemma System-1 thread; first validating spike |

### Tree rules

Issues form a tree. Prefer this shape:

| Level | Role |
|---|---|
| Epic (`size-L`, `epic`) | Goal, combined done-when, ordered child links |
| Capability (`size-M` or small `size-L`) | One decision area that still needs children (for example a model family) |
| Task (`size-S`, usually `worker-fit`) | One behaviour or one research artifact |

Hard stops:

- Prefer depth at most three (epic → capability → task).
- Prefer at most five to seven open children under one parent. Park the rest
  on the parent as “later”.
- Split when one accept-when would mix two proofs, two backends or two
  decisions.
- Research that cannot name an accept-when stays `judgment` until a specifier
  or supervisor names it.
- Never dispatch a tracker as one builder task.
- For Python packages: prefer **flat modules** (one concern per file at the
  layer). Flesh out each package `__init__.py` (re-exports + docs). Do not
  nest a new subpackage until the layer clearly needs it. See `CLAUDE.md`
  Architecture and the glossary term “flat module”.

Each child has its own acceptance. Narrow repairs stay as named slices on the
same issue when they share one acceptance boundary.

The parent lists child links, order and combined done-when. Each child names
its parent and non-goals.

Threads like “Gemma” break down on purpose: variants and GGUF, ollama path,
llama.cpp path, vLLM path, fit for typed generation, multimodal deferred.
Each path is a child with its own accept-when. The parent holds the verdict.

## 4. Accept a contract

When a concrete design question remains, run a read-only specifier (any
harness, usually **heavy** weight). Otherwise the supervisor writes the
contract.

Post a titled, revisioned comment on the issue. Keep it under 150 words.
**Ready when** and **Done when** in that comment are DoR and DoD for the
slice.

```text
Accepted specification — #<issue>, <slice>, revision <N>
Baseline: <revision; observed gap>
Classification: <judgment|worker-fit>, <size>, <weight hint>, parent
Decision and reason: <chosen behaviour or research deliverable>
Allowed edits / returns: <paths or "issue comment only">
Preserve / exclude: <tempting wrong scope>
Ready when: <DoR checklist for this slice, or "met">
Done when:
1. <observable result; command or required comment sections>
2. <unchanged behaviour or out of scope>
Open questions: <none, or blocked items>
Attribution: <who proposed and verified>
```

### Research return shape

A research `worker-fit` slice returns an issue comment (or a named path under
`scratchpad/` that the supervisor pastes). Require these sections unless the
contract drops one:

```text
## Research return — #<issue>
Sources: <URLs or paths; what was read>
Tried: <commands, models or pages exercised>
Findings: <facts only; cite sources>
Limits: <what this slice did not cover>
Open questions: <for the parent or next child>
```

Pin the contract comment URL in every worker brief. Paste the text when the
worker cannot run `gh`.

## 5. Dispatch and close the loop

Use [delegate a bounded change](delegate-work.md) for the brief and acceptance.
Record the run receipt on the issue. Link the commit with `Closes #N` or
`Refs #N`. Leave remaining parent lines open with an explicit status comment.
Verify leaves. Roll summaries up. Do not treat a child receipt as parent done.

When a judgment locks a durable architecture choice, file a child under the
ADR epic (or open that epic’s next scaffold/write task) so the decision lands
in `docs/adr/`. Do not write the ADR on the judgment issue alone once the ADR
log exists. Until the ADR epic is ready, the accepted judgment comment is
enough.

## Labels this repo uses

| Label | Meaning |
|---|---|
| `needs-triage` | Filed, not yet classified |
| `ready` | DoR met for the next step |
| `judgment` | Needs the supervising / orchestrating model |
| `worker-fit` | Mechanical; any verified worker harness |
| `size-S` / `size-M` / `size-L` | Split guidance |
| `P1`–`P4` | Priority |
| `epic` | Parent capability tracker |
| `task` | Bounded deliverable |
