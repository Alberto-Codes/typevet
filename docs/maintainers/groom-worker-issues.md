# Groom issues into worker tasks

Kind: how-to, for maintainers and supervisors.

GitHub issues are the durable bus across supervisor harnesses (Cursor, Claude
Code, Codex, Copilot, pi, …). Chat does not hand off. Use this procedure when
a goal is new, large, or still fuzzy. Sister projects follow the same shape
(see automarket's grooming how-to and judgevet's `judgment` / size labels).

## Why issue-first

A fresh session must recover the plan without the prior transcript. The issue
holds the ask, triage, accepted contract, child links, and acceptance
evidence. Workers receive a comment URL. Supervisors record receipts on the
same issue.

## 1. Capture the ask

Search open and closed issues before creating work:

```bash
gh issue list -R Alberto-Codes/typevet --state open
gh issue list -R Alberto-Codes/typevet --state all --search "<keywords>"
```

If nothing matches, open a parent issue (or draft the body for the user to
confirm). Record observed need, desired outcome, non-goals, and a provisional
done-when. Do not start research or implementation agents from chat alone.

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

## 3. Split before dispatch

Turn `size-L` and most `size-M` work into child issues or named slices.

| Parent | Children (examples for a research/port epic) |
|---|---|
| Port TypeLLM under hex, non-SGLang | Map portable core vs SGLang glue; backend bake-off; hex port sketch; first validating spike |

Each child has its own acceptance (command, comment body, or artifact path).
Never dispatch a tracker as one builder task. Narrow repairs stay as named
slices on the same issue when they share one acceptance boundary.

The parent lists child links, order and combined done-when. Each child names
its parent and non-goals.

## 4. Accept a contract

When a concrete design question remains, run a read-only specifier (any
harness, usually **heavy** weight). Otherwise the supervisor writes the
contract.

Post a titled, revisioned comment on the issue. Keep it under 150 words:

```text
Accepted specification — #<issue>, <slice>, revision <N>
Baseline: <revision; observed gap>
Classification: <judgment|worker-fit>, <size>, <weight hint>, parent
Decision and reason: <chosen behaviour or research deliverable>
Allowed edits / returns: <paths or "issue comment only">
Preserve / exclude: <tempting wrong scope>
Acceptance:
1. <observable result; command or required comment sections>
2. <unchanged behaviour or out of scope>
Open questions: <none, or blocked items>
Attribution: <who proposed and verified>
```

Pin that comment URL in every worker brief. Paste the text when the worker
cannot run `gh`.

## 5. Dispatch and close the loop

Use [delegate a bounded change](delegate-work.md) for the brief and acceptance.
Record the run receipt on the issue. Link the commit with `Closes #N` or
`Refs #N`. Leave remaining parent lines open with an explicit status comment.

## Labels this repo uses

| Label | Meaning |
|---|---|
| `needs-triage` | Filed, not yet classified |
| `ready` | Actionable now |
| `judgment` | Needs the supervising / orchestrating model |
| `worker-fit` | Mechanical; any verified worker harness |
| `size-S` / `size-M` / `size-L` | Split guidance |
| `P1`–`P4` | Priority |
| `epic` | Parent capability tracker |
| `task` | Bounded deliverable |
