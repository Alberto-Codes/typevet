# 0001: Runtime orchestration

Kind: reference.

Status: Accepted.
Decision date: 2026-09-26.
Recorded: 2026-09-26.

Acceptance authority: the prior supervisor accepted [issue 146, revision 2](https://github.com/Alberto-Codes/typevet/issues/146#issuecomment-5843139990).
That receipt records supervisor acceptance, not a human approval mark.
The current Codex supervisor authorized this record through [issue 137](https://github.com/Alberto-Codes/typevet/issues/137#issuecomment-5850163830).

## Context

Root modules mixed orchestration, field instructions and backend framing.
The accepted ownership map assigned each concern to an existing or explicitly named package.
This record preserves its runtime decision without extending that map.

## Decision

Use thin `typevet.runtime` facades for orchestration. Do not introduce a `typevet.engine` layer.
Keep pure domain rules in `typevet.domain` and port contracts in `typevet.ports`.
Keep backend framing in outbound adapters.

## Consequences

Runtime composes lower layers without owning evaluation workflows.
The [import policy](https://github.com/Alberto-Codes/typevet/blob/main/pyproject.toml) enforces the layer order.
Later, [commit 58bfc8a](https://github.com/Alberto-Codes/typevet/commit/58bfc8a616231d1d04d77a7ef91d6b87df868fc8) added the runtime-to-evaluation prohibition for [issue 190](https://github.com/Alberto-Codes/typevet/issues/190).
That guard is current enforcement, not part of the original decision.
Run `uv run lint-imports` to check these boundaries.

[Issue 148](https://github.com/Alberto-Codes/typevet/issues/148#issuecomment-5843140235) assigned the migration and compatibility requirements.
[Commit 68c0dda](https://github.com/Alberto-Codes/typevet/commit/68c0dda) implemented runtime ownership and Gemma framing separation.
The [current runtime package](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/runtime/__init__.py) exposes the orchestration surface.
This record makes no new runtime change and does not prove live model quality.
