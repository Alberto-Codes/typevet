# typevet

Kind: explanation, the project overview.

typevet evaluates and hardens type-safe generation around
[TypeLLM](https://github.com/TypeLLM/TypeLLM). The product surface is still
forming.

It shares the supervised-worker pattern with
[judgevet](https://github.com/Alberto-Codes/judgevet),
[finvet](https://github.com/Alberto-Codes/finvet),
[gepa-adk](https://github.com/Alberto-Codes/gepa-adk),
[automarket](https://github.com/Alberto-Codes/automarket) and
[docvet](https://github.com/Alberto-Codes/docvet).

## How work moves

**GitHub issues are the bus.** New goals become issues, then triage, size and
child tasks, then an accepted contract comment, then a worker brief. Chat does
not hand off across harnesses or fresh sessions.

Read [CLAUDE.md](CLAUDE.md) first in every agent session.
Then [groom worker issues](docs/maintainers/groom-worker-issues.md) and
[delegate a bounded change](docs/maintainers/delegate-work.md).

## Supervised workers

Role, harness and model weight are three separate choices. Supervisor and
worker harnesses are independent. `worker-fit` means any verified worker
harness may take the slice — not “use pi”.

Verified worker harnesses:

| Harness | Entry |
|---|---|
| pi | `delegate-to-pi` skill |
| Claude Code sub agents | `.claude/agents/` (`builder`, `acceptance-reviewer`, `specifier`) |
| Cursor CLI | print mode, guarded by `.cursor/cli.json` |

## Where to read next

- [docs/README.md](docs/README.md) indexes every page.
- [The glossary](docs/reference/glossary.md) defines each term.
- [Worker runs](docs/reference/worker-runs.md) for launch evidence and trailers.
