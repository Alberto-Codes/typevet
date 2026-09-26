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

## Supervised workers

Supervisor and worker harnesses are independent. Cursor (this session), Claude
Code, Codex, Copilot or pi may supervise; pick the worker separately.

Verified worker harnesses:

| Harness | Entry |
|---|---|
| pi | `delegate-to-pi` skill |
| Claude Code sub agents | `.claude/agents/` (`builder`, `acceptance-reviewer`, `specifier`) |
| Cursor CLI | print mode, guarded by `.cursor/cli.json` |

Read [CLAUDE.md](CLAUDE.md) for roles and bounds,
[docs/maintainers/delegate-work.md](docs/maintainers/delegate-work.md) for the
dispatch procedure, and
[docs/reference/worker-runs.md](docs/reference/worker-runs.md) for launch
evidence and commit trailers.

## Where to read next

- [docs/README.md](docs/README.md) indexes every page.
- [The glossary](docs/reference/glossary.md) defines each term.
