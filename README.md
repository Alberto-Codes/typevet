# typevet

Kind: explanation, the project overview.

typevet delivers type-safe structured generation under hexagonal architecture.
The MVP path is **local llama.cpp** with **Gemma 4** (schema in → validated
JSON out). That is the transport floor. The product spine follows
[TypeLLM](https://github.com/TypeLLM/TypeLLM)’s decision model (Jev-inspired)
so sisters like [judgevet](https://github.com/Alberto-Codes/judgevet) can
consume a local System One–shaped backend later. See
[TypeLLM, Jev and judgevet](docs/explanation/typellm-and-judgevet.md).
No SGLang dependency.

It shares the supervised-worker pattern with
[judgevet](https://github.com/Alberto-Codes/judgevet),
[finvet](https://github.com/Alberto-Codes/finvet),
[gepa-adk](https://github.com/Alberto-Codes/gepa-adk),
[automarket](https://github.com/Alberto-Codes/automarket) and
[docvet](https://github.com/Alberto-Codes/docvet).

## Quick start (MVP)

```bash
uv sync
uv run pre-commit install --hook-types pre-commit --hook-types pre-push --hook-types commit-msg
uv run pytest -q
# optional live (loads Gemma 4 on the local router; slow first load):
uv run pytest -m live -q
```

Pull requests and pushes to `main` run the same gates on GitHub Actions
(`.github/workflows/ci.yml`): pre-commit and pre-push hook stages from
`.pre-commit-config.yaml`, plus commit-message range checks. Live pytest is
excluded (`-m "not live"` in `pyproject.toml`).

See [Run Gemma 4 on llama.cpp](docs/how-to/run-gemma4-llamacpp.md).

## How work moves

**GitHub issues are the bus.** New goals become issues, then triage, size and
child tasks, then an accepted contract comment, then a worker brief. Chat does
not hand off across harnesses or fresh sessions.

Read [CLAUDE.md](CLAUDE.md) first in every agent session.
Then [groom worker issues](docs/maintainers/groom-worker-issues.md) and
[delegate a bounded change](docs/maintainers/delegate-work.md).

**Diátaxis**, **Conventional Commits 1.0.0** and the **ASD-STE100 local writing
profile** are law. See [the writing system](docs/reference/writing-system.md)
and [commits](docs/reference/commits.md).

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
