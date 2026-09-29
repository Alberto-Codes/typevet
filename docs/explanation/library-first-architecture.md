# Why the library is the artifact

Kind: explanation.

typevet ships an importable Python library first. A command-line or MCP inbound
adapter translates shell or agent inputs into the same ports the library uses.
The choice follows the same library-first pattern as
[judgevet](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/architecture.md).
It is not a claim that every caller must use Python imports today.

The package covers two paths. Grammar-JSON generation through
`GenerationPort` is the transport floor. Typed judgment through `JudgmentPort`
and candidate scoring is the product spine that sisters like judgevet could
consume. See [TypeLLM, Jev and judgevet](typellm-and-judgevet.md) and
[native typed judgments](native-typed-judgments.md). This page explains how
the package boundary supports both paths without mixing transport with product
spine.

## One contract, many entry points

| Caller | Entry | What the caller owns |
|---|---|---|
| Python application, generation | `generate(port, …)` or `port.generate(request)` | Adapter construction, lifetime, prompt/schema/model, and what to do with the result |
| Python application, judgment | `ScoringJudgmentAdapter(scorer, …).judge(state, questions, model)` or `judge_with_scoring` | Scoring adapter, tokenizer callback, questions, and what to do with the answers |
| Shell or CI, eval runs (checkout only) | `python -m typevet_evals.cli.eval_runner` | Arguments, `TYPEVET_LLAMA__*` environment, stdout/stderr, exit status |
| Shell, general command (not shipped) | `typevet` console script | No `[project.scripts]` entry exists |
| MCP host (not shipped) | stdio server | Host config, process lifetime, tool schemas |

The library paths and one module command ship today. The eval command is a
Python module entry, not a console script: installing the wheel puts no
`typevet` executable on `PATH`. The `cli` extra in `pyproject.toml` lists Typer
only, and no typevet module imports Typer. MCP is out of scope for the MVP
slice tracked under parent
[#29](https://github.com/Alberto-Codes/typevet/issues/29).

## Hex layers keep domain pure

Import direction follows the layers contract in `pyproject.toml`, enforced by
`lint-imports`. A layer may import any layer below it, never one above it:

```
typevet.runtime
typevet.adapters.inbound
typevet.adapters.outbound
typevet.adapters.diagnostics
typevet.testing
typevet.ports
typevet.domain      (no httpx, jsonschema, structlog, logging, or filesystem I/O)
```

Two forbidden contracts add to the order: `typevet.testing` never imports
`typevet.adapters`, and `typevet.domain` never imports I/O or transport
libraries.
The library also never imports `typevet_evals`, the workspace member that
holds the evaluation code.

**Domain** holds generation requests and results, judgment questions and
answers, scoring requests and results, errors, and the TypeLLM decision
compiler and executor. **Ports** declare `GenerationPort`,
`AsyncGenerationPort`, `JudgmentPort`, and `CandidateScoringPort`.
**Testing** supplies `StaticGenerationFake`, a port double that never imports
adapters. **Diagnostics** own stderr structlog events and redaction.
**Outbound adapters** own llama.cpp HTTP, candidate scoring, Gemma template
handling, the scoring-backed judgment adapter, and validating fakes.
**Inbound adapters** expose the `generate` and `run_sync` helpers, the
`TYPEVET_LLAMA__*` settings reader, and the eval command module.
**Evaluation** holds dataset loaders, the loader eval runner, and the TPJEP
runner ([#147](https://github.com/Alberto-Codes/typevet/issues/147)). It sits
beside the inbound adapters because the eval command drives it and its live
gate reads inbound settings. **Runtime** holds thin orchestration facades
(`decide_categorical`, `ScoringJudgmentAdapter`, `judge_with_scoring`,
`compose_scoring_prefix`) over the layers below
([#148](https://github.com/Alberto-Codes/typevet/issues/148)).

The root `eval_*` compatibility shims were removed
before 0.1.0. Import from the layers above. See
[supported imports](../reference/supported-imports.md#removed-before-010).

That split matches judgevet’s “ports separate callers from HTTP” story.
`JudgmentPort` sits beside `GenerationPort` in `typevet.ports`. The llama.cpp
scoring glue implements `CandidateScoringPort`; the judgment adapter consumes
that port and does not own HTTP.

## Composition root vs library constructor

A **composition root** is the place that reads configuration, builds adapters,
runs work, and closes resources when work ends. judgevet’s CLI and MCP entry
points are composition roots: they construct the HTTP adapter from settings and
pass it to the runner.

Direct library use moves that job to your application:

- Pass explicit constructor arguments. `LlamaCppGenerationAdapter` takes
  `base_url`, `timeout`, and an optional shared `httpx.Client`. It does not read
  environment variables for those values.
- Own adapter lifetime. Use `with LlamaCppGenerationAdapter(...) as port` or
  call `close()` when you manage the client yourself.
- Configure logging in the application. Importing typevet does not attach
  handlers to the root logger or configure structlog.

Composition roots read `TYPEVET_LLAMA__*` through
[`load_llama_settings`](../reference/configuration.md) and pass the values into
`LlamaCppGenerationAdapter` (or call `llama_cpp_adapter`). The eval command in
`typevet_evals.cli.eval_runner` is the in-repo example (in the `typevet-evals`
workspace member, not the wheel): it loads settings
once, builds the adapter in a `with` block, and closes it before exit. Library
constructors stay explicit so tests and embedders never depend on hidden
global configuration.

Settings and credentials stay on the inbound side. Outbound adapters accept only
values the composition root passes in. That boundary mirrors judgevet and
automarket and keeps contract tests deterministic.

## Two generation fakes, two jobs

| Double | Module | Role |
|---|---|---|
| `StaticGenerationFake` | `typevet.testing` | Fast port stub for inbound unit tests. No schema validation. |
| `FakeGenerationAdapter` | `typevet.adapters.outbound` | Validates against JSON Schema like a real adapter. Used in contract tests. |

Choose `StaticGenerationFake` when the test only needs a fixed mapping back
from `generate`. Choose `FakeGenerationAdapter` when the test must prove
schema validation behavior shared with llama.cpp. Offline judgment uses
`ScriptedScoringFake` from `typevet.testing` behind `ScoringJudgmentAdapter`;
see [first typed judgment offline](../tutorials/first-typed-judgment-offline.md).

## judgevet compatibility is a consumer story

judgevet callers depend on `SystemOnePort`, typed questions, and probability-
bearing answers. typevet’s `JudgmentPort` uses the same vocabulary (`Noul`,
`Choice`, `Score`, state, model) without importing judgevet. No typevet-backed
judgevet adapter ships. Library-first architecture means such an adapter
could wrap `JudgmentPort` without forcing judgevet to fork its domain.
Calibration and parity with hosted Jev remain open; see
[native typed judgments](native-typed-judgments.md#limitations).

For import paths and root exports, see
[supported imports](../reference/supported-imports.md).

## References

- Sister pattern: [judgevet architecture](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/architecture.md)
- Research baseline: issue [#33](https://github.com/Alberto-Codes/typevet/issues/33)
- Layer ownership: issue [#174](https://github.com/Alberto-Codes/typevet/issues/174)
- Product spine: [TypeLLM, Jev and judgevet](typellm-and-judgevet.md)
- Agent architecture law: [CLAUDE.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md) and [AGENTS.md](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md)
