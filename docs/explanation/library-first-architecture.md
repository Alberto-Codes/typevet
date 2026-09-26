# Why the library is the artifact

Kind: explanation.

typevet ships an importable Python library first. A future CLI or MCP inbound
adapter would translate shell or agent inputs into the same generation contract.
The choice follows the same library-first pattern as
[judgevet](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/architecture.md).
It is not a claim that every caller must use Python imports today.

The MVP proves constrained JSON on a local llama.cpp router. The long game is a
TypeLLM-style decision runtime that sisters like judgevet could consume. See
[TypeLLM, Jev and judgevet](typellm-and-judgevet.md). This page explains how
the package boundary supports that path without mixing transport with product
spine.

## One contract, many entry points (when they exist)

| Caller (target) | Entry | What the caller owns |
|---|---|---|
| Python application | `generate(port, …)` or `port.generate(request)` | Adapter construction, lifetime, prompt/schema/model, and what to do with the result |
| Shell or CI (planned) | `typevet` CLI via optional `cli` extra | Arguments, environment, stdout/stderr, exit status |
| MCP host (deferred) | stdio server (not shipped) | Host config, process lifetime, tool schemas |

Today only the library path is wired. The `cli` extra in `pyproject.toml` is a
placeholder dependency set, not a finished command. MCP is out of scope for
the MVP slice tracked under parent
[#29](https://github.com/Alberto-Codes/typevet/issues/29).

## Hex layers keep domain pure

Import direction follows fixed layers enforced by `lint-imports`:

```
adapters.inbound  →  adapters.outbound, ports, domain
adapters.outbound →  domain
testing           →  ports, domain  (never adapters)
ports             →  domain
domain            →  (no httpx, jsonschema, logging, or filesystem I/O)
```

**Domain** holds requests, results, errors, and the TypeLLM decision compiler.
**Ports** declare `GenerationPort`. **Outbound adapters** own llama.cpp HTTP
and validating fakes. **Inbound adapters** expose the library helper `generate`.
**Testing** supplies `StaticGenerationFake`, a port double that never imports
adapters.

That split matches judgevet’s “ports separate callers from HTTP” story. A future
judgment-shaped port would sit beside `GenerationPort`, not inside llama.cpp glue.

## Composition root vs library constructor

A **composition root** is the place that reads configuration, builds adapters,
runs work, and closes resources when work ends. judgevet’s CLI and MCP entry
points are composition roots: they construct the HTTP adapter from settings and
pass it to the runner.

Direct library use moves that job to your application:

- Pass explicit constructor arguments. `LlamaCppGenerationAdapter` takes
  `base_url`, `timeout`, and an optional shared `httpx.Client`. It does not read
  environment variables for those values today.
- Own adapter lifetime. Use `with LlamaCppGenerationAdapter(...) as port` or
  call `close()` when you manage the client yourself.
- Configure logging in the application. Importing typevet does not attach
  handlers to the root logger.

Composition roots read `TYPEVET_LLAMA__*` through
[`load_llama_settings`](../reference/configuration.md) and pass the values into
`LlamaCppGenerationAdapter` (or call `llama_cpp_adapter`). A future CLI or
MCP inbound adapter would construct the adapter once per process and close it at
shutdown. Library constructors stay explicit so tests and embedders never depend
on hidden global configuration.

Settings and credentials stay on the inbound side. Outbound adapters accept only
values the composition root passes in. That boundary mirrors judgevet and
automarket and keeps contract tests deterministic.

## Two fakes, two jobs

| Double | Module | Role |
|---|---|---|
| `StaticGenerationFake` | `typevet.testing` | Fast port stub for inbound unit tests. No schema validation. |
| `FakeGenerationAdapter` | `typevet.adapters.outbound` | Validates against JSON Schema like a real adapter. Used in contract tests. |

Choose `StaticGenerationFake` when the test only needs a fixed mapping back
from `generate`. Choose `FakeGenerationAdapter` when the test must prove
schema validation behavior shared with llama.cpp.

## judgevet compatibility is a consumer story

judgevet callers depend on `SystemOnePort`, typed questions, and probability-
bearing answers. typevet’s MVP `GenerationPort` is the grammar-JSON floor, not
that port yet. Library-first architecture means a future typevet-backed adapter
could implement or wrap a judgment port without forcing judgevet to fork its
domain. Probabilities, logprob scoring, and System One shape remain product
spine work tracked outside this page.

For import paths and root exports, see
[supported imports](../reference/supported-imports.md).

## References

- Sister pattern: [judgevet architecture](https://github.com/Alberto-Codes/judgevet/blob/main/docs/explanation/architecture.md)
- Research baseline: issue [#33](https://github.com/Alberto-Codes/typevet/issues/33)
- Product spine: [TypeLLM, Jev and judgevet](typellm-and-judgevet.md)
- Agent architecture law: [CLAUDE.md](../../CLAUDE.md) and [AGENTS.md](../../AGENTS.md)
