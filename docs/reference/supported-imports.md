# Supported imports and compatibility

Kind: reference. This page lists supported `from typevet…` import paths and
states compatibility expectations for the shipped wheel.

Parent theme: [#29](https://github.com/Alberto-Codes/typevet/issues/29).
Research baseline: [#33](https://github.com/Alberto-Codes/typevet/issues/33).

typevet ships one distribution and one version (`pyproject.toml` / package
metadata). Optional extras (`cli`) add dependencies only. They do not define
a separate release line.

## Import style

Prefer the package path that re-exports a name in that package’s `__all__`:

```python
from typevet.domain import compile_json_schema, GenerationRequest
from typevet.adapters.outbound import LlamaCppGenerationAdapter
```

Deep imports of the underlying module (for example
`typevet.domain.decision_compile`) remain valid for the same objects. They are
not the documented discovery path. Agents and new callers should start from the
package `__init__.py` docstring and this page.

Import-linter contracts in `pyproject.toml` enforce hex layers. Do not import
adapters from domain code or pull `typevet.testing` into adapters.

## Root `typevet`

The root `__all__` declares these supported names:

| Name | Role |
|---|---|
| `GenerationError` | Base failure for a generation call |
| `GenerationPort` | Structural protocol for typed generation |
| `GenerationRequest` | Prompt, schema, and model ask |
| `GenerationResult` | Validated structured value |
| `SchemaValidationError` | Output failed the requested schema |

The attribute `__version__` exists on the package (`"0.1.0"` today) but is
**not** listed in root `__all__` yet. Import it as
`from typevet import __version__` only when you accept that it may move into
`__all__` in a later release without a breaking deep-import change.

Convenience re-exports at the root mirror domain and ports. For compiler types
and decision helpers, import from `typevet.domain` instead of the root.

## `typevet.domain`

| Name | Role |
|---|---|
| `MAX_ENUM_CHOICES` | Upper bound on enum size when compiling |
| `MAX_PERMUTATIONS` | Upper bound on enum permutation budget |
| `Decision` | One compiled TypeLLM field from JSON Schema |
| `GenerationError` | Base generation failure |
| `GenerationRequest` | Prompt, schema, and model ask |
| `GenerationResult` | Validated structured value |
| `SchemaError` | Invalid or unsupported schema for compilation |
| `SchemaValidationError` | Output failed the requested schema |
| `compile_json_schema` | Compile object schema to decisions |
| `dependency_layers` | Topological layers for decision dependencies |

## `typevet.ports`

| Name | Role |
|---|---|
| `GenerationPort` | Structural protocol for typed generation |

## `typevet.adapters.inbound`

| Name | Role |
|---|---|
| `generate` | Build a `GenerationRequest` and invoke a port |

Signature: keyword-only `prompt`, `schema`, and `model` after the port argument.
Prefer this helper for library entry when you already hold a `GenerationPort`.

## `typevet.adapters.outbound`

| Name | Role |
|---|---|
| `FakeGenerationAdapter` | Offline adapter that validates a fixed or callable value |
| `LlamaCppGenerationAdapter` | OpenAI-compat llama.cpp router adapter |

Constructors take explicit arguments only (no settings module on the adapter).
See [library-first architecture](../explanation/library-first-architecture.md).

## `typevet.testing`

| Name | Role |
|---|---|
| `StaticGenerationFake` | Fixed-value port double without adapter imports |

Use for fast unit doubles. Use `FakeGenerationAdapter` when tests must exercise
schema validation like production outbound code.

## `typevet.adapters`

Organizational package only. It has no `__all__` exports. Import from
`typevet.adapters.inbound` or `typevet.adapters.outbound`.

## Typing and packaging

- `src/typevet/py.typed` marks the package as typed for consumers.
- Public API surface is the union of package `__all__` lists and documented
  module docstrings checked by docvet.
- Version string authority today is `pyproject.toml` / `[project].version` and
  `typevet.__version__`. A single-source release workflow may consolidate that
  later ([#29](https://github.com/Alberto-Codes/typevet/issues/29) follow-ups).

## 0.1.0 compatibility assessment

| Surface | Assessment | Notes |
|---|---|---|
| Library | Initial public hex surface | Root, domain, ports, inbound `generate`, outbound adapters, testing fake |
| CLI | Not shipped | `cli` extra lists Typer only. No command module in this release |
| MCP | Not shipped | No extra or entry point |
| Dependencies | `httpx`, `jsonschema` | Locked via `uv.lock` in development |

Additive changes should extend `__all__` and this page. Breaking renames or
removed exports require an explicit compatibility note in a future release
section on this page.

## Verification limits

Contract tests with `FakeGenerationAdapter` do not prove a live llama.cpp router
behaves correctly. Live tests use the `live` pytest marker and opt-in runs.
See [verification](../explanation/verification.md).
