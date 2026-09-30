# Python API reference

Kind: reference. These pages are generated from the docstrings of the public
packages. Each package has its own page. [Supported imports](../supported-imports.md)
lists the stable names and the import path for each name.

## Packages

| Page | Package | Contents |
|---|---|---|
| [Root package](root.md) | `typevet` | The names most callers import: the generation request and result, the errors, the generation ports and `decide_categorical`. |
| <span id="domain"></span>[Domain](domain.md) | `typevet.domain` | Models, errors, question types, decisions and schema compilation, with no I/O. |
| <span id="ports"></span>[Ports](ports.md) | `typevet.ports` | The protocols that adapters implement: generation, scoring, judgment and model framing. |
| <span id="runtime"></span>[Runtime](runtime.md) | `typevet.runtime` | Categorical decisions, typed judgment and the vLLM and Gemma 4 native vision sessions. |
| <span id="inbound-adapters"></span>[Inbound adapters](inbound.md) | `typevet.adapters.inbound` | The caller entry points: `generate`, settings loaders and backend selection. |

## Links from before the split

Before this split, one page held every package. A link to a symbol anchor
on that page, such as `reference/api/#typevet.domain.Noul`, now opens this
index and not the symbol. Use the table above to open the package page.
Links inside this site resolve to the package page.

## Source modules

The package docstrings link to these modules. Each entry opens the
module source.

- [](){#typevet._version} [`typevet._version`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/_version.py)
- [](){#typevet.domain.models} [`typevet.domain.models`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/domain/models.py)
- [](){#typevet.domain.errors} [`typevet.domain.errors`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/domain/errors.py)
- [](){#typevet.domain.decisions} [`typevet.domain.decisions`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/domain/decisions.py)
- [](){#typevet.domain.decision_compile} [`typevet.domain.decision_compile`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/domain/decision_compile.py)
- [](){#typevet.domain.media} [`typevet.domain.media`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/domain/media.py)
- [](){#typevet.domain.question_schema} [`typevet.domain.question_schema`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/domain/question_schema.py)
- [](){#typevet.ports.generation} [`typevet.ports.generation`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/ports/generation.py)
- [](){#typevet.ports.async_generation} [`typevet.ports.async_generation`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/ports/async_generation.py)
- [](){#typevet.ports.framing} [`typevet.ports.framing`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/ports/framing.py)
- [](){#typevet.runtime.categorical} [`typevet.runtime.categorical`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/runtime/categorical.py)
- [](){#typevet.runtime.judgment} [`typevet.runtime.judgment`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/runtime/judgment.py)
- [](){#typevet.adapters.inbound.api} [`typevet.adapters.inbound.api`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/inbound/api.py)
- [](){#typevet.adapters.inbound.helpers} [`typevet.adapters.inbound.helpers`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/inbound/helpers.py)
- [](){#typevet.adapters.inbound.settings} [`typevet.adapters.inbound.settings`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/inbound/settings.py)
- [](){#typevet.adapters.inbound.backend_settings} [`typevet.adapters.inbound.backend_settings`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/inbound/backend_settings.py)
- [](){#typevet.adapters.outbound} [`typevet.adapters.outbound`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/outbound/__init__.py)
- [](){#typevet.adapters.outbound.llama_cpp} [`typevet.adapters.outbound.llama_cpp`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/outbound/llama_cpp/__init__.py)
- [](){#typevet.adapters.outbound.vllm} [`typevet.adapters.outbound.vllm`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/outbound/vllm/__init__.py)
- [](){#typevet.adapters.diagnostics.settings} [`typevet.adapters.diagnostics.settings`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/settings.py)
- [](){#typevet.adapters.diagnostics.configure_from_environ} [`typevet.adapters.diagnostics.configure_from_environ`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/logs.py)
