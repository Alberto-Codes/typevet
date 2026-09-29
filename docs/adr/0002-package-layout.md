# 0002: Package layout and the evals workspace member

Kind: reference.

Status: Accepted.
Decision date: 2026-09-29.
Recorded: 2026-09-29.

Acceptance authority: the [user decisions of 2026-09-29](https://github.com/Alberto-Codes/typevet/issues/256#issuecomment-5898728398) set the narrow root and the workspace member.
[Revision 2 of the design](https://github.com/Alberto-Codes/typevet/issues/256#issuecomment-5898811486) applies those decisions.
It keeps the adapter layout of [revision 1](https://github.com/Alberto-Codes/typevet/issues/256#issuecomment-5898634867).
The supervisor accepted each child of [issue 256](https://github.com/Alberto-Codes/typevet/issues/256) after a fresh acceptance review, recorded on the issue.
Review findings set the rules for leaf modules, root tests and the root import.
The supervisor carried those findings to child C10.
These receipts record user decisions and supervisor acceptance, not a human approval mark on this record.

## Context

Before 0.1.0, the `src/typevet` root held 33 compatibility shims and the real `question_schema` module.
The outbound adapters were one flat list of llama.cpp, vLLM, Gemma, fake and HTTP helper modules.
The `typevet.evaluation` package mixed dataset loaders, harnesses, receipts and command-line tools with the library.
PyPI held only the `0.1.0.dev1` name placeholder, which pip skips by default. No outside importer of these paths is known.
After the first release, each path becomes public API and needs a deprecation period.

## Decision

### Library layout

Keep the fixed hex layers: `domain`, `ports`, `adapters`, `runtime` and `testing`.
Delete the root shims before 0.1.0, with no deprecation period.
Move `question_schema` into `typevet.domain`.
Keep a narrow root of ten names. Discover the other names through the package path.

Group the outbound modules of each serving backend into one package:
`typevet.adapters.outbound.llama_cpp` and `typevet.adapters.outbound.vllm`.
Inside a package, remove the family prefix from module names.
Use `http_mapping`, not `http`, so no module shadows the standard library.
Keep the shared outbound modules, the fakes and `typevet.adapters.outbound.gemma` flat at the outbound level.

Each package `__init__` has a docstring and an `__all__` that re-exports its public names.
The parent `typevet.adapters.outbound` keeps its nine names.
The backend package `__init__` files do not import the factory modules.
Callers import `gemma_native_vision_factory` and `judgment_factory` from their own modules.

The factory rule is a static layout rule, not an import-cost rule.
`import typevet` loads `typevet.runtime`, and the runtime loads both factories and `judgment_scoring`.
No test checks the modules that a root import loads.

### Workspace split

Move all evaluation code out of the library wheel.
The `evals/` directory is a uv workspace member with the distribution name `typevet-evals`.
Its import package is `typevet_evals`, and its classifier is `Private :: Do Not Upload`.
The library wheel holds only `typevet`.
The eval command-line modules, the wheel proof tools and each evaluation family live in the member.

The library never imports `typevet_evals`.
The root library tests may import `typevet_evals`.
`tests/live/gate.py` imports the live gate from `typevet_evals.runner.live_gate`.
`tests/unit/test_question_schema.py` checks the mapping against the dataset loaders in `typevet_evals.datasets`.
Library docstring examples use only library imports, because the member is not in the wheel.

### Import contracts

`pyproject.toml` holds these import-linter contracts for the layout:

| Contract | Rule |
|---|---|
| Hexagonal layers | Runtime, inbound, outbound, diagnostics, testing, ports and domain keep their order |
| The library does not import the evals | `typevet` does not import `typevet_evals` |
| Model framing stays off the serving backends | `typevet.adapters.outbound.gemma` imports neither backend package |
| Serving backends stay independent | The llama.cpp and vLLM packages do not import each other |
| Evaluation families | Each family imports only the families in lower layers |

The top layer of "Evaluation families" holds `cli` and three leaf modules.
The leaf modules are `gemma_native_vision_wheel_smoke`, `psai_vision_probability_evidence` and `wheel_isolated`.
No family imports them.

## Consequences

The [supported imports page](../reference/supported-imports.md) lists each package `__all__` in one table.
`tests/contract/test_supported_imports_surface.py` fails when a table and an `__all__` disagree.
It also fails when a re-export is not the same object as its definition.

The removed paths do not resolve. The supported imports page maps each removed path to its current home.
`tests/contract/test_package_layout.py` checks that the old paths do not resolve.
`tests/contract/test_import_boundary_negatives.py` injects forbidden edges and expects each contract to break.
Run `uv run lint-imports` to check the contracts.

The evaluation code has no compatibility promise.
Its fixtures stay in `tests/fixtures/` for 0.1.0, because recorded receipts name those paths.
A lighter root import needs its own decision and a test that reads `sys.modules` in a new interpreter.
This record makes no runtime change and does not prove live model quality.
