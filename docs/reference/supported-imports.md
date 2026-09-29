# Supported imports and compatibility

Kind: reference. This page lists supported `from typevet…` import paths and
states compatibility expectations for the shipped wheel.

Parent theme: [#29](https://github.com/Alberto-Codes/typevet/issues/29).
Research baseline: [#33](https://github.com/Alberto-Codes/typevet/issues/33).
Layer ownership: [#174](https://github.com/Alberto-Codes/typevet/issues/174).

typevet ships one distribution and one version (`pyproject.toml` / package
metadata). Optional extras (`cli`) add dependencies only. They do not define
a separate release line.

## Import style

Prefer the package path that re-exports a name in that package’s `__all__`:

```python
from typevet.domain import compile_json_schema, GenerationRequest
from typevet.adapters.outbound import LlamaCppGenerationAdapter
from typevet.runtime import ScoringJudgmentAdapter
```

Deep imports of the underlying module (for example
`typevet.domain.decision_compile`) remain valid for the same objects. They are
not the documented discovery path. Agents and new callers should start from the
package `__init__.py` docstring and this page.

Import-linter contracts in `pyproject.toml` enforce hex layers. Do not import
adapters from domain code or pull `typevet.testing` into adapters. See
[library-first architecture](../explanation/library-first-architecture.md#hex-layers-keep-domain-pure)
for the layer order.

## Root `typevet`

The root `__all__` declares these supported names:

| Name | Role |
|---|---|
| `AsyncGenerationPort` | Structural protocol for async typed generation |
| `BackendHttpError` | llama.cpp HTTP error status with a body snippet |
| `GenerationError` | Base failure for a generation call |
| `GenerationPort` | Structural protocol for typed generation |
| `GenerationRequest` | Prompt, schema, and model ask |
| `GenerationResult` | Validated structured value |
| `SchemaValidationError` | Output failed the requested schema |
| `TransportError` | HTTP client failure before a response |
| `__version__` | Installed distribution version string |
| `decide_categorical` | M1 categorical decision via injected scoring port (native ``context=`` or ``inject_prefix=True``) |

`__version__` comes from `importlib.metadata.version("typevet")`. An editable
checkout without distribution metadata falls back to `[project].version` in
`pyproject.toml`.

Convenience re-exports at the root mirror domain, ports, and runtime. For
compiler types, judgment types, and scoring types, import from
`typevet.domain` instead of the root.

## `typevet.domain`

| Name | Role |
|---|---|
| `MAX_ENUM_CHOICES` | Upper bound on enum size when compiling |
| `MAX_PERMUTATIONS` | Upper bound on enum permutation budget |
| `Decision` | One compiled TypeLLM field from JSON Schema |
| `GenerationError` | Base generation failure |
| `BackendHttpError`, `TransportError` | Generation failures raised by HTTP adapters |
| `GenerationUnsupportedCapabilityError` | The backend cannot honor the generation ask |
| `GemmaTemplateError` | Served template output is not a known Gemma or ChatML shape |
| `GenerationRequest` | Prompt, schema, and model ask |
| `GenerationResult` | Validated structured value |
| `SchemaError` | Invalid or unsupported schema for compilation |
| `SchemaValidationError` | Output failed the requested schema |
| `compile_json_schema` | Compile object schema to decisions |
| `dependency_layers` | Topological layers for decision dependencies |
| `Noul`, `Choice`, `Score`, `Question` | System One-shaped judgment questions |
| `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `Answer` | Typed judgment answers; `ScoreAnswer.score` is the probability-weighted expected rubric level (float), not the modal level |
| `JudgmentResponse`, `TokenUsage` | Judgment call result and token metadata |
| `JudgmentError`, `JudgmentValidationError` | Judgment failure types |
| `question_types` | Map question ids to wire type names |
| `CandidateScoringRequest`, `CandidateTokenSpec` | Candidate logprob scoring ask |
| `ImageInput` | One image to condition a judgment on |
| `MEDIA_MARKER`, `SUPPORTED_IMAGE_MIME_TYPES`, `count_media_markers` | Media marker and accepted image mime types |
| `CandidateScoringResult`, `ScoredCandidate`, `ScoringTermination` | Scoring result and metadata |
| `ScoreStage` | Pre- vs post-sampling stage enum |
| `ScoringError`, `ScoringValidationError`, `ScoringUnsupportedCapabilityError` | Scoring failure types |
| `build_and_validate_result` | Fail-closed result assembly for scoring |
| `CategoricalExecutionResult` | Greedy categorical execute outcome |
| `DecisionExecutionError` | Categorical execute rejected inputs |
| `execute_categorical_decision` | Choice/Bool execution; calls the injected scoring port and performs no I/O itself |
| `bind_control_candidates`, `judgment_original_labels` | Control-token binding for native questions |
| `normalize_noul`, `normalize_choice`, `normalize_score`, `normalize_question` | Native question → ``Decision`` |
| `question_record_to_property`, `question_records_to_json_schema`, `compile_question_records` | Question records → JSON Schema or ``Decision`` values; defined in `typevet.domain.question_schema` (see [question-schema-map.md](question-schema-map.md)) |

## `typevet.ports`

| Name | Role |
|---|---|
| `GenerationPort` | Structural protocol for typed generation |
| `AsyncGenerationPort` | Structural protocol for async typed generation |
| `JudgmentPort` | Structural protocol for System One-shaped judgment |
| `CandidateScoringPort`, `ScoringPort` | Structural protocol for candidate logprobs (`ScoringPort` is an alias) |
| `ModelFramingPort` | Structural protocol for model turn framing |

## `typevet.runtime`

Thin orchestration facades over domain, ports, and outbound adapters
([#148](https://github.com/Alberto-Codes/typevet/issues/148)).

| Name | Role |
|---|---|
| `ScoringJudgmentAdapter` | Sync ``JudgmentPort`` over ``CandidateScoringPort`` |
| `judge_with_scoring` | One-shot helper wrapping the adapter |
| `decide_categorical` | M1 categorical decision via injected scoring port |
| `compose_scoring_prefix` | Degraded ChatML scoring prefix composition |
| `open_vllm_judgment` | Context manager: judgment port over vLLM chat-completions scoring |
| `VllmJudgmentSession` | Session that `open_vllm_judgment` yields (`port`, `client`, `model`) |
| `vllm_tokenize` | Hook factory for vLLM `/tokenize` without special tokens |
| `open_gemma_native_vision_judgment` | Context manager: judgment port over the llama.cpp Gemma native vision factory |
| `GemmaNativeVisionSession` | Session that `open_gemma_native_vision_judgment` yields |
| `probe_gemma_native_vision_support` | Router probe for Gemma native vision support, without a long-lived port |

`open_vllm_judgment` takes keyword `client`, `model`, and optional `base_url`,
`tokenize_content` and `scoring_port_wrapper`. It sends no template probe.
Scoring and `/tokenize` requests go to `base_url`, or to the client
`base_url` when `base_url` is not set. The caller owns the client.

`typevet.adapters.inbound.backend_settings.open_judgment` selects the session
from `TYPEVET_BACKEND`: the Gemma native vision factory for `llama_cpp` (the
default), or `open_vllm_judgment` on the `TYPEVET_VLLM__*` client for `vllm`.

## `typevet.adapters.inbound`

| Name | Role |
|---|---|
| `generate` | Build a `GenerationRequest` and invoke a sync port |
| `run_sync` | Run an async generation coroutine from sync scripts |
| `LlamaSettings` | Frozen llama.cpp connection settings for composition roots |
| `load_llama_settings` | Read `TYPEVET_LLAMA__*` (and legacy aliases) from the environment |
| `llama_cpp_adapter` | Construct `LlamaCppGenerationAdapter` from `LlamaSettings` |
| `VllmSettings` | Frozen vLLM connection settings; `api_key` is not in `repr` |
| `load_vllm_settings` | Read `TYPEVET_VLLM__*` from the environment |
| `vllm_http_client` | Build the shared vLLM `httpx.Client` |
| `async_vllm_generation_adapter` | Construct `AsyncVllmGenerationAdapter` from `VllmSettings` |
| `load_backend` | Read `TYPEVET_BACKEND` |
| `generation_adapter` | Construct the generation adapter that `TYPEVET_BACKEND` selects |

Environment names and CLI hookup notes live in
[configuration.md](configuration.md).

Signature: keyword-only `prompt`, `schema`, and `model` after the port argument.
Prefer `generate` for library entry when you already hold a `GenerationPort`.
Prefer `run_sync(port.generate(request))` for `AsyncGenerationPort` in scripts
instead of duplicating sync wrappers on each adapter.

The inbound package holds no command-line module. The eval runner command is
in the `typevet-evals` workspace member. See [Not in the wheel](#not-in-the-wheel).

## `typevet.adapters.outbound`

| Name | Role |
|---|---|
| `FakeGenerationAdapter` | Offline adapter that validates a fixed or callable value |
| `AsyncFakeGenerationAdapter` | Offline async adapter that validates a fixed value |
| `LlamaCppGenerationAdapter` | OpenAI-compat llama.cpp router adapter |
| `AsyncLlamaCppGenerationAdapter` | Async OpenAI-compat llama.cpp router adapter |
| `LlamaCppCandidateScoringAdapter` | Pre-sampling ``/completion`` candidate scorer |
| `VllmGenerationAdapter` | vLLM structured-output generation adapter |
| `AsyncVllmGenerationAdapter` | Async vLLM generation adapter with a POST limit |
| `VllmCandidateScoringAdapter` | vLLM chat-completions logprob scorer |
| `ChatContentFraming` | Plain chat content framing for vLLM scoring |

Constructors take explicit arguments only (no settings module on the adapter).
See [library-first architecture](../explanation/library-first-architecture.md).

### `typevet.adapters.outbound.llama_cpp`

This package groups the llama.cpp serving-backend modules.

| Name | Role |
|---|---|
| `LlamaCppGenerationAdapter` | OpenAI-compat llama.cpp router adapter |
| `AsyncLlamaCppGenerationAdapter` | Async OpenAI-compat llama.cpp router adapter |
| `LlamaCppCandidateScoringAdapter` | Pre-sampling ``/completion`` candidate scorer |

The package does not import the Gemma native vision factory. Import the factory
from `typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory`.

### `typevet.adapters.outbound.vllm`

This package groups the vLLM serving-backend modules.

| Name | Role |
|---|---|
| `VllmGenerationAdapter` | vLLM structured-output generation adapter |
| `AsyncVllmGenerationAdapter` | Async vLLM generation adapter with a POST limit |
| `VllmCandidateScoringAdapter` | vLLM chat-completions logprob scorer |
| `ChatContentFraming` | Plain chat content framing for vLLM scoring |

The package does not import the vLLM judgment factory. Import the factory from
`typevet.adapters.outbound.vllm.judgment_factory`, or import
`open_vllm_judgment` from `typevet.runtime`.

### `typevet.adapters.outbound.gemma`

This package holds Gemma and ChatML served-template constants, template
classification and answer-binding helpers. It imports no serving backend.

| Name | Role |
|---|---|
| `CHATML_ASSISTANT_HEADER`, `CHATML_IM_START`, `CHATML_IM_END` | ChatML turn control strings |
| `GEMMA3_START_OF_TURN`, `GEMMA3_END_OF_TURN`, `GEMMA3_MODEL_TURN_HEADER` | Gemma 3 turn control strings |
| `GEMMA4_TURN_OPEN`, `GEMMA4_TURN_CLOSE`, `GEMMA4_MODEL_TURN_HEADER` | Gemma 4 turn control strings |
| `GEMMA4_CHANNEL_CLOSE`, `GEMMA4_THINK_TRIGGER`, `GEMMA4_NO_THINKING_PREFILL`, `GEMMA4_TOOL_RESPONSE` | Gemma 4 channel, thinking and tool control strings |
| `ServedTemplateClass` | Served-template family enum |
| `classify_served_template` | Classify `/apply-template` output |
| `stop_markers_for` | Stop-marker substrings for one template class |
| `label_embeds_control_fragment` | Tell whether a label embeds a reserved control fragment |
| `ThinkingDisposition` | Thinking-channel disposition enum |
| `AnswerAnchor`, `resolve_answer_anchor` | Byte and token boundary before the first candidate token, and its resolver |
| `bind_enum_label`, `bind_enum_labels` | Bind enum labels to exact token ids and `CandidateTokenSpec` rows |
| `termination_kind` | Classify how an assistant completion ends |
| `compose_scoring_prefix`, `compose_media_scoring_prefix` | Scoring prefix text: degraded ChatML, or native turns with media |

## `typevet.adapters.diagnostics`

| Name | Role |
|---|---|
| `LogSettings`, `load_log_settings` | Frozen log settings and the `TYPEVET_LOG__*` reader |
| `configure`, `configure_from_environ` | Apply log settings to stderr structlog |
| `bind_run_id`, `new_run_id` | Invocation id helpers |
| `generation_call_event`, `http_request_event` | Terminal diagnostic event context managers |
| `diagnostic_model` | Keep safe model aliases in event fields |
| `REDACTED`, `SECRET_KEYS` | Redaction placeholder and masked field names |

Importing the package does not configure structlog. See
[diagnostic events](diagnostic-events.md).

## `typevet.testing`

| Name | Role |
|---|---|
| `StaticGenerationFake` | Fixed-value port double without adapter imports |
| `ScriptedScoringFake` | Scripted logprob double for `CandidateScoringPort` |

Use for fast unit doubles and offline typed-judgment tutorials from an installed
wheel. Use `FakeGenerationAdapter` when tests must exercise schema validation
like production outbound code.

## `typevet.adapters`

Organizational package only. It has no `__all__` exports. Import from
`typevet.adapters.inbound`, `typevet.adapters.outbound`, or
`typevet.adapters.diagnostics`.

## Not in the wheel

The `typevet-evals` workspace member in `evals/` holds the evaluation code.
Its import package is `typevet_evals`. The member is never published: its
classifier is `Private :: Do Not Upload`. The library wheel holds only
`typevet`. The member has no compatibility promise. Import it by submodule.

| Module | Contents |
|---|---|
| `typevet_evals.cli` | Module entries `eval_runner` and `cord_semantic_acceptance` |
| `typevet_evals.datasets` | Dataset loaders, download helpers and the partner guard |
| `typevet_evals.runner`, `typevet_evals.tpjep` | Eval runner and TPJEP records, loader and runner |
| `typevet_evals.cord`, `typevet_evals.psai_vision_consumer`, `typevet_evals.instruction_variant` | Evaluation harness families |
| `typevet_evals.throughput`, `typevet_evals.vllm_acceptance` | Throughput and vLLM acceptance harnesses |
| `typevet_evals.experiment_identity`, `typevet_evals.outcome_replay_metrics`, `typevet_evals.psai_vision_probability_evidence` | Shared evaluation modules |
| `typevet_evals.wheel_isolated`, `typevet_evals.gemma_native_vision_wheel_smoke` | Wheel proof tooling |

The library never imports `typevet_evals`. An import-linter contract enforces
this rule. The root library tests may import `typevet_evals`. See
[ADR 0002](../adr/0002-package-layout.md).

## Removed before 0.1.0

typevet removed these root modules before its first release (#256). They only
re-exported names from their current home. Import from the current home.

| Removed module | Current home |
|---|---|
| `typevet.eval_runner_cli` | `typevet_evals.cli.eval_runner` (workspace member, not in the wheel) |
| `typevet.cord_semantic_acceptance_cli` | `typevet_evals.cli.cord_semantic_acceptance` (workspace member, not in the wheel) |
| `typevet.eval_runner`, `typevet.eval_runner_datasets`, `typevet.eval_runner_live_gate`, `typevet.eval_runner_report` | `typevet_evals.runner` (workspace member, not in the wheel) |
| `typevet.eval_tpjep_loader`, `typevet.eval_tpjep_outcome`, `typevet.eval_tpjep_records`, `typevet.eval_tpjep_runner` | `typevet_evals.tpjep` (workspace member, not in the wheel) |
| Other `typevet.eval_*` loaders, download helpers, and guards | One submodule of `typevet_evals.datasets` (workspace member, not in the wheel) |

typevet also moved the llama.cpp outbound modules into one package before its
first release (#256). The six old module files are removed, and five of the
old paths do not resolve. `typevet.adapters.outbound.llama_cpp` still resolves:
it is now a package that re-exports the three adapters.

| Removed module | Current home |
|---|---|
| `typevet.adapters.outbound.llama_cpp` (module) | `typevet.adapters.outbound.llama_cpp.generation` |
| `typevet.adapters.outbound.async_llama_cpp` | `typevet.adapters.outbound.llama_cpp.generation_async` |
| `typevet.adapters.outbound.llama_cpp_http` | `typevet.adapters.outbound.llama_cpp.http_mapping` |
| `typevet.adapters.outbound.llama_cpp_multimodal` | `typevet.adapters.outbound.llama_cpp.multimodal` |
| `typevet.adapters.outbound.llama_cpp_scoring` | `typevet.adapters.outbound.llama_cpp.scoring` |
| `typevet.adapters.outbound.gemma_native_vision_factory` | `typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory` |

typevet also moved the vLLM outbound modules into one package before its first
release (#256). The six old module files do not resolve.

| Removed module | Current home |
|---|---|
| `typevet.adapters.outbound.vllm_content` | `typevet.adapters.outbound.vllm.content` |
| `typevet.adapters.outbound.vllm_generation` | `typevet.adapters.outbound.vllm.generation` |
| `typevet.adapters.outbound.vllm_generation_async` | `typevet.adapters.outbound.vllm.generation_async` |
| `typevet.adapters.outbound.vllm_http` | `typevet.adapters.outbound.vllm.http_mapping` |
| `typevet.adapters.outbound.vllm_scoring` | `typevet.adapters.outbound.vllm.scoring` |
| `typevet.adapters.outbound.vllm_judgment_factory` | `typevet.adapters.outbound.vllm.judgment_factory` |

typevet also moved these evaluation modules out of the library into the
`typevet-evals` workspace member before its first release (#256). The wheel
does not hold them, and the old paths do not resolve.

| Removed module | Current home (workspace member, not in the wheel) |
|---|---|
| `typevet.adapters.inbound.cord_semantic_acceptance_cli` | `typevet_evals.cli.cord_semantic_acceptance` |
| `typevet.adapters.inbound.eval_cli` | `typevet_evals.cli.eval_runner` |
| `typevet.testing.wheel_isolated` | `typevet_evals.wheel_isolated` |
| `typevet.evaluation.gemma_native_vision_wheel_smoke` | `typevet_evals.gemma_native_vision_wheel_smoke` |
| `typevet.evaluation.instruction_variant_consumer_*` (8 modules) | `typevet_evals.instruction_variant.*`, with the prefix removed (for example `…_offline` → `offline`) |
| `typevet.evaluation.outcome_replay_metrics` | `typevet_evals.outcome_replay_metrics` |
| `typevet.evaluation.collections_metrics`, `…collections_throughput`, `…collections_workload`, `…public_workload` | `typevet_evals.throughput` (same module names) |
| `typevet.evaluation.vllm_acceptance`, `…vllm_acceptance_sets`, `…vllm_acceptance_transport` | `typevet_evals.vllm_acceptance.core`, `.sets`, `.transport` |
| `typevet.evaluation.psai_vision_consumer_*` (12 modules) | `typevet_evals.psai_vision_consumer.*`, with the prefix removed (for example `…_harness` → `harness`) |
| `typevet.evaluation.consumer_http_accounting` | `typevet_evals.psai_vision_consumer.http_accounting` |
| `typevet.evaluation.psai_vision_probability_evidence` | `typevet_evals.psai_vision_probability_evidence` |
| `typevet.evaluation.cord_*` (7 modules) | `typevet_evals.cord.*`, with the `cord_` prefix removed (for example `…_semantic_acceptance` → `semantic_acceptance`) |
| `typevet.evaluation.runner` and its 4 modules | `typevet_evals.runner` (same module names) |
| `typevet.evaluation.tpjep` and its 4 modules | `typevet_evals.tpjep` (same module names) |
| `typevet.evaluation.experiment_identity` | `typevet_evals.experiment_identity` |
| `typevet.evaluation.datasets` and its 22 modules, with the `clinc_*.json` data files | `typevet_evals.datasets` (same module names) |
| `typevet.evaluation` | No replacement. The package is removed; each evaluation family is a `typevet_evals` subpackage or module |

## Command-line entry

typevet declares no console script. `pyproject.toml` has no
`[project.scripts]` table, so installing the wheel does not put a `typevet`
command on `PATH`.

The wheel ships no command. In a checkout, the eval runner and the CORD
receipt acceptance check are Python module entries in the `typevet-evals`
workspace member:

```bash
uv run python -m typevet_evals.cli.eval_runner --help
uv run python -m typevet_evals.cli.cord_semantic_acceptance path/to/receipt.json
```

Without `TYPEVET_LLAMA__*` router settings, the command prints a skip
reason and exits `0`. See [live eval runner](eval-live-runner.md).

## Typing and packaging

- `src/typevet/py.typed` marks the package as typed for consumers.
- Public API surface is the union of package `__all__` lists and documented
  module docstrings checked by docvet.
- Version string authority is `[project].version` in `pyproject.toml`.
  `typevet.__version__` reads it back from distribution metadata. See
  [verify package typing and version](https://github.com/Alberto-Codes/typevet/blob/main/docs/maintainers/verify-package.md).

## 0.1.0 compatibility assessment

| Surface | Assessment | Notes |
|---|---|---|
| Library | Initial public hex surface | Root, domain, ports, runtime, inbound, outbound (with `llama_cpp`, `vllm` and `gemma`), diagnostics, testing |
| Evaluation | Not shipped | All evaluation code is in the `typevet-evals` workspace member. See [Not in the wheel](#not-in-the-wheel) |
| CLI | Not shipped | No console script and no module entry in the wheel. In a checkout: `python -m typevet_evals.cli.eval_runner` and `python -m typevet_evals.cli.cord_semantic_acceptance`. The `cli` extra lists Typer only and no module imports it |
| MCP | Not shipped | No extra or entry point |
| Dependencies | `httpx`, `jsonschema`, `structlog` | Locked via `uv.lock` in development |

Additive changes should extend `__all__` and this page. Breaking renames or
removed exports require an explicit compatibility note in a future release
section on this page.

## Verification limits

Contract tests with `FakeGenerationAdapter` do not prove a live llama.cpp router
behaves correctly. Live tests use the `live` pytest marker and opt-in runs.
See [verification](../explanation/verification.md).
