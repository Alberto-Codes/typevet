# Use typevet as a judgevet provider

Kind: how-to.

Use this page to send judgevet questions to a typevet backend instead of Jev.
The code that calls judgevet stays the same.
Only the provider changes.

The bridge is compatible in shape with Jev, not equivalent in judgment.
The answers come from your backend model, for example Gemma 4.
Calibration does not transfer between judge models.
Measure each judge model on your own task before you trust its numbers.

## Prerequisites

- A typevet backend that you can reach: llama.cpp or vLLM.
  See [Run Gemma 4 on llama.cpp](run-gemma4-llamacpp.md) or [Serve typevet on vLLM](serve-typevet-on-vllm.md).
- The `TYPEVET_BACKEND` and backend settings in the environment.
  See [configuration](../reference/configuration.md).
- For llama.cpp, `TYPEVET_LLAMA__MULTIMODAL_MODEL` set to the judgment model.
  `open_judgment` and the bridge probe this variable, not `TYPEVET_LLAMA__DEFAULT_MODEL`.
  It must name a Gemma 4 model whose template renders `<|turn>`.
  The default, `gemma-3-4b-it-q4km-mm`, is a Gemma 3 id and fails this check with `ValueError`.

  ```bash
  export TYPEVET_BACKEND=llama_cpp
  export TYPEVET_LLAMA__MULTIMODAL_MODEL='<your-gemma-4-multimodal-model-id>'
  ```

- judgevet 0.17 or 0.18. The `judgevet` extra pins `judgevet>=0.17,<0.19`.

## Install the extra

In your project directory, run one of these commands:

```bash
uv add "typevet[judgevet]"
```

```bash
pip install "typevet[judgevet]"
```

A bare `typevet` install does not install or import judgevet.
Only the bridge, `typevet.adapters.inbound.judgevet`, and its mapping module, `typevet.adapters.inbound.judgevet_mapping`, import judgevet.
Without the extra, that import raises `ImportError` and names `typevet[judgevet]`.

## Build the provider

`provider_factory` wraps a typevet session opener.
Each provider scope opens one session and closes it at exit.

```python
from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.adapters.inbound.judgevet import provider_factory

factory = provider_factory(open_judgment)
```

To hold one port for the whole program, wrap a session port yourself:

```python
from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort

with open_judgment() as session:
    port = TypevetSystemOnePort(session.port)
    response = port.system_one(state, questions, model=session.model)
```

For async callers, wrap the port in `AsyncTypevetSystemOnePort`.
Each call then runs on a worker thread.

## Swap the provider in a pipeline

Find the place where your pipeline builds its judgevet port.
Replace the Jev port with the typevet port.
Keep the questions, the policy and the answer handling.

```python
from judgevet.providers import provider_scope

with provider_scope(factory=factory) as port:
    response = port.system_one(state, questions, model="gemma-4-31b")
```

Pass the model id that your typevet session serves.
A pinned session refuses any other model id with `ProviderRequestError`.
`response.model` is the model id that the backend reports.

## Run the judgevet command line on typevet

Build the judgevet command with the factory:

```python
from judgevet.adapters.inbound.cli import create_cli_app

app = create_cli_app(provider_factory=factory)
```

Run `app` as a Typer application.
It takes the same arguments as the `judgevet` command.

## Serve judgevet MCP on typevet

Install `judgevet[mcp]` for the MCP runtime.
Then start the stdio server with the factory:

```python
from judgevet.adapters.inbound.mcp_entrypoint import main

raise SystemExit(main(provider_factory=factory, model="gemma-4-31b"))
```

The server opens one typevet session for its lifetime and closes it at shutdown.

## Know the declared capabilities

Each bridge port declares its capabilities in `port.bridge_capabilities`.
The bridge never infers a capability and never falls back.

| Capability | Declaration |
|---|---|
| Logprobs | Required. A backend that returns no logprobs raises `ProviderResponseError` or `ProviderCapabilityError` |
| Choice options and Score levels | At most 24 by default. A larger question raises `ProviderCapabilityError` before any backend call |
| Instructions | Text or absent. Object and array instructions raise `ProviderCapabilityError` |
| Images | None on `TypevetSystemOnePort`. `judge_with_images` refuses it with `ProviderCapabilityError` |

The default cap is typevet's native limit, `MAX_ENUM_CHOICES` (24 options).
Issue [#287](https://github.com/Alberto-Codes/typevet/issues/287) raised the native limit to 24.
Pass `max_choice_options` to declare a lower cap.

Text instructions are enough for question-wording evolution with gepa-adk.
gepa-adk evolves a wording value in a caller-owned mapping, and its `MappingComponentHandler` accepts only `str` values.
The evolution loop therefore sends text instructions to the provider.
Questions built outside the evolution loop can carry object or array instructions. The bridge refuses them.
Render such instructions to text before you call the bridge.
Issue [#291](https://github.com/Alberto-Codes/typevet/issues/291) records this decision.

To judge images, use `TypevetMediaSystemOnePort` with a vision backend.
Declare the image types in a `MediaCapabilities` value:

```python
from judgevet.media import MediaCapabilities

factory = provider_factory(open_judgment, media=MediaCapabilities({"image/png"}))
```

Each question can have its own images.
The media port groups the questions by their bound images and keeps the image order of each binding.
It sends one typevet judgment for each group. The backend still gets one scoring call per question.
Questions with no images go as one text judgment.
The port merges the answers into one response and adds up the token counts.
If the groups report different model ids, the port raises `ProviderResponseError`.
An image type outside the declared `MediaCapabilities` raises `ProviderCapabilityError` before any backend call.

## Evolved wording over the bridge

An evolution run can change more than one text part of a question.
Render each candidate to the existing question fields before you call the bridge.
The bridge needs no other field.

| Part | Component names | Question field |
|---|---|---|
| Question instructions | `instructions` | `instructions`, as text |
| Noul true and false texts | `criteria_true`, `criteria_false` | `criteria`, as `{"true": ..., "false": ...}` |

The bridge sends both fields to typevet unchanged.
This applies to a typed judgevet `Noul` and to a raw wire dict.
Object or array instructions still raise `ProviderCapabilityError`.
That error message does not repeat the question text.

The rendered option block and the context template never cross the bridge.
typevet renders them from the question, so a caller cannot change them through judgevet.
Issue [#364](https://github.com/Alberto-Codes/typevet/issues/364) tracks evolution of those parts.
[Judgment text parts](../reference/judgment-text-parts.md#component-names) lists each part and its component name.
Contract tests in `tests/contract/test_judgevet_bridge_evolved_wording.py` prove this on offline fakes.

## Set the off-option threshold and read the receipts

Pass `off_option_threshold` as a keyword, or as the `off_option_threshold` key of judgevet `provider_options`.
The text, media and async ports take both.
When the keyword is set, the keyword wins.

```python
response = port.system_one(
    state,
    questions,
    model="gemma-4-31b",
    provider_options={"off_option_threshold": 0.25},
)
receipt = response.receipts["q"]
```

The bridge reads no other option key. Any other key raises `ProviderCapabilityError` before any backend call.
A threshold that is not `None` or a number from 0 to 1 raises `ProviderRequestError` before any backend call.
This applies to the keyword and to the option value. The bridge checks the option value also when the keyword wins.
The error message does not show the value.

Each answer with a typevet off-option receipt has one entry in `response.receipts`, under the answer name.
The entry has `off_option_mass`, `off_option_threshold` and `off_option_flag`.
`off_option_mass` is the probability outside the option set.
`off_option_flag` is `True` only when the mass is above the threshold.
An answer with no receipt has no entry.

## Handle errors

The bridge maps each typevet error to a judgevet `ProviderError` subclass.
The judgevet error keeps the typevet message.
Its `__cause__` is the typevet error.

| typevet error | judgevet error |
|---|---|
| `TransportError` | `ProviderTransportError` |
| `BackendHttpError`, status 400 to 499 | `ProviderRequestError` |
| `BackendHttpError`, other status | `ProviderTransportError` |
| `GenerationUnsupportedCapabilityError`, `ScoringUnsupportedCapabilityError`, `GemmaTemplateError` | `ProviderCapabilityError` |
| `JudgmentValidationError`, `DecisionExecutionError` | `ProviderRequestError` |
| Any other `GenerationError` | `ProviderResponseError` |

Other exceptions keep their type.

## Evidence limits

Offline tests prove the bridge on typevet fakes.
The text, media and async ports pass the judgevet provider conformance kit, `judgevet.testing.conformance` from judgevet 0.17.0 and 0.18.0.
The tests also run judgevet's command line and MCP server on the bridge.
They prove no live model behaviour.
One live run per backend (llama.cpp and vLLM) asked one Noul, one Choice and one Score through the bridge and returned typed answers; see [#289](https://github.com/Alberto-Codes/typevet/issues/289). This is not a quality claim.

## Next steps

- Read why the bridge exists in [TypeLLM, Jev and judgevet](../explanation/typellm-and-judgevet.md).
- See the bridge import path in [supported imports](../reference/supported-imports.md#judgevet-bridge).
- See which text parts of a judgment call a caller can change in [judgment text parts](../reference/judgment-text-parts.md).
- See the judgevet side, with typevet as the example, in judgevet's [use a self-hosted provider](https://alberto-codes.github.io/judgevet/how-to/use-a-self-hosted-provider/).
