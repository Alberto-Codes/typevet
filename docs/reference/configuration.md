# Configuration

Kind: reference. Environment variables for composition roots (CLI, MCP, live
harness). Library callers pass explicit adapter constructor arguments instead.

Parent: [#46](https://github.com/Alberto-Codes/typevet/issues/46). Log settings
were added in [#30](https://github.com/Alberto-Codes/typevet/issues/30).

## Composition root vs library

| Layer | Reads environment | Module |
|---|---|---|
| Composition root | yes | [typevet.adapters.inbound.settings][] |
| Composition root | yes | [typevet.adapters.inbound.backend_settings][] |
| Outbound adapter | no | [typevet.adapters.outbound.llama_cpp][] |
| Diagnostics | yes (stderr only) | [typevet.adapters.diagnostics.settings][] |

Importing typevet does not read the environment. Call
[`load_llama_settings`][] or [`configure_from_environ`][] at process startup in
the inbound layer.

## llama.cpp router

[`LlamaSettings`][] holds connection options. [`load_llama_settings`][] reads
the mapping below. [`llama_cpp_adapter`][] passes the values into
`LlamaCppGenerationAdapter` without the adapter touching `os.environ`.

| Environment name | Field | Type | Default | Notes |
|---|---|---|---|---|
| `TYPEVET_LLAMA__BASE_URL` | `base_url` | URL string | `http://127.0.0.1:8090` | Trailing slash stripped |
| `TYPEVET_LLAMA__TIMEOUT` | `timeout` | float, seconds | `300` | Must be positive |
| `TYPEVET_LLAMA__DEFAULT_MODEL` | `default_model` | string or empty | none | Router model id for live pytest (any Gemma 4 GGUF alias); live skips when empty |
| `TYPEVET_LLAMA__MULTIMODAL_MODEL` | `multimodal_model` | string | `gemma-3-4b-it-q4km-mm` | Router model id for the [image-conditioned live smoke](../how-to/run-a-multimodal-live-smoke.md); the id must declare `image` input |
| `TYPEVET_LLAMA_URL` | `base_url` | URL string | (same) | Legacy alias when nested name unset |
| `TYPEVET_GEMMA_MODEL` | `default_model` | string | (same) | Legacy alias when nested name unset |

Nested names win when both nested and legacy names are set.

### Future CLI hookup

There is no shipped CLI yet (`pyproject.toml` optional extra `cli` only adds
Typer). When a CLI lands, it should:

1. Call `load_llama_settings()` once at startup (and `configure_from_environ()`
   for stderr diagnostics).
2. Build `llama_cpp_adapter(settings)` and pass the port into command handlers.
3. Close the adapter on process exit.

Until then, scripts and live tests act as the composition root using the same
helpers.

## Backend selection

[`generation_adapter`][] reads `TYPEVET_BACKEND` and builds one generation
adapter.

| Environment name | Values | Default | Notes |
|---|---|---|---|
| `TYPEVET_BACKEND` | `llama_cpp`, `vllm` | `llama_cpp` | Other values raise `ValueError` |

`llama_cpp` builds `llama_cpp_adapter(load_llama_settings())`. `vllm` builds a
`VllmGenerationAdapter` on the [`vllm_http_client`][] client. Closing that
adapter closes its client.

## vLLM server

[`VllmSettings`][] holds connection options. [`load_vllm_settings`][] reads the
mapping below.

| Environment name | Field | Type | Default | Notes |
|---|---|---|---|---|
| `TYPEVET_VLLM__BASE_URL` | `base_url` | URL string | none | Required; trailing slash stripped |
| `TYPEVET_VLLM__MODEL` | `model` | string | none | Required; served model name |
| `TYPEVET_VLLM__TIMEOUT` | `timeout` | float, seconds | `300` | Must be positive |
| `TYPEVET_VLLM__API_KEY` | `api_key` | string or empty | none | Sent as `Authorization: Bearer <key>` |

The key does not appear in `repr(VllmSettings)`. The client is built the same
way with or without a key. Only the `Authorization` header differs. So
`HTTPS_PROXY` and the other proxy variables apply in both cases. The key must
be ASCII. The adapter
from `generation_adapter` masks the key in errors. It checks each generation
error and its cause for the raw or JSON-escaped key. On a match, it raises the
same error type again. The new error shows `***` for the key and has no cause. A server that echoes
the `Authorization` header therefore cannot put the key into a
`BackendHttpError` message. Successful results are not changed. Error messages
name the variable, not its value.

## Diagnostic logging

See [Diagnostic events](diagnostic-events.md) for `TYPEVET_LOG__FORMAT`,
`TYPEVET_LOG__LEVEL`, and `TYPEVET_LOG__LOG_PROMPTS`.

## Related pages

- [Library-first architecture](../explanation/library-first-architecture.md)
- [Run Gemma 4 on llama.cpp](../how-to/run-gemma4-llamacpp.md)
- [Supported imports](supported-imports.md)
