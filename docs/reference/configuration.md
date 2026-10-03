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
[`load_llama_settings`][typevet.adapters.inbound.load_llama_settings] or [`configure_from_environ`][typevet.adapters.diagnostics.configure_from_environ] at process startup in
the inbound layer.

## llama.cpp router

[`LlamaSettings`][typevet.adapters.inbound.LlamaSettings] holds connection options. [`load_llama_settings`][typevet.adapters.inbound.load_llama_settings] reads
the mapping below. [`llama_cpp_adapter`][typevet.adapters.inbound.llama_cpp_adapter] passes the values into
`LlamaCppGenerationAdapter` without the adapter touching `os.environ`.

| Environment name | Field | Type | Default | Notes |
|---|---|---|---|---|
| `TYPEVET_LLAMA__BASE_URL` | `base_url` | URL string | `http://127.0.0.1:8090` | Trailing slash stripped |
| `TYPEVET_LLAMA__TIMEOUT` | `timeout` | float, seconds | `300` | Must be positive |
| `TYPEVET_LLAMA__DEFAULT_MODEL` | `default_model` | string or empty | none | Router model id for live pytest (any Gemma 4 GGUF alias); live skips when empty |
| `TYPEVET_LLAMA__MULTIMODAL_MODEL` | `multimodal_model` | string | `gemma-3-4b-it-q4km-mm` | Router model id for the [image-conditioned live smoke](../how-to/run-a-multimodal-live-smoke.md); the id must declare `image` input |
| `TYPEVET_LLAMA__API_KEY` | `api_key` | string or empty | none | Sent in `auth_header`; empty means no key; ASCII only; left out of `repr` |
| `TYPEVET_LLAMA__AUTH_HEADER` | `auth_header` | header name | `Authorization` | Empty means `Authorization`; must be an HTTP token and not protected |
| `TYPEVET_LLAMA__AUTH_SCHEME` | `auth_scheme` | token or empty | `Bearer` | Unset means `Bearer`; set but empty sends the key bare |
| `TYPEVET_LLAMA__HEADERS` | `headers` | JSON object of strings | none | Literal extra headers on every request; read-only; left out of `repr` |
| `TYPEVET_LLAMA__USER_AGENT` | `user_agent` | string or empty | httpx default | Empty means the httpx default |
| `TYPEVET_LLAMA_URL` | `base_url` | URL string | (same) | Legacy alias when nested name unset |
| `TYPEVET_GEMMA_MODEL` | `default_model` | string | (same) | Legacy alias when nested name unset |

Nested names win when both nested and legacy names are set.

The key and header variables follow the [vLLM](#vllm-server) rules
([#410](https://github.com/Alberto-Codes/typevet/issues/410)). The same header
checks run when `LlamaSettings` is built. Error messages name the variable,
never a value. These clients send the key and headers:

- [`llama_cpp_adapter`][typevet.adapters.inbound.llama_cpp_adapter], which owns
  its client and closes it.
- The `llama_cpp` branches of `generation_adapter` and `open_judgment`.
- `async_llama_http_client`, which the caller closes.

With a key or extra headers, typevet masks each error from that adapter. It
also masks each error from the `open_judgment` session open and its port. The
masked error is a copy with each key and header value replaced by `***`. The
copy has no cause or context. An adapter that a caller builds on
`async_llama_http_client` does not mask its errors.
`TYPEVET_LLAMA__REQUEST_ID_HEADER` does not exist: the request-id header is
for vLLM only.

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

[`generation_adapter`][typevet.adapters.inbound.generation_adapter] reads `TYPEVET_BACKEND` and builds one generation
adapter.

| Environment name | Values | Default | Notes |
|---|---|---|---|
| `TYPEVET_BACKEND` | `llama_cpp`, `vllm`, `fake` | `llama_cpp` | Other values raise `ValueError` |
| `TYPEVET_FAKE__DISTRIBUTIONS` | Path to a JSON file, or empty | none | Read only when the backend is `fake`. Questions the file does not name are uniform |

`llama_cpp` builds `llama_cpp_adapter(load_llama_settings())`. `vllm` builds a
`VllmGenerationAdapter` on the [`vllm_http_client`][typevet.adapters.inbound.vllm_http_client] client. Closing that
adapter closes its client. `fake` has no generation adapter, so
`generation_adapter` raises `ValueError`.

[`open_judgment`][typevet.adapters.inbound.open_judgment] reads the same variable and opens a judgment session.
For `fake`, the session is offline and its `model` is `fake`. Its port builds a
`ScriptedJudgmentFake` for each `judge` call.

The `TYPEVET_FAKE__DISTRIBUTIONS` file is a JSON object keyed by question name.

| Question | File value |
|---|---|
| `Noul` | A number, P(True) |
| `Choice` | An object of label to weight |
| `Score` | An object of level string, such as `"0"`, to weight |

A question that the file does not name gets a uniform distribution. When the
variable is unset or empty, all questions are uniform. A missing or invalid
file raises `ValueError`. The message names the variable and holds no file
content. An entry that parses but does not fit its question, such as a label
outside the criteria, raises `JudgmentValidationError` at `judge` time.

## vLLM server

[`VllmSettings`][typevet.adapters.inbound.VllmSettings] holds connection options. [`load_vllm_settings`][typevet.adapters.inbound.load_vllm_settings] reads the
mapping below.

| Environment name | Field | Type | Default | Notes |
|---|---|---|---|---|
| `TYPEVET_VLLM__BASE_URL` | `base_url` | URL string | none | Required; trailing slash stripped |
| `TYPEVET_VLLM__MODEL` | `model` | string | none | Required; served model name |
| `TYPEVET_VLLM__TIMEOUT` | `timeout` | float, seconds | `300` | Must be positive and finite; `nan`, `inf` and `-inf` fail |
| `TYPEVET_VLLM__API_KEY` | `api_key` | string or empty | none | Sent in `auth_header` after `auth_scheme`. The defaults give `Authorization: Bearer <key>` |
| `TYPEVET_VLLM__MAX_CONCURRENCY` | `max_concurrency` | integer | `1` | Must be a positive integer; POST limit for one `AsyncVllmGenerationAdapter` |
| `TYPEVET_VLLM__USER_AGENT` | `user_agent` | string or empty | none | Sent as `User-Agent` only when set; otherwise the httpx default |
| `TYPEVET_VLLM__AUTH_HEADER` | `auth_header` | header name or empty | `Authorization` | The header that carries the key. An empty value gives the default |
| `TYPEVET_VLLM__AUTH_SCHEME` | `auth_scheme` | token or empty | `Bearer` | Sent before the key and a space. A set but empty value sends the key alone |
| `TYPEVET_VLLM__HEADERS` | `headers` | JSON object of strings | `{}` | Extra headers on each request. Values are literal. Not in `repr` |
| `TYPEVET_VLLM__REQUEST_ID_HEADER` | `request_id_header` | header name or empty | none | When set, each request gets a new UUID4 hex value in this header |

The key does not appear in `repr(VllmSettings)`. The sync and async clients
are built the same way with or without a key. Only the auth header differs. So `HTTPS_PROXY` and the other proxy variables apply in both cases.
The key must be ASCII. The adapters from `generation_adapter` and
`async_vllm_generation_adapter` mask the key in errors. Each adapter checks
each generation error, its attributes and its cause for the raw or
JSON-escaped key. The attributes include strings inside a parsed payload, for
example the `payload` of a `SchemaValidationError`. On a match, the adapter
raises the same error type again. The new error shows `***` for the key and
has no cause or context. A server that echoes the `Authorization` header
therefore cannot put the key into a `BackendHttpError` message. Successful
results are not changed. Error messages name the variable, not its value. An
invalid `TYPEVET_VLLM__TIMEOUT` or `TYPEVET_VLLM__MAX_CONCURRENCY` error has
no cause, so a traceback does not show the value.

### API gateway headers

The four gateway variables serve a vLLM server behind an API gateway.
[Use a vLLM server behind an API gateway](../how-to/use-a-vllm-server-behind-an-api-gateway.md)
gives the steps. `VllmSettings` checks these rules when it is built:

| Rule | Limit |
|---|---|
| Header name, `auth_header` and `request_id_header` | An HTTP token of at most 128 bytes |
| `auth_scheme` | Empty or an HTTP token |
| Header value | ASCII characters 0x20 to 0x7E only, at most 2,048 bytes |
| Number of extra headers | At most 32 |
| Extra names and values together | At most 8,192 bytes |

`headers` must not name a hop-by-hop header, `Host`, `Content-Length`,
`Content-Type`, `Accept`, `Accept-Encoding`, `User-Agent` or the auth header.
The match ignores letter case. Set the `User-Agent` header with
`TYPEVET_VLLM__USER_AGENT` only. `auth_header` and `request_id_header` must
not name one of these protected headers either. `request_id_header` must not name the auth header or
an extra header. A failed rule raises `ValueError`. The message names the
field and never shows a value. A `TYPEVET_VLLM__HEADERS` value that is not a
JSON object of strings raises `ValueError` with no cause or context.

typevet sends each header value as written. It does not expand `$NAME` or run
`!command`. The adapters mask each extra header value in errors as a whole
token, not inside a longer word or number. The clients follow no redirect. A 3xx status raises
`BackendHttpError` without the `Location` header. An HTML error body is not
in the `BackendHttpError`, and its `body_snippet` is empty.

`generation_adapter` builds the sync adapter and does not read
`max_concurrency`. `async_vllm_generation_adapter` builds an
`AsyncVllmGenerationAdapter` from the `TYPEVET_VLLM__*` variables and does not
read `TYPEVET_BACKEND`. Its `httpx.AsyncClient` has the same base URL,
timeout, headers and proxy behaviour as the sync client, and
`max_concurrency` sets its POST limit. Closing the adapter closes its client.
The limit applies to one adapter only. Two adapters do not share it. With the
default of `1`, the adapter sends one request at a time. The adapter keeps one
limit for each event loop. The `httpx.AsyncClient` of this adapter binds to
the first event loop that uses it. The adapter records the first running loop
that calls `generate`. A call on a different loop, for example from a second
`asyncio.run`, raises `RuntimeError("build one adapter per event loop")`
before any request. This error is not a `GenerationError`. Build one adapter
for each event loop, for example inside each `asyncio.run` call. An
`AsyncVllmGenerationAdapter` built with `client=None` does the same check. An
adapter with an injected client does not.

### vLLM live acceptance run

The opt-in test `evals/tests/live/test_vllm_acceptance_live.py` reads the variables
above and these three. It skips unless `TYPEVET_REQUIRE_LIVE` is truthy. When
it is truthy and a required variable is missing, the test fails before any
network call.

| Environment name | Default | Notes |
|---|---|---|
| `TYPEVET_REQUIRE_LIVE` | none | Set to `1` to run the paid run |
| `TYPEVET_VLLM_RECEIPT` | none | Required; the file must not exist and the nearest existing parent directory must be writable |
| `TYPEVET_VLLM_POD_NOTES` | `unknown` | Free text for the receipt, such as GPU, flags and Hugging Face revision; never put the key here |

The CORD set sends an off-option threshold of `0.25` with each scored call.
The order set uses the same value when it runs the CORD combined arm again.
This value is the constant `CORD_OFF_OPTION_THRESHOLD` in
`typevet_evals.vllm_acceptance.sets`. No variable changes it. Each scored row
in the PSAI, CORD and order sets records `off_option_mass` and
`off_option_flag`. The PSAI set sends no threshold, so its flag stays `false`.
A row with no off-option receipt, no model call, or a call that failed,
records `null` and `false`. The vLLM
scorer always reports the mass as `null`, so the flag stays `false` on vLLM.

## Diagnostic logging

See [Diagnostic events](diagnostic-events.md) for `TYPEVET_LOG__FORMAT`,
`TYPEVET_LOG__LEVEL`, and `TYPEVET_LOG__LOG_PROMPTS`.

## Related pages

- [Library-first architecture](../explanation/library-first-architecture.md)
- [Use a vLLM server behind an API gateway](../how-to/use-a-vllm-server-behind-an-api-gateway.md)
- [Run Gemma 4 on llama.cpp](../how-to/run-gemma4-llamacpp.md)
- [Supported imports](supported-imports.md)
