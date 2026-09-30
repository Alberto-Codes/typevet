---
status: draft
---

# Security

Kind: reference. This page lists the data that typevet sends to a backend.
It also covers API keys, redaction and vulnerability reports.

Status: **draft**.

The facts below describe the library in `src/typevet` at the 0.1.0 release.
The evaluation code in `evals/` (`typevet_evals`) is not part of the wheel.
This page is an implementation review, not a security certification.

## What typevet sends and where

typevet sends HTTP requests to the backend base URL that you configure.
The environment can route them through a proxy.
The adapters take a client as `client=`. They send to their own `base_url`
argument and use the client's TLS and proxy options.
The Gemma vision factory takes a client as `http_client=`.
It sends `/apply-template` and `/tokenize` to the client's base URL, and
`/props` and scoring requests to `settings.base_url`.
The outbound adapters under `src/typevet/adapters/outbound/llama_cpp/` and
`src/typevet/adapters/outbound/vllm/` use `httpx` for each request.

| Backend | Base URL source | Paths |
|---|---|---|
| llama.cpp | `TYPEVET_LLAMA__BASE_URL` or the `base_url` argument | `v1/chat/completions`, `completion`, `props`, `apply-template`, `tokenize` |
| vLLM | `TYPEVET_VLLM__BASE_URL` or the `base_url` argument | `v1/chat/completions`, `tokenize` |

A request body holds the data for one call:

- The model name.
- The prompt or the rendered question text, which includes your `state`.
- The JSON Schema for a generation call.
- The candidate token IDs for a vLLM scoring call.
- Fixed sampling options, such as `temperature`.
- Each image that you attach, encoded as base64.

Both backends send the httpx default headers, such as `Accept`,
`Accept-Encoding`, `Connection` and `User-Agent: python-httpx/<version>`.
The vLLM client changes two headers:

- `Authorization: Bearer <key>`, only when you set a key.
- `User-Agent`: the value of `TYPEVET_VLLM__USER_AGENT`, or the httpx default.

The llama.cpp adapters send no key.

typevet sets no TLS or proxy options on the clients that it builds. On those
clients the httpx defaults apply. The client verifies certificates.
The client reads these environment variables:

- `HTTPS_PROXY` and the other proxy variables, which route requests.
- `SSL_CERT_FILE` and `SSL_CERT_DIR`, which replace the default trusted
  certificates.
- `SSLKEYLOGFILE`, through the Python `ssl` module. When it is set, the
  process writes TLS session keys to that file.

A plain HTTP URL sends the key and the request body without encryption. Use an
HTTPS URL or a local address.

The backend receives your prompts, `state` and images. Send only data that you
may disclose to that backend.

## What typevet does not send or store

- The library has no telemetry and no remote log export. No module in
  `src/typevet` opens a connection to a host other than the configured backend
  or a proxy from the environment.
- The library code writes no files. It has no cache, database or credential
  store. `SSLKEYLOGFILE` is the exception above. The `ssl` module writes that
  file, not typevet code.
- The typevet modules do not read the environment at import. The settings
  loaders read it only when you call them. Dependencies such as structlog can
  read variables at import.
- Diagnostic events appear only when structlog is configured. See
  [Redaction in diagnostic events](#redaction-in-diagnostic-events).

Your application, shell, log handlers and backend can still store prompts,
answers and errors. typevet does not control that storage.

## API keys

Only the vLLM path takes an API key. The settings are in
[Configuration](configuration.md#vllm-server).

| Surface | Behaviour |
|---|---|
| Setting | `TYPEVET_VLLM__API_KEY`. The value must be ASCII. An empty value sends no key. |
| `repr` | `repr(VllmSettings)` does not show the key. |
| Settings errors | An error names the variable, never its value. |
| Adapter errors | The adapters from `generation_adapter` and `async_vllm_generation_adapter` mask the key in a `GenerationError`. The port from `open_judgment` also masks it. When a key is set, each such error is a masked copy with no cause or context. |
| Receipts | The vLLM live acceptance run in `evals/` masks the key before it writes the receipt. |

When a key is set, each `GenerationError` from these wrappers is a masked
copy. The copy shows `***` in place of the raw or JSON-escaped key. The copy
has the same type, and it has no cause or context. The copy is made even when
the key text is absent. The httpx error in the cause chain holds the request,
and its headers hold `Authorization: Bearer <key>`. Thus the copy drops that
chain. Without a key, the wrappers raise the original error. Masking
applies to strings inside a dict, list or tuple, for example a parsed payload.
Masking applies only to a `GenerationError`. Other exceptions, such as a
`RuntimeError` from the httpx client, pass through without masking.
[Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md#key-and-network-behaviour)
gives the steps and the limits.

Known gaps:

- Masking exists only in the wrappers from `generation_adapter`,
  `async_vllm_generation_adapter` and `open_judgment`. A vLLM adapter or
  client that you build yourself does not mask the key. This includes the
  sync and async generation adapters, the scoring adapter and the judgment
  factory.
- Masking does not look inside a set or a bytes value.
  [#227](https://github.com/Alberto-Codes/typevet/issues/227) tracks this gap.
- The default pytest options omit `--showlocals`
  ([#251](https://github.com/Alberto-Codes/typevet/issues/251)). A run with
  `--showlocals` or `-l` can still print a key from a test environment.
- Masking does not protect tracebacks from other code, debuggers or memory
  dumps. The client holds the key as a plain string.
  The traceback of a masked error still reaches that client through the
  wrapper frame. A tool that captures frame objects can read the key.

## Redaction in diagnostic events

The event helpers emit an event whenever structlog is configured. The typevet
`configure` function sends events to stderr by default.
`configure(settings, stream=...)` selects another stream.
The typevet redaction processor applies only through `configure`. When your
application configures structlog itself, events use your processors and
output. They then get no typevet redaction.

The typevet redaction processor replaces these values with `***`:

- Values under the listed secret field names: `api_key`, `authorization`,
  `password`, `private_key`, `private_key_pem` and `token`.
- Values under prompt field names, such as `prompt` and `messages`, unless
  `TYPEVET_LOG__LOG_PROMPTS` is on.
- String values that look like PEM data.

The built-in events keep a closed set of fields. They exclude prompts,
schemas, headers, response bodies and exception messages. The outbound
adapters do not emit these events yet.
[Diagnostic events](diagnostic-events.md#redaction) lists each field name and
each rule.

Redaction is not a general secret detector. It does not mask a value under
another field name, for example `bearer`. It does not change `exc_info`, so a
traceback in an event can hold a key or a prompt.

## Partner data

See the [eval partner data policy](eval-partner-data-policy.md).

## Report a vulnerability

Use the private GitHub
[Report a vulnerability form](https://github.com/Alberto-Codes/typevet/security/advisories/new).
Sign in to GitHub to submit a report. Do not put a vulnerability, a key or
private data in a public issue.

Include the affected version, the impact and a minimal reproduction. Use
dummy keys and synthetic data. Reports apply to the latest release on PyPI.
This policy does not promise a response deadline or fixes for older releases.
