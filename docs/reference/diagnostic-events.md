# Diagnostic events

Kind: reference. This page is the closed-set event contract for generation and
HTTP adapters. Sister shape: judgevet
[`docs/reference/events.md`](https://github.com/Alberto-Codes/judgevet/blob/main/docs/reference/events.md).

Status: **current code** for the module, field filters, redaction and
environment settings ([#39](https://github.com/Alberto-Codes/typevet/issues/39)).
**sketch** for automatic emission from outbound generation adapters until they
wrap the context managers below.

Parent: [#29](https://github.com/Alberto-Codes/typevet/issues/29). Tracking:
[#40](https://github.com/Alberto-Codes/typevet/issues/40).

Structured stderr lines are **diagnostics**, not telemetry. Nothing in
`typevet.adapters.diagnostics` opens a remote sink or exports metrics.
Domain code must not import structlog; only the composition root and adapters
call [`configure`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/logs.py) and the event
helpers.

## Rendering and ownership

Library imports stay silent. Context managers emit only when
`structlog.is_configured()` is true. Applications and worker harnesses own
configuration.

The default log level is `info`. Built-in terminal events log at **debug**, so
set `TYPEVET_LOG__LEVEL=debug` to see `http.request` and `generation.call`.

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `TYPEVET_LOG__FORMAT` | `auto`, `json`, `console` | `auto` | `json` is one object per line. `auto` picks JSON when stderr is not a TTY. |
| `TYPEVET_LOG__LEVEL` | `debug`, `info`, `warning`, `error`, `critical` | `info` | Minimum severity kept after filtering. |
| `TYPEVET_LOG__LOG_PROMPTS` | truthy (`1`, `true`, `yes`, `on`) | off | When off, prompt-like field names are redacted. |

Use [`configure_from_environ`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/logs.py) or
[`configure`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/logs.py) with
[`LogSettings`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/settings.py) at the
composition root. See
[library-first architecture](../explanation/library-first-architecture.md).

Processor order merges context, applies the built-in field filter, adds
`level` and UTC ISO `timestamp`, then redacts and renders to stderr. JSON key
order is not part of the contract.

## Invocation correlation

Import [`bind_run_id`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/logs.py) and
[`new_run_id`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/logs.py) from
`typevet.adapters.diagnostics`. `new_run_id()` returns twelve lowercase hex
characters. `bind_run_id` attaches that value to built-in events through
[`current_run_id`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/fields.py).

Built-in events expose `run_id` in their closed field set. Arbitrary structlog
context keys are stripped from built-in events by
[`filter_event_fields`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/fields.py); use
the dedicated binding instead of ad hoc context fields.

## Built-in event names

Built-in names form a closed set:

| Event | One line per |
|---|---|
| `http.request` | Logical HTTP request (for example one POST to chat completions). |
| `generation.call` | Logical generation call through a port adapter. |

Application-defined events are separate. The field filter leaves unknown
`event` values intact (subject to redaction).

## HTTP terminal event (`http.request`)

Emit with [`http_request_event`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/http_events.py)
once per logical request, at debug level, when logging is configured.

Adapters update the yielded [`HttpRequestEvent`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/http_events.py)
before the block exits. Default `outcome` is `error` until the adapter sets
`success`. Exceptions propagate; the terminal line records `error_type` as the
exception class name (never message text).

With JSON rendering, built-in keys are exactly:

| Key | Type | Meaning |
|---|---|---|
| `event` | string | Always `http.request`. |
| `level` | string | Always `debug` for this helper. |
| `timestamp` | string | UTC ISO timestamp from the renderer. |
| `method` | string | HTTP method (default `POST`). |
| `path` | string | Path relative to the adapter base URL (default `v1/chat/completions`). |
| `model` | string or null | Filtered router alias; see model filter below. |
| `status_code` | integer or null | HTTP status when known. |
| `outcome` | string | `success` or `error`. |
| `error_type` | string or null | Exception class name on failure; null on success. |
| `run_id` | string or null | Bound invocation id; null when nothing is bound. |

Prompts, JSON schemas, headers, response bodies and exception messages are
excluded. They are not in the closed field set.

### Adapter wiring (**sketch**)

[`LlamaCppGenerationAdapter`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/outbound/llama_cpp.py) and
[`FakeGenerationAdapter`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/outbound/fake.py) do not yet
wrap `http_request_event`. The contract above is authoritative for the next
adapter change.

## Generation terminal event (`generation.call`)

Emit with [`generation_call_event`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/generation_events.py)
once per logical generation call, at debug level, when logging is configured.

Adapters update the yielded [`GenerationCallEvent`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/generation_events.py)
before the block exits. Exception handling matches `http.request`.

With JSON rendering, built-in keys are exactly:

| Key | Type | Meaning |
|---|---|---|
| `event` | string | Always `generation.call`. |
| `level` | string | Always `debug` for this helper. |
| `timestamp` | string | UTC ISO timestamp from the renderer. |
| `model` | string or null | Filtered router alias; see model filter below. |
| `outcome` | string | `success` or `error`. |
| `error_type` | string or null | Exception class name on failure; null on success. |
| `run_id` | string or null | Bound invocation id; null when nothing is bound. |

Prompt and schema payloads never appear in this event.

### Adapter wiring (**sketch**)

Outbound generation adapters do not yet wrap `generation_call_event`. Nest
HTTP and generation events when an adapter performs both (one `generation.call`
around the port work, one `http.request` around the POST).

## Model alias filter

[`diagnostic_model`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/fields.py) keeps
values that match ASCII
`^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$`. Other strings become null in
diagnostics only. Request and API model values are unchanged.

## Redaction

[`make_redact_processor`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/redaction.py)
masks:

- Field names in `SECRET_KEYS`: `api_key`, `authorization`, `password`,
  `private_key`, `private_key_pem`, `token` (case-insensitive).
- When `log_prompts` is false, names in `PROMPT_KEYS`: `content`, `messages`,
  `prompt`, `raw_text`, `system`, `user`.
- String values that look like PEM (`-----BEGIN`).

Masked scalars become `***` ([`REDACTED`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/diagnostics/redaction.py)).
Nested structures are walked; non-scalar leaves become type names.

Unit tests in [`test_diagnostics.py`](https://github.com/Alberto-Codes/typevet/blob/main/tests/unit/test_diagnostics.py)
cover secret keys, prompt keys and built-in field stripping.

## Public surface

Import from `typevet.adapters.diagnostics` (not the package root):

| Symbol | Role |
|---|---|
| `configure`, `configure_from_environ`, `bind_run_id`, `new_run_id` | Composition-root setup and correlation |
| `LogSettings`, `load_log_settings` | Settings types and env loader |
| `http_request_event`, `generation_call_event` | Terminal event context managers |
| `diagnostic_model` | Safe model alias helper |
| `REDACTED`, `SECRET_KEYS` | Redaction constants |

Implementation modules live under
[`src/typevet/adapters/diagnostics/`](https://github.com/Alberto-Codes/typevet/tree/main/src/typevet/adapters/diagnostics).

## Related pages

- [Errors](errors.md) — domain and adapter failure types referenced in
  `error_type`.
- [Supported imports](supported-imports.md) — library `__all__` surfaces.
- Sister: [judgevet diagnostic events](https://github.com/Alberto-Codes/judgevet/blob/main/docs/reference/events.md).
