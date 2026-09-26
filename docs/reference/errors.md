# Error reference

Kind: reference. This page maps domain failures, adapter behavior and retry
boundaries for typed generation. Sister shape: judgevet
[`docs/reference/errors.md`](https://github.com/Alberto-Codes/judgevet/blob/main/docs/reference/errors.md).

Status: **current code**. A future error-hierarchy ADR
([#43](https://github.com/Alberto-Codes/typevet/issues/43)) may refine names or
grouping; until then, treat the types below as authoritative.

Parent: [#29](https://github.com/Alberto-Codes/typevet/issues/29).

## Domain vs adapter errors

**Domain errors** describe typed generation and schema compilation inside
`typevet.domain`. They do not perform HTTP. Callers use them to classify bad
input, unsupported schema shapes and invalid model output.

**Adapter errors** are the same domain exception types raised from outbound
adapters (`FakeGenerationAdapter`, `LlamaCppGenerationAdapter`). Adapters
translate transport and parse failures into `GenerationError` and run
`jsonschema` validation into `SchemaValidationError`. They do not define a
parallel adapter-specific hierarchy.

The [generation port](../../src/typevet/ports/generation.py) documents the
two failure modes every implementation may raise: `GenerationError` and
`SchemaValidationError`.

## Generation failures

Export from `typevet` (package root) or `typevet.domain.errors`:

| Type | Parent | Meaning |
|---|---|---|
| `GenerationError` | `Exception` | A generation call failed before a valid `GenerationResult` existed (transport, parse, or empty fake). |
| `SchemaValidationError` | `GenerationError` | Parsed output failed JSON Schema validation. Optional `payload` holds the rejected value. |

`SchemaValidationError` stores the message in standard exception `args`. When
set, `payload` is the parsed object or mapping that failed validation (see
[`domain/errors.py`](../../src/typevet/domain/errors.py)).

### Request validation (not generation errors)

`GenerationRequest` rejects bad asks in `__post_init__` with ordinary Python
errors, not `GenerationError`:

| Condition | Type |
|---|---|
| Blank `prompt` or `model` | `ValueError` |
| `schema` is not a mapping | `TypeError` |
| `schema.type` is set and not `"object"` | `ValueError` |

Fix the request; do not treat these as retryable generation failures.

## Schema compilation failures

Import `SchemaError` from `typevet.domain` (not the package root):

| Type | Parent | Meaning |
|---|---|---|
| `SchemaError` | `ValueError` | The JSON Schema mapping is outside the supported compile subset (`compile_json_schema`, `Decision`, dependency layers). |

`compile_json_schema` and helpers in
[`decision_compile.py`](../../src/typevet/domain/decision_compile.py) raise
`SchemaError` with a human-readable message for unsupported keywords, bad
enums, dependency cycles and similar compile-time rules.

For some unsupported field types without a finite enum, compilation raises
`NotImplementedError` instead of `SchemaError`. That signals a missing feature,
not a malformed document the caller can correct by editing one field.

## llama.cpp adapter mapping

[`LlamaCppGenerationAdapter`](../../src/typevet/adapters/outbound/llama_cpp.py)
POSTs to `v1/chat/completions` with `response_format` `json_schema`. HTTP status
**400 and above** are treated as adapter failure (constant `_HTTP_ERROR_STATUS`).

| Condition | Raised type | Typical message prefix |
|---|---|---|
| `httpx.HTTPError` on POST | `GenerationError` | `llama.cpp request failed:` |
| HTTP status ≥ 400 | `GenerationError` | `llama.cpp HTTP {status}:` (body truncated to 500 chars) |
| Response body is not JSON | `GenerationError` | `llama.cpp returned non-JSON HTTP body` |
| Missing or empty `choices[0].message.content` | `GenerationError` | `llama.cpp response missing…` or `empty message content` |
| Message content is not valid JSON | `GenerationError` | `model content was not valid JSON` |
| Parsed JSON root is not an object | `SchemaValidationError` | `model JSON root must be an object` (`payload` set) |
| `jsonschema.validate` fails on parsed object | `SchemaValidationError` | `output failed schema:` (`payload` set) |

This table describes local mapping only. It does not assert which HTTP statuses
a given llama.cpp build returns for every failure mode. See
[Run Gemma 4 on llama.cpp](../how-to/run-gemma4-llamacpp.md) for the live path.

## Fake adapter mapping

[`FakeGenerationAdapter`](../../src/typevet/adapters/outbound/fake.py) validates
a fixed or callable mapping with the same `jsonschema` path as llama.cpp:

| Condition | Raised type |
|---|---|
| Constructor given no `value`, `responder`, or `fail` | `ValueError` |
| `fail=` configured | Raises that exception as-is (tests use `GenerationError`) |
| No value source at generate time | `GenerationError` |
| Value fails schema | `SchemaValidationError` (`fake output failed schema:`) |

## Retry and exception boundaries

typevet **does not** expose a `retryable` flag or a built-in retry loop on
generation adapters. Each `generate` call performs at most one HTTP round trip
(llama.cpp) or one validation pass (fake).

Use these boundaries when a caller adds retries:

| Failure | Retry at generation layer? | Notes |
|---|---|---|
| `SchemaValidationError` | No (fail-fast) | Same prompt and schema may repeat the same invalid output; fix schema, prompt, or model. Adapter docstring: fail-fast on schema mismatch. |
| `GenerationError` (HTTP 5xx, timeout, transport) | Optional caller policy | Not implemented in-repo; a supervisor may retry with backoff outside the adapter. |
| `GenerationError` (HTTP 4xx, bad JSON shape) | Usually no | Often indicates router config, model id, or a non-recoverable response shape. |
| `SchemaError`, `NotImplementedError` | No | Fix or narrow the schema before calling generation. |
| `GenerationRequest` `ValueError` / `TypeError` | No | Fix the request object. |

`except GenerationError` catches `SchemaValidationError` because it subclasses
`GenerationError`. Use `except SchemaValidationError` when validation failures
need distinct handling (for example logging `payload`).

Adapters do not catch arbitrary exceptions from `httpx` beyond mapping
`httpx.HTTPError` to `GenerationError`. Other library or application errors
propagate unless the caller handles them.

The following example runs offline and prints nothing when its assertions pass.
It demonstrates types only; it makes no network call.

```python
from typevet import GenerationError, SchemaValidationError
from typevet.domain import SchemaError, compile_json_schema

assert issubclass(SchemaValidationError, GenerationError)

err = SchemaValidationError("missing key", payload={"x": 1})
assert err.payload == {"x": 1}

try:
    compile_json_schema({"type": "array"})
except SchemaError:
    pass
else:
    raise AssertionError("expected SchemaError")
```

## Public surface

| Symbol | Import from |
|---|---|
| `GenerationError`, `SchemaValidationError` | `typevet` or `typevet.domain.errors` |
| `SchemaError`, `compile_json_schema`, `Decision` | `typevet.domain` |
| `GenerationPort` | `typevet.ports.generation` |
| `LlamaCppGenerationAdapter`, `FakeGenerationAdapter` | `typevet.adapters.outbound` |

Do not document exception types that are not raised by the current tree. When
[#43](https://github.com/Alberto-Codes/typevet/issues/43) lands, revise this
page to match the accepted hierarchy.
