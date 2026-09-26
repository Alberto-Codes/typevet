# Call typevet from Python

Kind: how-to.

Use typevet as a library: pick an outbound adapter, pass a JSON Schema object,
and get a validated mapping back. The shape matches how sister libraries such
as judgevet inject a port and keep domain types at the package root.

## Prerequisites

From the typevet checkout:

```bash
cd /path/to/typevet
uv sync
```

## Public surface

Import **domain types and errors** from the top-level package:

```python
from typevet import GenerationRequest, GenerationResult, GenerationPort
from typevet import GenerationError, SchemaValidationError
```

Import **adapters** from the outbound package (not from deep module paths in
application code):

```python
from typevet.adapters.outbound import FakeGenerationAdapter, LlamaCppGenerationAdapter
```

Optional **inbound helpers**:

```python
from typevet.adapters.inbound import generate, run_sync
```

`generate` builds a `GenerationRequest` for sync ports. `run_sync` runs async
port coroutines from scripts (for example `run_sync(port.generate(request))`)
without adding `*_sync` methods on adapters.

For live llama.cpp setup and Gemma 4 model ids, see
[Run Gemma 4 on llama.cpp](run-gemma4-llamacpp.md).

## Use-library shape (judgevet-style)

1. Construct a port that implements `GenerationPort` (fake or llama.cpp).
2. Call `port.generate(request)` or `generate(port, prompt=..., schema=..., model=...)`.
3. Read `result.value` — a mapping that already passed JSON Schema validation.

Anything that accepts `GenerationPort` stays testable: swap the adapter at the
edge without changing call sites.

## Runnable example — offline fake

`FakeGenerationAdapter` validates its return value against the schema. Use it
in unit tests, contract tests, and local scripts when you do not need a model.

```python
from typevet import GenerationRequest
from typevet.adapters.outbound import FakeGenerationAdapter

schema = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}

port = FakeGenerationAdapter(value={"answer": 42})
result = port.generate(
    GenerationRequest(
        prompt="What is the meaning of life?",
        schema=schema,
        model="fake",
    )
)
print(result.value)  # {"answer": 42}
```

Run it:

```bash
uv run python -c "
from typevet import GenerationRequest
from typevet.adapters.outbound import FakeGenerationAdapter

schema = {
    'type': 'object',
    'properties': {'answer': {'type': 'integer'}},
    'required': ['answer'],
    'additionalProperties': False,
}
port = FakeGenerationAdapter(value={'answer': 42})
result = port.generate(
    GenerationRequest(prompt='n?', schema=schema, model='fake')
)
assert result.value == {'answer': 42}
print('ok', result.value)
"
```

## Runnable example — llama.cpp adapter

`LlamaCppGenerationAdapter` POSTs to a local OpenAI-compat router with
`response_format` / `json_schema`, then validates the parsed object.

You need the router running (see the Gemma how-to). Default base URL is
`http://127.0.0.1:8090`.

```python
from typevet import GenerationRequest
from typevet.adapters.outbound import LlamaCppGenerationAdapter

schema = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}

with LlamaCppGenerationAdapter() as port:
    result = port.generate(
        GenerationRequest(
            prompt="Return whether 2+2 equals 4.",
            schema=schema,
            model="gemma-4-31b-24gib-kv11-decoder",
        )
    )
print(result.value)
```

Use a context manager (or call `close()`) when the adapter creates its own
HTTP client.

## `generate()` vs `port.generate()`

| Approach | Import | When to use |
|---|---|---|
| `port.generate(request)` | `GenerationRequest` + adapter | Default. Matches `GenerationPort` and judgevet port injection. |
| `generate(port, prompt=..., schema=..., model=...)` | `typevet.adapters.inbound.generate` | Same behaviour; avoids constructing `GenerationRequest` at the call site. |

Both paths end in `port.generate`. Pick one style per module and stay consistent.

## Async ports from a script

Async adapters implement `AsyncGenerationPort`. In library code, `await
port.generate(request)`. In a one-off script or CLI entry point, wrap the
coroutine with `run_sync`:

```python
from typevet import GenerationRequest
from typevet.adapters.inbound import run_sync
from typevet.adapters.outbound import AsyncFakeGenerationAdapter

schema = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
port = AsyncFakeGenerationAdapter(value={"ok": True})
result = run_sync(
    port.generate(GenerationRequest(prompt="Say ok.", schema=schema, model="fake"))
)
print(result.value)
```

Do not call `run_sync` from code that already runs inside an event loop; use
`await` there instead.

Deep imports such as `typevet.adapters.outbound.fake` or
`typevet.domain.models` are for typevet’s own tests and docs snippets. Prefer
the exports above in downstream libraries.

## Which fake to use

| Type | Import | Validates schema? | Use when |
|---|---|---|---|
| `FakeGenerationAdapter` | `typevet.adapters.outbound` | Yes | Contract tests, offline demos, asserting validation failures. |
| `StaticGenerationFake` | `typevet.testing` | No | Inbound unit tests that only need a fixed port double. |
| `ScriptedScoringFake` | `typevet.testing` | No | Offline typed judgment; ships in the wheel (see first typed judgment tutorial). |

`FakeGenerationAdapter` also accepts `responder=` (callable from request to
mapping) or `fail=` (raise a configured exception) for richer test scenarios.

## Errors

- `SchemaValidationError` — output JSON did not match the requested schema
  (fail-fast after parse).
- `GenerationError` — transport, HTTP, or non-JSON content from the backend.

Catch these at the boundary; keep domain logic free of HTTP details.
