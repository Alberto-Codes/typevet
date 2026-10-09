# Connect Gemma 4 native vision judgment

Kind: how-to.

Use one public runtime factory to compose `LlamaCppCandidateScoringAdapter` and
`ScoringJudgmentAdapter` for Gemma 4 native-turn vision. The factory lives in
`typevet.runtime` and does not import `typevet_evals`.

## Prerequisites

- llama.cpp router with a Gemma 4 multimodal model id and `--mmproj` loaded.
- Native template render from `POST /apply-template` (no ChatML markers).
- `TYPEVET_LLAMA__*` variables set for your router (see
  [Configuration](../reference/configuration.md)).

## Open a session

```python
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.runtime import open_gemma_native_vision_judgment

settings = load_llama_settings()
with open_gemma_native_vision_judgment(settings=settings) as session:
    response = session.port.judge(state, questions, session.model)
```

Pass ``session.model`` to ``judge``. The factory pins that id on
``session.port``; any other model argument raises ``JudgmentValidationError``
before tokenization or scoring.

Defaults:

- Model id: `settings.multimodal_model` (`TYPEVET_LLAMA__MULTIMODAL_MODEL`).
- Timeout: `settings.timeout` (`TYPEVET_LLAMA__TIMEOUT`, default 300s).
- Template: must classify as `NATIVE_GEMMA4_TURN` when `require_gemma4=True`.

Unsupported routers raise `ValueError` before the first `judge` call when:

- `/props` reports text-only modalities, or
- `/apply-template` is not a supported native Gemma turn family.

Pass `require_vision=False` to accept a text-only model.
The session then has `capability.vision` set to `False`.
Its port raises `ScoringUnsupportedCapabilityError` for an image judgment before any request.
`open_judgment` passes `require_vision=False`.

## Probe without holding a port

```python
from typevet.runtime import probe_gemma_native_vision_support

meta = probe_gemma_native_vision_support(settings=load_llama_settings())
print(meta["served"], meta["vision"])
```

## Lifecycle notes

- The context manager owns the scoring adapter lifecycle (`close()` on exit).
- Pass an existing `httpx.Client` only in tests via `http_client=`.
- Consumer live matrices may pass ``tokenize_content`` and
  ``scoring_port_wrapper`` hooks for dispatch ledgers without duplicating probe
  wiring.
- For long-running services, prefer one session per request or explicit client
  ownership documented in your composition root.

See also [Run the image-conditioned live smoke](run-a-multimodal-live-smoke.md)
for lower-level router checks.
