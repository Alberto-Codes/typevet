# Run Gemma 4 on local llama.cpp for typevet

Kind: how-to.

typevet needs **stock upstream** llama.cpp and **public Gemma 4 text GGUF** weights.
There are **no typevet forks or patches** of Gemma or llama.cpp for grammar-JSON.
The adapter sends standard OpenAI `/v1/chat/completions` with nested
`response_format.json_schema` only.

## Stock path (recommended)

1. Build or install **unmodified** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
   `llama-server` (binary or official container).

2. Use a **minimum server build** where nested `json_schema` is enforced (not
   ignored). Research [#99](https://github.com/Alberto-Codes/typevet/issues/99)
   treats **2025 H2+ releases** and builds from roughly **b4739+ / b4820+**
   as the floor (fixes for nested schema landed around
   [llama.cpp #11847](https://github.com/ggml-org/llama.cpp/issues/11847) /
   [#11988](https://github.com/ggml-org/llama.cpp/issues/11988)). Pin a exact
   tag here after your live smoke passes on your hardware.

3. Download a **Gemma 4 text** GGUF from a public catalog (for example Hugging
   Face community quant releases). Weights must include chat-template metadata,
   or pass `--chat-template` / `--chat-template-file` explicitly.

4. Start the server with **Jinja chat templates enabled** (`--jinja`; default
   on in current upstream trees). Example single-model serve:

   ```bash
   llama-server -m /path/to/gemma-4-text.gguf --jinja --host 127.0.0.1 --port 8090
   ```

   Multi-model routers are fine if the Gemma 4 id appears in `/v1/models`.

5. Confirm the model id:

   ```bash
   curl -s http://127.0.0.1:8090/v1/models | jq '.data[].id'
   ```

6. Set that id for live tests and scripts (any alias your server exposes):

   ```bash
   export TYPEVET_LLAMA__DEFAULT_MODEL='<your-gemma-4-model-id>'
   ```

Wire shape must stay **nested** OpenAI form (typevet already does this):

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "typevet_result",
      "schema": { }
    }
  }
}
```

Do **not** probe with the flat `{"type":"json_schema","schema":{…}}` shorthand
from some README examples; on several builds it returns 200 but **does not**
enforce grammar.

### Enforcement sanity (optional)

Same prompt once with `response_format` and once without. Constrained output
should differ. Matching unconstrained prose usually means a too-old server or
wrong wire shape, not a typevet bug.

### Number bounds and `multipleOf`

On the pinned build `b11243-fc07d781e`, the grammar does not enforce number
`minimum`, `maximum` or `multipleOf`
([#129 receipt](https://github.com/Alberto-Codes/typevet/issues/129#issuecomment-5892208050)).
The server can return a number outside the bounds, or no content for
`multipleOf`. typevet validates each reply with `jsonschema` on the client, so
the adapter raises `SchemaValidationError` for these values. Before the
request, the adapter also checks the schema against the JSON Schema meta-schema
and raises `ValueError` for a malformed schema. No request is sent. These
results apply to this build only.

### Thinking setting

The generation adapters send `"chat_template_kwargs": {"enable_thinking": false}`
with each request, as the vLLM adapters do. The value is a JSON boolean. On
build `b11243-fc07d781e`, the string `"false"` gets HTTP 400
([local probe](https://github.com/Alberto-Codes/typevet/issues/226#issuecomment-5897063440)).
That probe used one prompt on one Q2_K model file. There, the setting removed
the default thinking turn from the template. The probe does not prove that the
hidden completion tokens in the #129 receipt came from thinking.

## Call typevet

```bash
cd /path/to/typevet
uv sync
TYPEVET_LLAMA__DEFAULT_MODEL='<your-gemma-4-model-id>' \
  TYPEVET_LLAMA__TIMEOUT=600 \
  uv run pytest -m live -q
```

If `TYPEVET_LLAMA__DEFAULT_MODEL` is unset, live tests **skip**. If the router
is down or the id is missing from `/v1/models`, they **skip** as well.

Legacy names `TYPEVET_GEMMA_MODEL` and `TYPEVET_LLAMA_URL` still work. See
[Configuration](../reference/configuration.md).

### Pre-sampling candidate scoring (opt-in)

Enum / judgment **logprob** scoring uses stock llama.cpp ``POST /completion`` (not
chat completions): ``prompt`` is the rendered answer prefix, ``n_predict=0``,
``n_probs=262144`` (Gemma 4 vocab), ``post_sampling_probs=false``. Logprobs come
from ``completion_probabilities[0].top_logprobs`` (``id`` + ``logprob``, pre-sampling).

```bash
TYPEVET_LLAMA__DEFAULT_MODEL='<your-gemma-4-model-id>' \
  TYPEVET_LLAMA__TIMEOUT=600 \
  uv run pytest tests/live/test_llama_cpp_scoring_live.py -m live -q
```

To condition that scoring on an image, see
[Run the image-conditioned live smoke](run-a-multimodal-live-smoke.md). That
page holds the nested ``prompt`` shape and the marker the router randomizes.

Library entry: ``LlamaCppCandidateScoringAdapter`` from
``typevet.adapters.outbound``. For enum decisions, inject that adapter as
``scoring_port`` on ``typevet.runtime.categorical.decide_categorical`` (caller owns client lifetime).
Use ``inject_prefix=True`` with a template-derived ``prefix=`` when you already
applied ``/apply-template``; otherwise pass ``context=`` and let the library
compose the degraded ChatML scoring prefix.

```bash
TYPEVET_LLAMA__DEFAULT_MODEL='<your-gemma-4-model-id>' \
  TYPEVET_LLAMA__TIMEOUT=600 \
  uv run pytest tests/live/test_decide_categorical_live.py -m live -q
```

### Judgment template pin (opt-in)

Gemma enum / judgment scoring gates on the **rendered** chat prompt from
llama.cpp ``POST /apply-template`` (same Jinja path as completions). typevet
classifies that string into:

- **native Gemma4 turn** — ``<|turn>`` / ``<turn|>`` markers without ChatML
- **degraded ChatML** — ``<|im_start|>`` family without turn markers
- **unsupported** — mixed families, or neither marker set

Unsupported templates fail before scoring; native and degraded paths use
different answer-prefix anchors and stop markers
(``typevet.adapters.outbound.gemma.served_template``).

Live receipt (router up, model catalog reachable; pins
``gemma-4-31b-24gib-kv11-decoder`` on ``/apply-template`` as degraded ChatML):

```bash
TYPEVET_LLAMA__DEFAULT_MODEL='<your-gemma-4-model-id>' \
  uv run pytest tests/live/test_gemma_template_pin.py -m live -q
```

### Loader eval slice (opt-in)

After the schema smoke passes, run a tiny BoolQ or Banking77 slice through the
same adapter. The runner reports attempted, schema-valid, and gold-match counts
(structure + label agreement only — not ECE). See
[Live eval runner](../reference/eval-live-runner.md).

```bash
TYPEVET_LLAMA__DEFAULT_MODEL='<your-gemma-4-model-id>' \
  TYPEVET_LLAMA__TIMEOUT=600 \
  uv run python -m typevet_evals.cli.eval_runner --dataset boolq --limit 2
```


Or from Python:

```python
from typevet.adapters.outbound import LlamaCppGenerationAdapter
from typevet.domain.models import GenerationRequest

schema = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
model_id = "<your-gemma-4-model-id>"
with LlamaCppGenerationAdapter() as port:
    result = port.generate(
        GenerationRequest(
            prompt="Return whether 2+2 equals 4.",
            schema=schema,
            model=model_id,
        )
    )
print(result.value)
```

## Optional alternate: a router preset

You may use an operator-managed llama.cpp server instead of hand-running
`llama-server`. Such a router serves GGUF files from `<path>` on
`127.0.0.1:8090` with `--jinja` and operator presets. Refer to the
documentation of that server for install and tuning.

The preset id `gemma-4-31b-24gib-kv11-decoder` is a **VRAM-fit quant alias**
(same HTTP contract as any other Gemma 4 GGUF on the router). It is **not**
required for typevet and is not a grammar patch.

```bash
systemctl --user status <router-service>
TYPEVET_LLAMA__DEFAULT_MODEL=gemma-4-31b-24gib-kv11-decoder \
  uv run pytest -m live -q
```

First load of a large GGUF can take minutes; raise `TYPEVET_LLAMA__TIMEOUT` when
needed.

## Notes

- Keep schemas within llama.cpp grammar support (object root, properties,
  `enum`, integer bounds, `additionalProperties: false` match the live test).
- GPU images must match your hardware; CPU `:server` images may ignore `-ngl`.
- Preset and image choices live in operator repos, not in typevet.
