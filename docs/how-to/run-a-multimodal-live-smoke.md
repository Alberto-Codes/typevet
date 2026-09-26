# Run the image-conditioned live smoke

Kind: how-to.

This page runs the opt-in live check for image-conditioned candidate scoring.
The check proves the image changes the scored distribution. It does not measure
model quality.

## Before you start

1. Serve a llama.cpp model with an `--mmproj` projector. The router must list
   `image` under `architecture.input_modalities` for that model id.
2. Confirm the id and the modalities:

   ```bash
   curl -s http://127.0.0.1:8090/v1/models \
     | jq '.data[] | {id, mod: .architecture.input_modalities}'
   ```

3. Confirm the served template renders the native Gemma 3 turn. The output
   must start with `<start_of_turn>user` and hold no `<|im_start|>`:

   ```bash
   curl -s http://127.0.0.1:8090/apply-template \
     -d '{"model": "gemma-3-4b-it-q4km-mm", "add_generation_prompt": true,
          "messages": [{"role": "user", "content": "hello"}]}' \
     | jq -r .prompt
   ```

## Run the smoke

```bash
TYPEVET_LLAMA__MULTIMODAL_MODEL=gemma-3-4b-it-q4km-mm \
  TYPEVET_LLAMA__TIMEOUT=900 \
  uv run pytest tests/live/test_llama_cpp_multimodal_live.py -m live -q
```

`TYPEVET_LLAMA__MULTIMODAL_MODEL` defaults to `gemma-3-4b-it-q4km-mm`. See
[Configuration](../reference/configuration.md).

The test builds three synthetic single-colour PNG images. It asks one `Choice`
question about the fill colour four times: once with no image, then once per
colour. It classifies the served template once and passes it to every
request. It writes a receipt to `scratchpad/multimodal/live_receipt.json`.

## Read the result

| Result | Cause |
|---|---|
| Skip | The router is down, or the model id is not in the catalog |
| Fail on `text-only input modalities` | The router serves that id without a projector |
| `JudgmentValidationError` on `served template` | `/apply-template` renders a family other than native Gemma 3 |
| Pass | Each image moved the decision to its own colour |

A skip is not evidence. A text-only router **fails** the test, because upload
alone proves nothing about image conditioning.

## Call the library

A media call needs three things from the router before the first score:

1. The served template family. Render one turn through `POST /apply-template`
   and classify it with `classify_served_template`.
2. A tokenizer for the control strings, through `POST /tokenize`.
3. The media capability, which `LlamaCppCandidateScoringAdapter` probes from
   `GET /props` on the first image request.

Pass the family to `ScoringJudgmentAdapter` as `served_template`. Media
scoring needs a native turn family: `ServedTemplateClass.NATIVE_GEMMA3_TURN`
or `ServedTemplateClass.NATIVE_GEMMA4_TURN`. When you omit `served_template`,
or pass ChatML or an unsupported family, `judge` raises
`JudgmentValidationError` before any scoring request. The recipe below guards
on Gemma 3, because the smoke runs a Gemma 3 model.

```python
import httpx

from typevet.adapters.outbound import LlamaCppCandidateScoringAdapter
from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    classify_served_template,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain import Choice, ImageInput, JudgmentResponse

QUESTIONS = {
    "fill": Choice(
        criteria={"red": "The image is red.", "blue": "The image is blue."},
        instructions="What single colour fills the attached image?",
    )
}


def judge_image(client: httpx.Client, model: str, png_bytes: bytes) -> JudgmentResponse:
    rendered = (
        client.post(
            "/apply-template",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "add_generation_prompt": True,
            },
        )
        .raise_for_status()
        .json()["prompt"]
    )
    served = classify_served_template(rendered)
    if served is not ServedTemplateClass.NATIVE_GEMMA3_TURN:
        msg = f"{model} serves {served.value}; media needs native_gemma3_turn"
        raise RuntimeError(msg)

    def tokenize(text: str) -> tuple[int, ...]:
        body = client.post(
            "/tokenize",
            json={"model": model, "content": text, "add_special": False},
        )
        return tuple(body.raise_for_status().json()["tokens"])

    image = ImageInput(data=png_bytes, mime_type="image/png")
    with LlamaCppCandidateScoringAdapter(
        str(client.base_url), client=client
    ) as scoring:
        port = ScoringJudgmentAdapter(
            scoring,
            tokenize_content=tokenize,
            served_template=served,
        )
        return port.judge(
            "Look at the attached image.", QUESTIONS, model, media=(image,)
        )
```

Call it with one client for the router session:

```python
with httpx.Client(base_url="http://127.0.0.1:8090", timeout=900.0) as client:
    response = judge_image(client, "gemma-3-4b-it-q4km-mm", png_bytes)
print(response.choices["fill"].choice, response.usage.input_tokens)
```

`tests/contract/test_multimodal_howto_recipe.py` runs the `judge_image` block
on this page against an offline router. It fails when the block stops passing
`served_template`:

```bash
uv run pytest tests/contract/test_multimodal_howto_recipe.py -q
```

`ImageInput` accepts `image/png`, `image/jpeg` and `image/webp`. It rejects
empty bytes and every other mime type. A request must hold one `MEDIA_MARKER`
per image, or the domain raises `ScoringValidationError`.

With a native family, the adapter wraps every field prefix in that family's
turn, with or without images. An image-omitted request and an imaged request
then differ only in the media markers. Without a family, a text-only request
falls back to a ChatML prefix.

## Backend wire shape

Measured on router build `b11176-f805c57a2` with `gemma-3-4b-it-q4km-mm`.
`POST /completion` attaches an image only through the nested object prompt:

```json
{
  "prompt": {
    "prompt_string": "<__media_...__> Answer:",
    "multimodal_data": ["<raw base64>"]
  },
  "n_predict": 0,
  "n_probs": 262144,
  "cache_prompt": false
}
```

The adapter sends `"cache_prompt": false` on every `/completion` request, with
or without images. A cached KV prefix from an earlier request can otherwise
shift the scores.

Three findings govern the adapter:

- A top-level `multimodal_data` beside a string `prompt` returns HTTP 200 and
  **drops the image**. `tokens_evaluated` stays at the text count, and the
  model answers from its prior.
- `multimodal_data` entries are raw base64. A `data:` URI returns HTTP 400
  with `Failed to load image or audio file`.
- The server **randomizes** the media marker per instance. Read it from
  `GET /props?model=<id>` as `media_marker`. A hardcoded `<__media__>` returns
  HTTP 400 with `Failed to tokenize prompt`.

`LlamaCppCandidateScoringAdapter` handles all three. It probes `/props` once
per model id, refuses a text-only model with
`ScoringUnsupportedCapabilityError`, and substitutes the router marker for the
documented `MEDIA_MARKER`. A text-only request sends a string `prompt` and
never probes `/props`.

## Check the attachment yourself

A valid distribution is not proof of attachment. Read
`CandidateScoringResult.usage.input_tokens`, which carries `tokens_evaluated`.
One Gemma 3 image costs 256 prompt tokens. A dropped image raises the count by
about 30, the cost of the marker as plain text. The live test asserts a gap of
at least 200 tokens over the image-omitted baseline.

Historical run, recorded at revision `3ecea25` with `gemma-3-4b-it-q4km-mm`.
That revision predates `cache_prompt: false` (`5c5df49`) and native Gemma 3
turns (`9d8d818`, `aa1ad37`). The omitted request evaluated 103 prompt tokens
and each imaged request evaluated 362, a gap of 259. Run the smoke again before
you quote a count for the current revision.

The marker rotates when the router reloads the model. A stale marker returns
HTTP 400 `Failed to tokenize prompt`, so build a new adapter per session
rather than holding one across a reload.
