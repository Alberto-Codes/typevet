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
colour. It writes a receipt to `scratchpad/multimodal/live_receipt.json`.

## Read the result

| Result | Cause |
|---|---|
| Skip | The router is down, or the model id is not in the catalog |
| Fail on `text-only input modalities` | The router serves that id without a projector |
| Pass | Each image moved the decision to its own colour |

A skip is not evidence. A text-only router **fails** the test, because upload
alone proves nothing about image conditioning.

## Call the library

Pass images with the keyword-only `media` argument. The judgment adapter puts
one `MEDIA_MARKER` in every field prefix for each image.

```python
from typevet.adapters.outbound import LlamaCppCandidateScoringAdapter
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain import Choice, ImageInput

image = ImageInput(data=png_bytes, mime_type="image/png")
question = {"fill": Choice(criteria={"red": "Red", "blue": "Blue"})}
with LlamaCppCandidateScoringAdapter(n_vocab=262144) as scoring:
    port = ScoringJudgmentAdapter(scoring, tokenize_content=tokenize)
    response = port.judge("Look at the image.", question, model_id, media=(image,))
```

`ImageInput` accepts `image/png`, `image/jpeg` and `image/webp`. It rejects
empty bytes and every other mime type. A request must hold one `MEDIA_MARKER`
per image, or the domain raises `ScoringValidationError`.

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
  "n_probs": 262144
}
```

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
One Gemma 3 image costs 256 prompt tokens, so an attached image raises the
count by about 259 over the text baseline. A dropped image raises it by about
30, the cost of the marker as plain text. The live test asserts that gap.

The marker rotates when the router reloads the model. A stale marker returns
HTTP 400 `Failed to tokenize prompt`, so build a new adapter per session
rather than holding one across a reload.
