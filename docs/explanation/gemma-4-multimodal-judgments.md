# Gemma 4 multimodal judgments

Kind: explanation.

This page is for a platform engineer who must decide whether to send images with
a typed question. It explains how typevet carries an image to Gemma 4 on
llama.cpp and on vLLM. It also states what the receipts prove and where the
evidence stops. For steps, follow the linked how-to pages.

## What an image-conditioned typed judgment is

A typed judgment asks named `Noul`, `Choice` or `Score` questions about a
`state` and returns typed answers with probabilities. An image-conditioned
judgment adds one or more images to that call. Every scored field sees the same
images, in the same order.

The image helps when the answer is in the pixels and not in the text. Examples
are a receipt that supports or contradicts an expense claim, or a screenshot
that shows which site is open. When the text alone settles the question, an
image adds tokens and latency and no new evidence.

The answer is still a probability map over fixed labels. The model does not
write free text that you must parse.

## How images enter typevet

The domain type is `ImageInput`. It holds encoded bytes and a mime type. It
accepts `image/png`, `image/jpeg` and `image/webp`. It refuses empty bytes and
every other mime type with `ScoringValidationError`.

The caller passes images through the keyword-only `media` argument of
`JudgmentPort.judge`. `None` or an empty tuple is the text path. The judgment
adapter puts one `MEDIA_MARKER` (`<__media__>`) per image in front of the
rendered state, one marker per line. The marker order is the image order.
Caller text can hold the literal `<__media__>`, for example in a git diff or
a fetched page. The adapter neutralizes that text in the state, the
instructions and the criteria to `<\_\_media\_\_>` before it adds the real
markers. Caller text therefore cannot bind or demand an image (#433).

Two request types enforce a count check. `CandidateScoringRequest` and
`GenerationRequest` each count the markers in the prefix or prompt. When that
count differs from the number of images, the request fails before any network
call. This check stops an image from being sent without a place in the prompt.

## How each backend receives images

The two backends take the same domain request and send it on different wires.

| Topic | llama.cpp | vLLM |
|---|---|---|
| Session factory | `open_gemma_native_vision_judgment` | `open_vllm_judgment` |
| Endpoint | `POST /completion` | `POST /v1/chat/completions` |
| Image payload | Raw base64 in `prompt.multimodal_data`, beside `prompt_string` | One `image_url` block per image, with a `data:` URI |
| Media marker | Replaced with the router marker from `GET /props` | Split out. Each marker becomes one image block |
| Vision check | `/props` must report vision, or the factory raises `ValueError` | No probe. The server must accept images |
| Template | typevet composes the native Gemma 4 turn itself | The server applies its chat template |
| Thinking | Prefix ends with the no-thinking channel prefill | `chat_template_kwargs` sets `enable_thinking` to false |
| Scores | `n_probs` over the full vocabulary, 262144 entries | `logprob_token_ids`, at most 128 candidates |
| Prompt cache | `cache_prompt` is false on every request | No typevet setting |
| Image count limit | None in typevet | Tested server flag `--limit-mm-per-prompt {"image":2}` |
| Model pin | Alias `gemma-4-31b-kv9-q4km-mm`: file `gemma-4-31b-24gib-kv9-decoder.gguf`, ftype `Q2_K - Medium` (not Q4_K_M), 16.0 GB, with its projector ([#233](https://github.com/Alberto-Codes/typevet/issues/233)) | `google/gemma-4-31B-it`, BF16, vLLM 0.30.0 |

The alias name says Q4_K_M, but the file it loads is not Q4_K_M. On
2026-09-29 the router's `/props` reported ftype `Q2_K - Medium` for the
16.0 GB file `gemma-4-31b-24gib-kv9-decoder.gguf`. The router also applied a
`--chat-template-file` override and ran the stock image `server-cuda-b11243`
([#233](https://github.com/Alberto-Codes/typevet/issues/233)).

On llama.cpp, the model must load with its multimodal projector (`--mmproj`).
The router reports `modalities.vision` and a `media_marker` through
`GET /props`. The router randomizes that marker for each server instance.
The adapter caches the marker for each model id and substitutes it for
`MEDIA_MARKER`.

The nested prompt object is not a style choice. A top-level `multimodal_data`
beside a string prompt returns HTTP 200 and drops the image. The
[image-conditioned live smoke](../how-to/run-a-multimodal-live-smoke.md) records
that finding and the wire shape.

On vLLM, the server applies the chat template, so typevet sends plain content
without turn markers. The factory makes no HTTP call before the first `judge`
call. See [Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md) for the
server flags and the settings.

The typed-generation path differs. The vLLM generation adapter sends images as
content blocks. The llama.cpp generation adapters refuse a request with images
before any HTTP call.

## What is specific to Gemma 4

Gemma 4 uses a native turn format: `<|turn>user`, the user text, `<turn|>`,
then `<|turn>model`. On llama.cpp, typevet composes this prefix itself. The
media markers sit inside the user turn, before the state and the field
instructions. After the model header, the prefix adds the no-thinking prefill
`<|channel>thought\n<channel|>`. This matches `/apply-template` output with
thinking turned off.

The llama.cpp factory checks the served template before it builds a port. It
renders one turn through `POST /apply-template` and classifies the result. By
default it requires `native_gemma4_turn`. A ChatML render, a Gemma 3 render or
a mixed render raises `ValueError` before the first `judge` call.

The judgment adapter also fails closed. With a native Gemma 3 or Gemma 4
family, it wraps every prefix in that turn, with or without images. An imaged
prefix and an image-omitted prefix then differ only in the media markers.
Without a native family, a text-only request falls back to a degraded ChatML
prefix. A request with images and no native family raises
`JudgmentValidationError` before any scoring request. typevet never composes a
ChatML prefix around an image.

On vLLM, typevet does not classify the template. The server owns it, and the
tested pin is the official Gemma 4 checkpoint.

## What the receipts prove

Each figure below comes from one recorded run. None is a calibration claim.

**Image-conditioned live smoke, llama.cpp, Gemma 3.** On
`gemma-3-4b-it-q4km-mm`, the smoke asked one colour `Choice` question. The
[smoke how-to](../how-to/run-a-multimodal-live-smoke.md) names that model id. It
answered red, green and blue correctly, and the omitted answer differed from
the imaged answers. Prompt tokens went from 103 without an image to 362 with an
image, a gap of 259 ([#142 acceptance](https://github.com/Alberto-Codes/typevet/issues/142#issuecomment-5843678137)).
This run predates `cache_prompt: false` and native Gemma 3 turns.

**PSAI vision smoke, llama.cpp, Gemma 3.** Five public computer-use screenshots
ran present, omitted and swapped controls. Paired ordering held on 5 of 5
rows. The image added 259 tokens, and no omitted row counted as a hit
([#154 acceptance](https://github.com/Alberto-Codes/typevet/issues/154#issuecomment-5843944050)).
This run also predates `cache_prompt: false` and native Gemma 3 turns.
[Run the PSAI vision smoke](../how-to/run-the-psai-vision-smoke.md) names the
model id, `gemma-3-4b-it-q4km-mm`.

**CORD receipt images, llama.cpp, Gemma 4.** The CORD set holds six public
receipts and 18 synthetic claims. On the alias `gemma-4-31b-kv9-q4km-mm` with
template `native_gemma4_turn` and build `b11223-4da633776`, the run used 43
requests. That alias loads `gemma-4-31b-24gib-kv9-decoder.gguf`, ftype
`Q2_K - Medium`, 16.0 GB, with a `--chat-template-file` override
([#233](https://github.com/Alberto-Codes/typevet/issues/233)). All
five semantic checks passed, and the acceptance command exited 0
([#203 receipt](https://github.com/Alberto-Codes/typevet/issues/203#issuecomment-5882379255)).
See [Run the CORD expense smoke](../how-to/run-the-cord-expense-smoke.md).

| Check | Limit | Measured | n |
|---|---:|---:|---:|
| Answerable accuracy | floor 0.67 | 1 | 12 |
| Contradicted recall | floor 0.5 | 1 | 6 |
| False-clear rate | ceiling 0.25 | 0 | 12 |
| Insufficient abstention rate | floor 0.5 | 1 | 6 |
| Largest label share | ceiling 0.8 | 0.3333 | 18 |

The text-only arm of that run had accuracy 0.667 and contradicted recall 0. The
image drove the combined result. A Gemma 4 image has no fixed prompt cost.
Gemma 4 has a variable image-token budget that depends on the image. Each
receipt added a different number of tokens. In the #203 run, `combined` minus
`text_only` `tokens_evaluated` ranged from 228 to 1,108 tokens per receipt
([#203 per-claim table](https://github.com/Alberto-Codes/typevet/issues/203#issuecomment-5899231642)).
The [CORD how-to](../how-to/run-the-cord-expense-smoke.md) describes the
attachment check.

**vLLM acceptance, Gemma 4.** One run on the vLLM pin passed every
pre-registered gate ([#170 receipt](https://github.com/Alberto-Codes/typevet/issues/170#issuecomment-5884707915)).

| Set | Calls | Result |
|---|---:|---|
| PSAI image present | 4 | 4 of 4 correct. The image added about 265 tokens |
| PSAI image swapped | 4 | 4 of 4 passed the swap gate |
| PSAI image omitted | 4 | Not counted toward any gate. 1 of 4 matched gold by chance (`omitted_credited: 1`) |
| PSAI text only | 4 | 4 of 4 correct |
| CORD | 43 | Accuracy 1.0, contradicted recall 1.0, false-clear 0, abstention 1.0, label share 0.33 |
| CORD with labels in reverse order | 18 | Same five checks pass, with 0 label flips |

The swapped control is the strongest signal here. The text stays the same, and
only the image changes.

The two backends ran different weights: a `Q2_K - Medium` GGUF on llama.cpp and BF16 on vLLM.
The receipts do not compare backends, and a difference is not a backend effect.

## Limits

- **Image count.** typevet sets no cap on images per request. The llama.cpp
  scoring adapter sends every image it gets. The tested vLLM server allows two
  images per prompt. typevet does not check that limit before the request.
- **Image size.** `ImageInput` checks the mime type and non-empty bytes only.
  An 8 MiB payload passes. One live run on the llama.cpp alias answered
  synthetic images from 256 px to 4096 px on a side. Prompt tokens stopped at
  1160 from 2048 px up ([#204](https://github.com/Alberto-Codes/typevet/issues/204)).
  One run is not a pixel limit.
- **Long-running service.** Offline tests run 200 calls on one session and
  restart the router in the middle ([#204](https://github.com/Alberto-Codes/typevet/issues/204)).
  A call while the router is down raises `TransportError`
  ([#298](https://github.com/Alberto-Codes/typevet/issues/298)).
  The llama.cpp marker changes when the router reloads the model. A stale
  marker fails tokenization with HTTP 400. The scoring adapter then reads the
  new marker once and sends the request once more
  ([#322](https://github.com/Alberto-Codes/typevet/issues/322)). A second
  failure raises `BackendHttpError`.
  The [native vision how-to](../how-to/connect-gemma4-native-vision-judgment.md)
  describes session ownership.
- **One model pin per backend.** Evidence covers the alias
  `gemma-4-31b-kv9-q4km-mm` on llama.cpp and `google/gemma-4-31B-it` at one
  revision on vLLM 0.30.0. The alias loads a `Q2_K - Medium` file, not
  Q4_K_M: `gemma-4-31b-24gib-kv9-decoder.gguf`, 16.0 GB, with a
  `--chat-template-file` override
  ([#233](https://github.com/Alberto-Codes/typevet/issues/233)). Other
  models, quantizations and versions are not tested.
- **Default model id.** `TYPEVET_LLAMA__MULTIMODAL_MODEL` defaults to the Gemma 3
  id `gemma-3-4b-it-q4km-mm`
  ([settings](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/inbound/settings.py)). The native
  vision factory requires the Gemma 4 turn by default. Set that variable, or
  pass `model=`, to the Gemma 4 id. Otherwise, a router that serves the Gemma 3
  id makes the factory raise `ValueError` when it opens the session.
- **No general OCR claim.** The receipts cover six CORD receipts and a few
  screenshots. They say nothing about text extraction quality on other
  documents.
- **Small samples.** Each set is one run with n of 18 or fewer per check. Treat
  the numbers as smoke evidence, not as accuracy you can expect in production.

The [release support matrix](../reference/typed-judgment-release-support-matrix.md)
lists each runtime limit with the test that proves it.
