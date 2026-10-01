---
status: draft
---

# Limits and known gaps

Kind: explanation.

Status: **draft**.

This page collects the known limits of typevet 0.2.0 in one place.
Each section states one limit and links to the page or issue that is its source.
The source page holds the full detail. When this page and a source differ, the source wins.

## Page status

Some typevet pages carry a `status:` value in the front matter and a `Status:` line under the title.
The value tells you how far the page is backed by evidence.

| Status | Meaning |
|---|---|
| `sketch` | The page comes from first principles. No code or measurement verifies it yet. |
| `draft` | Working code or real measurements back part of the page. Details can change. |
| `stable` | The page was verified against the current state of `main`. |

A page moves to a higher status in the commit that lands the proof.
A page without a status line makes no status claim.

## Tested pins per backend

Each backend has a small number of tested pins, and each pin has one receipt.
A pin is not a minimum version.
It says nothing about other versions, quantizations or hardware.

- **vLLM.** Stock `vllm/vllm-openai:v0.30.0` with BF16 `google/gemma-4-31B-it` at one revision on one H100 80 GB.
  Other vLLM versions, models, precisions and GPUs are not tested.
- **llama.cpp.** One `llama-server` build for image input and one build for generation.
  The image-input row does not record its hardware.
- **No backend comparison.** The vLLM and llama.cpp rows use different weights.
  Do not attribute a result difference to the backend.
- **Untested on vLLM.** Quantized weights and mixed scoring batches are not tested.
  One live run on the tested pin read the `vllm:kv_cache_usage_perc` gauge ([#231](https://github.com/Alberto-Codes/typevet/issues/231)).

Sources: [tested serving pins](../reference/typed-judgment-release-support-matrix.md#tested-serving-pins),
[Serve typevet on vLLM, limits](../how-to/serve-typevet-on-vllm.md#limits).

## Model scope

The tested pins cover Gemma 4 31B only.
Some older smoke rows use a Gemma 3 model, but they are not release pins.
Other models are not tested.
The multimodal evidence covers one model pin per backend.

- Native `Choice` and `Score` support 24 options, the execute limit `MAX_ENUM_CHOICES`.
  The controls are `"0"` to `"9"`, then `"A"` to `"Z"`, so control binding can label 36 options.
  The binding limit depends on the tokenizer.
  The #286 check covered the local GGUF tokenizer and the cached Hugging Face tokenizer.
  25 to 36 options raise `DecisionExecutionError` before that question's scoring call.
  More than 36 options raise `JudgmentValidationError` before that question's scoring call.
  No calibration receipt exists for more than 10 options.
  One live 24-option run on the local Q2_K Gemma 4 pin gave a valid distribution (#288). One run is not calibration.
  The eval code can compute top-label ECE and class-wise ECE for any label set. The public workloads accept 24 options (#296).
  A class with fewer than 30 gold instances gets the status "insufficient N" and is not in the class-wise mean.
  No live run has used these metrics yet.
- `TYPEVET_LLAMA__MULTIMODAL_MODEL` defaults to a Gemma 3 id.
  If the router serves the Gemma 3 id, the native vision factory raises `ValueError`.
  Set the variable, or pass `model=`, to the Gemma 4 id.
- typevet does not promise byte parity with TypeLLM or SGLang.

Sources: [behaviour at HEAD](../reference/typed-judgment-release-support-matrix.md#behaviour-at-head-on-both-backends),
[Gemma 4 multimodal judgments, limits](gemma-4-multimodal-judgments.md#limits),
[Native typed judgments, limitations](native-typed-judgments.md#limitations).

## The local alias does not name its quantization

Several receipts name the local llama.cpp alias `gemma-4-31b-kv9-q4km-mm`, and the name says Q4_K_M.
On the operator's router, on 2026-09-29, that alias loaded `gemma-4-31b-24gib-kv9-decoder.gguf` (16.0 GB).
`/props` reported ftype `Q2_K - Medium`, not Q4_K_M.
The router also used a `--chat-template-file` override on the stock image `server-cuda-b11243`.
The docs now record this identity next to each mention of the alias.
The older receipts (#203 and others) did not record the file.
The file behind each of those runs is not proven.

Source: [#233](https://github.com/Alberto-Codes/typevet/issues/233).
The pin itself is in the [tested serving pins](../reference/typed-judgment-release-support-matrix.md#tested-serving-pins).

## What live receipts prove and do not prove

A live pass shows that one exercised call completed for the prompt, schema and model that the test used.
It does not show calibration, task accuracy or identical future answers.
A valid structure is not a correct answer.

- Smoke runs prove typed wiring, image attachment and controls.
  They do not prove model quality or calibration.
- Run-to-run spread is measured on one slice only: the seed-0 synthetic checks on llama.cpp.
  Five repeats gave identical answers, so those one-run numbers are stable to the reported precision.
  Other slices and vLLM have no repeats. See [#357](https://github.com/Alberto-Codes/typevet/issues/357).
- The samples are small. Each multimodal set has 18 or fewer items per check.
  Treat them as smoke evidence, not as accuracy you can expect in production.
- A saved receipt can pass pytest and still fail the semantic acceptance floors.
- The receipts cover a few CORD receipts and screenshots.
  They make no general OCR claim.

Sources: [Verified evidence and inferred claims](verification.md),
[Judgment live receipts](../reference/judgment-live-receipts.md),
[deliberate exclusions](../reference/typed-judgment-release-support-matrix.md#deliberate-exclusions).

## Performance

typevet has two throughput measurements on vLLM.
Each is one run on one H100 pod with one model pin.

- **Text judgments (#236).** At concurrency level 64, Banking77-480 ran at 39.63 records/s with 0 errors.
- **Image judgments (#336).** At 16 in flight, the full face, check and signature sets ran at 4.4 to 5.3 judgments/s.

Limits of the text measurement:

- The texts are short public texts of 89 to 254 mean prompt tokens per record.
  The throughput does not transfer to longer prompts.
- Client latency includes the RunPod proxy.
- DIFrauD SMS-500 failed calibration parity: ECE 0.1578 against a threshold of 0.10.
- The full Banking77 test split of 3,080 records was not measured.
  GPU memory was not measured.
- Cold start was 6 min 46 s on that pod.

Limits of both measurements:

- The receipts of both runs do not record the vLLM server flags.
  The flags come from the issue comments of each run.
  Receipts written after #341 hold a `server_args` block with the observed cache config.
- The pages make no claim about other GPUs, models, precisions or vLLM versions.

Sources: [Performance on one H100](../reference/performance.md#limits),
[Image judgment throughput on one H100](image-throughput-h100.md).

## Pre-1.0 API

typevet is pre-1.0, at version `0.2.0`.
The public surface is the union of the package `__all__` lists.
No page promises a stable API before 1.0.
The supported imports page requires a compatibility note for a breaking rename or a removed export.

- No async judgment API ships. Async generation adapters exist for llama.cpp and vLLM.
- No CLI and no MCP server ship in the wheel.
- Evaluation code is not in the wheel.

Sources: [0.1.0 compatibility assessment](../reference/supported-imports.md#010-compatibility-assessment),
[deliberate exclusions](../reference/typed-judgment-release-support-matrix.md#deliberate-exclusions).

## Runtime limits without a cap

typevet sets no cap on image bytes, image pixels or images per request.
`ImageInput` checks the mime type and non-empty bytes only, and an 8 MiB payload passes.
The tested vLLM server allows two images per prompt, and typevet does not check that before the request.
One live run on the llama.cpp vision alias answered synthetic images up to 4096 px on a side.
That run is one receipt, not a pixel limit.
Offline tests run 200 calls on one session and restart the router in the middle.
A call while the router is down raises `TransportError` ([#298](https://github.com/Alberto-Codes/typevet/issues/298)).
A router that closes a connection (new or reused) before a response head gets one retry on a new connection. A second close raises `TransportError` ([#305](https://github.com/Alberto-Codes/typevet/issues/305)).

Sources: [runtime limits and ownership](../reference/typed-judgment-release-support-matrix.md#runtime-limits-and-ownership),
[Gemma 4 multimodal judgments, limits](gemma-4-multimodal-judgments.md#limits).

## Open issues that affect hosting

Each item is an open gap at the time of writing.
None of them has a promised fix date.

- [#188](https://github.com/Alberto-Codes/typevet/issues/188) and [#193](https://github.com/Alberto-Codes/typevet/issues/193): the off-menu mass guard is opt-in and flags only.
  The scoring result reports this mass as `off_option_mass` ([#297](https://github.com/Alberto-Codes/typevet/issues/297)).
  [#207](https://github.com/Alberto-Codes/typevet/issues/207) calls the same quantity off-menu mass.
  On llama.cpp the value is 1 minus the sum of the raw candidate probabilities.
  The adapter reports it only when the response holds exactly `n_vocab` entries and their total is within 0.01 of 1; otherwise the value is `None`.
  `n_vocab` is the model vocabulary size.
  The adapter reads it from `meta.n_vocab` in `/v1/models` and keeps it once known.
  It makes at most 3 reads per model ([#321](https://github.com/Alberto-Codes/typevet/issues/321)).
  If the caller sets `n_vocab`, the adapter does not read it.
  When the server does not report the size, the value is `None`.
  A caller-set `n_vocab` that is smaller than the model vocabulary still passes the count.
  The Gemma native vision factory sets 262144 by default.
  On vLLM the value is always `None` (unavailable), because the response holds at most 128 token ids.
  By default typevet reports the value and does not act on it.
  A caller can pass `off_option_threshold` to `judge` ([#353](https://github.com/Alberto-Codes/typevet/issues/353)).
  An answer whose mass is above the threshold has `off_option_flag` set to `True`.
  The guard flags the answer and does not raise.
  `JudgmentResponse.off_option` keeps one receipt per answer with the mass, the threshold and the flag.
  A `None` mass never sets the flag, and the receipt records `off_option_mass: null`.
  So on vLLM the guard never flags.
  The `ModelFramingPort` docstring states the Gemma 4 no-thinking prefill requirement ([#235](https://github.com/Alberto-Codes/typevet/issues/235)).
  typevet checks the rendered framing prefix before the scoring call ([#354](https://github.com/Alberto-Codes/typevet/issues/354)).
  The check reads only the text after the field instructions, so state text cannot trigger it.
  When that text ends in a Gemma 4 model turn, the prefix must end with the prefill.
  Otherwise `GemmaTemplateError` names the framing class and holds no prompt text.
  A framing whose last turn is not a Gemma 4 model turn needs no prefill, so the check allows it.
  The check runs on every backend, vLLM included.
  On vLLM the scoring adapter sends `enable_thinking: false`.
  Thus a vLLM framing that sends plain chat content needs no prefill.
  A vLLM framing that renders its own Gemma 4 model turn still needs the prefill.

## Related pages

- [Typed-judgment release support matrix](../reference/typed-judgment-release-support-matrix.md)
- [Performance on one H100](../reference/performance.md)
- [Verified evidence and inferred claims](verification.md)
- [Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md)
