---
status: draft
---

# Native typed judgments in typevet

Kind: explanation.

Status: **draft**.

typevet ships **native** System One–shaped judgment: `Noul`, `Choice`, and
`Score` questions, normalized answers, and probability maps from candidate
scoring. The product runs on **local llama.cpp** with **Gemma 4** weights.
typevet owns this implementation. [TypeLLM](https://github.com/TypeLLM/TypeLLM)
and [judgevet](https://github.com/Alberto-Codes/judgevet) are **research and
protocol references**, not dependencies you must install to use typevet.

Grammar-JSON generation remains the transport floor for some paths. Typed
judgment is the spine judgevet-shaped callers need later. See
[TypeLLM, Jev and judgevet](typellm-and-judgevet.md) for the family map.

## What you get today

| Capability | Status |
|---|---|
| `JudgmentPort` + offline fakes | Yes — tutorial and contract fixtures |
| `ScoringJudgmentAdapter` over `CandidateScoringPort` | Yes — offline and llama.cpp scoring |
| TPJEP v0 eight-task smoke | Yes — offline runner + opt-in live pytest |
| Partner-derived six-message exploratory receipt | Documented on [#133](https://github.com/Alberto-Codes/typevet/issues/133); not a shipped CLI |
| Public calibration or ECE headline | No — see limitations |

## Where to start

1. [First typed judgment offline](../tutorials/first-typed-judgment-offline.md)
   — wire `ScriptedScoringFake` and `ScoringJudgmentAdapter` with no model.
2. [Run a small live judgment eval](../how-to/run-a-small-live-judgment-eval.md)
   — TPJEP eight or read the frozen partner-6 receipt.
3. [Judgment live receipts](../reference/judgment-live-receipts.md) — measured
   numbers and pins, with links to issue comments.
4. [Judgment text parts](../reference/judgment-text-parts.md) — each text part
   of a judgment call and its evolution status.

Structured JSON from `GenerationPort` is a sibling path. See
[Call typevet from Python](../how-to/call-typevet-from-python.md).

## Limitations

**Valid structure is not calibration.** Schema-valid or probability-valid
outputs do not prove task accuracy, ECE, or production readiness. Say which
eval counted gold only when the run used a loader with explicit gold-match or
broad-agreement rules.

**Template honesty.** On llama.cpp, the judgment session classifies the
served template through `/apply-template`. By default it accepts only native
Gemma 4 turns and fails before scoring on any other family
([Gemma 4 page](how-typevet-works-with-gemma-4.md#what-is-specific-to-gemma-4)).
The classifier reads the markers in the rendered output, not the template
source. The recorded llama.cpp [receipt](https://github.com/Alberto-Codes/typevet/issues/203#issuecomment-5882379255) used a `--chat-template-file` override
([#233](https://github.com/Alberto-Codes/typevet/issues/233)). Thus
`native_gemma4_turn` does not prove that the GGUF's own template ran. A
`ScoringJudgmentAdapter` built with neither a served family nor a framing
uses a **degraded ChatML** prefix for text only. With image input, it fails
instead
([`judgment_scoring.py`](https://github.com/Alberto-Codes/typevet/blob/main/src/typevet/adapters/outbound/judgment_scoring.py)).
On vLLM, the server applies
its own chat template, and typevet does not classify it
([support matrix](../reference/typed-judgment-release-support-matrix.md#multimodal-primary-gemma-4-native-vision)).
Receipts must name the template class. Do not treat a degraded template as
silent equivalence to hosted Jev.

**Small samples.** The partner-derived six-row receipt on
[#133](https://github.com/Alberto-Codes/typevet/issues/133#issuecomment-5843131304)
is exploratory. It does not close the broader #133 quality study.

**Binding and labels.** Control-token scoring requires model-facing options to
match scored tokens ([#152](https://github.com/Alberto-Codes/typevet/issues/152),
commit [`249f168`](https://github.com/Alberto-Codes/typevet/commit/249f168)).
Pre-fix all-`duplicate_charge` failures were a demonstrated binding mismatch,
not proof that post-fix runs are calibrated.

**No TypeLLM compatibility promise.** typevet reimplements the open-weight
decision ideas. It does not guarantee byte-for-byte TypeLLM or SGLang parity.
