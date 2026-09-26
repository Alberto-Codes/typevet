# Native typed judgments in typevet

Kind: explanation.

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
| finvet-derived six-message exploratory receipt | Documented on [#133](https://github.com/Alberto-Codes/typevet/issues/133); not a shipped CLI |
| Public calibration or ECE headline | No — see limitations |

## Where to start

1. [First typed judgment offline](../tutorials/first-typed-judgment-offline.md)
   — wire `ScriptedScoringFake` and `ScoringJudgmentAdapter` with no model.
2. [Run a small live judgment eval](../how-to/run-a-small-live-judgment-eval.md)
   — TPJEP eight or read the frozen finvet-6 receipt.
3. [Judgment live receipts](../reference/judgment-live-receipts.md) — measured
   numbers and pins, with links to issue comments.

Structured JSON from `GenerationPort` is a sibling path. See
[Call typevet from Python](../how-to/call-typevet-from-python.md).

## Limitations

**Valid structure is not calibration.** Schema-valid or probability-valid
outputs do not prove task accuracy, ECE, or production readiness. Say which
eval counted gold only when the run used a loader with explicit gold-match or
broad-agreement rules.

**Template honesty.** Gemma 4 on stock llama.cpp may use a **degraded ChatML**
template until native template work lands ([#129](https://github.com/Alberto-Codes/typevet/issues/129)).
Receipts must name the template class. Do not treat a degraded template as
silent equivalence to hosted Jev.

**Small samples.** The finvet-derived six-row receipt on
[#133](https://github.com/Alberto-Codes/typevet/issues/133#issuecomment-5843131304)
is exploratory. It does not close the broader #133 quality study.

**Binding and labels.** Control-token scoring requires model-facing options to
match scored tokens ([#152](https://github.com/Alberto-Codes/typevet/issues/152),
commit [`249f168`](https://github.com/Alberto-Codes/typevet/commit/249f168)).
Pre-fix all-`duplicate_charge` failures were a demonstrated binding mismatch,
not proof that post-fix runs are calibrated.

**No TypeLLM compatibility promise.** typevet reimplements the open-weight
decision ideas. It does not guarantee byte-for-byte TypeLLM or SGLang parity.
