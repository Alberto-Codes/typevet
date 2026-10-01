# Wording evolution on DIFrauD scam messages

Kind: explanation.

This page is for an engineer who reads the #309 receipts. It explains one
attempt to improve the wording of a typevet question with gepa-adk. It states
the rule that was fixed before the run, and it shows where the evidence
stops. It is not a how-to. The contract and the result are on
[#309](https://github.com/Alberto-Codes/typevet/issues/309), and the plan is on
[#259](https://github.com/Alberto-Codes/typevet/issues/259#issuecomment-5904177183).

All probabilities on this page are model confidence. They are not calibrated
frequencies.

## Why this was tried

The #236 run scored 500 DIFrauD SMS test rows with the `is_scam` question.
Its ECE was 0.158, which is high. The model was often confident and wrong
([#133](https://github.com/Alberto-Codes/typevet/issues/133),
[#236](https://github.com/Alberto-Codes/typevet/issues/236)).

finvet had the same kind of problem. It evolved the wording of one `Noul`
question with gepa-adk, and its held-out ECE moved from about 0.143 to 0.104
(finvet #8 and #14, as the
[#259 plan](https://github.com/Alberto-Codes/typevet/issues/259#issuecomment-5904177183) reports).
The #259 plan asked whether the same method helps typevet. This page is the
answer for one run.

## What was evolved

The evolved text is the `instructions` string of the DIFrauD `is_scam`
`Noul` question. Nothing else changed: not the criteria, not the schema and
not the model. Since
[#363](https://github.com/Alberto-Codes/typevet/issues/363), a run can also
evolve the `Noul` criteria texts. This run evolved the `instructions` only.
The #365 runs evolved the criteria texts. See
[evolving the criteria](gemma-and-jev-difraud.md#evolving-the-criteria-365).

Since [#369](https://github.com/Alberto-Codes/typevet/issues/369), a run can
also evolve the parts of a `Choice` seed, such as PubMedQA yes, no or maybe.
The transport then returns one probability per label. The reward is one minus
the multi-class Brier score, scaled to 0..1. The held-out check adds Cohen's
kappa and uses the same pass rule and bootstrap. No live `Choice` run exists
yet. The run is a separate pre-registered issue.

| Wording | Text | Length |
|---|---|---|
| Seed | `Is this message a scam, phishing or social-engineering attempt?` | 63 characters |
| Evolved | `Estimate probability (0-1) this message is a scam, phishing, or social-engineering attempt.` | 91 characters |

## How it was evolved

The evolution ran once, on the local llama.cpp router, at no cost.

| Topic | Value |
|---|---|
| Optimizer | gepa-adk 2.6.0, in the `evals` workspace member only |
| Score | 1 − Brier per row, summed over the selection rows |
| Judge | Router alias `gemma-4-31b-24gib-kv11-decoder`: Gemma 4 31B, `Q2_K` GGUF, build `b11277-eae11d221` |
| Reflector | Router alias `Qwen3.8-27B-UD-Q4_K_M`, a different model from the judge |
| Train rows | 1,000, a stratified subset of the DIFrauD SMS train split (seed 0) |
| Selection rows | 200, a stratified subset of the validation split (seed 0) |
| Reflection minibatch | 8 rows |
| Length cap | 94 characters, 1.5 times the seed length |
| Iterations | `max_iterations` 10, stop reason `max_iterations` |
| Wall time | about 33 minutes |

The reflector made six attempts. gepa-adk accepted one proposal, the first.
It refused four proposals at the minibatch gate, and one reflection timed
out. On the 200 selection rows, the summed score went from 184.46 to 187.20.

The run has three guards against overfitting:

- **Held-out rows were never scored during evolution.** The held-out set is
  the upstream SMS test split minus every row that #236 scored. No held-out
  row is in the selection set.
- **The #236 rows are out.** The earlier 500 rows cannot leak into the check.
- **The length cap is in the reflection prompt.** finvet found that gains
  followed text length. The prompt states "at most 94 characters". Without
  that line, the first proposal had 542 characters and was refused.

## The pre-registered rule

The rule was fixed on the issue before any run
([contract amendment A1](https://github.com/Alberto-Codes/typevet/issues/309#issuecomment-5912291566)).
It applies to the vLLM held-out result. The evolved wording passes only when
all four parts hold:

1. ECE drops by at least 0.03 from the seed wording.
2. The evolved ECE is 0.10 or less.
3. The Brier score drops.
4. Accuracy drops by 0.01 or less.

ECE uses 10 equal-width bins. A probability of 0.5 or more reads `scam`. The
receipts also report a paired bootstrap 95% interval of the ECE and Brier
differences (2,000 resamples, seed 0). The interval is context only. It does
not change the rule.

## The result

Each backend scored all 158 held-out rows once with each wording. That is 316
calls per backend. Of the 158 rows, 36 are scam.

| Backend | Wording | Accuracy | Brier | ECE |
|---|---|---|---|---|
| vLLM v0.30.0, `google/gemma-4-31B-it` BF16, revision `842da379`, one H100 (primary) | seed | 0.810 | 0.182 | 0.185 |
| | evolved | 0.823 | 0.169 | 0.173 |
| llama.cpp `b11277-eae11d221`, `Q2_K` (secondary) | seed | 0.918 | 0.074 | 0.113 |
| | evolved | 0.924 | 0.070 | 0.082 |

**vLLM, the primary check: fail.** The ECE dropped by 0.012, not 0.03. The
evolved ECE is 0.173, not 0.10 or less. The Brier score dropped by 0.013, and
accuracy rose by 0.013. Two of the four parts fail. The bootstrap intervals
are [−0.048, +0.020] for the ECE difference and [−0.047, +0.019] for the
Brier difference. Both include 0.

**llama.cpp, the secondary check: narrow pass.** The ECE dropped by 0.0302,
just over 0.03. The evolved ECE is 0.082. The Brier score dropped by 0.004,
and accuracy rose by 0.006. The ECE interval is [−0.052, −0.008]. The Brier
interval is [−0.024, +0.019] and includes 0.

The llama.cpp pass does not change the result. The rule names vLLM. Also,
`Q2_K` on llama.cpp is the model that the wording was evolved on, so a pass
there is the weaker test.

### Low power

The check has 158 rows, and only 36 are scam. With so few rows, a real
change of a few hundredths in ECE is hard to separate from noise. The wide
vLLM intervals show this. A fail here does not show that the evolved wording
is no better. A pass on llama.cpp does not show that it is better.

## The #133 re-measurement

The seed wording's held-out ECE is 0.185 on vLLM and 0.113 on llama.cpp. The
#236 figure was 0.158. The numbers do not compare directly, for two reasons:

- **Different rows.** #236 used 500 test rows. This check used the 158
  held-out rows, which exclude those 500.
- **Different backends and weights.** The 0.185 comes from vLLM BF16, and the
  0.113 comes from llama.cpp `Q2_K`.

So the seed wording is still overconfident on vLLM. The llama.cpp value,
0.113, is lower than 0.158, but it was measured on different rows.

## An unexplained backend gap

**Update, 2026-09-30:** the #329 runs used native Gemma 4 framing on both backends
([#329](https://github.com/Alberto-Codes/typevet/issues/329)).
The framing fix and its probe are on [#327](https://github.com/Alberto-Codes/typevet/issues/327).
On the same 158 rows, the seed wording gave accuracy 0.829 and ECE 0.170 on
llama.cpp `Q2_K`, and 0.810 and 0.185 on vLLM BF16.
So the gap below was mostly the ChatML framing on the `gemma-4-31b-24gib-kv11-decoder` alias.
The #329 runs used the alias `gemma-4-31b-kv9-q4km-mm`, also `Q2_K` but a different file
([#324](https://github.com/Alberto-Codes/typevet/issues/324),
[Gemma 4 and Jev on DIFrauD](gemma-and-jev-difraud.md)).

On the same 158 rows and the same seed wording, vLLM BF16 scored much worse
than llama.cpp `Q2_K`: accuracy 0.810 against 0.918, and Brier 0.182 against
0.074. Under different prompt framing, the vLLM BF16 run did worse than the
llama.cpp `Q2_K` run.

The two backends did not see the same prompt framing. The llama.cpp receipt
records `served_template` `degraded_chatml`. The router alias serves no
native Gemma 4 chat template, so typevet composes a ChatML prefix itself.
vLLM applied the model's native Gemma 4 chat template through
`/v1/chat/completions`. Its receipt value `vllm_chat` is a constant in the
live test, not a probe. The framing difference is a likely part of the gap.
No run has proved it to be the cause.
[#324](https://github.com/Alberto-Codes/typevet/issues/324) holds that
question.

## Two known confounds

- **Weights.** The wording was evolved on `Q2_K` weights and checked on
  BF16 weights.
- **Prompt framing.** The evolution and the llama.cpp check ran under the
  ChatML framing. The vLLM check ran under the native Gemma 4 template.

The evolved wording can fit one weight set or one framing and not transfer
to the other. One run on each backend cannot separate those causes. Together
with the gap above, this limits what the vLLM fail says about the wording.

## Compared with finvet

finvet reported a held-out ECE drop of about 0.04 (0.143 to 0.104). The
typevet vLLM drop is 0.012, and the llama.cpp drop is 0.030. The two projects
differ in data, model and backend, so this is not a like-for-like comparison.
The typevet result does not confirm the finvet result, and it does not
contradict it.

## What this result does not say

- **No product change.** typevet's shipped questions and prompts do not
  change. The evolved text lives in an evaluation receipt only.
- **No general claim.** The result does not show that wording evolution
  works, or that it does not work, in general.
- **One run.** There was one evolution run and one held-out check per
  backend. The receipts give no variance estimate across runs.
- **One question and one dataset.** The result says nothing about other
  questions, other data or other models.

## Latency and cost

One item is one call: one typed judgment with the one `is_scam` question.
Each held-out check made 316 calls, 158 rows times two wordings. The
held-out receipts give one `wall_seconds` value in `pins`. They give no
per-call latency and no input tokens, so this page shows no token count.
[#324](https://github.com/Alberto-Codes/typevet/issues/324) asks the next
run to record both per row.

| Backend | Items | Wall time | Mean seconds per item | Items per minute | Mean input tokens |
|---|---|---|---|---|---|
| llama.cpp | 316 | 371 s (6.2 min) | 1.17 | 51.1 | Not recorded |
| vLLM | 316 | 317 s (5.3 min) | 1.00 | 59.8 | Not recorded |

The receipt clock starts after the dataset loads. The supervisor's run logs
(not in the repository) give pytest wall times of 372 s on llama.cpp and
318 s on vLLM. Those include the dataset download, so they are upper bounds.

The evolution itself ran on the local llama.cpp router. Its artifact gives a
wall time of 1,958 s (32.6 min). That time includes the judge calls and the
reflector calls.

Read these numbers with their limits:

- **One request at a time.** Each run sent one call, then waited for the
  answer. This is not a throughput test. The H100 throughput reference is the
  #236 receipt on the [performance page](../reference/performance.md).
- **Network time is included on vLLM.** The vLLM requests went through the
  RunPod proxy to a pod in data center AP-IN-2. For about the first minute,
  the signature run (#319) used the same pod.

The H100 pod cost about $2.24 in total
([#316 pod record](https://github.com/Alberto-Codes/typevet/issues/316#issuecomment-5914299864)).
That cost is shared with the check run (#316) and the signature run (#319).
It is not the cost of this run alone. The local runs had no rental cost.

## Receipts

- [`wording_evolution_artifact.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording_evolution_artifact.json):
  the seed and evolved text, settings, iteration history and scores.
- [`wording_held_out_vllm.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording_held_out_vllm.json):
  the primary check, with per-row probabilities, metrics, intervals and
  verdict.
- [`wording_held_out_llama_cpp.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording_held_out_llama_cpp.json):
  the secondary check, in the same shape.

## Related pages

- [Gemma 4 and Jev on DIFrauD, each with its evolved wording](gemma-and-jev-difraud.md)
- [Limits and known gaps](limits.md)
- [Verified evidence and inferred claims](verification.md)
- [Native typed judgments](native-typed-judgments.md)
- [Judgment text parts](../reference/judgment-text-parts.md): each text part of a judgment call and its evolution status
