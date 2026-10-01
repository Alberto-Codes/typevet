# Gemma 4 and Jev on DIFrauD, each with its evolved wording

Kind: explanation.

This page is for an engineer who reads the #252 receipts. It explains one
small study. Two judges each got their own gepa-adk wording for the DIFrauD
`is_scam` question. Then each judge scored the same held-out rows with the
seed wording and with its own evolved wording.

The study follows the #309 page,
[Wording evolution on DIFrauD scam messages](wording-evolution-difraud.md).
It also corrects one reading on that page: the #309 backend gap was mostly
prompt framing. See [the framing correction](#the-framing-correction).
A later section reports the #365 follow-up, which evolved the criteria
texts. See [evolving the criteria](#evolving-the-criteria-365).

The plan is the
[#252 design](https://github.com/Alberto-Codes/typevet/issues/252#issuecomment-5915065291).
The runs are on [#328](https://github.com/Alberto-Codes/typevet/issues/328)
(evolution) and [#329](https://github.com/Alberto-Codes/typevet/issues/329)
(held-out scoring). All probabilities on this page are model confidence, not
calibrated frequencies.

## The question the study asks

The owner asked a practical question on #252. Does evolved Gemma 4 come
close enough to evolved Jev for this use case? The study reports numbers
for each judge. It has no pass rule. It makes no claim that one judge can
substitute for the other.

Each judge gets its own wording because the #252 amendment allows it. Each
judge starts from the same seed wording and meets the same labels. So the
comparison is "each judge at its best wording", not "both judges on one
text".

## The judges

The study has three judges. Gemma 4 on two backends counts as two judges.
The #252 design asks that each quant be reported as its own judge.

| Judge | Model identity in the receipt | Backend | Weights | Served template |
|---|---|---|---|---|
| Jev | Requested `jev-latest`, reported `jev-1.13.0` | Hosted, `https://api.typesafe.ai`, through judgevet | Not published | Not applicable (Jev HTTP) |
| Gemma 4 31B, llama.cpp | Alias `gemma-4-31b-kv9-q4km-mm` | Local llama.cpp, build `b11277-eae11d221` | File `gemma-4-31b-24gib-kv9-decoder.gguf`, ftype `Q2_K - Medium` | `native_gemma4_turn`, probed |
| Gemma 4 31B, vLLM | `google/gemma-4-31B-it`, revision `842da379` | vLLM v0.30.0 on one rented H100 | BF16 | `native_gemma4_turn`, probed |

The `q4km` in the llama.cpp alias does not describe the decoder weights. The
server reports the ftype `Q2_K - Medium` for that alias.

Both Gemma judges ran with the native Gemma 4 chat template. The receipts
record the template from a probe, not from a constant. This was the fix for
the [#324](https://github.com/Alberto-Codes/typevet/issues/324) confound.

## How each wording was evolved

Both evolutions used the same runner, settings and data. Only the judge
changed.

| Setting | Value |
|---|---|
| Optimizer | gepa-adk, in the `evals` workspace member only |
| Seed wording | `Is this message a scam, phishing or social-engineering attempt?` (63 characters) |
| Score | 1 − Brier per row, summed over the 200 validation rows |
| Reflector | Local Qwen3.8-27B (`Qwen3.8-27B-UD-Q4_K_M`), the same for both judges |
| Rows | 1,000 train rows and 200 validation rows, from the #307 splits |
| Length cap | 94 characters, 1.5 times the seed length |
| Iterations | 10, stop reason `max_iterations` |

The two runs:

| Topic | Jev | Gemma 4 on llama.cpp `Q2_K` |
|---|---|---|
| Evolved wording | `Predict probability (0-1) that message is scam/phishing/unsolicited promo, offer, call, alert` | `Label 1 only for obvious deceptive scam/phishing/social engineering; ignore personal text.` |
| Length | 93 characters | 90 characters |
| Accepted proposals | 3 (iterations 5, 7 and 8) | 4 (iterations 1, 2, 3 and 5) |
| Validation score, seed → evolved | 192.08 → 195.83 | 176.80 → 197.63 |
| Mean score per validation row | 0.960 → 0.979 | 0.884 → 0.988 |
| Judge calls | 2,280, 0 failed | 2,280, 0 failed |
| Judge input tokens | 705,494 | 211,306 |
| Concurrent calls | 1 | 5 |
| Median judge latency | 0.107 s | 2.325 s |
| Wall time | 1,630.5 s (27.2 min) | 2,025.3 s (33.8 min) |

The Jev run had a spend cap of 4,000 attempts. It used 2,280 attempts. No
call was refused after the cap, so the artifact is marked valid. The cap was
raised from 3,000 to 4,000 before the run, as #328 records.

### The two wordings moved in opposite directions

The Jev wording widens "scam". It adds unsolicited promotions, offers, calls
and alerts. This suggests that Jev scored the DIFrauD positive label as
spam-like, not only as deception.

The Gemma wording narrows "scam". It asks for obvious deception only, and it
tells the model to ignore personal text. With the seed wording, Gemma read
too many held-out rows as scam. It read 63 (llama.cpp) or 66 (vLLM) of 158
rows as scam, and 36 rows are scam. With the evolved wording, both read 37.

These are readings of two texts. The study did not test why each judge
preferred its text.

## The held-out result

Each judge scored the 158 held-out rows once with each wording. That is 316
calls per judge. Of the 158 rows, 36 are scam. Evolution never scored these
rows.

| Judge | Wording | Accuracy | Cohen's κ | Brier | ECE (10 bins) |
|---|---|---|---|---|---|
| Jev `jev-1.13.0` | seed | 0.956 | 0.873 | 0.038 | 0.089 |
| | evolved | 0.981 | 0.947 | 0.020 | 0.071 |
| Gemma 4 31B, llama.cpp `Q2_K` | seed | 0.829 | 0.616 | 0.152 | 0.170 |
| | evolved | 0.968 | 0.911 | 0.033 | 0.034 |
| Gemma 4 31B, vLLM BF16 | seed | 0.810 | 0.583 | 0.181 | 0.185 |
| | evolved | 0.968 | 0.911 | 0.032 | 0.034 |

A probability of 0.5 or more reads `scam`. κ is the agreement of those
readings with the labels, beyond chance.

The receipts also give a paired bootstrap 95% interval of evolved minus seed.
They use 2,000 resamples, seed 0. The intervals are context only.

| Judge | ECE difference | Brier difference |
|---|---|---|
| Jev | [−0.042, −0.002] | [−0.027, −0.009] |
| Gemma 4, llama.cpp `Q2_K` | [−0.194, −0.080] | [−0.173, −0.065] |
| Gemma 4, vLLM BF16 | [−0.213, −0.093] | [−0.210, −0.092] |

No interval includes 0. For each judge, on these rows, the evolved wording
scored better than the seed wording.

### Reading the numbers per judge

- **Jev** starts high with the seed wording. Its evolved wording adds about
  0.03 accuracy and 0.07 κ. Its ECE stays higher than Gemma's evolved ECE.
- **Gemma 4** starts much lower. Its evolved wording adds about 0.14 to 0.16
  accuracy and 0.30 to 0.33 κ. The gain is much larger than Jev's gain.
- **Evolved Jev against evolved Gemma:** Jev is ahead on accuracy, κ and
  Brier. Gemma is ahead on ECE. The gaps are small on 158 rows.

These are observations per judge on one row set. They do not show that one
judge can replace the other.

### The same Gemma result on two backends

The evolved Gemma wording gives the same result on llama.cpp `Q2_K` and vLLM
BF16. Both read the same class on all 158 rows. Accuracy (0.968), κ (0.911)
and ECE (0.034) are equal to three decimals. Brier is 0.033 against 0.032.

The wording was evolved on the `Q2_K` weights. On these rows, it transfers
to the BF16 weights with no loss.

The Gemma vLLM seed row also reproduces #309. The #309 vLLM receipt scored
the same 158 rows with the same seed wording. Both runs read the same class
on all 158 rows. Accuracy (0.810) and ECE (0.185) are equal. Brier is 0.181
here and 0.182 in #309.

## The framing correction

The #309 page reports a backend gap it could not explain. With the seed
wording, llama.cpp `Q2_K` scored accuracy 0.918 and Brier 0.074. vLLM BF16
scored 0.810 and 0.182. The #309 llama.cpp run used ChatML framing
(`degraded_chatml`). The vLLM run is assumed to have used the native Gemma 4
template. #309 did not probe it: its receipt holds the constant `vllm_chat`
([#324](https://github.com/Alberto-Codes/typevet/issues/324)). The #329 vLLM
run probed `native_gemma4_turn`. Its seed readings match #309 on all 158 rows,
which supports the assumption.

This study ran llama.cpp with the native template. The seed wording then
scored accuracy 0.829 and ECE 0.170. That is close to vLLM (0.810 and
0.185), and far from the ChatML run (0.918 and 0.113).

**Correction to #309:** the #309 backend gap was mostly the prompt framing.
The evidence does not point to the `Q2_K` and BF16 weights. With native
framing on both backends, the gap mostly goes away.
[#324](https://github.com/Alberto-Codes/typevet/issues/324) holds the evidence
and the resolution.

Two limits apply to this correction:

- **The llama.cpp alias also changed.** #309 used `gemma-4-31b-24gib-kv11-decoder`.
  This study used `gemma-4-31b-kv9-q4km-mm`. Both are `Q2_K` files, but not the
  same file. The #309 `kv11` file was not re-run with native framing.
- **The #309 vLLM template is assumed, not probed.** #309 recorded the constant
  `vllm_chat`. The 158 matching seed readings support the native template.
- **The #309 wording result stands as measured.** The #309 evolution ran under
  ChatML framing. This study does not re-score the #309 evolved text.

## What this study does not say

- **No substitution claim.** Jev and Gemma 4 share one judgevet interface.
  That does not make their judgments equal. The numbers are per judge.
- **No pass rule.** The study reports numbers. Nothing on this page passes
  or fails.
- **Low power.** The held-out set has 158 rows and 36 scam rows. A few
  rows can move accuracy by 0.01 to 0.02.
- **One run each.** There was one evolution and one held-out check per
  judge. The receipts give no variance across runs.
- **DIFrauD labels as gold.** The labels come from the DIFrauD dataset, not
  from a person on this project. The Jev wording suggests that the positive
  label includes spam-like promotions. A different gold set can give
  different numbers.
- **One question and one dataset.** The result says nothing about other
  questions, other data or other models.
- **No product change.** typevet's shipped questions do not change. The
  evolved texts live in evaluation receipts only.

## Latency and cost

Each held-out call is one typed judgment with the one `is_scam` question.
Each held-out run sent one call at a time. The sum of the per-call latencies
equals the wall time to about 0.2 s.

| Judge | Calls | Wall time | Median latency | Maximum latency | Calls per minute | Input tokens |
|---|---|---|---|---|---|---|
| Jev (hosted) | 316 | 33.7 s | 0.097 s | 0.283 s | 562.6 | 98,508 |
| Gemma 4, llama.cpp `Q2_K` (local) | 316 | 359.1 s (6.0 min) | 1.136 s | 1.267 s | 52.8 | 29,794 |
| Gemma 4, vLLM BF16 (H100) | 316 | 54.1 s | 0.171 s | 0.189 s | 350.5 | 29,794 |

Read these numbers with their limits:

- **Different token counts.** Jev and Gemma 4 use different request bodies
  and different tokenizers. The token column does not compare across the two
  model families. The two Gemma judges agree on it.
- **Network time is included.** Jev calls went to the hosted service. The vLLM
  calls went from the client in Arizona to a pod in US-MO-1 over a direct TCP
  port ([#336](https://github.com/Alberto-Codes/typevet/issues/336)).
- **One request at a time.** This is not a throughput test. #336 records
  H100 throughput for the image runs.

Cost for each judge:

- **Jev:** 2,280 attempts for the evolution and 316 for the held-out check.
  The held-out check had a cap of 400 attempts. No call was refused. The
  receipts give no dollar figure, and the study knows none. The count is the
  only cost record.
- **Gemma 4 on llama.cpp:** local GPU, no rental cost.
- **Gemma 4 on vLLM:** the held-out check ran on the H100 pod of #336. That
  pod ran for about 18.4 min at $3.49 per hour, about $1.07 in total. The
  cost is shared with the #336 image runs. It is not the cost of this check
  alone.

## Evolving the criteria (#365)

[#365](https://github.com/Alberto-Codes/typevet/issues/365) asked a follow-up
question. Does evolving the `Noul` criteria texts beat the #252
instructions-only result? Its rule was fixed before the runs. An arm wins
when held-out Brier drops by 0.01 or more and accuracy drops by 0.01 or less.
The reference is the evolved Gemma row on llama.cpp above (0.968 and 0.033).
An arm ties when both metrics stay within 0.01. Otherwise it loses.

The #252 seed instructions got two criteria texts, fixed before the runs:

- `criteria_true`: `The message tries to deceive the reader into money, credentials or an unsafe action.`
- `criteria_false`: `The message is an ordinary personal, commercial or informational text.`

The criteria change the seed prompt. So arm 0 scores the seed with criteria,
with no evolution. Arm A evolves the two criteria texts and freezes the
instructions. Arm B evolves all three parts. The judge, reflector, budget,
scorer and 158 held-out rows are those of the #252 llama.cpp run.

| Row | Parts evolved | Iterations (of 10) | Validation 1 − Brier | Held-out accuracy | Brier | ECE | κ | Verdict |
|---|---|---|---|---|---|---|---|---|
| #252 seed, no criteria | none | — | — | 0.829 | 0.152 | 0.170 | 0.616 | reference seed |
| #252 evolved | `instructions` | 10 | 0.988 | 0.968 | 0.033 | 0.034 | 0.911 | reference |
| Arm 0: seed with criteria | none | — | — | 0.918 | 0.067 | 0.075 | 0.785 | not judged |
| Arm A | `criteria_true`, `criteria_false` | 6 | 0.928 | 0.918 | 0.074 | 0.078 | 0.785 | loses (Brier +0.041, accuracy −0.050) |
| Arm B | `instructions`, `criteria_true`, `criteria_false` | 9 | 0.958 | 0.949 | 0.049 | 0.055 | 0.867 | loses (Brier +0.016, accuracy −0.019) |

Validation 1 − Brier is the mean score per validation row. The verdict
column compares each arm with the #252 evolved row.

### The two evolved texts

Each arm changed one part. The other parts kept their seed text. The
frozen-part check passed on both evolution artifacts.

| Arm | Part changed | Evolved text |
|---|---|---|
| A | `criteria_true` | `The message intentionally deceives or pressures the reader to send money, credentials, or take an unsafe action.` |
| B | `instructions` | `Is this an unsolicited scam, phishing, spam, or social-engineering message to the recipient?` |

Each receipt also gives a paired bootstrap 95% interval of evolved minus arm
0. They use 2,000 resamples, seed 0.

| Arm | ECE difference | Brier difference |
|---|---|---|
| A | [−0.001, +0.008] | [+0.001, +0.014] |
| B | [−0.045, 0.000] | [−0.041, +0.001] |

Arm A made held-out Brier worse. The arm B interval touches 0.

### The answer

The answer is no. Evolving the criteria, alone or with the instructions, did
not beat the #252 instructions-only result. This holds for this seed, budget
and scorer.

The criteria texts help on their own. Arm 0 recovers most of the seed gap
with no evolution. Accuracy moves from 0.829 to 0.918, and Brier from 0.152
to 0.067.

The shared reflection prompt did not improve the criteria texts. In arm A,
the one accepted change made held-out Brier worse. In arm B, the reflector
proposed each part in turn. It accepted two `instructions` proposals and
rejected all six criteria proposals. So with all parts open, the gain came
from `instructions` only. Arm B then stopped below the #252 level.

### Caveats

- **One run per arm.** There is no repeat. The reflector proposals can
  change from run to run.
- **Patience stop.** Arms A and B each ended after five rejected proposals in
  a row. Their receipts still record the stop reason `max_iterations`.
- **Shared machine.** Other jobs ran on the same machine. The wall times (A
  1,899.7 s, B 2,120.8 s) do not compare with #252.
- **Judge failures.** Arm A had 10 failed judge calls of 1,432. Arm B had 1
  of 1,664.
- **One shared reflection prompt.** All parts use the same reflection prompt
  until [gepa-adk #437](https://github.com/Alberto-Codes/gepa-adk/issues/437)
  adds a prompt per part.
- **The arm 0 evolved column is not an arm.** The arm 0 receipt also scores
  the #252 evolved instructions with the criteria (0.956, Brier 0.042, ECE
  0.043). Those instructions were evolved without criteria. The column is
  context only and enters no verdict.

## What comes next

A third judge is planned:
[#333](https://github.com/Alberto-Codes/typevet/issues/333) adds an Ollama
`nimble` judge. It depends on judgevet
[#267](https://github.com/Alberto-Codes/judgevet/issues/267). No run exists
yet, so this page makes no claim about it.

## Receipts

- [`wording252_evolution_jev.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording252_evolution_jev.json)
  and
  [`wording252_evolution_gemma.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording252_evolution_gemma.json):
  the seed and evolved texts, settings, iteration history, per-call records
  and spend.
- [`wording252_held_out_jev.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording252_held_out_jev.json),
  [`wording252_held_out_gemma_llama_cpp.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording252_held_out_gemma_llama_cpp.json)
  and
  [`wording252_held_out_gemma_vllm.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording252_held_out_gemma_vllm.json):
  per-row probabilities, metrics, intervals, per-call latency and pins.
- [`wording365_armA_evolution_gemma_llama_cpp.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording365_armA_evolution_gemma_llama_cpp.json)
  and
  [`wording365_armB_evolution_gemma_llama_cpp.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording365_armB_evolution_gemma_llama_cpp.json):
  the #365 evolution artifacts, with the evolved parts, seed parts, digests,
  settings, iteration history and per-call records.
- [`wording365_seed_criteria_held_out_gemma_llama_cpp.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording365_seed_criteria_held_out_gemma_llama_cpp.json),
  [`wording365_armA_held_out_gemma_llama_cpp.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording365_armA_held_out_gemma_llama_cpp.json)
  and
  [`wording365_armB_held_out_gemma_llama_cpp.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/difraud/receipts/wording365_armB_held_out_gemma_llama_cpp.json):
  the #365 held-out checks for arm 0, arm A and arm B, with per-row
  probabilities, metrics, intervals and pins.

## Related pages

- [Wording evolution on DIFrauD scam messages](wording-evolution-difraud.md)
- [TypeLLM, Jev and judgevet](typellm-and-judgevet.md)
- [Limits and known gaps](limits.md)
- [Verified evidence and inferred claims](verification.md)
