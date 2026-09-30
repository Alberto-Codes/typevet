# Post-hoc calibration of Noul probabilities

Kind: explanation.

This page is for an engineer who reads the typevet receipts. It explains one
offline study from
[#343](https://github.com/Alberto-Codes/typevet/issues/343). The study asks
one question. Can a small map, fitted after the model answers, fix the poor
calibration in the committed receipts? If it cannot, the owner must consider
fine-tuning.

The study made no model call. It read receipts that typevet had already
committed. The fitters are plain Python in `typevet_evals.calibration`. The
full result is in
[`post_hoc_receipt.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/calibration/post_hoc_receipt.json).

## Why calibration matters here

A Noul answer gives a probability. Calibration asks whether that
probability matches the observed frequency. Some receipts show a large gap.
The DIFrauD seed wording on Gemma has an ECE of 0.17 to 0.19 on three of its
five receipts. The CEDAR signature runs have an ECE of about 0.31.

A calibration map does not change the model. It changes only the number that
the caller reads. It is cheap to fit and cheap to apply.

## The plan, fixed before any fit

The
[#343 contract](https://github.com/Alberto-Codes/typevet/issues/343#issuecomment-5919241864)
set the plan before any fit. The study did not change it after the results.

**Series.** Each series is one probability column of one receipt. There are
16 series:

- DIFrauD, five held-out receipts, columns `seed` and `evolved` (158 rows).
- Generated checks, llama.cpp and vLLM (140 rows). The probability is the
  largest verdict probability. The label is whether the verdict was correct.
- CEDAR signatures, llama.cpp and vLLM (180 pairs).
- LFW faces, llama.cpp and vLLM (200 pairs).

**Split.** A row goes to the calibration half when
`int(sha256(id).hexdigest(), 16)` is even. Otherwise it goes to the
evaluation half. The fitters see only the calibration half. All metrics use
the evaluation half.

**Methods.** Each fitter clips probabilities to `[1e-6, 1 - 1e-6]` first.

| Method | Map | Fit |
|---|---|---|
| Temperature | `sigmoid(logit(p) / T)` | Golden-section search on `log T` in [-3, 3] |
| Platt | `sigmoid(a * logit(p) + b)` | Newton steps with step halving |
| Isotonic | Non-decreasing steps | Pool adjacent violators; value of the nearest lower knot |

**Metrics.** ECE uses 10 equal-width bins. It reuses the face-match ECE
function. The study also reports the Brier score and the accuracy at 0.5. A
bootstrap of 1,000 resamples gives a 95% interval of the ECE change. The
interval is context only.

**Rule.** A method meets the rule on a series when three things are true:

- ECE drops by 0.03 or more.
- The Brier score drops.
- Accuracy drops by 0.01 or less.

A series with an evaluation-half ECE below 0.05 before the fit does not count
toward the decision.

**Decision.** The pre-registration named six series with an ECE of 0.10 or
more. Two more series pass that bar under the same ECE convention (the #309
llama.cpp seed wording at 0.113 and the llama.cpp checks at 0.165); they were
not named, and adding them does not change the outcome, because isotonic
leaves both above 0.05. Post-hoc calibration is sufficient when one method
meets the rule on 4 or more of the six named series. That method must also leave the evaluation-half ECE below 0.05 on each
of those 4.

## Result on the six decision series

Each cell is before → after on the evaluation half. "Rule" is the
pre-registered rule. "Counts" means the rule is met and the ECE after is
below 0.05.

| Series | n cal / eval | Method | ECE | Brier | Accuracy | Rule | Counts |
|---|---|---|---|---|---|---|---|
| DIFrauD #252 Gemma llama.cpp, seed | 68 / 90 | Temperature | 0.198 → 0.200 | 0.173 → 0.148 | 0.811 → 0.811 | no | no |
| | | Platt | 0.198 → 0.071 | 0.173 → 0.074 | 0.811 → 0.889 | yes | no |
| | | Isotonic | 0.198 → 0.057 | 0.173 → 0.048 | 0.811 → 0.956 | yes | no |
| DIFrauD #252 Gemma vLLM, seed | 68 / 90 | Temperature | 0.214 → 0.202 | 0.209 → 0.153 | 0.778 → 0.778 | no | no |
| | | Platt | 0.214 → 0.020 | 0.209 → 0.083 | 0.778 → 0.889 | yes | yes |
| | | Isotonic | 0.214 → 0.020 | 0.209 → 0.082 | 0.778 → 0.889 | yes | yes |
| DIFrauD #309 vLLM, seed | 68 / 90 | Temperature | 0.215 → 0.202 | 0.210 → 0.154 | 0.778 → 0.778 | no | no |
| | | Platt | 0.215 → 0.024 | 0.210 → 0.083 | 0.778 → 0.889 | yes | yes |
| | | Isotonic | 0.215 → 0.012 | 0.210 → 0.081 | 0.778 → 0.889 | yes | yes |
| DIFrauD #309 vLLM, evolved | 68 / 90 | Temperature | 0.187 → 0.188 | 0.184 → 0.139 | 0.811 → 0.811 | no | no |
| | | Platt | 0.187 → 0.020 | 0.184 → 0.083 | 0.811 → 0.889 | yes | yes |
| | | Isotonic | 0.187 → 0.008 | 0.184 → 0.087 | 0.811 → 0.878 | yes | yes |
| CEDAR signatures, llama.cpp | 91 / 89 | Temperature | 0.306 → 0.236 | 0.302 → 0.202 | 0.674 → 0.674 | yes | no |
| | | Platt | 0.306 → 0.126 | 0.302 → 0.115 | 0.674 → 0.843 | yes | no |
| | | Isotonic | 0.306 → 0.035 | 0.302 → 0.096 | 0.674 → 0.876 | yes | yes |
| CEDAR signatures, vLLM | 91 / 89 | Temperature | 0.282 → 0.219 | 0.280 → 0.192 | 0.719 → 0.719 | yes | no |
| | | Platt | 0.282 → 0.099 | 0.280 → 0.128 | 0.719 → 0.831 | yes | no |
| | | Isotonic | 0.282 → 0.036 | 0.280 → 0.117 | 0.719 → 0.831 | yes | yes |

Isotonic counts on 5 of the 6 series. Platt counts on 3. Temperature counts
on none. So by the pre-registered rule, post-hoc calibration is sufficient.
The study does not recommend a fine-tuning issue.

## The other ten series

These series are reported but are not part of the decision. Each cell is the
evaluation-half ECE; the ECE before the fit is in the second column.

| Series | ECE before | Temperature | Platt | Isotonic |
|---|---|---|---|---|
| DIFrauD #252 Gemma llama.cpp, evolved | 0.033 (excluded) | 0.052 | 0.046 | 0.017 |
| DIFrauD #252 Gemma vLLM, evolved | 0.034 (excluded) | 0.047 | 0.047 | 0.021 |
| DIFrauD #252 Jev, seed | 0.103 | 0.052 (rule met) | 0.089 | 0.089 |
| DIFrauD #252 Jev, evolved | 0.076 | 0.029 (rule met) | 0.022 (rule met) | 0.033 |
| DIFrauD #309 llama.cpp, seed | 0.141 | 0.134 | 0.057 (rule met) | 0.061 (rule met) |
| DIFrauD #309 llama.cpp, evolved | 0.107 | 0.110 | 0.067 (rule met) | 0.076 |
| Checks, llama.cpp | 0.189 | 0.129 (rule met) | 0.110 (rule met) | 0.059 (rule met) |
| Checks, vLLM | 0.093 | 0.069 | 0.085 | 0.087 |
| LFW faces, llama.cpp | 0.080 | 0.074 | 0.051 | 0.054 |
| LFW faces, vLLM | 0.028 (excluded) | 0.030 | 0.017 | 0.014 |

## Why temperature fails

Temperature scaling has one parameter. It can only pull probabilities toward
0.5 or push them away. It cannot move the point where the map crosses 0.5.

The Gemma DIFrauD seed runs need that move. Their fitted temperatures are
about 4 to 6.4, but ECE barely changes. Platt adds an intercept, and isotonic
has no fixed shape. Both move the crossing point.

This is also why accuracy rises after Platt and isotonic. A map that moves the
crossing point changes which rows read positive at 0.5.

## Transfer between tasks

The study also fitted a map on one task and applied it to another. It used
DIFrauD #309 llama.cpp seed and CEDAR signatures on llama.cpp. No method met
the rule in either direction. The signature map made DIFrauD worse: ECE
rose from 0.141 to between 0.237 and 0.325.

So each task and backend needs its own map. One shared map does not work.

## Limits

- **Small halves.** Each evaluation half has 70 to 98 rows. A 10-bin ECE on
  so few rows is noisy. The bootstrap intervals are wide.
- **One split.** The study used one fixed split. A different split can give
  different numbers.
- **Isotonic gives hard values.** Some isotonic steps are exactly 0 or 1.
  Such a value claims certainty. Log loss is infinite when such a row is
  wrong.
- **Boundary fits.** The Jev evolved temperature fit reached the search
  bound, `T = exp(-3)`. The Jev seed Platt fit gave a very large slope. On those
  Jev calibration halves, the probabilities almost separate the labels.
- **Same distribution.** Each map was fitted and tested on rows from one
  receipt. A map can fail when the data or the wording changes.
- **No live check.** No model call used a fitted map. This page says nothing
  about a deployed map.

## Receipts

- [`post_hoc_receipt.json`](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/calibration/post_hoc_receipt.json):
  every series, method, fitted parameter, metric and interval, the transfer
  rows, the decision and the sha256 of each source receipt.
- The source receipts are under `evals/fixtures/difraud/receipts/`,
  `evals/fixtures/checks/receipts/`, `evals/fixtures/cedar/receipts/` and
  `evals/fixtures/lfw/receipts/`.

## Related pages

- [Gemma 4 and Jev on DIFrauD, each with its evolved wording](gemma-and-jev-difraud.md)
- [Wording evolution on DIFrauD scam messages](wording-evolution-difraud.md)
- [Two-image signature comparison](two-image-signature-comparison.md)
- [Limits and known gaps](limits.md)
