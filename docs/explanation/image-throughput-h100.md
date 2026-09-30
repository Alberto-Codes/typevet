# Image judgment throughput on one H100

Kind: explanation.

This page is for an enterprise reader who asks how many image judgments one
GPU can give. It explains one measured run of typevet image judgments on one
H100. It says what raised the rate, what did not, and whether the answers
changed under load. It also says what the run cannot tell you.

Every number on this page comes from the committed throughput receipts. Setup
facts that the receipts do not record come from the
[#336 run plan](https://github.com/Alberto-Codes/typevet/issues/336#issuecomment-5917957400)
and the
[#336 result comment](https://github.com/Alberto-Codes/typevet/issues/336#issuecomment-5918309968).
The run is one run per level, on one model, one GPU and one pin.

## The short answer

On this setup, 16 requests in flight gave the highest rate. The full sets ran
at 4.4 to 5.3 judgments per second. That is 23 to 42 times the rate of the
0.4.0 runs. Those runs sent one request at a time over a longer network path.

At 32 in flight, the rate fell and each judgment waited longer. The verdicts
for faces and checks did not change under load. For signatures, 1 to 3
near-tied verdicts moved under batching. These results are for this setup
only.

## What was measured

One judgment is one item: one typed judgment that asks all its questions. Each
set asks different questions about different images:

| Set | Images per judgment | Questions per judgment | Full set |
|---|---|---|---|
| Faces (LFW View 2) | 2 | 3 | 200 pairs |
| Checks (synthetic) | 1 | 4 | 140 cases |
| Signatures (CEDAR) | 2 | 3 | 180 pairs |

The [face page](two-image-face-matching.md), the
[check page](check-register-matching.md) and the
[signature page](two-image-signature-comparison.md) explain each set and its
questions.

typevet sends each question as a separate request. The request carries the
images again. A face judgment is thus three requests, each with both images.

### Setup

| Item | Value |
|---|---|
| GPU | One H100 SXM, RunPod secure cloud, data center US-MO-1 |
| Server image | `vllm/vllm-openai:v0.30.0` (receipts record build `0.30.0`) |
| Model | `google/gemma-4-31B-it`, BF16 |
| Model revision | `842da3794eaa0b77d5f08bae87a17459d91ff475` (receipt pin) |
| Server flags | `--enable-prefix-caching --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.95 --max-model-len 8192 --limit-mm-per-prompt {"image":2} --logprobs-mode raw_logprobs` |
| Client | Operator machine in Arizona, direct TCP port to the pod |
| Network baseline | Median 0.073 s for 20 `GET /v1/models` calls |
| Run date | 2026-09-30 |

The receipts do not record the server flags. The flags come from the #336 run
plan. The result comment omits `--logprobs-mode raw_logprobs`.

## Full sets at concurrency 16 and 32

Concurrency is the number of judgments in flight at one time. Latency is the
client time for one judgment, network included.

| Set | Concurrency | Wall time | Judgments/s | Images/s | Latency p50 / p95 |
|---|---|---|---|---|---|
| Faces | 16 | 45.0 s | 4.45 | 8.89 | 3.62 s / 3.81 s |
| Faces | 32 | 58.6 s | 3.42 | 6.83 | 9.30 s / 10.19 s |
| Checks | 16 | 26.4 s | 5.31 | 5.31 | 2.92 s / 3.42 s |
| Checks | 32 | 28.3 s | 4.94 | 4.94 | 6.52 s / 7.31 s |
| Signatures | 16 | 40.6 s | 4.44 | 8.87 | 3.53 s / 4.05 s |
| Signatures | 32 | 54.0 s | 3.33 | 6.67 | 9.92 s / 11.61 s |

No judgment failed or was discarded in any run.

## The concurrency sweep

The sweep used the same subset at each level. The subset is 60 face pairs,
63 check cases and 60 signature pairs.

| Set | 1 | 4 | 16 | 32 |
|---|---|---|---|---|
| Faces, judgments/s | 1.14 | 3.01 | 4.78 | 3.30 |
| Checks, judgments/s | 0.88 | 2.80 | 5.47 | 5.01 |
| Signatures, judgments/s | 1.09 | 3.11 | 4.49 | 3.28 |
| Faces, p50 latency | 0.84 s | 1.25 s | 3.19 s | 8.55 s |
| Checks, p50 latency | 1.13 s | 1.35 s | 2.73 s | 6.11 s |
| Signatures, p50 latency | 0.91 s | 1.27 s | 3.41 s | 8.50 s |

The rate rose from 1 to 16 in flight for every set. At 32 it fell for every
set. The full-set runs show the same fall.

## What raised throughput

Three changes raised the rate against the 0.4.0 runs:

- **Concurrency.** On the same pod, 16 in flight gave four to six and a half
  times the rate of 1.
- **Prefix caching.** The server reused cached prompt tokens across the
  separate question requests. At concurrency 16, 0.55 of prompt tokens came
  from the cache.
- **Client placement.** The client used a direct TCP port to a US pod. The
  0.4.0 runs used the RunPod proxy to a pod in AP-IN-2.

The receipts do not split the gain between these three changes. The sweep
isolates concurrency only. No run turned prefix caching off or moved the
client back.

## What did not: concurrency 32

At 32 in flight, every set ran slower than at 16. Latency more than doubled.
The server counters show two changes at 32. For faces and signatures,
requests waited in the server queue. For checks, the queue time stayed near
zero. The prefix-cache hit rate fell for all three sets. The receipts do not show why the
hit rate fell.

| Counter, full set | Faces c16 | Faces c32 | Checks c16 | Checks c32 | Signatures c16 | Signatures c32 |
|---|---|---|---|---|---|---|
| Requests | 600 | 600 | 560 | 560 | 540 | 540 |
| Prefix-cache hit rate | 0.550 | 0.041 | 0.550 | 0.382 | 0.544 | 0.066 |
| Multimodal cache hits | 921 of 1,200 | 1,200 of 1,200 | 483 of 560 | 560 of 560 | 888 of 1,080 | 1,080 of 1,080 |
| Preemptions | 0 | 0 | 0 | 0 | 0 | 0 |
| Queue time, sum | 0.018 s | 69.9 s | 0.012 s | 0.018 s | 0.011 s | 61.9 s |

At concurrency 16, the faces run used 217,600 cached tokens of 395,600 prompt
tokens. The queue time is the server sum over all requests. Near zero means
requests did not wait in the server queue.

## Comparison with the 0.4.0 runs

The 0.4.0 receipts are single-stream runs of the same model on vLLM. The check
and signature receipts pin the same revision. The face receipt records no
revision. These runs sent one request at a time through the RunPod proxy to
AP-IN-2. Prefix caching was off.

| Set | 0.4.0 judgments/min | Concurrency 16 judgments/min | Ratio |
|---|---|---|---|
| Faces | 11.4 | 267 | 23.4 |
| Checks | 7.6 | 319 | 41.7 |
| Signatures | 9.2 | 266 | 29.0 |

The ratio combines concurrency, prefix caching and the shorter network path.
Do not read it as the gain from any one of them.

## Answers under load

A higher rate is of no use if the answers change. The receipts let us compare
answers at each level.

**Within this session.** Faces and checks gave the same verdicts at every
level. The face `Noul` probabilities were identical at every level. The check
`payee_matches` probabilities were identical too. The check `amounts_match`
probabilities moved by small amounts between levels. The largest move was
0.00004 in the subset and 0.00006 in the full set.

For signatures, the `Noul` probabilities were identical at every level. Some
`Choice` verdicts moved:

- In the subset, 1 of 60 verdicts at 16 and 2 of 60 at 32 differ from
  concurrency 1.
- In the full set, 3 of 180 verdicts differ between 16 and 32.

All moved verdicts are on genuine-versus-skilled-forgery pairs. The `Choice`
options for these pairs are near-tied. The leading option has a probability of
about 0.65 or less in each receipt checked. Batching moved these pairs between
`same_writer` and `skilled_forgery_suspected`.

**Against the 0.4.0 receipts.** Verdicts are equal for 199 of 200 faces, 140
of 140 checks and 176 to 179 of 180 signatures. The face difference exists
already at concurrency 1. Three of the four signature differences also exist
at concurrency 1. These differences do not come from concurrency. The
receipts do not isolate the cause. The fourth signature difference is a
near-tied pair that moved at concurrency 16.

At concurrency 32, two of the three signature pairs that differ at
concurrency 1 return to the 0.4.0 verdict. These pairs are `original_27_19`
and `original_12_2`. At the same level, the prefix-cache hit rate falls.

The run plan also asked for `Noul` probabilities within 0.001 of 0.4.0. That
did not hold everywhere. In the full sets, 7 face pairs and 10 signature pairs
differ by more. The largest face difference is 0.36, on a pair whose verdict
did not change. One check value differs by 0.007.

The [#332 design](https://github.com/Alberto-Codes/typevet/issues/332#issuecomment-5917158229)
(item 3) said that a mismatch makes the run a bug. The #336 run plan said that
a mismatch is reported, not averaged away. This page reports it for that
reason.

## Image reuse and an open possibility

Each question is a separate request that sends the images again. The server
caches do the image reuse. At concurrency 16, most multimodal cache lookups hit.

A future code change could ask all questions of a judgment in one request.
That is an open possibility only. No receipt measures it.

The [#342 design](https://github.com/Alberto-Codes/typevet/issues/342#issuecomment-5919250989)
looked at that possibility and closed it for now. One prompt with several
scoring positions changes what the model conditions on, so its probabilities
would need new calibration evidence, and llama.cpp gives no prompt-side
logprobs. One shared prefix with batched continuations works on llama.cpp
only. The one small change left is to send the questions of one judgment at
the same time instead of one after the other. That change belongs to
[#121](https://github.com/Alberto-Codes/typevet/issues/121). It needs a new
concurrency 32 receipt before any claim.

## Cost

The pod ran from 19:13:28Z to about 19:31:50Z, about 18.4 minutes. At $3.49
per hour, that is about $1.07. The pod took 6.7 minutes to become ready. That
start-up time is part of the cost.

The same pod also ran the #329 Gemma held-out text receipt. The $1.07 is thus
not the cost of the image runs alone.

On this setup, 1,000 judgments at concurrency 16 in the full sets take this
much GPU time:

| Set | Seconds per 1,000 judgments | GPU cost per 1,000 judgments |
|---|---|---|
| Faces | 225 s | $0.22 |
| Checks | 188 s | $0.18 |
| Signatures | 225 s | $0.22 |

These figures exclude start-up time and idle time. They use the $3.49 hourly
price from the run plan.

## What you can take from this

- On this setup, about 16 judgments in flight gave the highest rate of the
  levels tried.
- On this setup, one H100 gave about 4.4 to 5.3 image judgments per second.
- On this setup, faces and checks gave the same verdicts at every level tried.
- Near-tied `Choice` answers can move under batching. Check them if a
  decision depends on them.

## What you cannot take from this

- **No general vLLM or GPU claim.** The run used one model, one GPU, one pin
  and one flag set.
- **No variance.** Each level ran once. The run has no repeats and no
  confidence interval.
- **No other levels.** The sweep tried 1, 4, 16 and 32 only. The best level
  can lie between them.
- **No split of the gain.** The receipts do not separate concurrency, caching
  and network.
- **No single-request latency claim.** Concurrency hides round-trip time. The
  client latency includes the network path from Arizona.
- **No other image sizes or prompts.** Other images, questions or token counts
  can give other rates.
- **No quality claim.** Throughput says nothing about whether the answers are
  correct. The set pages report accuracy.

## Receipts

| Set | Sweep receipts | Full-set receipts | 0.4.0 receipt |
|---|---|---|---|
| Faces | `evals/fixtures/lfw/receipts/face_match_vllm_throughput_c{1,4,16,32}.json` | `face_match_vllm_throughput_full_c{16,32}.json` | `face_match_vllm_receipt.json` |
| Checks | `evals/fixtures/checks/receipts/check_match_vllm_throughput_c{1,4,16,32}.json` | `check_match_vllm_throughput_full_c{16,32}.json` | `check_match_vllm_receipt.json` |
| Signatures | `evals/fixtures/cedar/receipts/signature_match_vllm_throughput_c{1,4,16,32}.json` | `signature_match_vllm_throughput_full_c{16,32}.json` | `signature_match_vllm.json` |

The design is on [#332](https://github.com/Alberto-Codes/typevet/issues/332),
with the
[client placement amendment](https://github.com/Alberto-Codes/typevet/issues/332#issuecomment-5917725324).
The text throughput run is on the
[performance page](../reference/performance.md).
