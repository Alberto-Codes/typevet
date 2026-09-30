# Two-image signature comparison

Kind: explanation.

This page is for an engineer who reads the signature-match receipts. It
explains what typevet measured with two signature images per call. It names
the data and pins, and it shows where the evidence stops. It is not a how-to.
The [CEDAR loader reference](../reference/eval-cedar-loader.md) names the
files, the cache and the request shape.

The owner approved CEDAR for one experiment: how typed two-image judgments
behave. There is no customer use or support.

## What was measured

Each pair is one typed judgment. The call carries both images, in order. Image
1 is a genuine signature and image 2 is the questioned signature. The call asks
three questions:

| Question id | Type | Answer |
|---|---|---|
| `same_writer` | `Noul` | Probability that the same person wrote both signatures |
| `verdict` | `Choice` | `same_writer`, `different_writer`, `skilled_forgery_suspected` or `cannot_tell` |
| `image_quality` | `Score` | 0 to 4, how clearly image 2 shows the signature |

The data is the CEDAR offline signature set: 55 writers, each with 24 genuine
signatures and 24 skilled forgeries. The loader fetches the archive from CEDAR
and checks a pinned SHA-256 digest. The slice holds 180 pairs, 60 of each kind,
drawn with seed 0:

| Pair kind | Image 2 |
|---|---|
| `genuine_genuine` | Another genuine signature by the same writer |
| `genuine_skilled` | A skilled forgery of the same writer's signature |
| `genuine_random` | A genuine signature by another writer |

A "random forgery" is a genuine signature by a different writer. Nobody tried
to copy the reference. The design is in the
[#304 design comment](https://github.com/Alberto-Codes/typevet/issues/304#issuecomment-5911811000).

## The two runs

Each backend ran the slice once. There are no repeats.

| Topic | llama.cpp | vLLM |
|---|---|---|
| Receipt | [signature_match_llama_cpp.json](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/cedar/receipts/signature_match_llama_cpp.json) | [signature_match_vllm.json](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/cedar/receipts/signature_match_vllm.json) |
| Server | Local build `b11277-eae11d221` | Stock vLLM v0.30.0, one H100 GPU on RunPod |
| Model | Alias `gemma-4-31b-kv9-q4km-mm`: Gemma 4 31B, `Q2_K` GGUF | `google/gemma-4-31B-it`, revision `842da379`, BF16 |
| Context | `n_ctx` 4096 tokens | `max_model_len` 8192 tokens |
| Pairs | 180, no backend failure | 180, no backend failure |
| Same-writer accuracy (`verdict`) | 0.744 | 0.761 |
| ROC-AUC of `same_writer`, all pairs | 0.926 | 0.913 |
| ROC-AUC, genuine against skilled | 0.853 | 0.827 |
| ROC-AUC, genuine against random | 1.000 | 1.000 |
| ECE (10 bins) | 0.319 | 0.307 |
| Skilled false accept, `same_writer` ≥ 0.5 | 0.983 (59 of 60) | 0.900 (54 of 60) |
| Skilled false accept, verdict `same_writer` | 0.650 (39 of 60) | 0.650 (39 of 60) |
| `Noul`–`Choice` agreement | 0.85 (153 of 180) | 0.90 (162 of 180) |
| `cannot_tell` rate | 0.0 | 0.0 |
| Mean latency | 4.31 s per pair | 6.53 s per pair |

Same-writer accuracy counts `skilled_forgery_suspected` and `different_writer`
both as "different writer". A skilled forgery is a different-writer pair.

The llama.cpp receipt pins the model alias, not the file type or a GGUF hash.
The `q4km` in the alias does not describe the decoder weights. The local router
reported the file type `Q2_K - Medium` for this alias
([#319 correction](https://github.com/Alberto-Codes/typevet/issues/319#issuecomment-5914410671)).

The verdicts by pair kind:

| Pair kind | Backend | `same_writer` | `skilled_forgery_suspected` | `different_writer` | `cannot_tell` |
|---|---|---|---|---|---|
| `genuine_genuine` | llama.cpp | 53 | 5 | 2 | 0 |
| `genuine_genuine` | vLLM | 56 | 4 | 0 | 0 |
| `genuine_skilled` | llama.cpp | 39 | 16 | 5 | 0 |
| `genuine_skilled` | vLLM | 39 | 14 | 7 | 0 |
| `genuine_random` | llama.cpp | 0 | 0 | 60 | 0 |
| `genuine_random` | vLLM | 0 | 0 | 60 | 0 |

The two backends ran different weights: a `Q2_K` GGUF on llama.cpp and BF16
on vLLM. Do not credit a gap between the columns to the backend. The weights
changed as well, and one run each cannot separate the two causes.

## What the answers show

On this slice, every random forgery was rejected. All 120 random pairs, 60 per backend, got
the verdict `different_writer`. Every one had a `same_writer` probability below
0.01. The ROC-AUC against random pairs is 1.000 on both backends.

Skilled forgeries are mostly accepted. On both backends, 39 of 60 skilled
forgeries got the verdict `same_writer`. The `Noul` accepted even more of them:
59 of 60 on llama.cpp and 54 of 60 on vLLM.

The `Choice` catches about a third of skilled forgeries. It rejected 21 of 60 on
each backend. The `Noul` almost never does: at the 0.5 threshold it rejected 1
of 60 on llama.cpp and 6 of 60 on vLLM. Every skilled forgery that the `Noul`
rejected, the `Choice` rejected too.

The two answers disagree most on skilled forgeries. On llama.cpp, 20 of the 27
disagreements are skilled pairs. On vLLM, 15 of the 18 are. In those pairs the
`verdict` says "different writer" while the `same_writer` probability stays
high. Of the 21 skilled pairs that the `Choice` rejected, 16 on llama.cpp and
14 on vLLM still had a probability of 0.9 or more.

The reliability bins show why the ECE is high. On llama.cpp, the top bin
(0.9 to 1.0) held 113 pairs with a mean probability of 0.999. Only 51% of them
were same-writer pairs: 58 genuine pairs and 55 skilled forgeries. On vLLM, the
top bin held 112 pairs with a mean of 0.999, and 53% were same-writer pairs.
The `Noul` gives a skilled forgery the same near-certain value as a genuine
pair.

The model never chose `cannot_tell` in `verdict`. On this slice, the option to
abstain was never used, so the slice says nothing about when the model
abstains.

The `image_quality` score does not separate the kinds. Every genuine and
skilled image 2 got level 4 on both backends. Among random pairs, one image got
level 0 on llama.cpp, and two images got level 3 on vLLM.

## What the Noul alone would miss

A caller that reads only the `same_writer` probability and accepts at 0.5 would
accept 90% to 98% of skilled forgeries on this slice. A higher threshold does
not help much. At 0.9, the `Noul` still accepted 55 of 60 skilled forgeries on
llama.cpp and 53 of 60 on vLLM.

One reading: the probability behaves like an answer to "do these look like the
same hand?". A skilled forger tries to make that answer yes. The `Choice` has an
explicit `skilled_forgery_suspected` option. That option gives the model a
place to say that a copy is a forgery. On skilled forgeries it used that option 16 and 14 times. It
is still wrong on 39 of 60 skilled forgeries. This reading is a hypothesis. The
runs did not test it.

The `Noul` ranks some skilled forgeries below genuine pairs, with a ROC-AUC of
0.853 and 0.827. But its values for both kinds sit near 1.0, so no single
threshold separates them. At a threshold, the `Noul` separates different
writers and not much else. Do not read one number as a verdict on a signature.
Read both answers. Treat disagreement between them as a sign that the pair is
hard, not as a detector.

## What the probability means

The `same_writer` probability is model confidence. It is not a match
percentage and not a forensic score. A value of 0.999 does not mean that 999 of
1000 such pairs share a writer. On this slice, about half of such pairs did not.

The ECE values of 0.319 and 0.307 hold for 180 CEDAR pairs only. They do not
show how the probability behaves on other signatures, other writers or other
pins.

## What this is not

This is not a fraud control and not a document examination. The evaluation
has no trained examiner, no original document, no pen-pressure or stroke-order
data and no chain of custody. Do not use these results to accept or refuse a
signature, a cheque, a contract or a person.

## Limits of the data

- **Real writers.** CEDAR signatures come from real people, not from a
  generator. The results describe these 55 writers and their forgers only.
- **Writer population and scans.** This page did not verify the writers'
  languages, scripts, ages or regions from CEDAR documentation. It did not
  verify the scanner or the collection period either. Treat these as unknown.
  Do not assume the results transfer to other scripts or other scan
  conditions.
- **Skilled forgers.** This page did not verify who made the forgeries or how
  much practice they had. The forgery skill level is unknown.
- **Possible memorization.** CEDAR is public. Its images may be in the model's
  training data. This was not tested.
- **One slice, one run.** Each backend ran 180 pairs once. The slice gives no
  variance estimate. Five writers appear twice in each kind.

## Data terms and storage

CEDAR publishes no licence. The CEDAR page links the archive under "Published
Data Sets" and needs no sign-in. It states no terms of use. typevet uses the
data for research only.

The repository stores no signature bytes. Receipts hold pair ids and typed
answers only. The loader fetches the archive from CEDAR at run time into a
cache outside the repository. It refuses a file whose SHA-256 differs from the
pinned value. The [loader reference](../reference/eval-cedar-loader.md) states
the policy for fixtures and tests.

## Signatures and biometric law

Illinois BIPA is the Illinois Biometric Information Privacy Act. Its
definition of "biometric identifier" (740 ILCS 14/10) excludes "writing
samples" and "written signatures".

GDPR can still apply. Article 4(14) and Recital 51 set the test. An image is
biometric data when specific technical means process it to identify a person. A
same-writer judgment can be such processing. Article 9 can then apply to the
signature image.

This page is not legal advice. It states why the repository keeps no signature
images, not whether a use is lawful.

## Related pages

- [CEDAR signature loader and signature-match request](../reference/eval-cedar-loader.md)
- [Two-image face matching](two-image-face-matching.md)
- [Gemma 4 multimodal judgments](gemma-4-multimodal-judgments.md)
- [Limits and known gaps](limits.md)
