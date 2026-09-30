# Two-image face matching

Kind: explanation.

This page is for an engineer who reads the face-match receipts. It explains
what typevet measured with two face images per call. It names the data and
pins, and it shows where the evidence stops. It is not a how-to. The
[LFW loader reference](../reference/eval-lfw-loader.md) names the files, the
cache and the request shape.

## What was measured

Each pair is one typed judgment. The call carries both images, in order, and
asks three questions:

| Question id | Type | Answer |
|---|---|---|
| `same_person` | `Noul` | Probability that the two faces are the same person |
| `verdict` | `Choice` | `same_person`, `different_person` or `cannot_tell` |
| `face_visibility` | `Score` | 0 to 4, how clearly image 2 shows the face |

The data is LFW View 2 (`pairs.txt`, 10 folds). The loader fetches the archive
from the scikit-learn figshare mirror and checks a pinned SHA-256 digest. The
slice holds 200 pairs: 100 same-person pairs and 100 different-person pairs,
20 per fold, drawn with seed 0. The design is in the
[#292 design comment](https://github.com/Alberto-Codes/typevet/issues/292#issuecomment-5904157545).

## The two runs

Each backend ran the slice once. There are no repeats.

| Topic | llama.cpp | vLLM |
|---|---|---|
| Receipt | [face_match_llama_cpp_receipt.json](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/lfw/receipts/face_match_llama_cpp_receipt.json) | [face_match_vllm_receipt.json](https://github.com/Alberto-Codes/typevet/blob/main/evals/fixtures/lfw/receipts/face_match_vllm_receipt.json) |
| Server | Local build `b11243-fc07d781e` | Stock vLLM v0.30.0, one H100 80 GB on RunPod |
| Model | Alias `gemma-4-31b-kv9-q4km-mm`: Gemma 4 31B, `Q2_K` GGUF | `google/gemma-4-31B-it`, revision `842da379`, BF16 |
| Context | `n_ctx` 4096 tokens | `max_model_len` 8192 tokens |
| Pairs | 200 | 200 |
| `verdict` accuracy | 0.955 (95 of 100 same, 96 of 100 different) | 0.975 (96 of 100 same, 99 of 100 different) |
| ROC-AUC | 0.9956 | 0.9993 |
| ECE | 0.0617 | 0.0236 |
| `cannot_tell` rate | 0.0 | 0.0 |
| `face_visibility`, same pairs | level 3: 5, level 4: 95 | level 3: 1, level 4: 99 |
| `face_visibility`, different pairs | level 3: 9, level 4: 91 | level 3: 3, level 4: 97 |
| Mean latency | 4.18 s per pair | 5.27 s per pair, through a remote proxy |

Two earlier llama.cpp attempts stopped on keep-alive disconnects
([#305](https://github.com/Alberto-Codes/typevet/issues/305)). The vLLM pod
cost about 1.69 USD, shared with
[#231](https://github.com/Alberto-Codes/typevet/issues/231).

The two backends ran different weights: a `Q2_K` GGUF on llama.cpp and BF16 on
vLLM. Do not credit the accuracy gap to the backend. The weights changed as
well, and one run each cannot separate the two causes.

## What the answers show

The probabilities cluster at the ends. On llama.cpp, 84 pairs had a
same-person probability below 0.1 (mean 0.0012), and 107 pairs had one of 0.9
or more (mean 0.996). Of those 107 pairs, 8 were different-person pairs.
The five same-person pairs that llama.cpp missed got the verdict
`different_person` with a verdict probability from 0.59 to 0.88.
Their same-person probability was 0.86 to 0.99.
The two answers disagree on those pairs.

The model never chose `cannot_tell` in `verdict`. On this slice, the option to
abstain was never used, so the slice says nothing about when the model
abstains.

The `face_visibility` score does not separate the classes. Almost every image
2 got level 4 in both classes. LFW images are mostly news photographs of public figures, so the faces are
almost always visible. The field needs harder images before it can show
anything.

## What the probability means

The same-person probability is model confidence. It is not a calibrated match
percentage. A value of 0.996 does not mean that 996 of 1000 such pairs show the
same person.

The ECE values are low on this slice. That result holds for 200 LFW pairs
only. It does not show that the probability is calibrated on other images,
other people or other pins.

## What this is not

This is not an identity-verification control and not a KYC control. The
evaluation has no ID document, no liveness check and no spoof check. Do not use
these results to admit or refuse a person.

## Limits of the data

- **Population.** LFW over-represents light-skinned, male, Western public
  figures photographed by news media. The results do not transfer to other
  populations.
- **Image kind.** The results do not transfer to selfies, ID-document photos
  or poor lighting.
- **Possible memorization.** The people in LFW are public figures. They may be
  in the model's training data. The model could recognize a known face instead
  of comparing two faces. This was not tested.
- **One slice, one run.** Each backend ran 200 pairs once. The slice gives no
  variance estimate.

## Face images are biometric data

A face image is biometric data under laws such as Illinois BIPA and GDPR
Article 9. BIPA is the Illinois Biometric Information Privacy Act.

The repository stores no face bytes. Receipts hold pair ids and typed answers
only. The loader fetches the images at run time into a cache outside the
repository. LFW has no formal licence. The photographers keep the image
copyright, and the dataset is for research use. The
[loader reference](../reference/eval-lfw-loader.md) states the policy for
fixtures and tests.

## Related pages

- [LFW View 2 loader and face-match request](../reference/eval-lfw-loader.md)
- [Gemma 4 multimodal judgments](gemma-4-multimodal-judgments.md)
- [Limits and known gaps](limits.md)
