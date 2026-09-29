# typevet

Kind: reference, the landing page. Choose a path by what you need to do.

typevet is a Python library that asks a model typed questions and returns typed answers.
The three question types are `Noul` (yes or no), `Choice` (one label) and `Score` (one rubric level).
typevet computes each answer from the model's next-token probabilities, read before sampling.
typevet also returns JSON objects that pass a JSON Schema you supply, or it raises an error.
The receipts cover Gemma 4 31B on llama.cpp for local work and on vLLM for hosting.

typevet is pre-1.0 and is not on PyPI yet.
The [full index](https://github.com/Alberto-Codes/typevet/blob/main/docs/README.md) lists every page and its kind.

## Evaluate typevet

- [How typevet works with Gemma 4](explanation/how-typevet-works-with-gemma-4.md):
  the scoring path, the two backends and what each receipt proves.
- [Verified evidence and inferred claims](explanation/verification.md):
  what unit, contract and live tests each prove.
- [Typed-judgment release support matrix](reference/typed-judgment-release-support-matrix.md):
  supported APIs, tested pins, evidence commands and exclusions.
- [Judgment live receipts](reference/judgment-live-receipts.md):
  measured live predictions, latencies and known limitations.
- [Performance on one H100](reference/performance.md):
  vLLM throughput per concurrency level, calibration per set, cost and limits.
- [Eval partner data policy](reference/eval-partner-data-policy.md):
  which datasets typevet may ship in public artifacts.

## Run offline, with no model

- [Install typevet](how-to/install.md): build a wheel and install it outside a checkout.
- [First typed judgment offline](tutorials/first-typed-judgment-offline.md):
  get one `Noul` answer from a scripted fake.
- [Call typevet from Python](how-to/call-typevet-from-python.md#runnable-example-offline-fake):
  generate a schema-bound object with the fake adapter.

## Host typevet on vLLM

- [Serve typevet on vLLM](how-to/serve-typevet-on-vllm.md):
  the tested vLLM 0.30.0 pin, the server flags and the settings.
- [Serve Gemma 4 31B on a rented H100](how-to/serve-gemma-4-31b-on-a-rented-h100.md):
  rent one RunPod H100, serve the model and run one judgment.

## Run typevet on llama.cpp locally

- [Run Gemma 4 on llama.cpp](how-to/run-gemma4-llamacpp.md):
  a stock `llama-server`, nested `json_schema` and opt-in live tests.

## Send images with a question

- [Gemma 4 multimodal judgments](explanation/gemma-4-multimodal-judgments.md):
  how images reach each backend, the receipts and the limits.
- [Connect Gemma 4 native vision judgment](how-to/connect-gemma4-native-vision-judgment.md):
  open a vision session on llama.cpp.
- [Run the image-conditioned live smoke](how-to/run-a-multimodal-live-smoke.md):
  check that an image changes the answer.

## Integrate typevet from Python

- [Call typevet from Python](how-to/call-typevet-from-python.md):
  the public surface, the adapters and the async ports.
- [Python API reference](reference/api.md): the public package docstrings.
- [Supported imports](reference/supported-imports.md): the import paths to use.
- [Configuration](reference/configuration.md): the `TYPEVET_*` settings.
- [Errors](reference/errors.md): the error classes and their causes.
- [Glossary](reference/glossary.md): the one meaning of each term.

## Contribute

- [Testing pyramid](reference/testing.md): the unit, contract and live layers.
- [Writing system](reference/writing-system.md): page kinds and prose rules.
- [Commit messages](reference/commits.md): the Conventional Commits vocabulary.
- [Contributor and agent notes](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md):
  the gates, the issue workflow and the non-negotiable rules.
