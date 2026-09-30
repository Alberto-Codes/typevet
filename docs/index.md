# typevet

Kind: reference, the landing page. Choose a path by what you need to do.

typevet is a Python library that asks a model typed questions and returns typed answers.
The three question types are `Noul` (yes or no), `Choice` (one label) and `Score` (one rubric level).
typevet computes each answer from the model's next-token probabilities, read before sampling.
typevet also returns JSON objects that pass a JSON Schema you supply, or it raises an error.
The receipts cover Gemma 4 31B on llama.cpp for local work and on vLLM for hosting.

typevet is pre-1.0. Install it from PyPI with `pip install typevet`.
The [full index](https://github.com/Alberto-Codes/typevet/blob/main/docs/README.md) lists every page and its kind.

[Start with the offline tutorial](tutorials/first-typed-judgment-offline.md){ .md-button .md-button--primary }

## Choose a path

<div class="grid cards" markdown>

-   ### :material-power-plug-off-outline:{ .lg .middle } Run offline, with no model

    ---

    Install typevet and get typed answers from a scripted fake.

    - [Install typevet](how-to/install.md)
    - [First typed judgment offline](tutorials/first-typed-judgment-offline.md)
    - [Generate a schema-bound object offline](how-to/call-typevet-from-python.md#runnable-example-offline-fake)

-   ### :material-clipboard-check-outline:{ .lg .middle } Evaluate typevet

    ---

    Read what typevet does, what each receipt proves and where the limits are.

    - [How typevet works with Gemma 4](explanation/how-typevet-works-with-gemma-4.md)
    - [Verified evidence and inferred claims](explanation/verification.md)
    - [Limits and known gaps](explanation/limits.md)
    - [Typed-judgment release support matrix](reference/typed-judgment-release-support-matrix.md)
    - [Judgment live receipts](reference/judgment-live-receipts.md)
    - [Performance on one H100](reference/performance.md)
    - [Security](reference/security.md)
    - [Eval partner data policy](reference/eval-partner-data-policy.md)

-   ### :material-language-python:{ .lg .middle } Integrate typevet from Python

    ---

    Find the public surface, the settings and the errors.

    - [Call typevet from Python](how-to/call-typevet-from-python.md)
    - [Python API reference](reference/api/index.md)
    - [Supported imports](reference/supported-imports.md)
    - [Configuration](reference/configuration.md)
    - [Errors](reference/errors.md)
    - [Glossary](reference/glossary.md)

-   ### :material-server-outline:{ .lg .middle } Host typevet on vLLM

    ---

    Serve the tested vLLM pin on one H100, your own or rented.

    - [Serve typevet on vLLM](how-to/serve-typevet-on-vllm.md)
    - [Serve Gemma 4 31B on a rented H100](how-to/serve-gemma-4-31b-on-a-rented-h100.md)

-   ### :material-laptop:{ .lg .middle } Run on llama.cpp locally

    ---

    Run a stock `llama-server` with nested `json_schema` and opt-in live tests.

    - [Run Gemma 4 on llama.cpp](how-to/run-gemma4-llamacpp.md)
    - [First typed judgment on llama.cpp](tutorials/first-typed-judgment-on-llama-cpp.md)

-   ### :material-image-outline:{ .lg .middle } Send images with a question

    ---

    Send an image to each backend and check on llama.cpp that it changes the answer.

    - [Gemma 4 multimodal judgments](explanation/gemma-4-multimodal-judgments.md)
    - [Connect Gemma 4 native vision judgment](how-to/connect-gemma4-native-vision-judgment.md)
    - [Run the image-conditioned live smoke](how-to/run-a-multimodal-live-smoke.md)

</div>

## Contribute

- [Testing pyramid](reference/testing.md): the unit, contract and live layers.
- [Writing system](reference/writing-system.md): page kinds and prose rules.
- [Commit messages](reference/commits.md): the Conventional Commits vocabulary.
- [Contributor and agent notes](https://github.com/Alberto-Codes/typevet/blob/main/CLAUDE.md):
  the gates, the issue workflow and the non-negotiable rules.
