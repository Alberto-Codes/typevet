[![CI](https://img.shields.io/github/actions/workflow/status/Alberto-Codes/typevet/ci.yml?branch=main&label=CI)](https://github.com/Alberto-Codes/typevet/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/Alberto-Codes/typevet/docs.yml?branch=main&label=docs)](https://alberto-codes.github.io/typevet/)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://github.com/Alberto-Codes/typevet/blob/main/pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![docs vetted](https://img.shields.io/badge/docs%20vetted-docvet-purple)](https://github.com/Alberto-Codes/docvet)

# typevet

Kind: landing page (the project overview; the one page that mixes kinds).

typevet is a Python library that asks a model typed questions and returns typed answers.
The three question types are `Noul` (yes or no), `Choice` (one label) and `Score` (one rubric level).
typevet computes each answer from the model's next-token probabilities, read before sampling.
typevet also returns JSON objects that pass a JSON Schema you supply, or it raises an error.
The receipts cover Gemma 4 31B on llama.cpp for local work and on vLLM for hosting.

Read the documentation at <https://alberto-codes.github.io/typevet/>.

## Status

- typevet is pre-1.0. The package version is `0.1.0`.
- typevet is not on PyPI yet.
  Build a wheel from a checkout and install it: see
  [Install typevet](docs/how-to/install.md).
- typevet requires Python 3.12 or later.
- Each backend has one tested model pin.
  The receipts give the full pin and its limits.

| Backend | Tested pin | Receipt |
|---|---|---|
| vLLM | `vllm/vllm-openai:v0.30.0`, BF16 `google/gemma-4-31B-it`, one H100 80 GB | [#170](https://github.com/Alberto-Codes/typevet/issues/170#issuecomment-5884707915) |
| llama.cpp | Build `b11223-4da633776`, local alias `gemma-4-31b-kv9-q4km-mm` | [#203](https://github.com/Alberto-Codes/typevet/issues/203#issuecomment-5882379255) |
| llama.cpp grammar | Build `b11243-fc07d781e`, Gemma 4 31B QAT Q4_0 GGUF | [#129](https://github.com/Alberto-Codes/typevet/issues/129#issuecomment-5892208050) |

The H100 throughput measurement is in progress
([#236](https://github.com/Alberto-Codes/typevet/issues/236)).
A valid structure does not prove accuracy or calibration.
The receipts are small samples.

## Quickstart

Get one offline typed judgment from a scripted fake. This step needs no model.

```bash
uv sync
uv run python -c "
from typevet.domain import Noul
from typevet.judge import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake
fake = ScriptedScoringFake(logprobs={'True': -0.2, 'False': -1.0})
port = ScoringJudgmentAdapter(fake, tokenize_content=lambda t: (ord(t[0]),))
r = port.judge('text', {'q': Noul(instructions='Ok?', criteria={'true': 'Y', 'false': 'N'})}, 'fake')
print('noul', r.nouls['q'].noul)
"
```

The command prints the probability of yes, near 0.69.
The [offline tutorial](docs/tutorials/first-typed-judgment-offline.md) explains each step.
Then connect a model server:

- To host typevet, follow [Serve typevet on vLLM](docs/how-to/serve-typevet-on-vllm.md).
- To run typevet locally, follow [Run Gemma 4 on llama.cpp](docs/how-to/run-gemma4-llamacpp.md).
- To call typevet from code, follow [Call typevet from Python](docs/how-to/call-typevet-from-python.md).

## Learn more

- [How typevet works with Gemma 4](docs/explanation/how-typevet-works-with-gemma-4.md)
  explains the scoring path, the two backends and the receipts.
- [Gemma 4 multimodal judgments](docs/explanation/gemma-4-multimodal-judgments.md)
  explains how images reach each backend, and the limits.
- [Native typed judgments](docs/explanation/native-typed-judgments.md) states the scope and the limitations.
- [The documentation index](docs/README.md) lists every page and its kind.

[TypeLLM](https://github.com/TypeLLM/TypeLLM) is a research reference for the decision model.
It is not a runtime dependency.

## For contributors

Read [CLAUDE.md](CLAUDE.md) first.
It states the gates, the issue workflow and the rules for agents and people.

```bash
uv sync
uv run pre-commit install -t pre-commit -t pre-push -t commit-msg
uv run pytest -q
```

The default test run skips live tests.
Pull requests and pushes to `main` run the hook stages in
[the CI workflow](.github/workflows/ci.yml).
[The writing system](docs/reference/writing-system.md) and
[the commit rules](docs/reference/commits.md) apply to every change.
