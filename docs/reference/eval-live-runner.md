# Live eval runner over typed loaders

Kind: reference.

Parent: [#98](https://github.com/Alberto-Codes/typevet/issues/98),
[#95](https://github.com/Alberto-Codes/typevet/issues/95). Depends on the
stock llama.cpp path ([#97](https://github.com/Alberto-Codes/typevet/issues/97)).

This runner drives a **tiny balanced slice** of landed loaders through
``GenerationPort`` (grammar-JSON floor). It reports:

| Counter | Meaning |
|---|---|
| ``attempted`` | ``generate`` calls made |
| ``schema_valid`` | Calls that returned a value validating against the task schema |
| ``gold_match`` | Schema-valid outputs equal to loader gold |

Banking77 uses ``reports_unauthorized`` Noul agreement vs the six-intent proxy
(``fraud`` → ``true``). BoolQ uses **exact match** on the ``answer`` Noul
(``no`` / ``yes``).

This measures **structure + accuracy on gold**. It is **not** calibration,
ECE, or probability quality ([#50](https://github.com/Alberto-Codes/typevet/issues/50)).

## Supported loaders

| Dataset | Default limit | Metric label |
|---|---|---|
| ``banking77`` | 4 (balanced fraud / not_fraud) | ``noul_agreement`` |
| ``boolq`` | 4 (balanced no / yes) | ``exact_match`` |

CI and offline unit tests use vendored fixtures under ``tests/fixtures/``.
Live runs may download public Hub slices when fixture text is not injected.

## Commands

Offline wiring (default CI):

```bash
uv run pytest tests/unit/test_eval_runner.py -m unit -q
```

Opt-in live (router + model required):

```bash
export TYPEVET_LLAMA__DEFAULT_MODEL='<your-gemma-4-model-id>'
TYPEVET_LLAMA__TIMEOUT=600 \
  uv run python -m typevet.eval_runner_cli --dataset boolq --limit 2
```

Proof runs that must not silently skip (nonzero on missing config, model,
workload, or incomplete schema-valid completion; wrong gold labels stay in
metrics only):

```bash
uv run python -m typevet.eval_runner_cli --require-live --dataset boolq --limit 2
```

Run both loaders in one invocation:

```bash
uv run python -m typevet.eval_runner_cli --dataset boolq --dataset banking77 --limit 2
```

Pytest live marker (BoolQ smoke):

```bash
uv run pytest tests/live/test_eval_runner_live.py -m live -q
```

When ``TYPEVET_LLAMA__DEFAULT_MODEL`` is unset or the router is down, the CLI
prints ``skip: …`` to stderr and exits ``0``; live pytest tests **skip**.

## Library API

```python
from typevet.eval_runner import run_eval_tasks
from typevet.eval_runner_datasets import load_eval_tasks

tasks = load_eval_tasks("boolq", limit=2, boolq_jsonl_text=open("…").read())
report = run_eval_tasks(port, tasks, model="your-model-id")
```

``port`` is any ``GenerationPort`` (``FakeGenerationAdapter`` offline,
``LlamaCppGenerationAdapter`` live).

## Related pages

- [Run Gemma 4 on llama.cpp](../how-to/run-gemma4-llamacpp.md)
- [Testing pyramid](testing.md)
- [Banking77 proxy and metrics](banking77-proxy-and-metrics.md)
- [BoolQ loader and answer Noul fixture](eval-boolq-loader.md)
