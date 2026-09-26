# First typed judgment offline

Kind: tutorial.

You will obtain one **Noul** answer: a yes-probability (`noul`). You will use
an offline fake only. You will not start llama.cpp in this tutorial.

## Prerequisites

Install typevet into a virtual environment. From a checkout:

```bash
cd /path/to/typevet
uv sync
```

From a built wheel outside the tree:

```bash
uv build --wheel --out-dir /tmp/typevet-wheel
uv pip install /tmp/typevet-wheel/typevet-*.whl
```

## Step 1 — Import the public surface

typevet exposes judgment types on the domain package, the scoring adapter on
`typevet.judge`, and the offline scoring fake on `typevet.testing`.

```python
from typevet.domain.judgment_questions import Noul
from typevet.judge import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake
```

## Step 2 — Script logprobs on the fake

`ScriptedScoringFake` implements `CandidateScoringPort`. It returns the logprobs
you configure for each candidate label.

```python
fake = ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0})
```

## Step 3 — Build `ScoringJudgmentAdapter`

The adapter implements `JudgmentPort`. Pass a `tokenize_content` callable.
This tutorial uses a one-byte stub. Live llama.cpp runs use `/tokenize` instead.

```python
port = ScoringJudgmentAdapter(fake, tokenize_content=lambda text: (ord(text[0]),))
```

## Step 4 — Call `judge`

Name one question id. Gold labels never belong in `state` for eval runners.

```python
questions = {
    "reports_unauthorized": Noul(
        instructions="Did the customer report unauthorized use?",
        criteria={"true": "Yes", "false": "No"},
    ),
}
response = port.judge(
    "I did not authorize this charge.",
    questions,
    "fake-judgment",
)
answer = response.nouls["reports_unauthorized"]
print(answer.noul)
```

You should see a finite `noul` near **0.69**. A `NoulAnswer` stores only
that yes-probability. `ChoiceAnswer` and `ScoreAnswer` carry full probability
maps. For `ScoreAnswer`, the public `score` field is the probability-weighted
expected rubric level (a float), not the winning level index.

## Step 5 — Run the same script from the shell

```bash
uv run python -c "
from typevet.domain.judgment_questions import Noul
from typevet.judge import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake

fake = ScriptedScoringFake(logprobs={'True': -0.2, 'False': -1.0})
port = ScoringJudgmentAdapter(fake, tokenize_content=lambda t: (ord(t[0]),))
response = port.judge(
    'I did not authorize this charge.',
    {
        'reports_unauthorized': Noul(
            instructions='Did the customer report unauthorized use?',
            criteria={'true': 'Yes', 'false': 'No'},
        ),
    },
    'fake-judgment',
)
assert 0.6 < response.nouls['reports_unauthorized'].noul < 0.75
print('ok', response.nouls['reports_unauthorized'].noul)
"
```

## Next steps

- Add `Choice` and `Score` in one call — see
  [contract fixtures](../../tests/fixtures/judgment_scoring_contract.py).
- Run a live small eval — [Run a small live judgment eval](../how-to/run-a-small-live-judgment-eval.md).
- Read [Native typed judgments](../explanation/native-typed-judgments.md) for
  product scope and limitations.
