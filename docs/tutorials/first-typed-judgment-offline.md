# First typed judgment offline

Kind: tutorial.

In this tutorial, you install typevet and ask one judgment call three typed
questions. You get back a **Noul** yes-probability, a **Choice** label and a
**Score** level. Everything runs on your machine. A scripted fake stands in
for the model, so you need no GPU, no server and no network after install.

You build one Python file, `first_typed_judgment.py`, one step at a time. Add
each code block to the end of that file.

## Step 1 — Install typevet

You need Python 3.12 or newer. Create a new directory and a virtual environment.
Then install typevet from PyPI:

```bash
mkdir typevet-first-run
cd typevet-first-run
python -m venv .venv
source .venv/bin/activate
pip install typevet
```

With uv, run `uv init` and then `uv add typevet` instead.

To test a commit that is not on PyPI, install from a checkout instead.
[Install typevet](../how-to/install.md#install-from-a-checkout) shows how.

## Step 2 — Import the public names

Create `first_typed_judgment.py` and add these imports:

```python
from typevet.domain import Choice, Noul, Score
from typevet.runtime import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake
```

Each name comes from a package path, not from a module inside it.
[Supported imports](../reference/supported-imports.md) lists these paths.

- `Noul`, `Choice` and `Score` are the three question types.
- `ScoringJudgmentAdapter` asks questions through a scoring backend.
- `ScriptedScoringFake` is an offline scoring backend that returns numbers you choose.

## Step 3 — Script the fake backend

A real backend returns one log-probability (logprob) for each candidate
answer. The fake returns the logprobs you give it. Add this block:

```python
fake = ScriptedScoringFake(
    logprobs={
        "True": -0.2,
        "False": -1.0,
        "fraud": -0.1,
        "billing": -2.5,
        "other": -3.0,
        "0": -2.0,
        "1": -1.0,
        "2": -0.3,
    },
)
```

The fake looks up each candidate by its label:

- The Noul candidates are `True` and `False`.
- The Choice candidates are the Choice labels: `fraud`, `billing` and `other`.
- The Score candidates are the rubric levels as text: `0`, `1` and `2`.

A higher logprob means a more likely answer. The fake raises an error when a
candidate has no logprob.

## Step 4 — Build the adapter

Add this block:

```python
port = ScoringJudgmentAdapter(
    fake,
    tokenize_content=lambda text: (ord(text[0]),),
)
```

The first argument is the scoring backend. Here it is the fake.

The adapter asks the model to answer with a control digit: `0`, `1`, `2` and
so on. `tokenize_content` turns each control digit into token ids. Each digit
must be exactly one token. A live backend uses the tokenizer of the served
model. The llama.cpp adapter calls the server `/tokenize` endpoint, for example.

The fake reads labels, not token ids. So a stub is enough here. The lambda
returns the character code of the first character: one token per digit.

## Step 5 — Ask three questions in one call

Add this block:

```python
questions = {
    "reports_unauthorized": Noul(
        instructions="Did the customer report unauthorized use?",
        criteria={"true": "Yes", "false": "No"},
    ),
    "route": Choice(
        instructions="Which team handles this message?",
        criteria={
            "fraud": "Fraud team",
            "billing": "Billing team",
            "other": "General support",
        },
    ),
    "urgency": Score(
        instructions="How urgent is this message?",
        criteria=["Low", "Medium", "High"],
    ),
}

response = port.judge(
    "I did not authorize this charge. Please block my card today.",
    questions,
    "fake-judgment",
)
```

`judge` takes three arguments:

1. `state` is the content to judge. Here it is one customer message.
2. `questions` maps a question id that you choose to a question.
3. `model` is the model id that the backend serves. The fake copies it into the response.

Each question has `instructions`, the text that the model reads. Its
`criteria` describe the answers:

- Noul `criteria` describe the `true` and `false` outcomes. They are optional.
- Choice `criteria` map each label to a description.
- Score `criteria` list the rubric levels in order, from level 0.

## Step 6 — Print the answers

Add this last block:

```python
print(f"model: {response.model}")

noul = response.nouls["reports_unauthorized"]
print(f"reports_unauthorized: noul={noul.noul:.2f}")

choice = response.choices["route"]
print(f"route: choice={choice.choice} confidence={choice.confidence:.2f}")

score = response.scores["urgency"]
print(f"urgency: score={score.score:.2f} confidence={score.confidence:.2f}")
```

Save the file and run it:

```bash
python first_typed_judgment.py
```

You see this output:

```text
model: fake-judgment
reports_unauthorized: noul=0.69
route: choice=fraud confidence=0.87
urgency: score=1.49 confidence=0.60
```

## What you just saw

The adapter asked the backend once for each question. For each question, it
turned the candidate logprobs into probabilities that add up to 1. Then it
built one typed answer:

- The **Noul** answer holds one number, `noul`: the probability of yes. The
  scripted logprobs give `True` a probability of 0.69.
- The **Choice** answer holds the most likely label (`fraud`), its
  probability as `confidence` and every label's probability in `probabilities`.
- The **Score** answer holds `score`, the expected rubric level weighted by
  probability. It is 1.49 here, between Medium and High. It is not the most
  likely level. That level is 2 (High), with a `confidence` of 0.60.

The response gives you each answer type through `nouls`, `choices` and
`scores`, keyed by question id. The numbers came from your script, not from a
model. So this run proves the wiring and the answer types, not model quality.

## Next: a real backend

A live run replaces the fake and the tokenizer stub with a served model. Your
questions carry over. First, [install typevet](../how-to/install.md) into your
own project. Then pick one backend:

- **vLLM.** [Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md) on a
  GPU host. Then follow its
  [Run one judgment call](../how-to/serve-typevet-on-vllm.md#run-one-judgment-call)
  section. There, `open_judgment()` builds the port for you. You pass
  `session.model` to `judge`. The session pins that id from `TYPEVET_VLLM__MODEL`.
- **llama.cpp.** [Run Gemma 4 on local llama.cpp](../how-to/run-gemma4-llamacpp.md)
  on your machine. Then follow
  [First typed judgment on llama.cpp](first-typed-judgment-on-llama-cpp.md).
  It asks these three questions against that server. To check more examples,
  [run a small live judgment eval](../how-to/run-a-small-live-judgment-eval.md)
  against that router. That eval runs judgments through pytest. It reads
  `TYPEVET_LLAMA__BASE_URL` and a pinned llama.cpp model id.
  For images with Gemma 4, `open_gemma_native_vision_judgment()` builds the port.
  [Connect Gemma 4 native vision judgment](../how-to/connect-gemma4-native-vision-judgment.md)
  shows how.

[Native typed judgments](../explanation/native-typed-judgments.md) explains
the product scope and its limits.
