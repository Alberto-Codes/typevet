# First typed judgment on llama.cpp

Kind: tutorial.

In this tutorial, you ask a real model the three typed questions from
[First typed judgment offline](first-typed-judgment-offline.md). A Gemma 4
model on your own llama.cpp server answers them. You get back a **Noul**
yes-probability, a **Choice** label and a **Score** level, as before. This
time, the numbers come from the model, not from a script.

You build one Python file, `first_llama_judgment.py`, one step at a time. Add
each code block to the end of that file.

## Before you start

You need three things:

- Python 3.12 or newer.
- A llama.cpp server on your machine that serves a Gemma 4 text model.
  [Run Gemma 4 on local llama.cpp](../how-to/run-gemma4-llamacpp.md) shows how
  to start one. This tutorial uses the router alias
  `gemma-4-31b-24gib-kv11-decoder` at `http://127.0.0.1:8090`. The steps were
  checked with that model only.
- The model id. The `id` fields in this list are the ids that your server serves:

```bash
curl -s http://127.0.0.1:8090/v1/models
```

To host the model on a GPU server instead, see
[Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md). That how-to has
its own judgment call.

## Step 1 — Install typevet

Create a new directory and a virtual environment. Then install typevet from
PyPI:

```bash
mkdir typevet-llama-run
cd typevet-llama-run
python -m venv .venv
source .venv/bin/activate
pip install typevet
```

With uv, run `uv init` and then `uv add typevet` instead.

We checked these steps against the code that became typevet 0.2.0.
A later release can print different numbers. To use the newest code, install
from a checkout.
[Install typevet](../how-to/install.md#install-from-a-checkout) shows how.

## Step 2 — Tell typevet where the server is

typevet reads the server address and the model id from `TYPEVET_LLAMA__*`
environment variables. Set them in the shell where you run the script:

```bash
export TYPEVET_LLAMA__BASE_URL=http://127.0.0.1:8090
export TYPEVET_LLAMA__DEFAULT_MODEL=gemma-4-31b-24gib-kv11-decoder
export TYPEVET_LLAMA__TIMEOUT=600
```

Use the model id from your own server. The timeout is in seconds. A large
model can take minutes to load on the first request.
[Configuration](../reference/configuration.md) lists every setting.

## Step 3 — Import the public names

Create `first_llama_judgment.py` and add these imports:

```python
import httpx

from typevet.adapters.inbound import load_llama_settings
from typevet.adapters.outbound import LlamaCppCandidateScoringAdapter
from typevet.domain import Choice, Noul, Score
from typevet.runtime import ScoringJudgmentAdapter
```

Two names are new since the offline tutorial:

- `load_llama_settings` reads the `TYPEVET_LLAMA__*` variables.
- `LlamaCppCandidateScoringAdapter` is the llama.cpp scoring backend. It
  replaces the scripted fake.

typevet installs `httpx`. You use it here to call the server tokenizer.

## Step 4 — Connect to the server

Add this block:

```python
settings = load_llama_settings()
model = settings.default_model
if model is None:
    raise SystemExit("Set TYPEVET_LLAMA__DEFAULT_MODEL to your Gemma 4 model id.")

client = httpx.Client(base_url=settings.base_url, timeout=settings.timeout)


def tokenize_content(text: str) -> tuple[int, ...]:
    reply = client.post(
        "/tokenize",
        json={"model": model, "content": text, "add_special": False},
    )
    return tuple(reply.raise_for_status().json()["tokens"])
```

In the offline tutorial, a lambda stood in for the tokenizer. Here,
`tokenize_content` asks the server `/tokenize` endpoint for the real token
ids. The adapter uses it to check that each control digit is exactly one
token.

## Step 5 — Build the adapter

Add this block:

```python
scoring = LlamaCppCandidateScoringAdapter(
    base_url=settings.base_url,
    timeout=settings.timeout,
)
port = ScoringJudgmentAdapter(scoring, tokenize_content=tokenize_content)
```

The scoring adapter sends each question to the server `/completion` endpoint.
It reads one logprob for each candidate answer before sampling. The judgment
adapter is the same class as in the offline tutorial. Only its backend and its
tokenizer changed.

## Step 6 — Ask the same three questions

Add this block. The questions and the message are the same as in the offline
tutorial:

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
    model,
)
```

The third argument is now the model id that your server serves. The server
uses it to pick the model.

## Step 7 — Print the answers

Add this last block:

```python
print(f"model: {response.model}")

noul = response.nouls["reports_unauthorized"]
print(f"reports_unauthorized: noul={noul.noul:.2f}")

choice = response.choices["route"]
print(f"route: choice={choice.choice} confidence={choice.confidence:.2f}")
for label, probability in choice.probabilities.items():
    print(f"  {label}: {probability:.2f}")

score = response.scores["urgency"]
print(f"urgency: score={score.score:.2f} confidence={score.confidence:.2f}")

scoring.close()
client.close()
```

The last two lines close the HTTP connections. Save the file and run it:

```bash
python first_llama_judgment.py
```

We saw this output in one run. Your numbers can differ with another model
file, quantization or server build:

```text
model: gemma-4-31b-24gib-kv11-decoder
reports_unauthorized: noul=0.99
route: choice=fraud confidence=1.00
  fraud: 1.00
  billing: 0.00
  other: 0.00
urgency: score=2.00 confidence=1.00
```

That run used llama.cpp build `b11243-fc07d781e` on one local GPU. The script
took about 4.5 seconds with the model already loaded.

## What you just saw

The model read the message once for each question. typevet turned its
logprobs into the same three answer types as in the offline run:

- The **Noul** answer says the customer reported unauthorized use, with a yes
  probability of 0.99.
- The **Choice** answer routes the message to `fraud`. The model put almost
  all its probability on that label.
- The **Score** answer is 2.00, the High level. The model put almost all its
  probability on level 2. So the expected level and the most likely level
  agree here. In the offline run, they differed.

Your code did not change between the two tutorials, apart from the backend and
the tokenizer. The questions, the call and the answer types stayed the same.

One run on one message says that the wiring works against a real model. It
does not measure accuracy or calibration. For a bounded check across more
examples, see
[Run a small live judgment eval](../how-to/run-a-small-live-judgment-eval.md).

## Next

- [Serve typevet on vLLM](../how-to/serve-typevet-on-vllm.md) runs the same
  questions on a GPU host. There, `open_judgment()` builds the port for you.
- [Connect Gemma 4 native vision judgment](../how-to/connect-gemma4-native-vision-judgment.md)
  adds an image to a question. It needs a Gemma 4 model file with vision
  support. The text-only model in this tutorial does not have it.
- [Native typed judgments](../explanation/native-typed-judgments.md) explains
  the product scope and its limits.
