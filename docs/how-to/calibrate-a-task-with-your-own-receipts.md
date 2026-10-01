# Calibrate a task with your own receipts

Kind: how-to.

Fit a calibration map on one labelled receipt for your task. Write the map,
pin its sha256 and wrap your judgment port with it. The steps run offline. No
step calls a model. Background is in
[Post-hoc calibration of Noul probabilities](../explanation/post-hoc-calibration.md).
The work is tracked on [#352](https://github.com/Alberto-Codes/typevet/issues/352).

## Prerequisites

1. Clone the repository and run `uv sync`.
2. Have one receipt for your task with a row id, a probability and a gold label in each row.
3. Know the model id and the serving backend that produced the receipt.

A map is valid for one task, one model and one backend. The wrapper refuses
any other combination. A map does not transfer between them.

## Build the series

The writer lives in the `typevet-evals` workspace member, not in the wheel.

For a Noul question, build one binary `Series` from the receipt rows:

```python
import json
from pathlib import Path

from typevet_evals.calibration import Series

receipt = Path("receipts/fraud_messages.json")
rows = json.loads(receipt.read_text(encoding="utf-8"))["rows"]
series = Series(
    "fraud/fraud_messages",
    "receipts/fraud_messages.json",
    tuple(str(r["id"]) for r in rows),
    tuple(float(r["p_true"]) for r in rows),
    tuple(bool(r["gold"]) for r in rows),
)
```

The row keys in this example are placeholders. Use the keys of your receipt.

For a Score question, pool the levels into one series:

```python
from typevet_evals.calibration_artifact import score_level_series

series = score_level_series(
    "essays/quality",
    "receipts/essays.json",
    ids=[r["id"] for r in rows],
    probabilities=[{int(k): p for k, p in r["levels"].items()} for r in rows],
    gold=[int(r["gold_level"]) for r in rows],
)
```

Score levels are 0-based. The level keys run from `"0"` to `"n-1"`.

`score_level_series` repeats each row id once per level. The split hashes
the row id, so all levels of one row stay in one half.

## Fit and write the map

```python
from typevet_evals.calibration_artifact import (
    calibration_map_artifact,
    write_calibration_map,
)

artifact = calibration_map_artifact(
    series,
    method="isotonic",
    task_id="fraud-message",
    model="google/gemma-4-31B-it",
    backend="vllm",
    receipt=receipt,
)
digest = write_calibration_map("maps/fraud.json", artifact)
print(digest)
```

`method` is `temperature`, `platt` or `isotonic`. The fit uses the
calibration half only. The `evaluation` block holds the evaluation-half ECE,
Brier score and accuracy before and after the fit. It also holds `rule_met`,
the #343 rule result. The wrapper records a map that failed the rule. It
does not refuse it.

For a Score map, pass `levels=n`, with `n` the number of Score levels. The
artifact then carries a top-level `"levels": n` field. A Noul map has no
`levels` field. Both kinds use the schema `typevet.calibration_map/1`.

`fitted_on.receipt_sha256` is the lower-case hex sha256 of the receipt
bytes. `write_calibration_map` returns the sha256 of the bytes it wrote.

## Pin the digest

Store the printed digest next to your configuration. The reader refuses the
file when its bytes have another sha256. Write the map again after any edit,
and pin the new digest.

## Wrap the judgment port

```python
from typevet.adapters.inbound import load_calibration_map
from typevet.domain import Noul
from typevet.runtime import CalibratedJudgment

maps = {"fraud": load_calibration_map("maps/fraud.json", sha256=digest)}
port = CalibratedJudgment(inner, maps, task_id="fraud-message", backend="vllm")
response = port.judge(message, {"fraud": Noul()}, "google/gemma-4-31B-it")
```

`inner` is the judgment port that you already use. The wrapper refuses a map
for another task or backend when you construct it. It refuses a response
from another model.

For a Score question, the wrapper calibrates each level as
[Use a calibration map](../explanation/post-hoc-calibration.md#use-a-calibration-map)
describes. The wrapper refuses a Score map whose `levels` differs from the
question. It also refuses a map for a Choice question.

## Read the record

```python
record = response.calibration["fraud"]
print(record.raw, record.calibrated, record.method, record.map_sha256)
```

The answer carries the calibrated value. The record keeps the raw value next
to it. For a Score question, `raw` and `calibrated` hold the score, and
`raw_levels` and `calibrated_levels` hold the level probabilities.

## Limits

- **No Score evidence.** The #343 study fitted binary series only. No Score
  map has a held-out result. The Score path is a tested mechanism.
- **Small halves.** A receipt with 200 rows gives about 100 evaluation rows.
  A 10-bin ECE on so few rows is noisy.
- **One split.** The split is fixed by the row ids. Another split can give
  other numbers.
- **No live check.** No model call has used a written map yet.

## Related pages

- [Post-hoc calibration of Noul probabilities](../explanation/post-hoc-calibration.md)
- [Supported imports](../reference/supported-imports.md)
- [Call typevet from Python](call-typevet-from-python.md)
