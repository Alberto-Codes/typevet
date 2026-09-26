"""Confusion-based semantic metrics for CORD expense routed verdicts ([#184][i184]).

Examples:
    ```python
    from typevet.evaluation.cord_semantic_metrics import semantic_metrics

    gold = {"R01-C1": "supported", "R01-C2": "contradicted"}
    predicted = {"R01-C1": "supported", "R01-C2": "contradicted"}
    metrics = semantic_metrics(gold, predicted)
    assert metrics["accuracy"] == 1.0
    ```

See Also:
    - [typevet.evaluation.cord_semantic_acceptance][]: offline acceptance floors
    - [typevet.evaluation.datasets.cord_expense][]: gold verdict vocabulary

[i184]: https://github.com/Alberto-Codes/typevet/issues/184
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from typevet.evaluation.datasets.cord_expense import INSUFFICIENT, SUPPORTED, VERDICTS


def semantic_metrics(
    gold: Mapping[str, str], predicted: Mapping[str, str]
) -> dict[str, object]:
    """Score routed verdicts against gold verdicts for one modality.

    Args:
        gold: Gold verdict per claim id.
        predicted: Routed verdict per claim id.

    Returns:
        Accuracy, per-verdict recall, macro recall, confusion counts, the
        insufficient abstention rate and the false-supported count.
    """
    confusion = {g: dict.fromkeys(VERDICTS, 0) for g in VERDICTS}
    for claim_id, verdict in gold.items():
        confusion[verdict][predicted[claim_id]] += 1
    totals = Counter(gold.values())
    recall = {v: confusion[v][v] / totals[v] for v in VERDICTS if totals[v]}
    correct = sum(confusion[v][v] for v in VERDICTS)
    false_supported = sum(confusion[v][SUPPORTED] for v in VERDICTS if v != SUPPORTED)
    return {
        "n": len(gold),
        "accuracy": correct / len(gold),
        "recall": recall,
        "macro_recall": sum(recall.values()) / len(recall),
        "insufficient_abstention_rate": recall.get(INSUFFICIENT, 0.0),
        "false_supported": false_supported,
        "confusion_gold_by_predicted": confusion,
        "predicted_counts": dict(Counter(predicted.values())),
    }
