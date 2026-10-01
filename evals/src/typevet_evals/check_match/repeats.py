"""Repeat-run variance of one fixed-weight checks slice (#357).

The study runs the same slice several times with the same weights, seed,
prompts and code paths. ``repeat_summary`` reads the receipts and applies
the pre-registered statistics. A run whose code-path digests differ from
the majority leaves the statistics and is reported by digest name only.

Over the remaining runs the summary gives the spread of accuracy, false
clear rate and each ``Noul`` ECE, a seeded bootstrap interval of pooled
accuracy, the per-case verdict agreement and the largest difference of
any recorded probability. That last number proves bit-equal runs.

Attributes:
    REPEAT_RESAMPLES (int): Bootstrap resamples over pooled case rows.
    REPEAT_SEED (int): Seed of ``resample_indices``.
    REPEAT_LEVEL (float): Level of the percentile interval.
    STABILITY_RANGE_LIMIT (float): A range below this value is stable.

Examples:
    ```python
    from typevet_evals.check_match import stability_verdict

    assert stability_verdict(0.0, {"payee_matches": 0.0})
    ```

See Also:
    - [typevet_evals.check_match.runner][]: the receipt this module reads
    - [typevet_evals.check_match.orderings][]: another receipt analysis
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from typevet_evals.wording.metrics import percentile_interval, resample_indices

REPEAT_RESAMPLES: Final[int] = 10_000
REPEAT_SEED: Final[int] = 0
REPEAT_LEVEL: Final[float] = 0.95
STABILITY_RANGE_LIMIT: Final[float] = 0.01

_PROBABILITY_MAPS: Final[tuple[str, ...]] = (
    "verdict_probabilities",
    "legibility_probabilities",
)
_PROBABILITY_SCALARS: Final[tuple[str, ...]] = (
    "payee_confidence",
    "amounts_confidence",
)


def _digests(receipt: Mapping[str, Any]) -> dict[str, str]:
    return dict(receipt["identity"]["code_path_digests"])


def digest_outliers(receipts: Sequence[Mapping[str, Any]]) -> dict[int, list[str]]:
    """Return the runs whose code-path digests differ from the majority.

    The majority is the most common digest set; a tie goes to the earliest
    run. The result names digest keys only, never digest values.

    Args:
        receipts: Receipt bodies in run order.

    Returns:
        Run index to the sorted names of the differing digests.
    """
    keys = [tuple(sorted(_digests(r).items())) for r in receipts]
    if not keys:
        return {}
    reference = dict(Counter(keys).most_common(1)[0][0])
    outliers: dict[int, list[str]] = {}
    for index, receipt in enumerate(receipts):
        digests = _digests(receipt)
        names = set(digests) | set(reference)
        differing = sorted(n for n in names if digests.get(n) != reference.get(n))
        if differing:
            outliers[index] = differing
    return outliers


def _cases(receipts: Sequence[Mapping[str, Any]]) -> list[dict[str, Mapping]]:
    tables = [{c["case_id"]: c for c in r["cases"]} for r in receipts]
    if any(set(t) != set(tables[0]) for t in tables):
        msg = "case ids differ between the receipts"
        raise ValueError(msg)
    return tables


def case_agreement(receipts: Sequence[Mapping[str, Any]]) -> float:
    """Return the share of cases whose typed verdict is equal in every run.

    Args:
        receipts: Receipt bodies with the same case ids.

    Returns:
        The agreement share, 0 to 1.

    Raises:
        ValueError: When the receipts do not share their case ids.
    """
    tables = _cases(receipts)
    same = sum(len({t[cid]["verdict"] for t in tables}) == 1 for cid in tables[0])
    return same / len(tables[0])


def _probabilities(case: Mapping[str, Any]) -> dict[str, float]:
    values = {name: float(case[name]) for name in _PROBABILITY_SCALARS}
    for field in _PROBABILITY_MAPS:
        for label, value in case[field].items():
            values[f"{field}.{label}"] = float(value)
    return values


def max_probability_difference(receipts: Sequence[Mapping[str, Any]]) -> float:
    """Return the largest absolute difference of one recorded probability.

    Each case contributes its verdict and legibility probabilities and both
    ``Noul`` confidences. The difference is the max minus the min over runs.

    Args:
        receipts: Receipt bodies with the same case ids.

    Returns:
        The largest difference; 0.0 means bit-equal probabilities.

    Raises:
        ValueError: When the receipts do not share their case ids.
    """
    tables = _cases(receipts)
    largest = 0.0
    for cid in tables[0]:
        rows = [_probabilities(t[cid]) for t in tables]
        for name in rows[0]:
            values = [row[name] for row in rows]
            largest = max(largest, max(values) - min(values))
    return largest


def pooled_accuracy_interval(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the bootstrap percentile interval of pooled accuracy.

    The rows are the ``correct`` flags of every case of every run.
    ``REPEAT_RESAMPLES`` resamples come from ``resample_indices`` with
    ``REPEAT_SEED``.

    Args:
        receipts: Receipt bodies.

    Returns:
        ``low``, ``high``, ``level``, ``resamples``, ``seed`` and ``rows``.
    """
    rows = [int(bool(c["correct"])) for r in receipts for c in r["cases"]]
    count = len(rows)
    shares = [
        math.fsum(
            rows[i] for i in resample_indices(count, seed=REPEAT_SEED, resample=b)
        )
        / count
        for b in range(REPEAT_RESAMPLES)
    ]
    interval = percentile_interval(shares, level=REPEAT_LEVEL)
    return {
        "low": interval.low,
        "high": interval.high,
        "level": REPEAT_LEVEL,
        "resamples": REPEAT_RESAMPLES,
        "seed": REPEAT_SEED,
        "rows": count,
    }


def stability_verdict(accuracy_range: float, ece_ranges: Mapping[str, float]) -> bool:
    """Apply the pre-registered stability rule.

    Args:
        accuracy_range: Max minus min accuracy over the runs.
        ece_ranges: Max minus min ECE over the runs, per ``Noul``.

    Returns:
        True when every range is below ``STABILITY_RANGE_LIMIT``.
    """
    ranges = [accuracy_range, *ece_ranges.values()]
    return all(value < STABILITY_RANGE_LIMIT for value in ranges)


def _spread(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def _run_row(name: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    metrics = receipt["metrics"]
    return {
        "run": name,
        "commit": receipt["identity"]["baseline_commit"],
        "accuracy": metrics["accuracy"],
        "false_clear_rate": metrics["false_clear_rate"],
        "ece": {noul: v["ece"] for noul, v in metrics["nouls"].items()},
        "wall_seconds": receipt["wall_seconds"],
    }


def _statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nouls = list(rows[0]["ece"])
    return {
        "accuracy": _spread([r["accuracy"] for r in rows]),
        "false_clear_rate": _spread([r["false_clear_rate"] for r in rows]),
        "ece": {n: _spread([r["ece"][n] for r in rows]) for n in nouls},
    }


def repeat_summary(paths: Sequence[Path]) -> dict[str, Any]:
    """Read repeat receipts and apply the pre-registered statistics (#357).

    Runs with odd code-path digests stay in ``runs`` with ``included``
    false. Each run row also gives its agreement and largest probability
    difference against the first included run.

    Args:
        paths: Receipt files in run order; each stem names its run.

    Returns:
        ``runs``, ``included``, ``excluded``, ``statistics``,
        ``accuracy_interval``, ``agreement``, ``max_abs_difference`` and
        ``stable``.

    Raises:
        ValueError: When no receipt is given or the case ids differ.
    """
    if not paths:
        msg = "repeat_summary needs at least one receipt"
        raise ValueError(msg)
    receipts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    names = [Path(p).stem for p in paths]
    outliers = digest_outliers(receipts)
    kept = [r for i, r in enumerate(receipts) if i not in outliers]
    runs = [_run_row(n, r) for n, r in zip(names, receipts, strict=True)]
    for index, (row, receipt) in enumerate(zip(runs, receipts, strict=True)):
        row["included"] = index not in outliers
        row["agreement"] = case_agreement([kept[0], receipt])
        row["max_abs_difference"] = max_probability_difference([kept[0], receipt])
    included = [row for row in runs if row["included"]]
    stats = _statistics(included)
    ece_ranges = {n: s["range"] for n, s in stats["ece"].items()}
    return {
        "runs": runs,
        "included": [row["run"] for row in included],
        "excluded": [
            {"run": names[i], "differing_digests": keys} for i, keys in outliers.items()
        ],
        "statistics": stats,
        "accuracy_interval": pooled_accuracy_interval(kept),
        "agreement": case_agreement(kept),
        "max_abs_difference": max_probability_difference(kept),
        "stable": stability_verdict(stats["accuracy"]["range"], ece_ranges),
    }
