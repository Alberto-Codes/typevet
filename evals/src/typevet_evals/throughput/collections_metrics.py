"""vLLM ``/metrics`` deltas and client latency for the collections run ([#236][i236]).

The implementation lives in [typevet_evals.serving_metrics][] (#335), below
the evaluation families, so the image runners share it. This module keeps
the #236 names for the throughput package.

The series names come from vLLM ``v0.30.0`` ``vllm/v1/metrics/loggers.py``:
``vllm:e2e_request_latency_seconds`` (line 842),
``vllm:request_queue_time_seconds`` (line 852),
``vllm:request_prefill_time_seconds`` (line 872), ``vllm:prefix_cache_queries``
(line 590) and ``vllm:prefix_cache_hits`` (line 601). The counters are
``prometheus_client`` counters, so the text exposition adds ``_total``; the
parser also accepts the bare name. Samples with the same name and ``le`` label
are summed across the ``model_name`` and ``engine`` labels. A series that is
absent from either reading is recorded as ``unknown``. Server percentiles are
the upper edge of the first bucket that holds the nearest rank, so they are
bounds, not exact values; ``+Inf`` means the rank is above the last edge.

Attributes:
    UNKNOWN (str): Value recorded for an absent series.
    HISTOGRAMS (dict[str, str]): Receipt key to vLLM histogram name.
    PREFIX_HITS (str): vLLM prefix-cache hit counter name.
    PREFIX_QUERIES (str): vLLM prefix-cache query counter name.
    QUANTILES (tuple[tuple[str, float], ...]): Reported percentiles.

Examples:
    ```python
    from typevet_evals.throughput.collections_metrics import latency, server_delta

    assert latency([0.2, 0.1])["p50"] == 0.1
    assert server_delta(None, None)["e2e"] == "unknown"
    ```

See Also:
    - [typevet_evals.serving_metrics][]: the shared implementation
    - [typevet_evals.throughput.collections_throughput][]: the runner that reads them
    - [typevet_evals.vllm_acceptance.core][]: ``kv_cache_usage`` gauge read

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from typevet_evals import serving_metrics as _shared
from typevet_evals.serving_metrics import (
    HISTOGRAMS,
    PREFIX_HITS,
    PREFIX_QUERIES,
    QUANTILES,
    UNKNOWN,
    Snapshot,
)

__all__ = [
    "HISTOGRAMS",
    "PREFIX_HITS",
    "PREFIX_QUERIES",
    "QUANTILES",
    "UNKNOWN",
    "Snapshot",
    "histogram_delta",
    "latency",
    "nearest_rank",
    "prefix_hit_rate",
    "read_metrics",
    "server_delta",
    "snapshot",
]


def nearest_rank(ordered: Sequence[float], quantile: float) -> float | None:
    """Return the nearest-rank quantile of sorted values.

    Delegates to ``typevet_evals.serving_metrics.nearest_rank``.

    Args:
        ordered: Values in ascending order.
        quantile: Quantile in (0, 1].

    Returns:
        The value at rank ``ceil(quantile * n)``, or ``None`` when empty.
    """
    return _shared.nearest_rank(ordered, quantile)


def latency(samples: Sequence[float]) -> dict[str, Any]:
    """Summarize client seconds with nearest-rank p50, p95 and p99.

    Delegates to ``typevet_evals.serving_metrics.latency``.

    Args:
        samples: Seconds per record.

    Returns:
        ``n`` and ``p50``, ``p95``, ``p99``; each is ``None`` when empty.
    """
    return _shared.latency(samples)


def read_metrics(client: httpx.Client) -> str | None:
    """Read the ``/metrics`` text once (one metadata call).

    Delegates to ``typevet_evals.serving_metrics.read_metrics``.

    Args:
        client: Session client.

    Returns:
        The response text, or ``None`` when the request fails or is not 200.
    """
    return _shared.read_metrics(client)


def snapshot(text: str) -> Snapshot:
    """Parse Prometheus text into sums keyed by sample name and ``le``.

    Delegates to ``typevet_evals.serving_metrics.snapshot``.

    Args:
        text: ``/metrics`` exposition text.

    Returns:
        ``{(name, le or None): value}`` summed over the other labels.
    """
    return _shared.snapshot(text)


def histogram_delta(before: Snapshot, after: Snapshot, name: str) -> Any:
    """Return the change of one histogram between two readings.

    Delegates to ``typevet_evals.serving_metrics.histogram_delta``.

    Args:
        before: Reading before the level.
        after: Reading after the level.
        name: vLLM histogram name without suffix.

    Returns:
        The count, sum and bucket-bound percentiles, or ``unknown``.
    """
    return _shared.histogram_delta(before, after, name)


def prefix_hit_rate(before: Snapshot, after: Snapshot) -> float | str:
    """Return Δ prefix-cache hits over Δ prefix-cache queries.

    Delegates to ``typevet_evals.serving_metrics.prefix_hit_rate``.

    Args:
        before: Reading before the level.
        after: Reading after the level.

    Returns:
        The hit rate, or ``unknown`` when a counter is absent or no query ran.
    """
    return _shared.prefix_hit_rate(before, after)


def server_delta(before: str | None, after: str | None) -> dict[str, Any]:
    """Return the server-side histogram and prefix-cache deltas.

    Delegates to ``typevet_evals.serving_metrics.server_delta``.

    Args:
        before: ``/metrics`` text before the level, or ``None``.
        after: ``/metrics`` text after the level, or ``None``.

    Returns:
        ``e2e``, ``queue``, ``prefill`` and ``prefix_cache_hit_rate``.
    """
    return _shared.server_delta(before, after)
