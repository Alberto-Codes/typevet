"""The ``server_args`` receipt block: observed cache config and stated flags ([#341][i341]).

vLLM exposes its ``CacheConfig`` on ``/metrics`` as the gauge
``vllm:cache_config_info``. The gauge value is always ``1``; each
``CacheConfig`` field is a label, for example ``block_size``,
``enable_prefix_caching`` and ``gpu_memory_utilization``. The Prometheus
logger also adds the ``engine`` label. ``cache_config`` keeps every label,
``engine`` too, so the receipt shows which engine the reading came from. The
values stay strings, as the server wrote them.

Scheduler and model flags, for example ``--max-num-seqs``,
``--max-num-batched-tokens`` and ``--logprobs-mode``, are not on
``/metrics``. The ``/server_info`` route needs ``VLLM_SERVER_DEV_MODE=1``,
which vLLM does not allow in production, so typevet does not read it. The
caller can state the full command line in ``TYPEVET_VLLM_SERVER_ARGS``. The
receipt keeps that text verbatim as ``caller_stated``, and never parses it.
Thus a reader can tell the observed values from the stated values.

Attributes:
    SERVER_ARGS_ENV (str): Variable that holds the caller-stated server
        flags.
    CACHE_CONFIG_INFO (str): vLLM gauge whose labels are the cache config.
    CACHE_CONFIG_SOURCE (str): Where ``cache_config`` comes from.

Examples:
    ```python
    from typevet_evals.throughput.server_args import cache_config, server_args_block

    text = 'vllm:cache_config_info{block_size="16",engine="0"} 1.0'
    block = server_args_block(cache_config(text), "--max-num-seqs 64")
    assert block["cache_config"] == {"block_size": "16", "engine": "0"}
    ```

See Also:
    - [typevet_evals.serving_metrics][]: the ``/metrics`` read and deltas
    - [typevet_evals.throughput.collections_throughput][]: the sweep receipt

[i341]: https://github.com/Alberto-Codes/typevet/issues/341
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

SERVER_ARGS_ENV: Final[str] = "TYPEVET_VLLM_SERVER_ARGS"
CACHE_CONFIG_INFO: Final[str] = "vllm:cache_config_info"
CACHE_CONFIG_SOURCE: Final[str] = "metrics"
_LINE: Final = re.compile(
    rf"^{re.escape(CACHE_CONFIG_INFO)}\{{(?P<labels>.*)\}}\s+\S+\s*$"
)
_LABEL: Final = re.compile(r'(?P<name>[A-Za-z_]\w*)="(?P<value>(?:[^"\\]|\\.)*)"')
_ESCAPES: Final[dict[str, str]] = {"\\\\": "\\", '\\"': '"', "\\n": "\n"}
_ESCAPE: Final = re.compile(r"\\[\\\"n]")


def _unescape(value: str) -> str:
    return _ESCAPE.sub(lambda m: _ESCAPES[m.group(0)], value)


def cache_config(text: str | None) -> dict[str, str] | None:
    """Return the labels of the first ``vllm:cache_config_info`` sample.

    Args:
        text: ``/metrics`` exposition text, or ``None`` when the read failed.

    Returns:
        Label name to label value, values kept as strings, or ``None`` when
        the text or the gauge is absent.
    """
    for line in (text or "").splitlines():
        match = _LINE.match(line)
        if match is not None:
            pairs = _LABEL.finditer(match.group("labels"))
            return {p.group("name"): _unescape(p.group("value")) for p in pairs}
    return None


def first_cache_config(texts: Iterable[str | None]) -> dict[str, str] | None:
    """Return the cache config from the first reading that holds it.

    Args:
        texts: ``/metrics`` readings in order, for example before and after
            a run; ``None`` for a failed read.

    Returns:
        The labels from ``cache_config``, or ``None`` when no reading holds
        the gauge.
    """
    for text in texts:
        labels = cache_config(text)
        if labels is not None:
            return labels
    return None


def server_args_block(
    cache: Mapping[str, str] | None, caller_stated: str | None
) -> dict[str, Any]:
    """Return the ``server_args`` receipt block.

    Args:
        cache: Labels from ``cache_config``, or ``None``.
        caller_stated: Text of ``TYPEVET_VLLM_SERVER_ARGS``, or ``None``.

    Returns:
        ``cache_config`` (the labels or ``None``), ``cache_config_source``
        (``metrics``) and ``caller_stated`` (the text or ``None``).
    """
    return {
        "cache_config": None if cache is None else dict(cache),
        "cache_config_source": CACHE_CONFIG_SOURCE,
        "caller_stated": caller_stated,
    }


def stated_server_args(environ: Mapping[str, str]) -> str | None:
    """Read the caller-stated server flags verbatim.

    Args:
        environ: Environment variables.

    Returns:
        The value of ``SERVER_ARGS_ENV`` unchanged, or ``None`` when it is
        not set.
    """
    return environ.get(SERVER_ARGS_ENV)
