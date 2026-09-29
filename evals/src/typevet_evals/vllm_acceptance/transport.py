"""Call caps, the counting transport and the ``/metrics`` read for vLLM runs.

``CountingTransport`` counts model, tokenizer and metadata calls and raises
``CallCapReached`` before a request that would pass a cap. Identical
``/tokenize`` bodies are answered from a per-run memo, because the tokenizer
is fixed for one served model. ``kv_cache_usage`` parses the whole
``/metrics`` text for the ``vllm:kv_cache_usage_perc`` gauge;
``CountingTransport.wait_for`` lets a runner read it while requests are in
flight (#216). ``typevet_evals.vllm_acceptance.core`` re-exports these names
(#229).

Examples:
    ```python
    import httpx

    from typevet_evals.vllm_acceptance.transport import (
        CallCaps,
        CountingTransport,
    )

    counter = CountingTransport(httpx.HTTPTransport(), CallCaps(model=5))
    ```

See Also:
    - [typevet_evals.vllm_acceptance.core][]: the acceptance run and receipt
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Final

import httpx

_KINDS: Final[tuple[str, ...]] = ("model", "tokenizer", "metadata")
_METADATA: Final[tuple[str, ...]] = ("/version", "/v1/models", "/metrics")
_KV_GAUGE: Final = re.compile(r"(?m)^vllm:kv_cache_usage_perc(?:\{[^}\n]*\})? +(\S+)")


@dataclass(frozen=True, slots=True)
class CallCaps:
    """Hard call limits per request kind.

    Attributes:
        model (int): Chat-completions calls.
        tokenizer (int): ``/tokenize`` and ``/detokenize`` calls.
        metadata (int): ``/version``, ``/v1/models`` and ``/metrics`` calls.

    Examples:
        ```python
        CallCaps(model=5)
        ```
    """

    model: int = 100
    tokenizer: int = 128
    metadata: int = 10


class AcceptanceStoppedError(RuntimeError):
    """A stop rule fired; the run ends and still returns a receipt.

    Examples:
        ```python
        AcceptanceStoppedError("preflight denied")
        ```
    """


class CallCapReached(AcceptanceStoppedError):
    """A request would pass its call cap; the run stops before sending it.

    Examples:
        ```python
        CallCapReached("model call cap 100 reached")
        ```
    """


class CountingTransport(httpx.BaseTransport):
    """Transport wrapper that counts calls, enforces caps and memoizes tokenize.

    A condition guards the counters. Each counted request notifies it, so
    ``wait_for`` can block until a number of requests were sent.

    Attributes:
        caps (CallCaps): Limits per request kind.
        calls (dict[str, int]): Requests sent to the server per kind.
        tokenizer_memo_hits (int): ``/tokenize`` answers served from the memo.

    Examples:
        ```python
        CountingTransport(httpx.HTTPTransport(), CallCaps())
        ```
    """

    def __init__(self, inner: httpx.BaseTransport, caps: CallCaps) -> None:
        """Wrap ``inner`` with counters for ``caps`` and their condition.

        Args:
            inner: Transport that sends requests; the caller closes it.
            caps: Limits per request kind.
        """
        self._inner = inner
        self.caps = caps
        self.calls = dict.fromkeys(_KINDS, 0)
        self.tokenizer_memo_hits = 0
        self._memo: dict[bytes, bytes] = {}
        self._lock = threading.Condition()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Count and forward one request, or answer it from the tokenize memo.

        Each counted request notifies the threads that ``wait_for`` blocks.

        Args:
            request: Outgoing request.

        Returns:
            The server response, or a memoized ``/tokenize`` reply.

        Raises:
            CallCapReached: When the request kind is at its cap.
        """
        kind = _kind(request.url.path)
        key = request.url.path.encode() + request.read() if kind == "tokenizer" else b""
        with self._lock:
            if key in self._memo:
                self.tokenizer_memo_hits += 1
                return httpx.Response(200, content=self._memo[key], request=request)
            limit = getattr(self.caps, kind)
            if self.calls[kind] >= limit:
                msg = f"{kind} call cap {limit} reached"
                raise CallCapReached(msg)
            self.calls[kind] += 1
            self._lock.notify_all()
        response = self._inner.handle_request(request)
        if key and response.status_code == httpx.codes.OK:
            content = response.read()
            with self._lock:
                self._memo[key] = content
        return response

    def wait_for(self, kind: str, count: int, timeout: float) -> bool:
        """Block until ``count`` requests of ``kind`` were sent, or ``timeout``.

        Args:
            kind: Request kind: ``model``, ``tokenizer`` or ``metadata``.
            count: Number of sent requests to wait for.
            timeout: Most seconds to wait.

        Returns:
            ``True`` when the count was reached, ``False`` on timeout.
        """
        with self._lock:
            return self._lock.wait_for(lambda: self.calls[kind] >= count, timeout)

    def close(self) -> None:
        """Leave the inner transport open; the caller owns it."""


def _kind(path: str) -> str:
    if path.endswith(("/tokenize", "/detokenize")):
        return "tokenizer"
    return "metadata" if path.endswith(_METADATA) else "model"


def kv_cache_usage(client: httpx.Client) -> float | str:
    """Read the KV-cache usage gauge from ``/metrics`` (one metadata call).

    The whole ``/metrics`` text is parsed. Only the vLLM v0.30.0 gauge
    ``vllm:kv_cache_usage_perc`` (1 means 100 percent) matches (#216).

    Args:
        client: Session client.

    Returns:
        The first gauge value, or ``unknown`` when it is absent, unreadable
        or the request fails.
    """
    try:
        match = _KV_GAUGE.search(client.get("/metrics").text)
        return float(match.group(1)) if match else "unknown"
    except (httpx.HTTPError, ValueError):
        return "unknown"
