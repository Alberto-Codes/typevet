"""Judge image requests in slice order with an optional thread pool (#334).

``judge_in_order`` is the shared loop of the face, check and signature
runners. With ``concurrency`` 1 it judges one request at a time, as before.
With a larger value it keeps that many judgments in flight on a thread pool
over the same sync judgment port. The results keep slice order.

The first backend failure (``GenerationError``) in slice order stops the run.
No request is sent after a failure; judgments already in flight finish. The
result keeps only the requests before the failing index, so a run at any
concurrency records the same outcomes and failure as a one-at-a-time run
when the backend answers each request the same way. The failure record
counts the ``discarded`` judgments (#335): completions and failures after
the failing index that reached the server but are dropped by design.

Attributes:
    IMAGE_CONCURRENCY_ENV (str): Variable the live image runs read for the
        concurrency.

Examples:
    ```python
    from typevet_evals.face_match.pool import judge_in_order

    batch = judge_in_order(requests, lambda r: port.judge(...), concurrency=4)
    ```

See Also:
    - [typevet_evals.face_match.runner][]: the face-match runner
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Final

from typevet.domain import JudgmentResponse
from typevet.domain.errors import GenerationError

IMAGE_CONCURRENCY_ENV: Final[str] = "TYPEVET_IMAGE_CONCURRENCY"


@dataclass(frozen=True, slots=True)
class Judged[R]:
    """One judged request with its response and call time.

    Attributes:
        request (R): The request that was judged.
        response (JudgmentResponse): Typed answers from the port.
        latency_seconds (float): Wall time of the judgment call.

    Examples:
        ```python
        judged.latency_seconds
        ```
    """

    request: R
    response: JudgmentResponse
    latency_seconds: float


@dataclass(frozen=True, slots=True)
class JudgmentFailure[R]:
    """The backend failure that stopped a run.

    Attributes:
        index (int): Slice index of the failing request.
        request (R): The failing request.
        error (GenerationError): The failure the port raised.
        discarded (int): Judgments after ``index`` that finished or failed
            but are dropped by design; ``0`` for a one-at-a-time run.

    Examples:
        ```python
        failure.record("pair_id", failure.request.pair_id)
        ```
    """

    index: int
    request: R
    error: GenerationError
    discarded: int = 0

    def record(self, id_key: str, id_value: str) -> dict[str, object]:
        """Return the receipt mapping for the failure.

        Args:
            id_key: Receipt key of the request id, for example ``pair_id``.
            id_value: Id of the failing request.

        Returns:
            Index, request id, error class, message and discarded count.
        """
        return {
            "index": self.index,
            id_key: id_value,
            "error_class": type(self.error).__name__,
            "message": str(self.error),
            "discarded": self.discarded,
        }


@dataclass(frozen=True, slots=True)
class JudgedBatch[R]:
    """Judged requests in slice order and the failure that stopped them.

    Attributes:
        judged (tuple[Judged[R], ...]): Requests before the failure.
        failure (JudgmentFailure[R] | None): The first failure in slice
            order; ``None`` when every request ran.

    Examples:
        ```python
        [j.request for j in batch.judged]
        ```
    """

    judged: tuple[Judged[R], ...]
    failure: JudgmentFailure[R] | None


def image_concurrency(environ: Mapping[str, str]) -> int:
    """Read the run concurrency from ``IMAGE_CONCURRENCY_ENV``.

    Args:
        environ: Environment variables.

    Returns:
        The concurrency; ``1`` when the variable is missing or blank.

    Raises:
        ValueError: When the value is not a whole number of at least 1.
    """
    raw = environ.get(IMAGE_CONCURRENCY_ENV, "").strip() or "1"
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        msg = f"{IMAGE_CONCURRENCY_ENV} must be a whole number of at least 1"
        raise ValueError(msg)
    return value


def judge_in_order[R](
    requests: Iterable[R],
    call: Callable[[R], JudgmentResponse],
    *,
    concurrency: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> JudgedBatch[R]:
    """Judge each request once and stop at the first backend failure.

    Args:
        requests: Requests in slice order.
        call: Sends one request to the judgment port.
        concurrency: Most judgments in flight at one time. ``1`` judges one
            request at a time on the calling thread.
        clock: Monotonic clock in seconds.

    Returns:
        The judged requests before the first failure, in slice order, and
        that failure.

    Raises:
        ValueError: When ``concurrency`` is less than 1.
    """
    if concurrency < 1:
        msg = f"concurrency must be at least 1, got {concurrency}"
        raise ValueError(msg)
    if concurrency == 1:
        return _judge_sequential(requests, call, clock)
    return _judge_pooled(list(requests), call, concurrency, clock)


def _timed[R](
    call: Callable[[R], JudgmentResponse],
    request: R,
    clock: Callable[[], float],
) -> Judged[R]:
    started = clock()
    response = call(request)
    return Judged(request, response, clock() - started)


def _judge_sequential[R](
    requests: Iterable[R],
    call: Callable[[R], JudgmentResponse],
    clock: Callable[[], float],
) -> JudgedBatch[R]:
    judged: list[Judged[R]] = []
    for index, request in enumerate(requests):
        try:
            judged.append(_timed(call, request, clock))
        except GenerationError as exc:
            return JudgedBatch(tuple(judged), JudgmentFailure(index, request, exc))
    return JudgedBatch(tuple(judged), None)


def _judge_pooled[R](
    requests: Sequence[R],
    call: Callable[[R], JudgmentResponse],
    concurrency: int,
    clock: Callable[[], float],
) -> JudgedBatch[R]:
    def _judge_one(index: int) -> Judged[R]:
        return _timed(call, requests[index], clock)

    done: dict[int, Judged[R]] = {}
    failures: dict[int, GenerationError] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: dict[Future[Judged[R]], int] = {}
        next_index = 0
        while next_index < len(requests) or pending:
            while next_index < len(requests) and len(pending) < concurrency:
                pending[pool.submit(_judge_one, next_index)] = next_index
                next_index += 1
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                index = pending.pop(future)
                try:
                    done[index] = future.result()
                except GenerationError as exc:
                    failures[index] = exc
            if failures:
                # The one stop guard: send nothing after a failure.
                next_index = len(requests)
    first = min(failures, default=None)
    stop = len(requests) if first is None else first
    judged = tuple(done[i] for i in range(stop))
    if first is None:
        return JudgedBatch(judged, None)
    discarded = sum(1 for i in (*done, *failures) if i > first)
    failure = JudgmentFailure(first, requests[first], failures[first], discarded)
    return JudgedBatch(judged, failure)
