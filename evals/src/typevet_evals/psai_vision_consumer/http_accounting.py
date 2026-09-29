"""Predispatch accounting shared by the two frozen consumer protocols.

Examples:
    ```python
    ledger = DispatchAccounting()
    assert ledger.accounting()["dispatch_accounting_version"] == 1
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.dispatch][]: port wrappers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import httpx

from typevet_evals.psai_vision_consumer.accounting import ConsumerCallBudgetError


@dataclass(slots=True)
class DispatchAccounting:
    """Attempt reservations and independent success counters for one run.

    Attributes:
        judgment_calls (int): Successful judgment calls.
        scoring_requests (int): Successful scoring calls.
        judgment_attempts (int): Admitted judgment entries, including failures.
        scoring_attempts (int): Admitted scoring entries, including failures.
        auxiliary_metadata_http (int): Admitted metadata requests.
        auxiliary_tokenizer_http (int): Admitted tokenizer requests.
        completion_http (int): Admitted completion requests.
        failed_attempts (int): Legacy failed port or negative-probe counter.
        limits (tuple[int, ...]): Judgment, scoring, metadata, tokenizer, completion caps.

    Examples:
        ```python
        ledger = DispatchAccounting()
        ledger.reserve("completion_http", 16)
        ```
    """

    judgment_calls: int = 0
    scoring_requests: int = 0
    judgment_attempts: int = 0
    scoring_attempts: int = 0
    auxiliary_metadata_http: int = 0
    auxiliary_tokenizer_http: int = 0
    completion_http: int = 0
    failed_attempts: int = 0
    limits: ClassVar[tuple[int, ...]] = (14, 16, 3, 128, 16)

    @property
    def auxiliary_http_total(self) -> int:
        """Return observed metadata and tokenizer attempts."""
        return self.auxiliary_metadata_http + self.auxiliary_tokenizer_http

    def reserve(self, name: str, ceiling: int) -> None:
        """Reserve a slot before dispatch; failed reservations change nothing.

        Raises:
            ConsumerCallBudgetError: When the next attempt exceeds its ceiling.
        """
        count = getattr(self, name) + 1
        if count > ceiling:
            msg = f"{name} {count} exceed budget {ceiling}"
            raise ConsumerCallBudgetError(msg)
        setattr(self, name, count)

    def before_judgment_dispatch(self) -> None:
        """Reserve an admitted judgment attempt."""
        self.reserve("judgment_attempts", self.limits[0])

    def before_scoring_dispatch(self) -> None:
        """Reserve an admitted scoring attempt."""
        self.reserve("scoring_attempts", self.limits[1])

    def before_metadata_http(self) -> None:
        """Reserve an admitted metadata request."""
        self.reserve("auxiliary_metadata_http", self.limits[2])

    def before_tokenizer_http(self) -> None:
        """Reserve an admitted tokenizer request."""
        self.reserve("auxiliary_tokenizer_http", self.limits[3])

    def before_http(self, request: httpx.Request) -> None:
        """Count HTTP immediately before the client dispatches the request.

        Args:
            request: Request supplied by the client request event hook.
        """
        path = request.url.path.rstrip("/")
        if path == "/tokenize":
            self.before_tokenizer_http()
        elif path == "/completion":
            self.reserve("completion_http", self.limits[4])
        else:
            self.before_metadata_http()

    def record_judgment_success(self) -> None:
        """Record a returned typed judgment."""
        self.judgment_calls += 1

    def record_scoring_success(self) -> None:
        """Record a returned scoring result."""
        self.scoring_requests += 1

    def record_failed_attempt(self) -> None:
        """Retain a failed port entry in the legacy failure counter."""
        self.failed_attempts += 1

    def accounting(self) -> dict[str, object]:
        """Return the versioned attempt and success receipt blocks.

        Returns:
            JSON-compatible accounting fields, independent of scheduled totals.
        """
        return {
            "dispatch_accounting_version": 1,
            "dispatch_accounting": {
                "attempts": {
                    "judgment": self.judgment_attempts,
                    "scoring": self.scoring_attempts,
                    "metadata_http": self.auxiliary_metadata_http,
                    "tokenizer_http": self.auxiliary_tokenizer_http,
                    "completion_http": self.completion_http,
                },
                "successes": {
                    "judgment": self.judgment_calls,
                    "scoring": self.scoring_requests,
                },
            },
        }
