"""Bad-input demonstrations for the terminal demo.

Each demonstration is one call that typevet must refuse before any HTTP call:
an image with an unsupported mime type, an image with empty bytes and, on a
session that pins a model id, a judgment for a different model id.
``refuse_bad_inputs`` runs each call, prints the error that refused it and
counts the HTTP requests that the calls made. ``CountingTransport`` records
each HTTP request, so the count of requests before and after is the proof.

Examples:
    ```python
    log: list[str] = []
    entry = refuse_bad_inputs(bad_input_cases(None), log)
    assert entry["http_calls"] == 0
    ```

See Also:
    - examples/terminal-demo/run.py: The demo that prints this section.
    - [typevet.domain.ImageInput][]: The image value that refuses bad bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from typevet.domain import ImageInput, JudgmentValidationError, ScoringValidationError


class CountingTransport(httpx.HTTPTransport):
    """HTTP transport that records each request path in a shared log.

    Attributes:
        log (list[str]): Request lines, one for each request sent.

    Examples:
        ```python
        log: list[str] = []
        transport = CountingTransport(log)
        ```
    """

    def __init__(self, log: list[str]) -> None:
        """Store the shared log.

        Args:
            log: List that receives one line for each request.
        """
        super().__init__()
        self.log = log

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Record the request, then send it.

        Args:
            request: Outgoing request.

        Returns:
            The server response.
        """
        self.log.append(f"{request.method} {request.url.path}")
        return super().handle_request(request)


# One demonstration: (label to print, call that must raise).
BadCase = tuple[str, Callable[[], object]]


def bad_input_cases(pinned_call: Callable[[], object] | None) -> list[BadCase]:
    """Return the bad-input demonstrations to run.

    Args:
        pinned_call: A judgment for a model id that the session does not pin,
            or ``None`` when the session pins no model id (the fake backend).

    Returns:
        The two image cases, then the model id case when ``pinned_call`` is set.
    """
    cases: list[BadCase] = [
        (
            "ImageInput with mime type image/gif",
            lambda: ImageInput(data=b"GIF89a", mime_type="image/gif"),
        ),
        (
            "ImageInput with empty bytes",
            lambda: ImageInput(data=b"", mime_type="image/png"),
        ),
    ]
    if pinned_call is not None:
        cases.append(
            ("judge() with a model id the session is not pinned to", pinned_call)
        )
    return cases


def refuse_bad_inputs(cases: list[BadCase], http_log: list[str]) -> dict[str, Any]:
    """Run each bad-input case, print how it was refused and count HTTP calls.

    Args:
        cases: Demonstrations from ``bad_input_cases``.
        http_log: Shared HTTP request log.

    Returns:
        The receipt entry for the refused inputs.
    """
    n0 = len(http_log)
    rejects: list[dict[str, str]] = []
    for label, fn in cases:
        try:
            fn()
        except (ScoringValidationError, JudgmentValidationError) as exc:
            print(f"  {label}")
            print(f"    -> {type(exc).__name__}:")
            print(f"       {exc}")
            rejects.append(
                {"case": label, "error": type(exc).__name__, "message": str(exc)}
            )
        else:
            print(f"  {label}: NOT rejected (unexpected)")
            rejects.append({"case": label, "error": "none"})
    extra = len(http_log) - n0
    print(f"  HTTP calls made by these {len(cases)} attempts: {extra}")
    return {"cases": rejects, "http_calls": extra}
