"""Offline HTTP fixtures for the installed bridge contract.

See Also:
    - [typevet_consumer_bridge.adapter][]: Factory composition under test.


Examples:
    ```bash
    uv run pytest -q integrations/consumer_bridge/tests
    ```
"""

import json

import httpx
import pytest


class Router:
    """Deterministic native Gemma endpoint fixture.

    Attributes:
        requests (list[tuple[str, dict]]): Ordered paths and decoded request bodies.
        positive (bool): Whether the true control candidate is preferred.
        failure (str): Optional endpoint fault.

    Examples:
        ```python
        router = Router()
        router.positive = False
        ```
    """

    def __init__(self) -> None:
        """Start with positive Noul evidence and no faults."""
        self.requests: list[tuple[str, dict]] = []
        self.positive = True
        self.failure = ""

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Exercise the real runtime HTTP protocol with fixed log probabilities.

        Returns:
            A deterministic endpoint response.
        """
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.requests.append((path, body))
        if path == "/props":
            props = [] if self.failure == "props" else {"modalities": {"vision": True}}
            return httpx.Response(200, json=props)
        if path == "/apply-template":
            prompt = "<|turn>user\nhello<turn|>\n<|turn>model\n"
            if self.failure == "template":
                prompt = "<|im_start|>user\nhello"
            return httpx.Response(200, json={"prompt": prompt})
        if path == "/tokenize":
            token = 101 if body["content"].endswith("1") else 100
            return httpx.Response(200, json={"tokens": [token]})
        if path == "/completion":
            if self.failure == "completion":
                return httpx.Response(503, text="private backend body")
            preferred = 101 if self.positive else 100
            top = [
                {"id": token, "logprob": 0.0 if token == preferred else -5.0}
                for token in (100, 101)
            ]
            return httpx.Response(
                200,
                json={
                    "completion_probabilities": [{"top_logprobs": top}],
                },
            )
        return httpx.Response(404)


@pytest.fixture
def router() -> Router:
    """Return one isolated router fixture.

    Returns:
        A new request ledger and HTTP handler.
    """
    return Router()
