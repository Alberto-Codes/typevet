"""Verify owned, caller-owned and borrowed resource behavior.

Examples:
    ```bash
    uv run pytest -q integrations/consumer_bridge/tests/unit/test_lifecycle.py
    ```

See Also:
    - [typevet_consumer_bridge.adapter][]: Public conversion and ownership.
"""

from dataclasses import replace
from unittest.mock import Mock, patch

import httpx
import pytest
from conftest import Router
from judgevet import Noul
from typevet_consumer_bridge import (
    BridgeCapabilityError,
    BridgeRequestError,
    BridgeResponseError,
    BridgeSettings,
    BridgeTransportError,
    BridgeUnavailableError,
    TypevetSystemOneAdapter,
    open_typevet_system_one,
)

from typevet.domain import JudgmentResponse
from typevet.runtime import open_gemma_native_vision_judgment

pytestmark = pytest.mark.unit
SETTINGS = BridgeSettings(
    base_url="http://offline", timeout=5.0, multimodal_model="model"
)


def require(condition: bool, message: str) -> None:
    """Fail a named ownership assertion.

    Raises:
        AssertionError: If the named contract assertion fails.
    """
    if not condition:
        raise AssertionError(message)


@pytest.mark.parametrize("caller_owned", [True, False])
@pytest.mark.parametrize(
    "failure",
    [
        "success",
        "props",
        "template",
        "completion",
        "output",
        "caller-value",
        "caller-base",
    ],
)
def test_ownership_all_exits(router: Router, caller_owned: bool, failure: str) -> None:
    """Every owned context closes exactly once while caller clients survive."""
    router.failure = failure
    transport = httpx.MockTransport(router.handle)
    client = httpx.Client(transport=transport, base_url="http://offline")
    adapters = []
    caller_error = (
        ValueError("caller private value")
        if failure == "caller-value"
        else KeyboardInterrupt()
    )
    expected = {
        "props": BridgeResponseError,
        "template": BridgeCapabilityError,
        "completion": BridgeTransportError,
        "output": BridgeResponseError,
        "caller-value": ValueError,
        "caller-base": KeyboardInterrupt,
    }

    def exercise() -> None:
        """Run the real runtime context with the selected failure path.

        Raises:
            BaseException: When the selected caller-body fault is active.
        """
        with open_typevet_system_one(
            settings=SETTINGS, http_client=client if caller_owned else None
        ) as adapter:
            adapters.append(adapter)
            if failure.startswith("caller-"):
                raise caller_error
            if failure == "output":
                with patch(
                    "typevet.adapters.outbound.judgment_scoring.ScoringJudgmentAdapter.judge",
                    return_value=JudgmentResponse(model="wrong"),
                ):
                    adapter.system_one("state", {"q": Noul()}, "model")
            else:
                adapter.system_one("state", {"q": Noul()}, "model")

    try:
        with (
            patch(
                "typevet.adapters.outbound.gemma_native_vision_factory.httpx.Client",
                return_value=client,
            ),
            patch.object(transport, "close", wraps=transport.close) as close,
        ):
            if failure == "success":
                exercise()
            else:
                with pytest.raises(expected[failure]) as caught:
                    exercise()
                if failure.startswith("caller-"):
                    require(caught.value is caller_error, "caller exception identity")
            require(
                close.call_count == (0 if caller_owned else 1),
                "exact owned cleanup count",
            )
            require(client.is_closed is not caller_owned, "client ownership")
        for adapter in adapters:
            with pytest.raises(BridgeUnavailableError):
                adapter.system_one("state", {"q": Noul()}, "model")
            adapter.close()
            with pytest.raises(BridgeUnavailableError):
                adapter.system_one("state", {"q": Noul()}, "model")
    finally:
        client.close()


def test_borrowed_session(router: Router) -> None:
    """Closing a borrowed adapter keeps both public session and client usable."""
    with (
        httpx.Client(
            transport=httpx.MockTransport(router.handle), base_url="http://offline"
        ) as client,
        open_gemma_native_vision_judgment(
            settings=SETTINGS, http_client=client
        ) as session,
    ):
        with TypevetSystemOneAdapter(session) as adapter:
            adapter.system_one("state", {"q": Noul()}, "model")
        adapter.close()
        with pytest.raises(BridgeUnavailableError):
            adapter.system_one("state", {"q": Noul()}, "model")
        with TypevetSystemOneAdapter(session) as second:
            second.system_one("state", {"q": Noul()}, "model")
        require(not client.is_closed, "borrowed client")


def test_invalid_request_does_not_call_port(router: Router) -> None:
    """Whole-request validation precedes the judgment port call."""
    with (
        httpx.Client(
            transport=httpx.MockTransport(router.handle), base_url="http://offline"
        ) as client,
        open_gemma_native_vision_judgment(
            settings=SETTINGS, http_client=client
        ) as session,
    ):
        port = Mock()
        adapter = TypevetSystemOneAdapter(replace(session, port=port))
        with pytest.raises(BridgeRequestError):
            adapter.system_one(
                "state",
                {"first": Noul(), "second": Noul(criteria={"bad": "bad"})},
                "model",
            )
        port.judge.assert_not_called()
