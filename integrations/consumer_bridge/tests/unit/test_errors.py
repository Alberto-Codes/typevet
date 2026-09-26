"""Test safe mapped exceptions only at named bridge boundaries.

Examples:
    ```bash
    uv run pytest -q integrations/consumer_bridge/tests/unit/test_errors.py
    ```

See Also:
    - [typevet_consumer_bridge.adapter][]: Public conversion and ownership.
"""

from collections.abc import Callable
from dataclasses import replace
from json import JSONDecodeError
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from conftest import Router
from judgevet import JudgevetError, Noul
from typevet_consumer_bridge import (
    BridgeCapabilityError,
    BridgeError,
    BridgeRequestError,
    BridgeResponseError,
    BridgeSettings,
    BridgeTransportError,
    TypevetSystemOneAdapter,
    open_typevet_system_one,
)

from typevet.domain import (
    BackendHttpError,
    GemmaTemplateError,
    GenerationError,
    JudgmentValidationError,
    SchemaValidationError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
    TransportError,
)
from typevet.runtime import open_gemma_native_vision_judgment

pytestmark = pytest.mark.unit
SETTINGS = BridgeSettings(
    base_url="http://offline", timeout=5.0, multimodal_model="model"
)


def require(condition: bool, message: str) -> None:
    """Fail a named error assertion.

    Raises:
        AssertionError: If the named contract assertion fails.
    """
    if not condition:
        raise AssertionError(message)


@pytest.mark.parametrize(
    "error, expected",
    [
        (JudgmentValidationError("secret"), BridgeRequestError),
        (ScoringUnsupportedCapabilityError("secret"), BridgeCapabilityError),
        (GemmaTemplateError("secret"), BridgeCapabilityError),
        (ScoringValidationError("secret"), BridgeResponseError),
        (SchemaValidationError("secret"), BridgeResponseError),
        (TransportError("secret"), BridgeTransportError),
        (
            BackendHttpError("secret", status_code=503, body_snippet="secret"),
            BridgeTransportError,
        ),
        (httpx.ConnectError("secret"), BridgeTransportError),
    ],
)
def test_judgment_error_boundary(
    router: Router, error: Exception, expected: type[BridgeError]
) -> None:
    """Known engine errors expose fixed safe messages and hidden chaining."""
    with (
        httpx.Client(
            transport=httpx.MockTransport(router.handle), base_url="http://offline"
        ) as client,
        open_gemma_native_vision_judgment(
            settings=SETTINGS, http_client=client
        ) as session,
    ):
        port = Mock()
        port.judge.side_effect = error
        adapter = TypevetSystemOneAdapter(replace(session, port=port))
        with pytest.raises(expected) as caught:
            adapter.system_one(
                "private state",
                {"q": Noul(instructions="private instruction")},
                "model",
            )
    require(isinstance(caught.value, JudgevetError), "consumer error base")
    require("secret" not in str(caught.value), "safe message")
    require(caught.value.__suppress_context__, "suppressed known backend chain")


@pytest.mark.parametrize(
    "error",
    [
        ValueError("unexpected"),
        TypeError("unexpected"),
        RuntimeError("unexpected"),
        KeyboardInterrupt(),
    ],
)
def test_unexpected_judgment_errors(router: Router, error: BaseException) -> None:
    """Unexpected errors retain their identity."""
    with (
        httpx.Client(
            transport=httpx.MockTransport(router.handle), base_url="http://offline"
        ) as client,
        open_gemma_native_vision_judgment(
            settings=SETTINGS, http_client=client
        ) as session,
    ):
        port = Mock()
        port.judge.side_effect = error
        with pytest.raises(type(error)) as caught:
            TypevetSystemOneAdapter(replace(session, port=port)).system_one(
                "state", {"q": Noul()}, "model"
            )
    require(caught.value is error, "unchanged unexpected exception")


@pytest.mark.parametrize(
    "failure, expected",
    [
        ("props", BridgeResponseError),
        ("template", BridgeCapabilityError),
        ("completion", BridgeTransportError),
    ],
)
def test_real_factory_error_boundary(
    router: Router, failure: str, expected: type[BridgeError]
) -> None:
    """Real metadata and transport faults use the accepted bridge categories."""
    router.failure = failure
    with httpx.Client(
        transport=httpx.MockTransport(router.handle), base_url="http://offline"
    ) as client:
        with (
            pytest.raises(expected),
            open_typevet_system_one(settings=SETTINGS, http_client=client) as adapter,
        ):
            adapter.system_one("state", {"q": Noul()}, "model")
        require(not client.is_closed, "caller client on error")


@pytest.mark.parametrize(
    "values",
    [
        ("", 1.0, "model"),
        ("http://offline", 0, "model"),
        ("http://offline", float("nan"), "model"),
        ("http://offline", True, "model"),
        ("http://offline", 1.0, " "),
    ],
)
def test_invalid_settings(values: tuple) -> None:
    """Validate settings before factory entry."""
    with pytest.raises(BridgeRequestError):
        BridgeSettings(*values)


@pytest.mark.parametrize("model", ["", " ", 12])
def test_explicit_invalid_model(model: Any) -> None:
    """An explicit invalid model never falls back to configured identity.

    Raises:
        AssertionError: If invalid configuration reaches factory entry.
    """
    with (
        pytest.raises(BridgeRequestError),
        open_typevet_system_one(settings=SETTINGS, model=model),
    ):
        raise AssertionError("factory must not enter")


@pytest.mark.parametrize("callback", ["transport", "request-hook", "response-hook"])
@pytest.mark.parametrize(
    "error",
    [
        TypeError("unexpected private callback error"),
        KeyError("unexpected private callback error"),
        JSONDecodeError("private callback JSON", "private", 0),
        GenerationError("unexpected private callback error"),
        ValueError("model reports text-only input modalities"),
        RuntimeError("unexpected private callback error"),
    ],
)
def test_unexpected_entry_callback_error(callback: str, error: Exception) -> None:
    """Caller callbacks retain exception identity even for metadata-like errors."""
    router = Router()

    def fail(value: httpx.Request | httpx.Response) -> None:
        """Raise a caller-owned error on the template metadata operation.

        Raises:
            Exception: The parameterized caller error on template requests.
        """
        if value.url.path == "/apply-template":
            raise error

    def handle(request: httpx.Request) -> httpx.Response:
        """Inject a transport callback fault or return ordinary metadata.

        Returns:
            A valid metadata response when no fault is active.
        """
        if callback == "transport":
            fail(request)
        return router.handle(request)

    hooks: dict[str, list[Callable[..., Any]]] = {}
    if callback == "request-hook":
        hooks["request"] = [fail]
    elif callback == "response-hook":
        hooks["response"] = [fail]
    with httpx.Client(
        transport=httpx.MockTransport(handle),
        base_url="http://offline",
        event_hooks=hooks,
    ) as client:
        with (
            pytest.raises(type(error)) as caught,
            open_typevet_system_one(settings=SETTINGS, http_client=client),
        ):
            pytest.fail("unexpected entry")
        require(caught.value is error, "unexpected callback error unchanged")
        require(not client.is_closed, "caller client survives callback error")


@pytest.mark.parametrize(
    "failure",
    [
        "transport",
        "invalid-json",
        "text-only",
        "missing-prompt",
        "template-json",
        "template-list",
        "template-null",
        "template-number",
    ],
)
def test_entry_metadata_boundaries(failure: str) -> None:
    """Classify concrete setup failures without changing client ownership."""
    router = Router()

    def handle(request: httpx.Request) -> httpx.Response:
        """Return one concrete metadata fault through the real runtime.

        Returns:
            The selected metadata response.

        Raises:
            httpx.ConnectError: For the transport fault.
        """
        if failure == "transport":
            raise httpx.ConnectError("private URL")
        if failure == "invalid-json":
            return httpx.Response(200, text="not JSON private body")
        if failure == "text-only":
            return httpx.Response(200, json={"modalities": {"vision": False}})
        if request.url.path == "/apply-template":
            if failure == "template-json":
                return httpx.Response(200, text="not JSON private template")
            payloads = {
                "template-list": [],
                "template-null": {"prompt": None},
                "template-number": {"prompt": 12},
            }
            return httpx.Response(200, json=payloads.get(failure, {}))
        return router.handle(request)

    expected = {
        "transport": BridgeTransportError,
        "invalid-json": BridgeResponseError,
        "text-only": BridgeCapabilityError,
        "missing-prompt": BridgeResponseError,
        "template-json": BridgeResponseError,
        "template-list": BridgeResponseError,
        "template-null": BridgeResponseError,
        "template-number": BridgeResponseError,
    }
    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url="http://offline"
    ) as client:
        with (
            pytest.raises(expected[failure]),
            open_typevet_system_one(settings=SETTINGS, http_client=client),
        ):
            pytest.fail("invalid metadata reached body")
        require(not client.is_closed, "caller client survives setup faults")


def test_invalid_settings_object() -> None:
    """Reject a foreign settings object before entry IO."""
    value: Any = object()
    with pytest.raises(BridgeRequestError), open_typevet_system_one(settings=value):
        pytest.fail("invalid settings reached factory")
