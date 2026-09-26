"""Contract tests: runtime Gemma native vision factory ([#174][i174], [#196][i196]).

The factory preserves client ownership on normal exit, judgment errors, and probe failures.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_runtime_gemma_vision_factory.py
    ```

See Also:
    - [typevet.runtime.llama_cpp_gemma_vision][]: factory helpers

[i174]: https://github.com/Alberto-Codes/typevet/issues/174
[i196]: https://github.com/Alberto-Codes/typevet/issues/196
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from tests.fixtures.gemma_vision_two_model_negative import OTHER_MODEL, PINNED_MODEL
from typevet.adapters.inbound.settings import LlamaSettings
from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.runtime.llama_cpp_gemma_vision import (
    open_gemma_native_vision_judgment,
    probe_gemma_native_vision_support,
)

pytestmark = pytest.mark.contract

_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_ROUTER_MARKER = "<__media_contract__>"


class _Router:
    """Offline router stub for factory contract tests.

    Attributes:
        paths (list[str]): Request paths observed in call order.
        completion_calls (int): Count of ``/completion`` requests served.

    Examples:
        ```python
        router = _Router()
        assert router.completion_calls == 0
        ```
    """

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.completion_calls = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return canned llama.cpp responses for template, props, and completion.

        Returns:
            Synthetic ``httpx.Response`` for supported routes.
        """
        path = request.url.path
        self.paths.append(path)
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": _GEMMA4_RENDERED})
        if path == "/props":
            return httpx.Response(
                200,
                json={
                    "modalities": {"vision": True, "audio": False},
                    "media_marker": _ROUTER_MARKER,
                },
            )
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            token_id = 100 + (sum(map(ord, content)) % 50)
            return httpx.Response(200, json={"tokens": [token_id]})
        if path == "/completion":
            self.completion_calls += 1
            top = [
                {"id": token_id, "logprob": -0.2 - (token_id % 5) * 0.15}
                for token_id in range(100, 180)
            ]
            return httpx.Response(
                200,
                json={
                    "completion_probabilities": [{"top_logprobs": top}],
                    "tokens_evaluated": 12,
                },
            )
        return httpx.Response(404)


def _settings() -> LlamaSettings:
    return LlamaSettings(base_url="http://offline-router", timeout=30.0)


def _mixed_questions() -> dict[str, Noul | Choice | Score]:
    return {
        "billing": Noul(instructions="Billing issue?"),
        "route": Choice(
            criteria={"billing": "Money", "technical": "Bugs"},
            instructions="Pick:",
        ),
        "quality": Score(
            criteria=["Poor", "Fair", "Good"],
            instructions="Rate:",
        ),
    }


@pytest.mark.unit
def test_probe_and_open_require_gemma4_native_turn() -> None:
    """Factory accepts Gemma 4 native template and exposes a judgment port."""
    router = _Router()
    transport = httpx.MockTransport(router.handle)
    with (
        httpx.Client(transport=transport, base_url="http://offline-router") as client,
        open_gemma_native_vision_judgment(
            settings=_settings(),
            model=PINNED_MODEL,
            http_client=client,
        ) as session,
    ):
        meta = probe_gemma_native_vision_support(
            settings=_settings(),
            model=PINNED_MODEL,
            http_client=client,
        )
        assert meta["ok"] is True
        assert meta["served"] == "native_gemma4_turn"
        assert session.served.value == "native_gemma4_turn"
        assert session.port is not None
        assert session.model == PINNED_MODEL


@pytest.mark.unit
def test_open_rejects_chatml_template() -> None:
    """Unsupported template families fail before scoring."""

    def handle(request: httpx.Request) -> httpx.Response:
        """Return vision-capable props and a ChatML template render.

        Returns:
            Synthetic ``httpx.Response`` for supported routes.
        """
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "modalities": {"vision": True, "audio": False},
                    "media_marker": _ROUTER_MARKER,
                },
            )
        if request.url.path == "/apply-template":
            return httpx.Response(
                200,
                json={"prompt": "<|im_start|>user\nhello"},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handle)
    with (
        httpx.Client(transport=transport, base_url="http://offline-router") as client,
        pytest.raises(ValueError, match="NATIVE_GEMMA4_TURN"),
    ):
        probe_gemma_native_vision_support(
            settings=_settings(),
            model=PINNED_MODEL,
            http_client=client,
        )


@pytest.mark.unit
def test_factory_port_returns_typed_noul_choice_score() -> None:
    """``session.port.judge`` returns native answer types through the factory."""
    router = _Router()
    transport = httpx.MockTransport(router.handle)
    with (
        httpx.Client(transport=transport, base_url="http://offline-router") as client,
        open_gemma_native_vision_judgment(
            settings=_settings(),
            model=PINNED_MODEL,
            http_client=client,
        ) as session,
    ):
        response = session.port.judge(
            "Charged twice.",
            _mixed_questions(),
            PINNED_MODEL,
        )
    billing = response.answers["billing"]
    route = response.answers["route"]
    quality = response.answers["quality"]
    assert isinstance(billing, NoulAnswer)
    assert isinstance(route, ChoiceAnswer)
    assert isinstance(quality, ScoreAnswer)
    assert 0.0 <= billing.noul <= 1.0
    assert route.choice in {"billing", "technical"}
    assert quality.legend == {0: "Poor", 1: "Fair", 2: "Good"}
    assert response.model == PINNED_MODEL
    assert router.completion_calls == 3


@pytest.mark.unit
def test_factory_rejects_other_model_before_tokenize() -> None:
    """Pinned factory ports reject mismatched model ids before ``/tokenize``."""
    router = _Router()
    transport = httpx.MockTransport(router.handle)
    with (
        httpx.Client(transport=transport, base_url="http://offline-router") as client,
        open_gemma_native_vision_judgment(
            settings=_settings(),
            model=PINNED_MODEL,
            http_client=client,
        ) as session,
    ):
        paths_before = list(router.paths)
        with pytest.raises(JudgmentValidationError, match="does not match pinned"):
            session.port.judge(
                "state",
                {"billing": Noul(instructions="Billing?")},
                OTHER_MODEL,
            )
        assert "/tokenize" not in router.paths[len(paths_before) :]
        assert router.completion_calls == 0


@pytest.mark.unit
def test_factory_closes_scoring_on_success_and_validation_error() -> None:
    """Factory ``finally`` closes the owned scoring adapter on success and error."""
    router = _Router()
    transport = httpx.MockTransport(router.handle)
    close_calls: list[str] = []
    real_close = LlamaCppCandidateScoringAdapter.close

    def track_close(self: LlamaCppCandidateScoringAdapter) -> None:
        """Record ``close`` and delegate to the adapter implementation."""
        close_calls.append("close")
        real_close(self)

    with (
        httpx.Client(transport=transport, base_url="http://offline-router") as client,
        patch.object(LlamaCppCandidateScoringAdapter, "close", track_close),
        open_gemma_native_vision_judgment(
            settings=_settings(),
            model=PINNED_MODEL,
            http_client=client,
        ) as session,
    ):
        session.port.judge(
            "Charged twice.",
            {"billing": Noul(instructions="Billing?")},
            PINNED_MODEL,
        )
        with pytest.raises(JudgmentValidationError):
            session.port.judge(
                "state",
                {"billing": Noul(instructions="Billing?")},
                OTHER_MODEL,
            )
    assert close_calls == ["close"]


@pytest.mark.parametrize("caller_owned", [False, True], ids=["owned", "caller-owned"])
@pytest.mark.parametrize(
    "outcome", ["success", "judgment-error", "props-error", "template-error"]
)
def test_factory_http_client_ownership(caller_owned: bool, outcome: str) -> None:
    """Factory closes owned clients and preserves caller clients across exit paths."""
    router = _Router()

    def handle(request: httpx.Request) -> httpx.Response:
        """Return router responses or fail the selected setup probe.

        Returns:
            A response that exercises the selected factory exit path.
        """
        if outcome == "props-error" and request.url.path == "/props":
            return httpx.Response(200, json={"modalities": {"vision": False}})
        if outcome == "template-error" and request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<|im_start|>user\\nhello"})
        return router.handle(request)

    client = httpx.Client(
        transport=httpx.MockTransport(handle), base_url="http://offline-router"
    )

    def exercise() -> None:
        """Run a real factory judgment and let errors escape its context."""
        with open_gemma_native_vision_judgment(
            settings=_settings(),
            model=PINNED_MODEL,
            http_client=client if caller_owned else None,
        ) as session:
            assert session.client is client
            assert not client.is_closed
            response = session.port.judge(
                "Charged twice.",
                {"billing": Noul(instructions="Billing?")},
                OTHER_MODEL if outcome == "judgment-error" else PINNED_MODEL,
            )
            assert isinstance(response.answers["billing"], NoulAnswer)

    try:
        with patch(
            "typevet.adapters.outbound.gemma_native_vision_factory.httpx.Client",
            return_value=client,
        ) as constructor:
            if outcome == "success":
                exercise()
                assert router.completion_calls == 1
            elif outcome == "judgment-error":
                with pytest.raises(
                    JudgmentValidationError, match="does not match pinned"
                ):
                    exercise()
            else:
                expected = (
                    "text-only input modalities"
                    if outcome == "props-error"
                    else "NATIVE_GEMMA4_TURN"
                )
                with pytest.raises(ValueError, match=expected):
                    exercise()
            if caller_owned:
                constructor.assert_not_called()
            else:
                constructor.assert_called_once_with(
                    base_url="http://offline-router", timeout=30.0
                )
        assert client.is_closed is not caller_owned
        if caller_owned:
            assert client.get("/still-usable").status_code == 404
        if outcome != "success":
            assert router.completion_calls == 0
    finally:
        client.close()
