"""Contract tests for the vLLM judgment factory ([#169][i169]).

Scoring replies are the redacted vLLM v0.30.0 probe files P2
(``text_three_way``) and P4 (``image_three_way``) from #168. The ``/tokenize``
replies are synthetic (``tokenize_ordinals``) and use the ordinal ids from
#168 finding 4. The request and reply shape follows vLLM v0.30.0
``vllm/entrypoints/serve/tokenize/protocol.py``.

[i169]: https://github.com/Alberto-Codes/typevet/issues/169
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

import typevet.runtime
from tests.fixtures.synthetic_images import solid_image
from typevet.adapters.inbound import backend_settings
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.vllm_scoring import ChatContentFraming
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    JudgmentValidationError,
)
from typevet.domain.judgment_questions import Choice, Noul
from typevet.ports.scoring import CandidateScoringPort
from typevet.runtime.vllm_judgment import (
    VllmJudgmentSession,
    open_vllm_judgment,
    vllm_tokenize,
)
from typevet.testing import ScriptedScoringFake

pytestmark = pytest.mark.contract

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_MODEL = "served-judge"
_HOST = "http://vllm.test:8000"
_TURN_MARKERS = ("<|turn>", "<turn|>", "<start_of_turn>", "<end_of_turn>")
_STATE = "Expense claim for this receipt: total 2,352,460."
_KEY = "sk-SENTINEL-169D"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _logprob_by_id(name: str) -> dict[int, float]:
    content = _fixture(name)["response"]["choices"][0]["logprobs"]["content"]
    return {
        int(item["token"].removeprefix("token_id:")): item["logprob"]
        for item in content[0]["top_logprobs"]
    }


class _Server:
    """MockTransport handler for ``/tokenize`` and chat completions."""

    def __init__(self, scoring: str = "text_three_way") -> None:
        self.requests: list[httpx.Request] = []
        self._scoring = _fixture(scoring)["response"]
        self._tokens = _fixture("tokenize_ordinals")["responses"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content.decode())
        if request.url.path == "/tokenize":
            return httpx.Response(200, json=self._tokens[body["prompt"]])
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=self._scoring)
        return httpx.Response(404, json={"error": "unexpected path"})

    def bodies(self, path: str) -> list[dict[str, Any]]:
        return [
            json.loads(r.content.decode()) for r in self.requests if r.url.path == path
        ]


class _Recording:
    """Scoring wrapper that records each request's candidates."""

    def __init__(self, inner: CandidateScoringPort) -> None:
        self.inner = inner
        self.calls: list[CandidateScoringRequest] = []

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        self.calls.append(request)
        return self.inner.score_candidates(request)


def _client(server: _Server, base_url: str = _HOST) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server), base_url=base_url)


def _scripted(recording: _Recording, fixture: str) -> dict[str, float]:
    by_id = _logprob_by_id(fixture)
    return {
        spec.label: by_id[spec.token_ids[0]]
        for call in recording.calls
        for spec in call.candidates
    }


def _fake_tokenize(text: str) -> tuple[int, ...]:
    return tuple(_fixture("tokenize_ordinals")["responses"][text]["tokens"])


@contextmanager
def _session(server: _Server) -> Iterator[tuple[VllmJudgmentSession, _Recording]]:
    holder: list[_Recording] = []

    def wrap(port: CandidateScoringPort) -> CandidateScoringPort:
        holder.append(_Recording(port))
        return holder[0]

    with (
        _client(server) as client,
        open_vllm_judgment(
            client=client, model=_MODEL, scoring_port_wrapper=wrap
        ) as session,
    ):
        yield session, holder[0]


def _assert_no_gemma_or_probe(server: _Server) -> None:
    for request in server.requests:
        text = request.content.decode()
        assert not any(marker in text for marker in _TURN_MARKERS)
        assert request.url.path in {"/tokenize", "/v1/chat/completions"}
        assert f"{request.url.scheme}://{request.url.netloc.decode()}" == _HOST


def test_text_noul_over_p2_equals_scripted_fake() -> None:
    server = _Server("text_three_way")
    question = {"flagged": Noul(instructions="Does the receipt support it?")}
    with _session(server) as (session, recording):
        answer = session.port.judge(_STATE, question, _MODEL)
    fake = ScriptedScoringFake(logprobs=_scripted(recording, "text_three_way"))
    fake_port = ScoringJudgmentAdapter(
        fake, tokenize_content=_fake_tokenize, framing=ChatContentFraming()
    )
    assert answer.nouls == fake_port.judge(_STATE, question, _MODEL).nouls
    [scored] = server.bodies("/v1/chat/completions")
    assert scored["messages"][0]["content"] == fake.calls[0].prefix
    assert server.bodies("/tokenize")
    _assert_no_gemma_or_probe(server)


def test_image_choice_over_p4_equals_scripted_fake() -> None:
    server = _Server("image_three_way")
    names = ("supported", "contradicted", "insufficient")
    question = {"verdict": Choice(criteria={n: f"Receipt: {n}" for n in names})}
    media = (solid_image("blue"),)
    with _session(server) as (session, recording):
        answer = session.port.judge(_STATE, question, _MODEL, media=media)
    fake = ScriptedScoringFake(logprobs=_scripted(recording, "image_three_way"))
    fake_port = ScoringJudgmentAdapter(
        fake, tokenize_content=_fake_tokenize, framing=ChatContentFraming()
    )
    expected = fake_port.judge(_STATE, question, _MODEL, media=media)
    assert answer.choices == expected.choices
    assert answer.choices["verdict"].choice == "supported"
    [scored] = server.bodies("/v1/chat/completions")
    blocks = scored["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image_url", "text"]
    assert scored["logprob_token_ids"] == [236771, 236770, 236778]
    _assert_no_gemma_or_probe(server)


def test_tokenize_hook_posts_documented_body_and_returns_tokens() -> None:
    server = _Server()
    with _client(server) as client:
        assert vllm_tokenize(client, _MODEL)("3") == (236800,)
    [request] = server.requests
    assert str(request.url) == f"{_HOST}/tokenize"
    assert json.loads(request.content) == {
        "model": _MODEL,
        "prompt": "3",
        "add_special_tokens": False,
    }


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(500, text="boom"), BackendHttpError),
        (httpx.Response(200, text="not json"), GenerationError),
        (httpx.Response(200, json={"count": 1}), GenerationError),
        (httpx.Response(200, json={"tokens": ["a"]}), GenerationError),
        (httpx.Response(200, json=[1]), GenerationError),
    ],
    ids=["status", "non_json", "no_tokens", "non_int", "non_object"],
)
def test_tokenize_hook_errors_go_through_vllm_http(
    response: httpx.Response, error: type[Exception]
) -> None:
    transport = httpx.MockTransport(lambda _request: response)
    with httpx.Client(transport=transport, base_url=_HOST) as client:
        hook = vllm_tokenize(client, _MODEL)
        with pytest.raises(error, match="vLLM"):
            hook("0")


def test_foreign_model_rejected_before_io() -> None:
    server = _Server()
    with (
        _client(server) as client,
        open_vllm_judgment(client=client, model=_MODEL) as session,
        pytest.raises(JudgmentValidationError, match="pinned"),
    ):
        session.port.judge(_STATE, {"f": Noul()}, "other-model")
    assert server.requests == []
    assert session.model == _MODEL
    assert session.client is client


def test_configured_base_url_wins_over_client_base_url() -> None:
    server = _Server()
    with (
        _client(server, base_url="http://elsewhere.test") as client,
        open_vllm_judgment(client=client, model=_MODEL, base_url=_HOST) as session,
    ):
        session.port.judge(_STATE, {"f": Noul()}, _MODEL)
    assert server.requests
    _assert_no_gemma_or_probe(server)


def test_client_without_base_url_needs_explicit_base_url() -> None:
    with (
        httpx.Client() as client,
        pytest.raises(ValueError, match="base_url"),
        open_vllm_judgment(client=client, model=_MODEL),
    ):
        pass


def test_explicit_tokenize_hook_skips_tokenize_endpoint() -> None:
    server = _Server()
    with (
        _client(server) as client,
        open_vllm_judgment(
            client=client, model=_MODEL, tokenize_content=_fake_tokenize
        ) as session,
    ):
        session.port.judge(_STATE, {"f": Noul()}, _MODEL)
    assert server.bodies("/tokenize") == []
    assert len(server.bodies("/v1/chat/completions")) == 1


def _vllm_env(**extra: str) -> dict[str, str]:
    return {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": _HOST,
        "TYPEVET_VLLM__MODEL": _MODEL,
        **extra,
    }


def test_open_judgment_selects_vllm_with_auth_header() -> None:
    server = _Server()
    env = _vllm_env(TYPEVET_VLLM__API_KEY=_KEY)
    transport = httpx.MockTransport(server)
    with backend_settings.open_judgment(env, transport=transport) as session:
        assert isinstance(session, VllmJudgmentSession)
        session.port.judge(_STATE, {"f": Noul()}, _MODEL)
    assert session.client.is_closed
    assert {r.headers["Authorization"] for r in server.requests} == {f"Bearer {_KEY}"}
    _assert_no_gemma_or_probe(server)


def test_open_judgment_masks_key_echoed_in_401() -> None:
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {request.headers['Authorization']}")

    env = _vllm_env(TYPEVET_VLLM__API_KEY=_KEY)
    transport = httpx.MockTransport(echo)
    with (
        backend_settings.open_judgment(env, transport=transport) as session,
        pytest.raises(BackendHttpError) as info,
    ):
        session.port.judge(_STATE, {"f": Noul()}, _MODEL)
    error = info.value
    assert _KEY not in str(error)
    assert _KEY not in (error.body_snippet or "")
    assert backend_settings.MASK in str(error)
    assert error.status_code == 401
    assert error.__cause__ is None
    assert error.__context__ is None


def test_open_judgment_passes_keyless_errors_unchanged() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(503, text="down"))
    with (
        backend_settings.open_judgment(_vllm_env(), transport=transport) as session,
        pytest.raises(BackendHttpError, match="down"),
    ):
        session.port.judge(_STATE, {"f": Noul()}, _MODEL)


def test_open_judgment_defaults_to_gemma_native_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    @contextmanager
    def fake_open(*, settings: Any) -> Iterator[str]:
        seen.append(settings)
        yield "llama-session"

    monkeypatch.setattr(
        backend_settings, "open_gemma_native_vision_judgment", fake_open
    )
    env = {"TYPEVET_LLAMA__MULTIMODAL_MODEL": "mm-model"}
    with backend_settings.open_judgment(env) as session:
        assert session == "llama-session"
    [settings] = seen
    assert settings.multimodal_model == "mm-model"


def test_runtime_package_re_exports_vllm_judgment() -> None:
    assert typevet.runtime.open_vllm_judgment is open_vllm_judgment
    assert typevet.runtime.vllm_tokenize is vllm_tokenize
    assert typevet.runtime.VllmJudgmentSession is VllmJudgmentSession
