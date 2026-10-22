"""Unit tests: offline proof for the #445 judgevet live harness.

Exercises the real live pytest entry and receipt helper with
``httpx.MockTransport``. No live HTTP. Covers model selection, ChatML framing
opt-in, catalog mass, first full top-N, the empty-think control replay, strict
live gating, the four-attempt budget, secret scrubbing, compressed responses,
failure receipts and transport close.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_judgevet_live_harness.py
    ```

See Also:
    - [tests.live.judgevet_llama_receipt][]: Recording transport and receipt
    - [tests.live.test_judgevet_bridge_live][]: Live entry that uses the helper

[i445]: https://github.com/Alberto-Codes/typevet/issues/445
"""

from __future__ import annotations

import gzip
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from _pytest.outcomes import Failed, Skipped

from tests.fixtures.gemma_template_contract import (
    degraded_chatml_rendered,
    native_gemma4_rendered,
)
from tests.live import test_judgevet_bridge_live as live
from typevet.adapters.outbound.gemma import (
    CHATML_ASSISTANT_HEADER,
    CHATML_EMPTY_THINK_PREFILLS,
    CHATML_IM_END,
    CHATML_IM_START,
)
from typevet.adapters.outbound.llama_cpp.http_mapping import (
    _CLOSED_BEFORE_RESPONSE,
)

pytestmark = pytest.mark.unit

_MODEL = "qwen-chatml-offline"
_VOCAB = 97
_EMPTY_THINK = CHATML_EMPTY_THINK_PREFILLS[0]
_MARKERS = (CHATML_IM_START, CHATML_IM_END, "<think>", "</think>")
_CONTROLS = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Inside ``_VOCAB`` and not equal to the digit indexes 0/1.
_CAND_FALSE = 17
_CAND_TRUE = 23
_KEY = "sk-HARNESS-SENTINEL-445"
_DESIRED_MODEL = "desired-qwen-445"
_PINNED_GEMMA = "gemma-4-31b-24gib-kv11-decoder"


def _harness() -> Any:
    """Import the live receipt helper (present after the harness edit)."""
    return importlib.import_module("tests.live.judgevet_llama_receipt")


def _token_id(content: str) -> int:
    """Map offline tokenize content to a stable single-token id.

    Digits zero and one use non-ordinal ids inside ``_VOCAB`` so hardcoding
    candidate indexes cannot pass. Other controls keep their ordinal indexes
    so Choice/Score binding stays inside the catalog.
    """
    if content == "0":
        return _CAND_FALSE
    if content == "1":
        return _CAND_TRUE
    if content in _MARKERS:
        return 80 + _MARKERS.index(content)
    if content in _CONTROLS:
        return _CONTROLS.index(content)
    return 40 + sum(map(ord, content)) % 40


def _normalized_top(
    count: int,
    *,
    false_p: float,
    true_p: float,
    false_id: int = _CAND_FALSE,
    true_id: int = _CAND_TRUE,
) -> list[dict[str, float | int | str]]:
    """Return a full-vocabulary top_logprobs list that sums to 1.0."""
    rest = (1.0 - false_p - true_p) / (count - 2)
    entries: list[dict[str, float | int | str]] = []
    for token_id in range(count):
        if token_id == false_id:
            prob = false_p
        elif token_id == true_id:
            prob = true_p
        else:
            prob = rest
        entries.append({"id": token_id, "logprob": math.log(prob)})
    return entries


@dataclass
class _RouterOptions:
    """Optional offline router knobs for counterexample cases."""

    n_vocab: int | None = _VOCAB
    completion_top_count: int | None = None
    echo_secret_in_top: str | None = None
    control_mass_mode: str = "rise"
    gzip_completions: bool = False
    fail_first_completions: int = 0
    include_meta: bool = True


class _OfflineRouter:
    """Offline llama.cpp router for the live harness unit suite."""

    def __init__(
        self,
        template: str,
        model_id: str,
        options: _RouterOptions | None = None,
        *,
        props: dict[str, Any] | None = None,
    ) -> None:
        self.template = template
        self.model_id = model_id
        self.options = options or _RouterOptions()
        self.props = props or {
            "modalities": {"vision": False, "audio": False},
            "model_alias": model_id,
            "build_info": "offline-build-445",
        }
        self.paths: list[str] = []
        self.completion_bodies: list[dict[str, Any]] = []
        self._completion_forwards = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Serve catalog, template, tokenize and completion bodies."""
        path = request.url.path
        self.paths.append(path)
        if path == "/props":
            return httpx.Response(200, json=self.props)
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": self.template})
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            return httpx.Response(200, json={"tokens": [_token_id(content)]})
        if path == "/v1/models":
            return self._models_response()
        if path == "/completion":
            return self._completion_response(request)
        return httpx.Response(404)

    def _models_response(self) -> httpx.Response:
        entry: dict[str, Any] = {"id": self.model_id}
        if self.options.include_meta and self.options.n_vocab is not None:
            entry["meta"] = {"n_vocab": self.options.n_vocab}
        return httpx.Response(200, json={"object": "list", "data": [entry]})

    def _completion_response(self, request: httpx.Request) -> httpx.Response:
        self._completion_forwards += 1
        if self._completion_forwards <= self.options.fail_first_completions:
            raise httpx.RemoteProtocolError(_CLOSED_BEFORE_RESPONSE, request=request)
        body = json.loads(request.content.decode())
        self.completion_bodies.append(body)
        prompt = body["prompt"]
        assert isinstance(prompt, str)
        top_count = self.options.completion_top_count
        if top_count is None:
            top_count = (
                self.options.n_vocab if self.options.n_vocab is not None else _VOCAB
            )
        top = self._top_for_prompt(prompt, top_count)
        if self.options.echo_secret_in_top is not None and top:
            top[0] = {**top[0], "token": f"echo:{self.options.echo_secret_in_top}"}
        payload = {"completion_probabilities": [{"top_logprobs": top}]}
        raw = json.dumps(payload).encode()
        if self.options.gzip_completions:
            compressed = gzip.compress(raw)
            return httpx.Response(
                200,
                headers={
                    "content-encoding": "gzip",
                    "content-type": "application/json",
                    "content-length": str(len(compressed)),
                },
                content=compressed,
            )
        return httpx.Response(200, json=payload)

    def _top_for_prompt(
        self, prompt: str, top_count: int
    ) -> list[dict[str, float | int | str]]:
        scored = prompt.endswith(_EMPTY_THINK)
        flat_or_scored = scored or self.options.control_mass_mode == "flat"
        if flat_or_scored:
            return _normalized_top(top_count, false_p=0.5, true_p=0.3)
        if self.options.control_mass_mode == "drop":
            return _normalized_top(top_count, false_p=0.6, true_p=0.3)
        return _normalized_top(top_count, false_p=0.1, true_p=0.1)


class _ClosingTransport(httpx.BaseTransport):
    """Transport that records ``close`` and forwards to a handler."""

    def __init__(self, handler: Any) -> None:
        self.closed = False
        self._handler = handler

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)

    def close(self) -> None:
        self.closed = True


def _framed_environ(**extra: str) -> dict[str, str]:
    environ = {
        "TYPEVET_BACKEND": "llama_cpp",
        "TYPEVET_LLAMA__BASE_URL": "http://router.test",
        "TYPEVET_LLAMA__DEFAULT_MODEL": _MODEL,
        "TYPEVET_LLAMA__MULTIMODAL_MODEL": "unused-mm",
        "TYPEVET_LIVE_LLAMA_FRAMING": "chatml",
    }
    environ.update(extra)
    return environ


def _unframed_environ(model: str, **extra: str) -> dict[str, str]:
    environ = {
        "TYPEVET_BACKEND": "llama_cpp",
        "TYPEVET_LLAMA__BASE_URL": "http://router.test",
        "TYPEVET_LLAMA__DEFAULT_MODEL": model,
        "TYPEVET_LLAMA__MULTIMODAL_MODEL": "unused-mm",
    }
    environ.update(extra)
    return environ


def _run_helper(
    router: _OfflineRouter,
    environ: dict[str, str] | None = None,
    *,
    inner: httpx.BaseTransport | None = None,
    receipt: dict[str, object] | None = None,
) -> tuple[Any, dict[str, object]]:
    harness = _harness()
    transport = inner or httpx.MockTransport(router.handle)
    recorder = harness.RecordingTransport(transport, max_completions=4)
    kwargs: dict[str, Any] = {"transport": recorder}
    if receipt is not None:
        kwargs["receipt"] = receipt
    return recorder, harness.run_llama_bridge_receipt(
        environ or _framed_environ(), **kwargs
    )


def _install_env(monkeypatch: pytest.MonkeyPatch, environ: dict[str, str]) -> None:
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TYPEVET_GEMMA_MODEL", raising=False)


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def test_live_entry_selects_default_model_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real llama pytest entry uses DEFAULT_MODEL under MockTransport."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _DESIRED_MODEL)
    inner = _ClosingTransport(router.handle)
    recorder = _harness().RecordingTransport(inner, max_completions=4)
    monkeypatch.setattr(live, "gate_live", lambda settings: None)
    _install_env(
        monkeypatch,
        {
            **_framed_environ(TYPEVET_LLAMA__DEFAULT_MODEL=_DESIRED_MODEL),
            "TYPEVET_LIVE_RECEIPT_DIR": str(tmp_path),
        },
    )
    live._run_llama_live_case(transport=recorder)
    receipt_path = tmp_path / "judgevet_bridge_llama_cpp_receipt.json"
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["requested_model"] == _DESIRED_MODEL
    assert written["model"] == _DESIRED_MODEL
    assert written["server"]["served_template_class"] == "degraded_chatml"
    assert _PINNED_GEMMA not in json.dumps(written)
    assert recorder.completion_attempts == 4
    assert inner.closed is True


def test_framed_harness_scores_and_control_with_mock_transport() -> None:
    """Framed offline harness records mass, full top-N and a rising control."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    recorder, receipt = _run_helper(router)
    assert receipt["requested_model"] == _MODEL
    assert receipt["model"] == _MODEL
    server = _mapping(receipt["server"])
    assert server["served_template_class"] == "degraded_chatml"
    assert server["n_vocab"] == _VOCAB
    assert recorder.completion_attempts == 4
    assert recorder.first_completion_request is not None
    prompt = recorder.first_completion_request["prompt"]
    assert isinstance(prompt, str)
    assert prompt.endswith(f"{CHATML_ASSISTANT_HEADER}{_EMPTY_THINK}")
    first_top = receipt["first_top_n"]
    assert isinstance(first_top, list)
    assert len(first_top) == _VOCAB
    masses = _mapping(receipt["off_option_masses"])
    for name in ("duplicate_charge", "route", "urgency"):
        mass = masses[name]
        assert isinstance(mass, float)
        assert math.isfinite(mass) and 0.0 <= mass <= 1.0
    assert masses["duplicate_charge"] == pytest.approx(0.2, abs=1e-9)
    control = _mapping(receipt["control"])
    assert control["prompt"].endswith(CHATML_ASSISTANT_HEADER)
    assert not control["prompt"].endswith(_EMPTY_THINK)
    assert control["prompt_delta"] == _EMPTY_THINK
    assert control["off_option_mass"] > masses["duplicate_charge"]
    assert control["candidate_ids"] == {"0": _CAND_FALSE, "1": _CAND_TRUE}
    assert set(_mapping(receipt["marker_token_ids"])) == set(_MARKERS)
    assert set(_mapping(receipt["control_token_ids"])) == set(_CONTROLS)


def test_all_markers_tokenized_before_first_score() -> None:
    """ChatML markers and controls are tokenized before any ``/completion``."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    _run_helper(router)
    first_completion = router.paths.index("/completion")
    before = router.paths[:first_completion]
    assert before.count("/tokenize") >= len(_MARKERS) + len(_CONTROLS)


def test_exact_framing_suffix_on_first_completion() -> None:
    """The first scored prompt ends with the empty-think ChatML suffix."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    recorder, _receipt = _run_helper(router)
    body = recorder.first_completion_request
    assert body is not None
    assert body["prompt"].endswith(f"{CHATML_ASSISTANT_HEADER}{_EMPTY_THINK}")


def test_metadata_driven_mass_matches_catalog_vocab() -> None:
    """Off-option mass uses the catalog vocabulary size, not a hardcoded size."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(n_vocab=_VOCAB),
    )
    _recorder, receipt = _run_helper(router)
    assert _mapping(receipt["server"])["n_vocab"] == _VOCAB
    assert _mapping(receipt["off_option_masses"])["duplicate_charge"] == pytest.approx(
        0.2, abs=1e-9
    )


def test_catalog_length_mismatch_fails() -> None:
    """A top-N length that disagrees with catalog ``n_vocab`` fails."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(n_vocab=_VOCAB, completion_top_count=_VOCAB + 5),
    )
    with pytest.raises(Exception, match=r"off_option|n_vocab|expected") as caught:
        _run_helper(router)
    assert str(_VOCAB) in str(caught.value) or "off_option" in str(caught.value)


def test_control_body_strips_only_empty_think_suffix() -> None:
    """The control replay changes only the empty-think suffix on the prompt."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    recorder, receipt = _run_helper(router)
    first = recorder.first_completion_request
    assert first is not None
    control_body = _mapping(receipt["control"])["request"]
    assert isinstance(control_body, dict)
    for key, value in first.items():
        if key == "prompt":
            continue
        assert control_body[key] == value
    assert first["prompt"].endswith(_EMPTY_THINK)
    assert control_body["prompt"] == first["prompt"][: -len(_EMPTY_THINK)]


def test_budget_stops_fifth_completion_before_forward() -> None:
    """A fifth ``/completion`` is refused before the inner transport sees it."""
    harness = _harness()
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    inner_paths: list[str] = []

    def counting(request: httpx.Request) -> httpx.Response:
        inner_paths.append(request.url.path)
        return router.handle(request)

    recorder = harness.RecordingTransport(
        httpx.MockTransport(counting), max_completions=4
    )
    harness.run_llama_bridge_receipt(_framed_environ(), transport=recorder)
    assert recorder.completion_attempts == 4
    assert inner_paths.count("/completion") == 4
    with pytest.raises(harness.CompletionBudgetError, match="4"):
        recorder.handle_request(
            httpx.Request("POST", "http://router.test/completion", json={"prompt": "x"})
        )
    assert inner_paths.count("/completion") == 4
    assert recorder.completion_attempts == 5


def test_failed_attempt_counts_toward_budget() -> None:
    """Early-close retries consume the four-attempt budget before control."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(fail_first_completions=1),
    )
    with pytest.raises((httpx.HTTPError, RuntimeError, OSError)):
        _run_helper(router)
    assert router._completion_forwards == 4


def test_transport_close_on_successful_run() -> None:
    """A successful helper run closes the owned transport."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    inner = _ClosingTransport(router.handle)
    harness = _harness()
    recorder = harness.RecordingTransport(inner, max_completions=4)
    harness.run_llama_bridge_receipt(_framed_environ(), transport=recorder)
    assert inner.closed is True


def test_transport_close_on_failed_run() -> None:
    """A failed helper run still closes the owned transport."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(control_mass_mode="flat"),
    )
    inner = _ClosingTransport(router.handle)
    harness = _harness()
    recorder = harness.RecordingTransport(inner, max_completions=4)
    with pytest.raises(AssertionError, match="control"):
        harness.run_llama_bridge_receipt(_framed_environ(), transport=recorder)
    assert inner.closed is True


def test_recorder_rebuilds_gzip_completion_without_stale_encoding() -> None:
    """Decoded first completions must not keep Content-Encoding/Length."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(gzip_completions=True),
    )
    recorder, receipt = _run_helper(router)
    assert recorder.completion_attempts == 4
    assert isinstance(receipt["first_top_n"], list)
    assert len(receipt["first_top_n"]) == _VOCAB


def test_secret_echo_scrubbed_before_receipt_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A synthetic secret echoed in top-N tokens never reaches disk or stdout."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(echo_secret_in_top=_KEY),
    )
    monkeypatch.setattr(live, "gate_live", lambda settings: None)
    _install_env(
        monkeypatch,
        {
            **_framed_environ(),
            "TYPEVET_LLAMA__API_KEY": _KEY,
            "TYPEVET_LIVE_RECEIPT_DIR": str(tmp_path),
        },
    )
    inner = httpx.MockTransport(router.handle)
    recorder = _harness().RecordingTransport(inner, max_completions=4)
    live._run_llama_live_case(transport=recorder)
    text = (tmp_path / "judgevet_bridge_llama_cpp_receipt.json").read_text(
        encoding="utf-8"
    )
    assert _KEY not in text
    written = json.loads(text)
    assert isinstance(written["first_top_n"], list)
    assert len(written["first_top_n"]) == _VOCAB
    assert all(isinstance(entry.get("id"), int) for entry in written["first_top_n"])
    assert all(
        isinstance(entry.get("logprob"), int | float)
        and math.isfinite(float(entry["logprob"]))
        for entry in written["first_top_n"]
    )


def test_unknown_framing_refuses_before_any_request() -> None:
    """An unknown ``TYPEVET_LIVE_LLAMA_FRAMING`` value refuses before I/O."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    recorder = _harness().RecordingTransport(
        httpx.MockTransport(router.handle), max_completions=4
    )
    with pytest.raises(ValueError, match="TYPEVET_LIVE_LLAMA_FRAMING"):
        _harness().run_llama_bridge_receipt(
            {**_framed_environ(), "TYPEVET_LIVE_LLAMA_FRAMING": "auto"},
            transport=recorder,
        )
    assert router.paths == []
    assert recorder.completion_attempts == 0


def test_framed_missing_default_model_refuses_before_any_request() -> None:
    """Framed harness without ``DEFAULT_MODEL`` refuses before any request."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    recorder = _harness().RecordingTransport(
        httpx.MockTransport(router.handle), max_completions=4
    )
    environ = {
        "TYPEVET_BACKEND": "llama_cpp",
        "TYPEVET_LLAMA__BASE_URL": "http://router.test",
        "TYPEVET_LLAMA__MULTIMODAL_MODEL": "unused-mm",
        "TYPEVET_LIVE_LLAMA_FRAMING": "chatml",
    }
    with pytest.raises(ValueError, match="DEFAULT_MODEL"):
        _harness().run_llama_bridge_receipt(environ, transport=recorder)
    assert router.paths == []


def test_missing_base_url_fails_under_require_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TYPEVET_REQUIRE_LIVE=1`` fails on the real llama live entry."""
    monkeypatch.setenv("TYPEVET_REQUIRE_LIVE", "1")
    monkeypatch.delenv("TYPEVET_LLAMA__BASE_URL", raising=False)
    monkeypatch.delenv("TYPEVET_LLAMA_URL", raising=False)
    with pytest.raises(Failed, match="TYPEVET_LLAMA__BASE_URL"):
        live._run_llama_live_case()


def test_missing_base_url_skips_without_require_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``TYPEVET_REQUIRE_LIVE``, a missing base URL skips."""
    monkeypatch.delenv("TYPEVET_REQUIRE_LIVE", raising=False)
    monkeypatch.delenv("TYPEVET_LLAMA__BASE_URL", raising=False)
    monkeypatch.delenv("TYPEVET_LLAMA_URL", raising=False)
    with pytest.raises(Skipped, match="TYPEVET_LLAMA__BASE_URL"):
        live._run_llama_live_case()


def test_unframed_preserves_historical_gemma_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent framing keeps the historical Gemma model when DEFAULT_MODEL is unset."""
    monkeypatch.delenv("TYPEVET_LIVE_LLAMA_FRAMING", raising=False)
    monkeypatch.delenv("TYPEVET_LLAMA__DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("TYPEVET_GEMMA_MODEL", raising=False)
    historical = live._HISTORICAL_GEMMA
    router = _OfflineRouter(native_gemma4_rendered("hello"), historical)
    recorder, receipt = _run_helper(
        router,
        {
            "TYPEVET_BACKEND": "llama_cpp",
            "TYPEVET_LLAMA__BASE_URL": "http://router.test",
        },
    )
    assert receipt["requested_model"] == historical
    assert receipt["model"] == historical
    assert receipt["control"] is None
    assert recorder.first_completion_request is not None
    prompt = recorder.first_completion_request["prompt"]
    assert isinstance(prompt, str)
    assert not prompt.endswith(_EMPTY_THINK)


def test_unframed_default_model_selected() -> None:
    """Unframed runs still honour ``TYPEVET_LLAMA__DEFAULT_MODEL``."""
    model = "gemma-other-445"
    router = _OfflineRouter(native_gemma4_rendered("hello"), model)
    _recorder, receipt = _run_helper(router, _unframed_environ(model))
    assert receipt["requested_model"] == model
    assert receipt["model"] == model
    assert receipt["control"] is None
    assert _mapping(receipt["off_option_masses"])["duplicate_charge"] is None
    assert router.completion_bodies[0]["model"] == model


def test_default_model_selection_overrides_historical_fallback() -> None:
    """``TYPEVET_LLAMA__DEFAULT_MODEL`` selects the live model when set."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    _recorder, receipt = _run_helper(router)
    assert receipt["requested_model"] == _MODEL
    assert _PINNED_GEMMA not in json.dumps(receipt)


def test_vllm_entry_still_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parametrized live entry still routes the vLLM branch offline."""
    called: list[str] = []

    def fake_vllm() -> None:
        called.append("vllm")

    monkeypatch.setattr(live, "_run_vllm_live_case", fake_vllm)
    live.test_judgevet_bridge_live_receipt("vllm")
    assert called == ["vllm"]


def test_pytest_entry_routes_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    """``test_judgevet_bridge_live_receipt('llama_cpp')`` hits the llama entry."""
    called: list[str] = []

    def tracking_case(*, transport: Any = None) -> dict[str, object]:
        del transport
        called.append("llama")
        return {"model": _MODEL}

    monkeypatch.setattr(live, "_run_llama_live_case", tracking_case)
    live.test_judgevet_bridge_live_receipt("llama_cpp")
    assert called == ["llama"]


def test_probabilities_and_score_bounds_recorded() -> None:
    """Receipt keeps Noul P(0)/P(1), Choice keys, Score legend and bounds."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    _recorder, receipt = _run_helper(router)
    answers = receipt["answers"]
    assert isinstance(answers, dict)
    noul = answers["duplicate_charge"]
    assert isinstance(noul, dict)
    assert set(noul["probabilities"]) == {"0", "1"}
    assert abs(sum(noul["probabilities"].values()) - 1.0) < 1e-6
    assert noul["probabilities"]["1"] == pytest.approx(float(noul["noul"]), abs=1e-9)
    choice = answers["route"]
    assert set(choice["probabilities"]) == {"billing", "technical", "sales"}
    score = answers["urgency"]
    assert isinstance(score, dict)
    assert len(score["probabilities"]) == 5
    assert len(score["legend"]) == 5
    assert set(score["probabilities"]) == set(score["legend"])
    levels = sorted(int(level) for level in score["probabilities"])
    assert (
        math.isfinite(float(score["score"]))
        and levels[0] <= float(score["score"]) <= levels[-1]
    )


def test_missing_catalog_n_vocab_refuses_before_scoring() -> None:
    """Framed harness refuses before ``/completion`` when catalog n_vocab is absent."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(include_meta=False, n_vocab=None),
    )
    with pytest.raises(AssertionError, match="n_vocab"):
        _run_helper(router)
    assert router.paths.count("/completion") == 0


def test_control_mass_must_rise() -> None:
    """Non-increasing control off-option mass fails the harness."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(control_mass_mode="flat"),
    )
    with pytest.raises(AssertionError, match="control"):
        _run_helper(router)


def test_duplicate_token_id_fails() -> None:
    """Duplicate token ids in a control distribution fail."""
    harness = _harness()
    top = _normalized_top(_VOCAB, false_p=0.1, true_p=0.1)
    top[-1] = dict(top[2])
    with pytest.raises(AssertionError, match="duplicate"):
        harness.off_option_mass_from_top(
            top, candidate_ids=(_CAND_FALSE, _CAND_TRUE), n_vocab=_VOCAB
        )


def test_missing_candidate_id_fails() -> None:
    """A control distribution that omits a candidate id fails."""
    harness = _harness()
    top = [
        entry if entry["id"] != _CAND_TRUE else {"id": 999, "logprob": entry["logprob"]}
        for entry in _normalized_top(_VOCAB, false_p=0.1, true_p=0.1)
    ]
    with pytest.raises(AssertionError, match="missing candidate"):
        harness.off_option_mass_from_top(
            top, candidate_ids=(_CAND_FALSE, _CAND_TRUE), n_vocab=_VOCAB
        )


def test_unnormalized_distribution_fails() -> None:
    """A control distribution outside production mass tolerance fails."""
    harness = _harness()
    top = [
        {"id": entry["id"], "logprob": float(entry["logprob"]) - 0.2}
        for entry in _normalized_top(_VOCAB, false_p=0.1, true_p=0.1)
    ]
    with pytest.raises(AssertionError, match="mass"):
        harness.off_option_mass_from_top(
            top, candidate_ids=(_CAND_FALSE, _CAND_TRUE), n_vocab=_VOCAB
        )


def test_live_entry_writes_receipt_on_control_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-increasing control still emits a partial receipt, then fails."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(control_mass_mode="flat"),
    )
    monkeypatch.setattr(live, "gate_live", lambda settings: None)
    _install_env(
        monkeypatch,
        {**_framed_environ(), "TYPEVET_LIVE_RECEIPT_DIR": str(tmp_path)},
    )
    recorder = _harness().RecordingTransport(
        httpx.MockTransport(router.handle), max_completions=4
    )
    with pytest.raises(AssertionError, match="control"):
        live._run_llama_live_case(transport=recorder)
    receipt_path = tmp_path / "judgevet_bridge_llama_cpp_receipt.json"
    assert receipt_path.is_file()
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert (
        written.get("first_top_n_count") == _VOCAB
        or len(written.get("first_top_n", [])) == _VOCAB
    )
    assert "failure" in written
    assert written["failure"]["type"] == "AssertionError"
    control = written["control"]
    assert isinstance(control, dict)
    assert "response" in control
    assert router.paths.count("/completion") == 4


def test_helper_mutates_receipt_argument_on_failure() -> None:
    """The helper fills a caller-owned receipt before re-raising."""
    router = _OfflineRouter(
        degraded_chatml_rendered("hello"),
        _MODEL,
        _RouterOptions(control_mass_mode="flat"),
    )
    receipt: dict[str, object] = {}
    with pytest.raises(AssertionError, match="control"):
        _run_helper(router, receipt=receipt)
    assert receipt["model"] == _MODEL
    assert "failure" in receipt
    assert isinstance(receipt["first_top_n"], list)
    assert isinstance(receipt["control"], dict)


def test_tokenized_candidate_ids_not_hardcoded_indexes() -> None:
    """Control candidate ids come from tokenize, not literal ``0``/``1`` indexes."""
    router = _OfflineRouter(degraded_chatml_rendered("hello"), _MODEL)
    _recorder, receipt = _run_helper(router)
    candidate_ids = _mapping(_mapping(receipt["control"])["candidate_ids"])
    assert candidate_ids["0"] == _CAND_FALSE
    assert candidate_ids["1"] == _CAND_TRUE
    assert _CAND_FALSE != 0
    assert _CAND_TRUE != 1
