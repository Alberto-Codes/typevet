"""Offline tests for the #170 vLLM acceptance harness ([#170][i170]).

The mock server answers with the redacted vLLM v0.30.0 probe files from #168:
P4 (``image_three_way``) is the template for every scoring reply, P10
(``http400_invalid_schema``) is the invalid-schema reply (since #226 the
adapter rejects that schema before any POST) and P11
(``generation_enum_image``) is the image generation reply. ``/tokenize``
replies come from ``tokenize_ordinals``. ``/version``, ``/v1/models`` and
``/metrics`` replies are synthetic. The answers need not be correct; the
tests prove a complete receipt, the call caps and key masking.

[i170]: https://github.com/Alberto-Codes/typevet/issues/170
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.live.test_gemma4_llama_cpp import SCHEMA
from typevet.adapters.inbound.backend_settings import (
    load_vllm_settings,
    vllm_http_client,
)
from typevet_evals.cli.cord_semantic_acceptance import main as cord_cli
from typevet_evals.datasets.cord_expense import load_expense_cases
from typevet_evals.experiment_identity import read_baseline_commit
from typevet_evals.vllm_acceptance import core as vllm_acceptance
from typevet_evals.vllm_acceptance import sets as vllm_acceptance_sets
from typevet_evals.vllm_acceptance.core import (
    OMITTED_RULE,
    AcceptanceInputs,
    CallCaps,
    cord_passed,
    latency_summary,
    live_gate_reason,
    psai_gates,
    run_acceptance,
    write_receipt,
)
from typevet_evals.vllm_acceptance.sets import (
    DEVIATIONS,
    SET_RUNNERS,
    TEXT_SCHEMA,
)

pytestmark = pytest.mark.unit

_TESTS = Path(__file__).resolve().parents[3] / "tests"
_REPO = _TESTS.parent
_VLLM = _TESTS / "fixtures" / "vllm"
_MODEL = "gemma-4-31b-it"
_KEY = "sk-SENTINEL-170A"
_SETS = {"generation", "psai", "cord", "order", "concurrency"}


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_VLLM / f"{name}.json").read_text(encoding="utf-8"))


def _env(**extra: str) -> dict[str, str]:
    return {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": "http://vllm.test:8000",
        "TYPEVET_VLLM__MODEL": _MODEL,
        "TYPEVET_VLLM__API_KEY": _KEY,
        "TYPEVET_VLLM_POD_NOTES": "H100 SXM; offline test",
        **extra,
    }


def _inputs() -> AcceptanceInputs:
    return AcceptanceInputs(fixtures_root=_TESTS / "fixtures", repo_root=_REPO)


class _Vllm:
    """MockTransport handler that mimics the vLLM endpoints the harness calls."""

    def __init__(
        self,
        *,
        echo_key: bool = False,
        statuses: dict[str, int] | None = None,
        served: str = _MODEL,
        fail_chat: Exception | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.echo_key = echo_key
        self.statuses = statuses or {}
        self.served = served
        self.fail_chat = fail_chat
        self._tokens = _fixture("tokenize_ordinals")["responses"]
        self._scoring = _fixture("image_three_way")["response"]
        self._invalid = _fixture("http400_invalid_schema")["response"]
        self._image = _fixture("generation_enum_image")["response"]

    def count(self, kind: str) -> int:
        paths = [r.url.path for r in self.requests]
        if kind == "tokenizer":
            return paths.count("/tokenize")
        if kind == "metadata":
            return sum(p in {"/version", "/v1/models", "/metrics"} for p in paths)
        return paths.count("/v1/chat/completions")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        auth = request.headers.get("authorization", "") if self.echo_key else ""
        if request.method == "GET":
            return self._get(request.url.path, auth)
        if self.fail_chat is not None and request.url.path == "/v1/chat/completions":
            raise self.fail_chat
        body = json.loads(request.content.decode())
        if request.url.path == "/tokenize":
            return httpx.Response(200, json=self._tokenize(body["prompt"]))
        if "logprob_token_ids" in body:
            return httpx.Response(200, json=self._score(body))
        return self._generate(body, auth)

    def _get(self, path: str, auth: str) -> httpx.Response:
        if path in self.statuses:
            return httpx.Response(self.statuses[path], json={"error": "denied"})
        if path == "/version":
            return httpx.Response(200, json={"version": "0.30.0", "echo": auth})
        if path == "/v1/models":
            data = [{"id": self.served, "root": "google/gemma-4-31B-it"}]
            return httpx.Response(200, json={"object": "list", "data": data})
        text = 'vllm:kv_cache_usage_perc{engine="0"} 0.125\n'
        return httpx.Response(200, text=text)

    def _tokenize(self, prompt: str) -> dict[str, Any]:
        if prompt in self._tokens:
            return self._tokens[prompt]
        return {"count": 1, "tokens": [200000 + zlib.crc32(prompt.encode()) % 50000]}

    def _score(self, body: dict[str, Any]) -> dict[str, Any]:
        content = body["messages"][0]["content"]
        parts = [] if isinstance(content, str) else content
        images = sum(part.get("type") == "image_url" for part in parts)
        reply = copy.deepcopy(self._scoring)
        top = [
            {"token": f"token_id:{token}", "logprob": -0.5 - rank}
            for rank, token in enumerate(body["logprob_token_ids"])
        ]
        reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = top
        reply["usage"] = {"prompt_tokens": 90 + 413 * images, "completion_tokens": 1}
        return reply

    def _generate(self, body: dict[str, Any], auth: str) -> httpx.Response:
        schema = body["structured_outputs"]["json"]
        if schema.get("required") == 5:
            error = copy.deepcopy(self._invalid)
            error["error"]["message"] += f" {auth}"
            return httpx.Response(400, json=error)
        if "verdict" in schema["properties"]:
            return httpx.Response(200, json=self._image)
        reply = copy.deepcopy(self._image)
        value = {"sentiment": "pos", "confidence": 90}
        reply["choices"][0]["message"]["content"] = json.dumps(value)
        return httpx.Response(200, json=reply)


def _run(
    server: _Vllm, caps: CallCaps | None = None, receipt_path: Path | None = None
) -> dict[str, Any]:
    transport = httpx.MockTransport(server)
    return run_acceptance(
        _env(),
        _inputs(),
        transport=transport,
        runners=SET_RUNNERS,
        deviations=DEVIATIONS,
        caps=caps,
        receipt_path=receipt_path,
    )


def test_offline_run_writes_a_receipt_with_every_set(tmp_path: Path) -> None:
    server = _Vllm()
    receipt = _run(server)
    path = tmp_path / "receipt.json"
    digest = write_receipt(path, receipt, api_key=_KEY)

    assert set(receipt["sets"]) == _SETS
    assert receipt["stopped"] is None
    assert receipt["deviations"] == list(DEVIATIONS)
    assert len(receipt["deviations"]) == 7
    assert any("60-minute" in note for note in receipt["deviations"])
    assert any("cmcc8u6yd00wr1p1yj7aot3ae" in note for note in receipt["deviations"])
    assert receipt["tokenizer_memo_hits"] > 0
    assert receipt["calls"] == {
        kind: server.count(kind) for kind in ("model", "tokenizer", "metadata")
    }
    # Generation posts 2 of its 3 asks; P10 stops before the POST (#226).
    assert receipt["calls"]["model"] == 2 + 16 + 43 + 18 + 8
    assert receipt["calls"]["metadata"] == 4
    assert receipt["calls"]["tokenizer"] <= CallCaps().tokenizer
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for name in _SETS:
        latency = receipt["sets"][name]["latency"]
        assert latency["n"] > 0
        assert latency["p50"] <= latency["p95"]


def test_offline_receipt_records_pins_and_set_details(tmp_path: Path) -> None:
    receipt = _run(_Vllm())
    pins = receipt["pins"]
    assert pins["version"]["body"]["version"] == "0.30.0"
    assert pins["served_models"] == [{"id": _MODEL, "root": "google/gemma-4-31B-it"}]
    assert pins["configured_model"] == _MODEL
    assert pins["source_sha"] == read_baseline_commit(_REPO)
    assert pins["pod_notes"] == "H100 SXM; offline test"
    assert pins["kv_cache_usage"] == 0.125
    sets = receipt["sets"]
    invalid = sets["generation"]["rows"][2]
    assert invalid["error"].startswith("ValueError: schema is not a valid JSON Schema:")
    assert invalid["posts"] == 0
    assert invalid["value"] is None
    assert sets["generation"]["passed"] is True
    assert len(sets["psai"]["rows"]) == 16
    assert sets["psai"]["coverage"] == {"calls": 16, "covered": 16}
    assert len(sets["cord"]["combined"]) == 18
    assert len(sets["order"]["combined"]) == 18
    assert sets["order"]["label_order"] == [
        "match",
        "mismatch",
        "insufficient_evidence",
    ]
    assert isinstance(sets["order"]["flips"], int)
    assert sets["concurrency"]["errors"] == 0
    assert sets["concurrency"]["kv_cache_in_flight"] == 0.125
    assert sets["psai"]["omitted_rule"] == OMITTED_RULE
    assert len(sets["concurrency"]["rows"]) == 8
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt, api_key=_KEY)
    assert cord_cli([str(path)]) in (0, 1)


@pytest.mark.parametrize(
    ("caps", "kind"),
    [
        (CallCaps(model=5), "model"),
        (CallCaps(tokenizer=1), "tokenizer"),
        (CallCaps(metadata=1), "metadata"),
    ],
)
def test_call_cap_stops_the_run_and_still_writes_a_receipt(
    tmp_path: Path, caps: CallCaps, kind: str
) -> None:
    server = _Vllm()
    receipt = _run(server, caps)
    limit = getattr(caps, kind)

    assert receipt["stopped"] == f"{kind} call cap {limit} reached"
    assert server.count(kind) == limit
    assert receipt["calls"][kind] == limit
    assert receipt["passed"] is False
    write_receipt(tmp_path / "receipt.json", receipt, api_key=_KEY)
    assert json.loads((tmp_path / "receipt.json").read_text())["stopped"]


def test_write_receipt_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_receipt(path, {"issue": 170}, api_key=None)


def test_receipt_never_holds_the_api_key(tmp_path: Path) -> None:
    receipt = _run(_Vllm(echo_key=True))
    error = receipt["sets"]["generation"]["rows"][2]["error"]
    assert _KEY not in error
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt, api_key=_KEY)
    assert _KEY not in path.read_text(encoding="utf-8")


def _captured_headers(env: dict[str, str]) -> httpx.Headers:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    settings = load_vllm_settings(env)
    with vllm_http_client(settings, transport=httpx.MockTransport(handler)) as client:
        client.get("/version")
    return seen[0].headers


def test_user_agent_header_is_sent_only_when_set() -> None:
    headers = _captured_headers(_env(TYPEVET_VLLM__USER_AGENT="curl/8.9.1"))
    assert headers["user-agent"] == "curl/8.9.1"
    for env in (_env(), _env(TYPEVET_VLLM__USER_AGENT="  ")):
        assert load_vllm_settings(env).user_agent is None
        assert _captured_headers(env)["user-agent"].startswith("python-httpx/")


def _constant_cord_receipt(tmp_path: Path, label: str) -> Path:
    manifest = _TESTS / "fixtures" / "cord" / "expense_smoke" / "manifest.json"
    cases = load_expense_cases(manifest.read_text(encoding="utf-8"))
    receipt = {
        "cases": [
            {"claim_id": c.claim_id, "expected_verdict": c.expected_verdict}
            for c in cases
        ],
        "combined": {c.claim_id: {"label": label} for c in cases},
    }
    path = tmp_path / f"{label}.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


@pytest.mark.parametrize("label", ["match", "insufficient_evidence"])
def test_constant_cord_baselines_fail_the_cli(tmp_path: Path, label: str) -> None:
    assert cord_cli([str(_constant_cord_receipt(tmp_path, label))]) == 1


def _psai_rows(present: dict[str, str], swapped: dict[str, str]) -> list[dict]:
    golds = {"C01": "true", "C03": "true", "C08": "false", "C10": "false"}
    rows: list[dict[str, Any]] = []
    for case, gold in golds.items():
        for condition, label, tokens in (
            ("omitted", "true", 90),
            ("present", present[case], 503),
            ("swapped", swapped[case], 503),
        ):
            other = "false" if label == "true" else "true"
            rows.append(
                {
                    "case_id": case,
                    "condition": condition,
                    "gold": gold,
                    "label": label,
                    "probabilities": {label: 0.9, other: 0.1},
                    "tokens_evaluated": tokens,
                    "error": None,
                }
            )
    for index, gold in enumerate(("false", "false", "BROWSER_TASK", "BROWSER_TASK")):
        rows.append(
            {
                "case_id": f"TR{index}",
                "condition": "text_only",
                "gold": gold,
                "label": gold,
                "probabilities": {gold: 0.9, "other": 0.1},
                "tokens_evaluated": 130,
                "error": None,
            }
        )
    return rows


def test_psai_gates_pass_image_conditioned_answers() -> None:
    gold = {"C01": "true", "C03": "true", "C08": "false", "C10": "false"}
    flipped = {k: "false" if v == "true" else "true" for k, v in gold.items()}
    outcome = psai_gates(_psai_rows(gold, flipped))
    assert outcome == {
        "present": 4,
        "swapped": 4,
        "text": 4,
        "omitted_credited": 2,
        "omitted_rule": OMITTED_RULE,
        "covered": True,
        "passed": True,
    }


def test_constant_psai_answer_fails_the_gates() -> None:
    constant = dict.fromkeys(("C01", "C03", "C08", "C10"), "true")
    outcome = psai_gates(_psai_rows(constant, constant))
    assert outcome["present"] == 2
    assert outcome["swapped"] == 0
    assert outcome["passed"] is False


def test_psai_gates_reject_a_dropped_image() -> None:
    gold = {"C01": "true", "C03": "true", "C08": "false", "C10": "false"}
    flipped = {k: "false" if v == "true" else "true" for k, v in gold.items()}
    rows = _psai_rows(gold, flipped)
    for row in rows:
        if row["condition"] == "present":
            row["tokens_evaluated"] = 120
    rows[-1]["error"] = "ScoringValidationError: missing id"
    outcome = psai_gates(rows)
    assert outcome["present"] == 0
    assert outcome["covered"] is False
    assert outcome["passed"] is False


def test_latency_summary_uses_nearest_rank() -> None:
    samples = [float(n) for n in range(1, 21)]
    assert latency_summary(samples) == {"n": 20, "p50": 10.0, "p95": 19.0}
    assert latency_summary([]) == {"n": 0, "p50": None, "p95": None}


def test_text_schema_is_the_frozen_llama_cpp_schema() -> None:
    assert TEXT_SCHEMA == SCHEMA


def test_live_gate_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPEVET_REQUIRE_LIVE", raising=False)
    assert "TYPEVET_REQUIRE_LIVE" in str(live_gate_reason(_env()))
    monkeypatch.setenv("TYPEVET_REQUIRE_LIVE", "1")
    assert "TYPEVET_VLLM_RECEIPT" in str(live_gate_reason(_env()))
    ready = _env(TYPEVET_VLLM_RECEIPT="scratchpad/vllm/receipt.json")
    assert live_gate_reason(ready) is None
    missing = {k: v for k, v in ready.items() if k != "TYPEVET_VLLM__BASE_URL"}
    assert "TYPEVET_VLLM__BASE_URL" in str(live_gate_reason(missing))
    llama = {**ready, "TYPEVET_BACKEND": "llama_cpp"}
    assert "TYPEVET_BACKEND" in str(live_gate_reason(llama))


@pytest.mark.parametrize(
    ("server", "reason"),
    [
        (_Vllm(statuses={"/version": 500}), "preflight: /version returned 500"),
        (_Vllm(statuses={"/v1/models": 404}), "preflight: /v1/models returned 404"),
        (_Vllm(served="other-model"), "does not list TYPEVET_VLLM__MODEL"),
        (
            _Vllm(statuses={"/version": 401, "/v1/models": 403}),
            "preflight denied: /version and /v1/models returned 401 or 403",
        ),
    ],
)
def test_preflight_stops_before_any_model_call(server: _Vllm, reason: str) -> None:
    receipt = _run(server)
    assert reason in str(receipt["stopped"])
    assert server.count("model") == 0
    assert receipt["calls"]["model"] == 0
    assert receipt["passed"] is False
    assert receipt["pins"]["configured_model"] == _MODEL


def test_any_error_still_writes_a_masked_receipt(tmp_path: Path) -> None:
    server = _Vllm(fail_chat=RuntimeError(f"socket died near {_KEY}"))
    path = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError):
        _run(server, receipt_path=path)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["stopped"] == "error: RuntimeError"
    assert written["error"]["type"] == "RuntimeError"
    assert "socket died near ***" in written["error"]["message"]
    assert _KEY not in path.read_text(encoding="utf-8")
    assert written["pins"]["configured_model"] == _MODEL


def test_run_writes_the_receipt_path_on_success(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "receipt.json"
    receipt = _run(_Vllm(), receipt_path=path)
    assert json.loads(path.read_text(encoding="utf-8"))["calls"] == receipt["calls"]


def test_cord_passes_only_with_full_coverage() -> None:
    full = {"accepted": True, "coverage": {"calls": 43, "covered": 43}}
    short = {"accepted": True, "coverage": {"calls": 43, "covered": 42}}
    assert cord_passed(full) is True
    assert cord_passed(short) is False
    assert cord_passed({**full, "accepted": False}) is False


def test_live_gate_rejects_unusable_receipt_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPEVET_REQUIRE_LIVE", "1")
    existing = tmp_path / "old.json"
    existing.write_text("{}", encoding="utf-8")
    reason = live_gate_reason(_env(TYPEVET_VLLM_RECEIPT=str(existing)))
    assert "already exists" in str(reason)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        target = str(locked / "sub" / "receipt.json")
        reason = live_gate_reason(_env(TYPEVET_VLLM_RECEIPT=target))
    finally:
        locked.chmod(0o700)
    assert "not writable" in str(reason)
    fresh = str(tmp_path / "new" / "receipt.json")
    assert live_gate_reason(_env(TYPEVET_VLLM_RECEIPT=fresh)) is None


def test_error_message_is_masked_before_write_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, Any]] = []

    def spy(path: Path, receipt: dict[str, Any], *, api_key: str | None) -> str:
        captured.append(copy.deepcopy(receipt))
        return ""

    monkeypatch.setattr(vllm_acceptance, "write_receipt", spy)
    server = _Vllm(fail_chat=RuntimeError(f"socket died near {_KEY}"))
    with pytest.raises(RuntimeError):
        _run(server, receipt_path=tmp_path / "receipt.json")
    assert len(captured) == 1
    message = captured[0]["error"]["message"]
    assert "socket died near" in message
    assert _KEY not in message


def test_write_receipt_masks_the_raw_and_json_escaped_key(tmp_path: Path) -> None:
    key = 'sk-"QUOTED"\\SENTINEL'
    escaped = json.dumps(key)[1:-1]
    path = tmp_path / "receipt.json"
    write_receipt(path, {"x": key}, api_key=key)
    text = path.read_text(encoding="utf-8")
    assert escaped not in text
    assert key not in text
    assert json.loads(text) == {"x": "***"}
    plain = tmp_path / "plain.json"
    write_receipt(plain, {"x": f"near {_KEY}"}, api_key=_KEY)
    assert _KEY not in plain.read_text(encoding="utf-8")


def test_swap_needs_a_correct_present_answer_to_count_as_moved() -> None:
    present = {
        "gold": "true",
        "label": "false",
        "probabilities": {"true": 0.4, "false": 0.6},
    }
    swapped = {"label": "false", "probabilities": {"true": 0.4, "false": 0.6}}
    assert vllm_acceptance._swap_ok(present, swapped) is False


_GC_PREAMBLE = "".join(
    f"# HELP python_gc_objects_collected_total Objects collected in gen {n}.\n"
    f"# TYPE python_gc_objects_collected_total counter\n"
    f'python_gc_objects_collected_total{{generation="{n}"}} {n}.0\n'
    for n in range(3)
)
_KV_LINE = 'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25\n'


def test_kv_cache_usage_reads_the_gauge_after_a_long_preamble() -> None:
    assert len(_GC_PREAMBLE) > 200
    body = _GC_PREAMBLE + _KV_LINE

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://vllm.test") as client:
        assert vllm_acceptance.kv_cache_usage(client) == 0.25


class _HeldChat(_Vllm):
    """Stub that holds chat replies until ``/metrics`` arrives (5 s timeout)."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._metrics = threading.Event()
        self.in_flight = 0
        self.in_flight_at_metrics: int | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            with self._lock:
                if self.in_flight_at_metrics is None:
                    self.in_flight_at_metrics = self.in_flight
            self._metrics.set()
            self.requests.append(request)
            return httpx.Response(200, text=_GC_PREAMBLE + _KV_LINE)
        if request.url.path != "/v1/chat/completions":
            return super().__call__(request)
        with self._lock:
            self.in_flight += 1
        self._metrics.wait(timeout=5.0)
        try:
            return super().__call__(request)
        finally:
            with self._lock:
                self.in_flight -= 1


def test_concurrency_reads_kv_cache_while_requests_are_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = vllm_acceptance_sets._generate

    def delayed(*args: Any) -> dict[str, Any]:
        time.sleep(0.2)
        return original(*args)

    monkeypatch.setattr(vllm_acceptance_sets, "_generate", delayed)
    server = _HeldChat()
    runners = [("concurrency", vllm_acceptance_sets._concurrency_set)]
    receipt = run_acceptance(
        _env(), _inputs(), transport=httpx.MockTransport(server), runners=runners
    )
    assert receipt["stopped"] is None
    assert server.in_flight_at_metrics is not None
    assert server.in_flight_at_metrics >= 1
    assert receipt["sets"]["concurrency"]["kv_cache_in_flight"] == 0.25
    assert receipt["calls"]["metadata"] == 4
