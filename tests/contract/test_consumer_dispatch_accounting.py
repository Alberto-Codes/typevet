"""Real-router dispatch accounting contracts with an offline HTTP transport."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from typevet.adapters.inbound.settings import LlamaSettings
from typevet.domain.errors import GenerationError, JudgmentError
from typevet.domain.judgment_questions import Noul
from typevet.evaluation.instruction_variant_consumer_live import (
    run_live_instruction_variant_proof,
)
from typevet.evaluation.instruction_variant_consumer_live_router import (
    run_live_variant_matrix,
)
from typevet.evaluation.instruction_variant_consumer_protocol import (
    VariantDispatchLedger,
)
from typevet.evaluation.psai_vision_consumer_dispatch import (
    ConsumerDispatchLedger,
    wrap_judgment_port,
    wrap_scoring_port,
)
from typevet.evaluation.psai_vision_consumer_live import run_live_consumer_proof
from typevet.evaluation.psai_vision_consumer_live_router import run_consumer_live_matrix
from typevet.runtime import open_gemma_native_vision_judgment

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[1] / "fixtures/psai/vision_smoke"


class Router:
    """Collect real factory HTTP requests and optionally fail one endpoint."""

    def __init__(self, failure: str = "") -> None:
        """Initialize capture state and optional injected failure."""
        self.paths: list[str] = []
        self.bodies: list[dict] = []
        self.failure = failure

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return protocol fixtures, keeping every admitted HTTP attempt."""
        path = request.url.path
        self.paths.append(path)
        if path == self.failure.removesuffix(":transport"):
            if self.failure.endswith(":transport"):
                raise httpx.ConnectError("injected outage", request=request)
            return httpx.Response(503)
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/props":
            return httpx.Response(
                200,
                json={"modalities": {"vision": True}, "media_marker": "<image-test>"},
            )
        if path == "/apply-template":
            return httpx.Response(
                200, json={"prompt": "<|turn>user\nhello<turn|>\n<|turn>model\n"}
            )
        body = json.loads(request.content)
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": [ord(body["content"][0])]})
        self.bodies.append(body)
        return httpx.Response(
            200,
            json={
                "completion_probabilities": [
                    {
                        "top_logprobs": [
                            {"id": i, "logprob": -i / 100} for i in range(256)
                        ]
                    }
                ]
            },
        )


def run(
    router: Router,
    variant: bool,
    ledger: ConsumerDispatchLedger | VariantDispatchLedger | None = None,
    out_dir: Path | None = None,
):
    """Run each real matrix with only network and environment substituted."""
    client = httpx.Client(
        base_url="http://offline-router", transport=httpx.MockTransport(router.handle)
    )
    settings = LlamaSettings(
        base_url="http://offline-router", default_model="gemma-test"
    )
    with (
        patch("httpx.Client", return_value=client),
        patch.dict("os.environ", {"TYPEVET_REQUIRE_LIVE": "1"}),
        patch(
            "typevet.evaluation.psai_vision_consumer_live.load_llama_settings",
            return_value=settings,
        ),
        patch(
            "typevet.evaluation.instruction_variant_consumer_live_router.load_llama_settings",
            return_value=settings,
        ),
    ):
        if out_dir is not None:
            if variant:
                return run_live_instruction_variant_proof(
                    fixture_root=ROOT, out_dir=out_dir
                )
            return run_live_consumer_proof(fixture_root=ROOT, out_dir=out_dir)
        if variant:
            assert ledger is None or isinstance(ledger, VariantDispatchLedger)
            return run_live_variant_matrix(
                fixture_root=ROOT,
                seed_instruction="Unique seed instruction",
                candidate_instruction="Unique candidate instruction",
                ledger=ledger,
            )
        assert ledger is None or isinstance(ledger, ConsumerDispatchLedger)
        return run_consumer_live_matrix(
            settings=settings, model="gemma-test", fixture_root=ROOT, ledger=ledger
        )


@pytest.mark.parametrize("variant", [False, True])
def test_actual_factory_has_no_hidden_metadata(variant: bool) -> None:
    """Metadata totals include factory setup and a single cached props probe."""
    router = Router()
    result = run(router, variant)
    assert router.paths.count("/props") == 1
    assert router.paths.count("/apply-template") == 1
    assert result.ledger.auxiliary_metadata_http == (2 if variant else 3)
    assert result.ledger.completion_http == len(router.bodies)
    assert result.ledger.scoring_attempts == len(router.bodies)
    assert result.ledger.judgment_attempts == (4 if variant else 14)
    assert all(
        "<image-test>" in body["prompt"]["prompt_string"]
        for body in router.bodies
        if isinstance(body["prompt"], dict)
    )


@pytest.mark.parametrize("variant", [False, True])
@pytest.mark.parametrize("endpoint", ["/props", "/tokenize", "/completion"])
@pytest.mark.parametrize("transport", [False, True])
def test_failed_dispatch_retained_externally(
    variant: bool, endpoint: str, transport: bool
) -> None:
    """A failed admitted HTTP request remains visible to the consumer."""
    router = Router(endpoint + (":transport" if transport else ""))
    ledger = VariantDispatchLedger() if variant else ConsumerDispatchLedger()
    with pytest.raises((httpx.HTTPError, GenerationError, JudgmentError)):
        run(router, variant, ledger)
    assert endpoint in router.paths
    assert ledger.auxiliary_metadata_http == sum(
        path in {"/health", "/props", "/apply-template"} for path in router.paths
    )
    if endpoint != "/props":
        assert ledger.judgment_attempts == 1
        assert ledger.judgment_calls == 0
    if endpoint == "/completion":
        assert ledger.scoring_attempts == ledger.completion_http == 1
        assert ledger.scoring_requests == 0


@pytest.mark.parametrize("variant", [False, True])
@pytest.mark.parametrize(
    "counter,limit",
    [
        ("judgment_attempts", (14, 4)),
        ("scoring_attempts", (16, 8)),
        ("auxiliary_metadata_http", (3, 2)),
        ("auxiliary_tokenizer_http", (128, 8)),
        ("completion_http", (16, 8)),
    ],
)
def test_exhausted_budget_refuses_next_io(
    variant: bool, counter: str, limit: tuple[int, int]
) -> None:
    """Every exhausted dimension refuses dispatch at its own boundary."""
    ledger = VariantDispatchLedger() if variant else ConsumerDispatchLedger()
    setattr(ledger, counter, limit[int(variant)])
    router = Router()
    with pytest.raises(ValueError, match="budget"):
        run(router, variant, ledger)
    assert not router.bodies
    if counter == "auxiliary_metadata_http":
        assert router.paths == []
    if counter == "judgment_attempts":
        assert "/tokenize" not in router.paths
    assert getattr(ledger, counter) == limit[int(variant)]


@pytest.mark.parametrize("variant", [False, True])
@pytest.mark.parametrize(
    "endpoint", ["/props", "/tokenize", "/completion", "/completion:transport"]
)
def test_orchestrator_retains_failure_receipt(
    variant: bool, endpoint: str, tmp_path: Path
) -> None:
    """Public orchestration persists admitted attempts after real factory errors."""
    router = Router(endpoint)
    result = run(router, variant, out_dir=tmp_path)
    assert result.exit_code == 1
    assert result.receipt_path is not None
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["dispatch_accounting_version"] == 1
    errors = (
        receipt["acceptance_failures"]
        if variant
        else receipt["failure_attempt"]["checks_failed"]
    )
    assert any("503" in error or "injected outage" in error for error in errors)
    if variant:
        assert receipt["instruction_variant_protocol_revision"] == 2
    attempts = receipt["dispatch_accounting"]["attempts"]
    assert attempts["metadata_http"] == sum(
        path in {"/health", "/props", "/apply-template"} for path in router.paths
    )
    assert attempts["completion_http"] == router.paths.count("/completion")
    assert attempts["tokenizer_http"] == router.paths.count("/tokenize")
    assert receipt["dispatch_accounting"]["successes"] == {"judgment": 0, "scoring": 0}


@pytest.mark.parametrize("variant", [False, True])
def test_wire_retains_images_instructions_and_typed_answers(variant: bool) -> None:
    """Real factory matrices transmit fixture bytes and caller instructions."""
    router = Router()
    result = run(router, variant)
    expected_images = {path.read_bytes() for path in ROOT.glob("*.png")}
    media_bodies = [body for body in router.bodies if isinstance(body["prompt"], dict)]
    assert media_bodies
    for body in media_bodies:
        prompt = body["prompt"]
        assert prompt["multimodal_data"]
        assert all(
            base64.b64decode(encoded) in expected_images
            for encoded in prompt["multimodal_data"]
        )
        assert "<image-test>" in prompt["prompt_string"]
    if variant:
        assert len(router.bodies) == 4
        for index, body in enumerate(router.bodies):
            assert (
                "Unique seed instruction"
                if index < 2
                else "Unique candidate instruction"
            ) in body["prompt"]["prompt_string"]
        rows = result.seed_rows + result.candidate_rows
    else:
        assert any(isinstance(body["prompt"], str) for body in router.bodies)
        rows = result.matrix_rows
    assert all(row["answers"] for row in rows)
    answers = [answer for row in rows for answer in row["answers"].values()]
    assert {answer["kind"] for answer in answers} == (
        {"Noul"} if variant else {"Noul", "Choice"}
    )
    assert all(
        0 <= answer["noul"] <= 1 for answer in answers if answer["kind"] == "Noul"
    )
    assert all(
        answer["choice"] in answer["probabilities"]
        for answer in answers
        if answer["kind"] == "Choice"
    )


@pytest.mark.parametrize("variant", [False, True])
def test_failed_judgments_exhaust_slots_in_one_factory_session(variant: bool) -> None:
    """Repeated completion errors consume judgment slots and cannot reopen them."""
    router = Router("/completion")
    ledger = VariantDispatchLedger() if variant else ConsumerDispatchLedger()
    with (
        httpx.Client(
            base_url="http://offline-router",
            transport=httpx.MockTransport(router.handle),
            event_hooks={"request": [ledger.before_http]},
        ) as client,
        open_gemma_native_vision_judgment(
            settings=LlamaSettings(base_url="http://offline-router"),
            model="gemma-test",
            http_client=client,
            scoring_port_wrapper=lambda port: wrap_scoring_port(port, ledger),
        ) as session,
    ):
        port = wrap_judgment_port(session.port, ledger)
        for _ in range(ledger.limits[0]):
            with pytest.raises(GenerationError):
                port.judge(
                    "caller state",
                    {"answer": Noul(instructions="caller instruction")},
                    session.model,
                )
        before = list(router.paths)
        with pytest.raises(ValueError, match="budget"):
            port.judge("caller state", {"answer": Noul()}, session.model)
        assert router.paths == before
    assert (
        ledger.judgment_attempts
        == ledger.scoring_attempts
        == ledger.completion_http
        == ledger.limits[0]
    )
    assert ledger.judgment_calls == ledger.scoring_requests == 0
