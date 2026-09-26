"""Contract tests for the PSAI consumer proof harness entrypoint ([#177][i177])."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import typevet.evaluation.psai_vision_consumer_harness as consumer_harness
from typevet.adapters.inbound.cord_semantic_acceptance_cli import main as cord_cli_main
from typevet.evaluation.psai_vision_consumer_harness import (
    consumer_proof_main,
    cord_semantic_cli_exit_code,
    run_offline_consumer_proof,
)

pytestmark = pytest.mark.contract

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
)
CORRECTION = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "consumer"
    / "live-receipt-v1-accounting-correction.json"
)
CORD_PASS = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cord"
    / "semantic_acceptance"
    / "gemma4_post187_combined_pass.json"
)


def test_offline_harness_entrypoint_exits_zero_on_pass() -> None:
    """Real harness CLI returns 0 when offline acceptance passes."""
    assert consumer_proof_main(["--fixture-root", str(FIXTURE_ROOT)]) == 0


def test_offline_harness_observes_sixteen_scoring_requests() -> None:
    """Matrix run records 16 score_candidates calls for rev 1."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    assert result.receipt["scoring_request_count"] == 16
    assert result.receipt["scoring_requests_observed"] == 16


def test_offline_harness_entrypoint_exits_one_when_forced_fail() -> None:
    """Real harness CLI returns 1 when paired ordering is forced to fail."""
    assert (
        consumer_proof_main(
            ["--fixture-root", str(FIXTURE_ROOT), "--force-fail"],
        )
        == 1
    )


def test_offline_harness_entrypoint_exits_one_when_capability_vision_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Harness maps capability denial to exit 1 through acceptance evaluation."""
    real_payload = consumer_harness._offline_receipt_payload

    def _payload_with_denied_capability(
        assembly: consumer_harness._OfflineReceiptAssembly,
    ) -> dict[str, object]:
        receipt = real_payload(assembly)
        receipt["capability"] = {"vision": False, "offline_stub": True}
        return receipt

    monkeypatch.setattr(
        consumer_harness,
        "_offline_receipt_payload",
        _payload_with_denied_capability,
    )
    assert consumer_proof_main(["--fixture-root", str(FIXTURE_ROOT)]) == 1


def test_offline_harness_entrypoint_exits_one_when_control_negative_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Harness maps unsupported-capability negative failure to exit 1."""
    real_payload = consumer_harness._offline_receipt_payload

    def _payload_with_bad_negative(
        assembly: consumer_harness._OfflineReceiptAssembly,
    ) -> dict[str, object]:
        receipt = real_payload(assembly)
        receipt["unsupported_capability_negative"] = {"ok": False}
        return receipt

    monkeypatch.setattr(
        consumer_harness,
        "_offline_receipt_payload",
        _payload_with_bad_negative,
    )
    assert consumer_proof_main(["--fixture-root", str(FIXTURE_ROOT)]) == 1


def test_offline_harness_entrypoint_exits_one_when_matrix_row_missing_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Harness maps empty answers on a matrix row to exit 1."""
    real_payload = consumer_harness._offline_receipt_payload

    def _payload_with_empty_answers(
        assembly: consumer_harness._OfflineReceiptAssembly,
    ) -> dict[str, object]:
        receipt = real_payload(assembly)
        rows = list(receipt.get("matrix_rows") or [])
        if rows:
            rows[0] = {"leg": "visual", "answers": {}}
        receipt["matrix_rows"] = rows
        return receipt

    monkeypatch.setattr(
        consumer_harness,
        "_offline_receipt_payload",
        _payload_with_empty_answers,
    )
    assert consumer_proof_main(["--fixture-root", str(FIXTURE_ROOT)]) == 1


def test_offline_harness_entrypoint_honors_evaluate_consumer_receipt_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing acceptance wiring would return 0 on an otherwise passing offline run."""
    monkeypatch.setattr(
        consumer_harness,
        "evaluate_consumer_receipt_acceptance",
        lambda _receipt: (False, ["injected failure"]),
    )
    assert consumer_proof_main(["--fixture-root", str(FIXTURE_ROOT)]) == 1


def test_offline_harness_retains_choice_and_noul_on_annotation_rows() -> None:
    """Annotation matrix rows retain both Choice and Noul answers."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    annotation_rows = [
        row for row in result.receipt["matrix_rows"] if row.get("leg") == "annotation"
    ]
    assert len(annotation_rows) == 2
    for row in annotation_rows:
        answers = row["answers"]
        assert "category" in answers
        assert answers["category"]["kind"] == "Choice"
        assert "requires_login" in answers
        assert answers["requires_login"]["kind"] == "Noul"


def test_replay_uses_committed_vision_smoke_only() -> None:
    """Replay path uses tests/fixtures/psai/vision_smoke without scratchpad graft."""
    manifest = FIXTURE_ROOT / "manifest.json"
    assert manifest.is_file()
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    assert "scratchpad" not in result.receipt["fixture_root"]
    assert Path(result.receipt["fixture_root"]).resolve() == FIXTURE_ROOT.resolve()
    text = manifest.read_text(encoding="utf-8")
    assert "cmcc8u6ym018l1p1yxhf18gc2" in text


def test_historical_accounting_sidecar_documents_fourteen_vs_sixteen() -> None:
    """Sidecar corrects legacy 14 scoring_calls_used vs 16 scoring requests."""
    payload = json.loads(CORRECTION.read_text(encoding="utf-8"))
    assert payload["judgment_call_count"] == 14
    assert payload["scoring_request_count"] == 16
    assert payload["legacy_field_scoring_calls_used"] == 14


@pytest.mark.skipif(not CORD_PASS.is_file(), reason="CORD PASS fixture")
def test_cord_semantic_cli_exit_codes_unchanged() -> None:
    """#184 CLI still returns 0 on the vendored PASS combined receipt."""
    assert cord_semantic_cli_exit_code(CORD_PASS) == 0
    assert cord_cli_main([str(CORD_PASS)]) == 0
