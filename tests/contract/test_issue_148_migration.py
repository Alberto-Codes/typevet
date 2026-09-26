"""Contract tests for #148 package migration shims and prompt bytes."""

from __future__ import annotations

import importlib
import inspect

import pytest

from typevet.domain import decision_execute as decision_execute_mod
from typevet.domain.decisions import Decision
from typevet.field_prompt import compose_scoring_prefix, render_field_instructions
from typevet.ports.scoring import CandidateScoringPort

_GOLDEN_PREFIX = (
    "<|im_start|>user\nTask body.\n\n"
    "label: Pick one.\n\n"
    "Options:\n"
    "- a\n"
    "- b<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@pytest.mark.contract
def test_field_prompt_root_shim_exports() -> None:
    fp = importlib.import_module("typevet.field_prompt")
    assert callable(fp.render_field_instructions)
    assert callable(fp.compose_scoring_prefix)
    assert callable(fp.gold_reference_markers)
    assert callable(fp.choice_criteria_from_schema)


@pytest.mark.contract
def test_gemma_root_shims_export_public_names() -> None:
    served = importlib.import_module("typevet.gemma_served_template")
    binding = importlib.import_module("typevet.gemma_answer_binding")
    assert hasattr(served, "classify_served_template")
    assert hasattr(served, "CHATML_ASSISTANT_HEADER")
    assert hasattr(binding, "resolve_answer_anchor")
    assert hasattr(binding, "ThinkingDisposition")


@pytest.mark.contract
def test_runtime_facade_modules_importable() -> None:
    importlib.import_module("typevet.runtime.categorical")
    importlib.import_module("typevet.runtime.judgment")
    importlib.import_module("typevet.runtime.scoring_prefix")
    importlib.import_module("typevet.adapters.outbound.gemma.served_template")


@pytest.mark.contract
def test_compose_scoring_prefix_bytes_unchanged() -> None:
    decision = Decision("label", "Pick one.", ("a", "b"), syntax="Choice")
    block = render_field_instructions(decision)
    prefix = compose_scoring_prefix(context="Task body.", field_block=block)
    assert prefix == _GOLDEN_PREFIX


@pytest.mark.contract
def test_candidate_scoring_port_single_owner() -> None:
    assert not hasattr(decision_execute_mod, "CandidateScoringPort")
    sig = inspect.signature(decision_execute_mod.execute_categorical_decision)
    port_param = sig.parameters["port"]
    assert "CandidateScoringPort" in str(port_param.annotation)
    assert CandidateScoringPort.__module__ == "typevet.ports.scoring"
