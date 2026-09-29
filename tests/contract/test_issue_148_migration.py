"""Contract tests for #148 package migration and prompt bytes.

The root shims are removed (#256); see ``tests/contract/test_package_layout.py``.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from typevet.adapters.outbound.gemma.scoring_prefix import compose_scoring_prefix
from typevet.domain import decision_execute as decision_execute_mod
from typevet.domain.decisions import Decision
from typevet.domain.field_instructions import render_field_instructions
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
