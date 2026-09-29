"""Contract tests for the library package layout before 0.1.0 (#256).

The root library shims and the root ``eval_*`` shims are removed, and
``question_schema`` lives in ``typevet.domain``. Each old root path must not
resolve, and each new path must import.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

pytestmark = pytest.mark.contract

REMOVED_ROOT_MODULES: tuple[str, ...] = (
    "typevet.cord_semantic_acceptance_cli",
    "typevet.decide_categorical",
    "typevet.eval_banking77",
    "typevet.eval_boolq",
    "typevet.eval_boolq_download",
    "typevet.eval_civil_comments",
    "typevet.eval_clinc",
    "typevet.eval_clinc_download",
    "typevet.eval_clinc_rows",
    "typevet.eval_clinc_shard",
    "typevet.eval_difraud",
    "typevet.eval_go_emotions",
    "typevet.eval_go_emotions_download",
    "typevet.eval_hyperpartisan",
    "typevet.eval_partner_guard",
    "typevet.eval_psai",
    "typevet.eval_psai_download",
    "typevet.eval_psai_schema",
    "typevet.eval_psai_stream",
    "typevet.eval_pubmedqa",
    "typevet.eval_runner",
    "typevet.eval_runner_cli",
    "typevet.eval_runner_datasets",
    "typevet.eval_runner_live_gate",
    "typevet.eval_runner_report",
    "typevet.eval_tpjep_loader",
    "typevet.eval_tpjep_outcome",
    "typevet.eval_tpjep_records",
    "typevet.eval_tpjep_runner",
    "typevet.field_prompt",
    "typevet.gemma_answer_binding",
    "typevet.gemma_served_template",
    "typevet.judge",
    "typevet.question_schema",
)

NEW_MODULES: tuple[str, ...] = ("typevet.domain.question_schema",)

QUESTION_SCHEMA_NAMES: tuple[str, ...] = (
    "compile_question_records",
    "question_record_to_property",
    "question_records_to_json_schema",
)


@pytest.mark.parametrize("old_name", REMOVED_ROOT_MODULES)
def test_removed_root_module_does_not_resolve(old_name: str) -> None:
    assert importlib.util.find_spec(old_name) is None


@pytest.mark.parametrize("new_name", NEW_MODULES)
def test_new_module_imports(new_name: str) -> None:
    importlib.import_module(new_name)


@pytest.mark.parametrize("name", QUESTION_SCHEMA_NAMES)
def test_domain_reexports_question_schema_names(name: str) -> None:
    domain = importlib.import_module("typevet.domain")
    module = importlib.import_module("typevet.domain.question_schema")
    assert name in domain.__all__
    assert name in module.__all__
    assert getattr(domain, name) is getattr(module, name)
