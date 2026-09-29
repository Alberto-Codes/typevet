"""Contract: the instruction-variant family lives in the evals member (#256 E4).

The eight ``typevet.evaluation.instruction_variant_consumer_*`` modules move
to ``typevet_evals.instruction_variant`` and ``outcome_replay_metrics`` moves
to the member root. Each old path must not resolve, and each new path must
import.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

pytestmark = pytest.mark.contract

# Old module path -> new module path.
MOVED_MODULES: dict[str, str] = {
    "typevet.evaluation.instruction_variant_consumer_live": (
        "typevet_evals.instruction_variant.live"
    ),
    "typevet.evaluation.instruction_variant_consumer_live_router": (
        "typevet_evals.instruction_variant.live_router"
    ),
    "typevet.evaluation.instruction_variant_consumer_matrix": (
        "typevet_evals.instruction_variant.matrix"
    ),
    "typevet.evaluation.instruction_variant_consumer_offline": (
        "typevet_evals.instruction_variant.offline"
    ),
    "typevet.evaluation.instruction_variant_consumer_proof": (
        "typevet_evals.instruction_variant.proof"
    ),
    "typevet.evaluation.instruction_variant_consumer_protocol": (
        "typevet_evals.instruction_variant.protocol"
    ),
    "typevet.evaluation.instruction_variant_consumer_receipt": (
        "typevet_evals.instruction_variant.receipt"
    ),
    "typevet.evaluation.instruction_variant_consumer_run": (
        "typevet_evals.instruction_variant.run"
    ),
    "typevet.evaluation.outcome_replay_metrics": (
        "typevet_evals.outcome_replay_metrics"
    ),
}


def _resolves(name: str) -> bool:
    """Return whether ``name`` resolves; a removed parent package means no."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize("old_name", sorted(MOVED_MODULES))
def test_old_module_path_does_not_resolve(old_name: str) -> None:
    assert not _resolves(old_name)


@pytest.mark.parametrize("new_name", sorted(MOVED_MODULES.values()))
def test_new_module_path_imports(new_name: str) -> None:
    importlib.import_module(new_name)


def test_package_reexports_are_the_defining_objects() -> None:
    package = importlib.import_module("typevet_evals.instruction_variant")
    assert package.__doc__
    assert package.__all__
    for name in package.__all__:
        exported = getattr(package, name)
        defining = importlib.import_module(exported.__module__)
        assert getattr(defining, name) is exported
