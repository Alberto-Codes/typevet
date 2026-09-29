"""The throughput and vLLM acceptance families live in the evals member (#256 E3)."""

from __future__ import annotations

import importlib
import importlib.util
import inspect

import pytest

_MOVED = {
    "typevet.evaluation.collections_metrics": "typevet_evals.throughput.collections_metrics",
    "typevet.evaluation.collections_throughput": (
        "typevet_evals.throughput.collections_throughput"
    ),
    "typevet.evaluation.collections_workload": "typevet_evals.throughput.collections_workload",
    "typevet.evaluation.public_workload": "typevet_evals.throughput.public_workload",
    "typevet.evaluation.vllm_acceptance": "typevet_evals.vllm_acceptance.core",
    "typevet.evaluation.vllm_acceptance_sets": "typevet_evals.vllm_acceptance.sets",
    "typevet.evaluation.vllm_acceptance_transport": "typevet_evals.vllm_acceptance.transport",
}


def _resolves(name: str) -> bool:
    """Return whether ``name`` resolves; a removed parent package means no."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.unit
@pytest.mark.parametrize(("old", "new"), sorted(_MOVED.items()))
def test_family_module_moved_to_the_member(old: str, new: str) -> None:
    assert not _resolves(old)
    assert importlib.import_module(new).__doc__


@pytest.mark.unit
@pytest.mark.parametrize(
    "package", ["typevet_evals.throughput", "typevet_evals.vllm_acceptance"]
)
def test_family_package_re_exports_its_public_names(package: str) -> None:
    module = importlib.import_module(package)
    assert module.__doc__
    assert module.__all__
    for name in module.__all__:
        value = getattr(module, name)
        if inspect.isclass(value) or inspect.isfunction(value):
            assert value.__module__.startswith(f"{package}."), name
