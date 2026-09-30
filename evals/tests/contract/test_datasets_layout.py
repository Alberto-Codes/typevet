"""Contract: the dataset loaders live in the evals member (#256 E8).

The ``typevet.evaluation.datasets`` package moves to ``typevet_evals.datasets``
with its JSON data files, and the ``typevet.evaluation`` package is removed
from the library. Each old path must not resolve, each new path must import,
and the package ``__all__`` lists its dataset submodules.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pkgutil
import zipfile
from pathlib import Path

import pytest

from typevet_evals.wheel_isolated import build_member_wheel_to_directory

pytestmark = pytest.mark.contract

_DATASET_MODULES = (
    "banking77",
    "boolq",
    "boolq_download",
    "civil_comments",
    "clinc",
    "clinc_download",
    "clinc_rows",
    "clinc_shard",
    "cord_expense",
    "difraud",
    "go_emotions",
    "go_emotions_download",
    "hyperpartisan",
    "lfw",
    "partner_guard",
    "psai",
    "psai_download",
    "psai_evidence_pilot",
    "psai_schema",
    "psai_stream",
    "psai_vision",
    "psai_vision_controls",
    "pubmedqa",
)
_DATA_FILES = ("clinc_domains.json", "clinc_plus_intent_names.json")

# Old module path -> new module path.
MOVED_MODULES: dict[str, str] = {
    "typevet.evaluation.datasets": "typevet_evals.datasets",
    **{
        f"typevet.evaluation.datasets.{name}": f"typevet_evals.datasets.{name}"
        for name in _DATASET_MODULES
    },
}


def _resolves(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def test_evaluation_package_is_removed_from_the_library() -> None:
    assert not _resolves("typevet.evaluation")


@pytest.mark.parametrize("old_name", sorted(MOVED_MODULES))
def test_old_module_path_does_not_resolve(old_name: str) -> None:
    assert not _resolves(old_name)


@pytest.mark.parametrize("new_name", sorted(MOVED_MODULES.values()))
def test_new_module_path_imports(new_name: str) -> None:
    importlib.import_module(new_name)


def test_datasets_all_lists_every_dataset_submodule() -> None:
    package = importlib.import_module("typevet_evals.datasets")
    assert package.__doc__
    found = sorted(info.name for info in pkgutil.iter_modules(package.__path__))
    assert found == sorted(_DATASET_MODULES)
    assert package.__all__ == list(_DATASET_MODULES)


@pytest.mark.parametrize("name", _DATASET_MODULES)
def test_datasets_reexports_are_the_submodules(name: str) -> None:
    package = importlib.import_module("typevet_evals.datasets")
    submodule = importlib.import_module(f"typevet_evals.datasets.{name}")
    assert getattr(package, name) is submodule


@pytest.mark.parametrize("data_file", _DATA_FILES)
def test_data_file_loads_from_the_source_tree(data_file: str) -> None:
    package = importlib.import_module("typevet_evals.datasets")
    package_dir = Path(package.__file__ or "").resolve().parent
    assert json.loads((package_dir / data_file).read_text(encoding="utf-8"))


def test_member_wheel_ships_the_dataset_data_files(tmp_path: Path) -> None:
    try:
        build_member_wheel_to_directory(tmp_path)
    except RuntimeError as exc:
        if str(exc) == "uv not on PATH":
            pytest.skip(str(exc))
        raise
    wheels = sorted(tmp_path.glob("typevet_evals-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
    for data_file in _DATA_FILES:
        assert f"typevet_evals/datasets/{data_file}" in names
