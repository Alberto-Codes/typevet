"""Resolve the package version from distribution metadata.

When the wheel or sdist is installed, ``__version__`` comes from
``importlib.metadata``. In editable checkouts without metadata, it falls back
to ``[project].version`` in ``pyproject.toml``.

Examples:
    ```python
    import typevet
    from importlib.metadata import version

    assert typevet.__version__ == version("typevet")
    ```

See Also:
    - [typevet][]: Re-exports ``__version__`` in ``__all__``
"""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["__version__"]


def _read_pyproject_version() -> str:
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = data["project"]["version"]
    if not isinstance(project_version, str):
        msg = "pyproject [project].version must be a string"
        raise TypeError(msg)
    return project_version


def _package_version() -> str:
    try:
        return version("typevet")
    except PackageNotFoundError:
        return _read_pyproject_version()


__version__ = _package_version()
