"""Import probes that run in a fresh spawned interpreter (#284).

This module imports nothing from judgevet, so a spawned child that loads it
starts with judgevet absent from ``sys.modules``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Sequence
from types import ModuleType

BARE_PACKAGES: tuple[str, ...] = (
    "typevet",
    "typevet.adapters.diagnostics",
    "typevet.adapters.inbound",
    "typevet.adapters.outbound",
    "typevet.domain",
    "typevet.ports",
    "typevet.runtime",
    "typevet.testing",
)
"""Every public typevet package that a bare install imports."""


def judgevet_modules_after_bare_import() -> list[str]:
    """Import every public typevet package, then list loaded judgevet modules.

    Returns:
        The judgevet module names in ``sys.modules``; empty when none loaded.
    """
    for name in BARE_PACKAGES:
        importlib.import_module(name)
    return sorted(name for name in sys.modules if name.split(".")[0] == "judgevet")


class _BlockJudgevet(importlib.abc.MetaPathFinder):
    """Refuse to find judgevet, as if the extra were not installed."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Raise for judgevet and defer every other module to later finders.

        Args:
            fullname: The module being imported.
            path: The parent package path, unused.
            target: The module being reloaded, unused.

        Returns:
            None for every module that is not judgevet.

        Raises:
            ModuleNotFoundError: The module is judgevet or one of its children.
        """
        if fullname.split(".", maxsplit=1)[0] == "judgevet":
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def bridge_import_error_without_judgevet() -> str:
    """Import the bridge while judgevet is blocked, as in a bare install.

    Returns:
        The ImportError message, or an empty string if the import succeeded.
    """
    sys.meta_path.insert(0, _BlockJudgevet())
    try:
        importlib.import_module("typevet.adapters.inbound.judgevet")
    except ImportError as error:
        return str(error)
    return ""
