"""Compatibility shim for the partner data guard (#147).

Prefer importing from ``typevet.evaluation.datasets.partner_guard`` in new code.

Examples:
    ```python
    from typevet.eval_partner_guard import normalize_posix
    ```

See Also:
    - [typevet.evaluation.datasets.partner_guard][]: New home for this module
"""

from typevet.evaluation.datasets.partner_guard import (
    CONTENT_ALLOWLIST,
    FORBIDDEN_IMPORT_MARKERS,
    FORBIDDEN_PATH_MARKERS,
    PACKAGING_FILES,
    normalize_posix,
    packaging_line_is_forbidden,
    path_is_forbidden,
    scan_packaging_config,
    scan_tracked_content,
    scan_tracked_paths,
    scan_tree_paths,
)

__all__ = [
    "CONTENT_ALLOWLIST",
    "FORBIDDEN_IMPORT_MARKERS",
    "FORBIDDEN_PATH_MARKERS",
    "PACKAGING_FILES",
    "normalize_posix",
    "packaging_line_is_forbidden",
    "path_is_forbidden",
    "scan_packaging_config",
    "scan_tracked_content",
    "scan_tracked_paths",
    "scan_tree_paths",
]
