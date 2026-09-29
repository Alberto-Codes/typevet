"""Build the typevet wheel into a directory (packaging tests only).

Examples:
    Build into a temp directory from the repo root:

    ```console
    $ uv run python scripts/build_wheel_for_tests.py /tmp/typevet-dist
    ```

See Also:
    - [typevet_evals.wheel_isolated][]: implementation module
    - [tests.unit.test_package_wheel][]: Wheel content checks
"""

from __future__ import annotations

import sys

from typevet_evals.wheel_isolated import (
    build_wheel_to_directory,
    main,
    run_isolated_wheel_python,
)

__all__ = ["build_wheel_to_directory", "main", "run_isolated_wheel_python"]

if __name__ == "__main__":
    sys.exit(main())
