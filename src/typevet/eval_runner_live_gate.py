"""Compatibility shim for the live eval gate (#147).

Prefer importing from ``typevet.evaluation.runner.live_gate`` in new code.

Examples:
    ```python
    from typevet.eval_runner_live_gate import live_skip_reason
    ```

See Also:
    - [typevet.evaluation.runner.live_gate][]: New home for this module
"""

from typevet.evaluation.runner.live_gate import (
    LlamaSettings,
    live_skip_reason,
)

__all__ = [
    "LlamaSettings",
    "live_skip_reason",
]
