"""Runtime re-export for ChatML scoring-prefix composition (#148).

Examples:
    ```python
    from typevet.runtime.scoring_prefix import compose_scoring_prefix
    ```

See Also:
    - [typevet.adapters.outbound.gemma.scoring_prefix][]: Canonical implementation
"""

from typevet.adapters.outbound.gemma.scoring_prefix import compose_scoring_prefix

__all__ = ["compose_scoring_prefix"]
