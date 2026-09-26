"""Compatibility shim for M1 categorical decisions (#148).

Examples:
    ```python
    from typevet import decide_categorical
    ```

See Also:
    - [typevet.runtime.categorical][]: Canonical implementation
"""

from typevet.runtime.categorical import decide_categorical

__all__ = ["decide_categorical"]
