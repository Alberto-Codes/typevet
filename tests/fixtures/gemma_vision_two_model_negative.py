"""Two-model negative fixture for factory model pin ([#196][i196]).

Examples:
    ```python
    from tests.fixtures.gemma_vision_two_model_negative import (
        PINNED_MODEL,
        OTHER_MODEL,
    )

    assert PINNED_MODEL != OTHER_MODEL
    ```

See Also:
    - [tests.contract.test_runtime_gemma_vision_factory][]: factory contract tests

[i196]: https://github.com/Alberto-Codes/typevet/issues/196
"""

from __future__ import annotations

PINNED_MODEL = "gemma-4-31b-kv9-q4km-mm"
OTHER_MODEL = "gemma-4-31b-kv9-q4km-mm-alt"
