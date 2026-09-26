r"""Check a saved CORD combined receipt against #161 semantic floors (#184).

Examples:
    ```console
    $ uv run python scripts/check_cord_semantic_acceptance.py \\
        tests/fixtures/cord/expense_smoke/gemma4_kv9_direct_receipt.json
    ```

See Also:
    - [typevet.adapters.inbound.cord_semantic_acceptance_cli][]: CLI implementation
    - [typevet.evaluation.cord_semantic_acceptance][]: acceptance floors
"""

from __future__ import annotations

from typevet.adapters.inbound.cord_semantic_acceptance_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
