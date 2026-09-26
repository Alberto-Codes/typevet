r"""Compatibility shim for the CORD semantic acceptance operator CLI (#184).

Examples:
    ```console
    $ uv run python -m typevet.cord_semantic_acceptance_cli \\
        tests/fixtures/cord/semantic_acceptance/labeled_synthetic_pass.json
    ```

See Also:
    - [typevet.adapters.inbound.cord_semantic_acceptance_cli][]: CLI implementation
"""

from typevet.adapters.inbound.cord_semantic_acceptance_cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
