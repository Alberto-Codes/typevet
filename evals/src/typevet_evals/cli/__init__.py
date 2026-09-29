"""Module-entry commands over the evaluation tooling (#256 E2).

Each submodule runs with ``uv run python -m typevet_evals.cli.<name>``. The
typevet wheel ships no command.

Attributes:
    __all__ (list[str]): The command submodule names.

Examples:
    ```console
    $ uv run python -m typevet_evals.cli.eval_runner --help
    ```

See Also:
    - [typevet_evals.cli.eval_runner][]: Loader eval runner command
"""

__all__ = ["eval_runner"]
