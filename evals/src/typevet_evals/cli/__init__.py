r"""Module-entry commands over the evaluation tooling (#256 E2, E5).

Each submodule runs with ``uv run python -m typevet_evals.cli.<name>``. The
typevet wheel ships no command.

Attributes:
    __all__ (list[str]): The command submodule names.

Examples:
    ```console
    $ uv run python -m typevet_evals.cli.eval_runner --help
    $ uv run python -m typevet_evals.cli.cord_semantic_acceptance \\
        tests/fixtures/cord/semantic_acceptance/labeled_synthetic_pass.json
    $ uv run python -m typevet_evals.cli.check_sheets --seed 1 --out DIR
    $ uv run python -m typevet_evals.cli.doodle_duel --receipt out.json
    $ uv run python -m typevet_evals.cli.doodle_duel_play \\
        --source-receipt out.json --duel-receipt duel.json
    ```

See Also:
    - [typevet_evals.cli.eval_runner][]: Loader eval runner command
    - [typevet_evals.cli.cord_semantic_acceptance][]: CORD receipt acceptance
    - [typevet_evals.cli.check_sheets][]: Offline check contact sheets
    - [typevet_evals.cli.doodle_duel][]: Quick, Draw! doodle Choice run
    - [typevet_evals.cli.doodle_duel_play][]: Doodle duel against a receipt
"""

__all__ = [
    "check_sheets",
    "cord_semantic_acceptance",
    "doodle_duel",
    "doodle_duel_play",
    "eval_runner",
]
