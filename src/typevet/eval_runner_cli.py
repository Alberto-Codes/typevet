r"""Compatibility shim for the live eval runner CLI (#147).

The module path stays runnable so pinned commands keep working. Prefer
``typevet.adapters.inbound.eval_cli`` in new code.

Examples:
    ```console
    $ uv run python -m typevet.eval_runner_cli --dataset boolq --limit 2
    dataset=boolq\tmetric=exact_match\tlimit=2\tattempted=2\tschema_valid=2\tgold_match=1
    ```

See Also:
    - [typevet.adapters.inbound.eval_cli][]: New home for this module
"""

from typevet.adapters.inbound.eval_cli import (
    SUPPORTED_DATASETS,
    EvalDatasetName,
    EvalRunReport,
    GenerationPort,
    format_report,
    live_skip_reason,
    llama_cpp_adapter,
    load_eval_tasks,
    load_llama_settings,
    main,
    run_eval_tasks,
)

__all__ = [
    "SUPPORTED_DATASETS",
    "EvalDatasetName",
    "EvalRunReport",
    "GenerationPort",
    "format_report",
    "live_skip_reason",
    "llama_cpp_adapter",
    "load_eval_tasks",
    "load_llama_settings",
    "main",
    "run_eval_tasks",
]

if __name__ == "__main__":
    raise SystemExit(main())
