"""Evaluation harnesses: dataset loaders, eval runners and TPJEP (#147).

Evaluation code drives the library from the outside. It may import adapters,
ports and the domain; nothing in the domain may import it.

Examples:
    ```python
    from typevet.evaluation.runner import load_eval_tasks
    from typevet.evaluation.tpjep import load_eight_task_fixture
    ```

See Also:
    - [typevet.evaluation.datasets][]: Dataset loaders and download helpers
    - [typevet.evaluation.runner][]: Loader eval runner, gate and reports
    - [typevet.evaluation.tpjep][]: TPJEP fixture, records and runner
    - [typevet.adapters.inbound.eval_cli][]: CLI entry for the eval runner

Attributes:
    None: This package provides organizational structure only.
"""
