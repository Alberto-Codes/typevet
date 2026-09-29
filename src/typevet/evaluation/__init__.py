"""Evaluation harnesses: dataset loaders (#147).

Evaluation code drives the library from the outside. It may import adapters,
ports and the domain; nothing in the domain may import it. The eval runner,
the TPJEP protocol and the experiment identity helpers now live in the
``typevet-evals`` workspace member (``typevet_evals``, #256).

Examples:
    ```python
    from typevet.evaluation.datasets import boolq
    ```

See Also:
    - [typevet.evaluation.datasets][]: Dataset loaders and download helpers

Attributes:
    None: This package provides organizational structure only.
"""
