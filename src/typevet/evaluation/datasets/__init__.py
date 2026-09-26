"""Dataset loaders, download helpers and data policy guards (#147).

Each dataset keeps its own module so a loader change touches one file. Import
the submodule you need; this package stays import-light on purpose.

Examples:
    ```python
    from typevet.evaluation.datasets.boolq import load_tier_a
    from typevet.evaluation.datasets.clinc import load_plus_split
    ```

See Also:
    - [typevet.evaluation.datasets.banking77][]: Banking77 fraud-intent proxy
    - [typevet.evaluation.datasets.boolq][]: BoolQ tiers and task export
    - [typevet.evaluation.datasets.civil_comments][]: Civil Comments toxicity proxy
    - [typevet.evaluation.datasets.clinc][]: CLINC150 ``plus`` shards
    - [typevet.evaluation.datasets.difraud][]: DIFrauD scam domains
    - [typevet.evaluation.datasets.go_emotions][]: GoEmotions single-label subset
    - [typevet.evaluation.datasets.hyperpartisan][]: Hyperpartisan holdout split
    - [typevet.evaluation.datasets.psai][]: PSAI metadata decisions
    - [typevet.evaluation.datasets.psai_vision][]: PSAI screenshot fixtures
    - [typevet.evaluation.datasets.psai_vision_controls][]: image control matrix
    - [typevet.evaluation.datasets.pubmedqa][]: PubMedQA labeled subset
    - [typevet.evaluation.datasets.partner_guard][]: Partner data path guard

Attributes:
    None: This package provides organizational structure only.
"""
