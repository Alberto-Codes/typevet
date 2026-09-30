"""Dataset loaders, download helpers and data policy guards (#147, #256).

Each dataset keeps its own module so a loader change touches one file. The
package re-exports the dataset submodules, not their names: constants such as
``DATASET_ID`` repeat across modules, so one flat namespace would collide.
The ``clinc_*.json`` data files ship next to ``clinc_shard``.

Examples:
    ```python
    from typevet_evals import datasets
    from typevet_evals.datasets import boolq
    from typevet_evals.datasets.clinc import load_plus_split

    assert "boolq" in datasets.__all__
    ```

See Also:
    - [typevet_evals.datasets.banking77][]: Banking77 fraud-intent proxy
    - [typevet_evals.datasets.boolq][]: BoolQ tiers and task export
    - [typevet_evals.datasets.civil_comments][]: Civil Comments toxicity proxy
    - [typevet_evals.datasets.clinc][]: CLINC150 ``plus`` shards
    - [typevet_evals.datasets.cord_expense][]: CORD expense claim cases
    - [typevet_evals.datasets.difraud][]: DIFrauD scam domains
    - [typevet_evals.datasets.go_emotions][]: GoEmotions single-label subset
    - [typevet_evals.datasets.hyperpartisan][]: Hyperpartisan holdout split
    - [typevet_evals.datasets.lfw][]: LFW View 2 face pairs
    - [typevet_evals.datasets.psai][]: PSAI metadata decisions
    - [typevet_evals.datasets.psai_evidence_pilot][]: evidence pilot manifest
    - [typevet_evals.datasets.psai_vision][]: PSAI screenshot fixtures
    - [typevet_evals.datasets.psai_vision_controls][]: image control matrix
    - [typevet_evals.datasets.pubmedqa][]: PubMedQA labeled subset
    - [typevet_evals.datasets.partner_guard][]: Partner data path guard

Attributes:
    banking77 (module): Banking77 loader.
    boolq (module): BoolQ loader and task export.
    boolq_download (module): BoolQ validation JSONL download.
    civil_comments (module): Civil Comments loader.
    clinc (module): CLINC150 ``plus`` loader.
    clinc_download (module): CLINC150 JSONL download.
    clinc_rows (module): CLINC150 row mapping.
    clinc_shard (module): CLINC150 domain catalog and schemas.
    cord_expense (module): CORD expense claim cases.
    difraud (module): DIFrauD loader.
    go_emotions (module): GoEmotions loader.
    go_emotions_download (module): GoEmotions download.
    hyperpartisan (module): Hyperpartisan loader.
    lfw (module): LFW View 2 pairs, slice and archive reader.
    partner_guard (module): Partner data path guard.
    psai (module): PSAI metadata loader.
    psai_download (module): PSAI parquet download.
    psai_evidence_pilot (module): PSAI evidence pilot manifest.
    psai_schema (module): PSAI metadata decision schema.
    psai_stream (module): PSAI dedupe, shuffle and JSONL.
    psai_vision (module): PSAI screenshot fixtures.
    psai_vision_controls (module): PSAI image control matrix.
    pubmedqa (module): PubMedQA loader.
"""

from typevet_evals.datasets import (
    banking77,
    boolq,
    boolq_download,
    civil_comments,
    clinc,
    clinc_download,
    clinc_rows,
    clinc_shard,
    cord_expense,
    difraud,
    go_emotions,
    go_emotions_download,
    hyperpartisan,
    lfw,
    partner_guard,
    psai,
    psai_download,
    psai_evidence_pilot,
    psai_schema,
    psai_stream,
    psai_vision,
    psai_vision_controls,
    pubmedqa,
)

__all__ = [
    "banking77",
    "boolq",
    "boolq_download",
    "civil_comments",
    "clinc",
    "clinc_download",
    "clinc_rows",
    "clinc_shard",
    "cord_expense",
    "difraud",
    "go_emotions",
    "go_emotions_download",
    "hyperpartisan",
    "lfw",
    "partner_guard",
    "psai",
    "psai_download",
    "psai_evidence_pilot",
    "psai_schema",
    "psai_stream",
    "psai_vision",
    "psai_vision_controls",
    "pubmedqa",
]
