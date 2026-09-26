"""Versioned PSAI metadata Decision schemas and label catalogs (#56 / #71).

Examples:
    Read the combined Decision schema constant:

    ```python
    from typevet.eval_psai_schema import METADATA_DECISIONS_SCHEMA

    assert "category" in METADATA_DECISIONS_SCHEMA["properties"]
    ```

See Also:
    - [typevet.eval_psai][]: loader and export
    - docs/reference/eval-psai-metadata-map.md
"""

from __future__ import annotations

from typing import Any, Final

SOURCE: Final[str] = "psai"
FAMILY: Final[str] = "psai"
SPLIT: Final[str] = "train"
DATASET_ID: Final[str] = "anaisleila/computer-use-data-psai"
DATASET_LICENSE: Final[str] = "MIT"
DEFAULT_SAMPLE_SEED: Final[int] = 0
CONTRACT_ROW_MIN: Final[int] = 12
CONTRACT_ROW_MAX: Final[int] = 24
TASK_NAME_COLUMN: Final[str] = "task_name"
PRIMARY_NOUL_NAME: Final[str] = "requires_login"
SUB_CATEGORY_CHOICE_NAME: Final[str] = "sub_category"
OTHER_SUB_CATEGORY: Final[str] = "OTHER"
METADATA_SCHEMA_VERSION: Final[str] = "1"

CATEGORY_LABELS: Final[tuple[str, ...]] = ("BROWSER_TASK", "COMPUTER_TASK")
DIFFICULTY_LABELS: Final[tuple[str, ...]] = ("EASY", "MEDIUM", "HARD")
BENCHMARK_LABELS: Final[tuple[str, ...]] = (
    "Web Bench",
    "Proprietary",
    "Web Voyager",
)
APP_TYPE_LABELS: Final[tuple[str, ...]] = ("SINGLE_APP", "MULTI_APP")
OS_LABELS: Final[tuple[str, ...]] = (
    "CROSS_PLATFORM",
    "WINDOWS",
    "MAC",
    "LINUX",
)
SUB_CATEGORY_LABELS: Final[tuple[str, ...]] = (
    "Development & Tech Services",
    "Document Editing",
    "Education & Learning",
    "Email Ops",
    "File Management",
    "Finance & Banking",
    "Health & Wellness",
    "Navigation & Maps",
    "News & Media",
    "Other / Miscellaneous",
    "Presentations",
    "Search & Research",
    "Shopping & E-commerce",
    "Social Media & Communication",
    "Utilities",
    "Utilities & Tools",
    "Web Settings",
    OTHER_SUB_CATEGORY,
)

BANNED_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "category",
        "difficulty",
        "benchmark",
        "appType",
        "os",
        "requires_login",
        "subCategory",
        "sub_category",
        "expected",
        "label",
        "ground_truth",
        "unique_data_id",
        "screenshots",
        "events",
    }
)

METADATA_DECISIONS_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": list(CATEGORY_LABELS),
            "instructions": "Is this a browser task or a computer (desktop) task?",
        },
        "difficulty": {
            "type": "string",
            "enum": list(DIFFICULTY_LABELS),
            "instructions": "What difficulty label applies to this task?",
        },
        "benchmark": {
            "type": "string",
            "enum": list(BENCHMARK_LABELS),
            "instructions": "Which upstream benchmark bucket sourced this task?",
        },
        "appType": {
            "type": "string",
            "enum": list(APP_TYPE_LABELS),
            "instructions": "Does the task stay in one app or span multiple apps?",
        },
        "os": {
            "type": "string",
            "enum": list(OS_LABELS),
            "instructions": "Which OS scope label applies to this task?",
        },
        PRIMARY_NOUL_NAME: {
            "type": "boolean",
            "instructions": "Does the task require the user to be logged in?",
        },
        SUB_CATEGORY_CHOICE_NAME: {
            "type": "string",
            "enum": list(SUB_CATEGORY_LABELS),
            "instructions": (
                "Which sub-category best describes the task (first Hub tag)?"
            ),
        },
    },
    "required": [
        "category",
        "difficulty",
        "benchmark",
        "appType",
        "os",
        PRIMARY_NOUL_NAME,
        SUB_CATEGORY_CHOICE_NAME,
    ],
    "additionalProperties": False,
}

METADATA_MANIFEST: Final[dict[str, Any]] = {
    "manifest_version": 1,
    "loader_issue": 71,
    "design_issue": 56,
    "research_issue": 52,
    "dataset_id": DATASET_ID,
    "split": SPLIT,
    "license": DATASET_LICENSE,
    "default_sample_seed": DEFAULT_SAMPLE_SEED,
    "contract_row_bounds": {"min": CONTRACT_ROW_MIN, "max": CONTRACT_ROW_MAX},
    "non_goals": [
        "screenshot/video/DOM decode (#57)",
        "event-count Score in v1",
        "full 49GB corpus in CI",
        "ECE or calibration claims (#50)",
    ],
}
