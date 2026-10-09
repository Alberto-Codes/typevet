"""Public-dataset workloads for the throughput runner ([#236][i236]).

Banking77 (CC BY 4.0, PolyAI) and DIFrauD SMS (MIT) become text records with a
``positive`` label, the partner questions and a parity baseline. The question
text is verbatim from the partner project's question set at a pinned commit. The
data is not in the repository. ``fetch_public_data`` downloads the two test
files with the typevet loaders into one directory, and
``load_public_workloads`` reads them from that directory offline.

Attributes:
    PUBLIC_DATASET_ENV (str): Env var for the directory with both data files.
    BANKING77_FILE (str): File name of the Banking77 test CSV.
    DIFRAUD_FILE (str): File name of the DIFrauD SMS test JSONL.
    MAX_CHOICE_OPTIONS (int): Highest option count of one Choice; equal to
        typevet's ``MAX_ENUM_CHOICES`` (24, #296).
    DIFRAUD_LIMIT (int): DIFrauD SMS rows in the parity set.
    SEED (int): Seed for the loaders' hash-based order.
    BANKING77_BASELINE (Baseline): Partner ECE 0.17 on the balanced 480 rows.
    DIFRAUD_BASELINE (Baseline): Partner ECE 0.07 on 500 SMS rows, 98 scam.

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.throughput.public_workload import load_public_workloads

    sets = load_public_workloads(Path("scratchpad/public-data"))
    work = sets["banking77_balanced"]
    ```

See Also:
    - [typevet_evals.throughput.collections_throughput][]: the runner
    - [typevet_evals.datasets.banking77][]: Banking77 loader
    - [typevet_evals.datasets.difraud][]: DIFrauD loader

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

from typevet.domain import MAX_ENUM_CHOICES
from typevet.domain.judgment_questions import Choice, Noul
from typevet_evals.datasets import banking77, difraud
from typevet_evals.throughput.collections_workload import Baseline

PUBLIC_DATASET_ENV: Final[str] = "TYPEVET_PUBLIC_DATASET"
BANKING77_FILE: Final[str] = "banking77_test.csv"
DIFRAUD_FILE: Final[str] = "difraud_sms_test.jsonl"
MAX_CHOICE_OPTIONS: Final[int] = MAX_ENUM_CHOICES
DIFRAUD_LIMIT: Final[int] = 500
SEED: Final[int] = 0
BANKING77_BASELINE: Final[Baseline] = Baseline(ece=0.17, base_rate=0.5, max_ece=0.20)
DIFRAUD_BASELINE: Final[Baseline] = Baseline(ece=0.07, base_rate=98 / 500, max_ece=0.10)

_BANKING77_QUESTIONS: Final[dict[str, Noul | Choice]] = {
    "reports_unauthorized": Noul(
        instructions="Does the customer report a transaction they did not authorize?"
    ),
    "fraud_type": Choice(
        criteria={
            "unauthorized_transaction": "The customer reports a charge or withdrawal "
            "they did not make.",
            "duplicate_charge": "The customer reports being charged more than once "
            "for one purchase.",
            "phishing_or_scam": "The customer was deceived into paying or sharing "
            "credentials.",
            "account_takeover": "Someone else gained control of the customer's "
            "account or card.",
            "not_fraud": "The message concerns an ordinary request with no fraud.",
            "unclear": "The message does not say enough to decide.",
        },
        instructions="Which kind of fraud, if any, does the customer describe?",
    ),
}
_DIFRAUD_QUESTIONS: Final[dict[str, Noul | Choice]] = {
    "is_scam": Noul(
        instructions="Is this message a scam, phishing or social-engineering attempt?"
    ),
}


@dataclass(frozen=True, slots=True)
class PublicRecord:
    """One public record: the text sent to the port and its label.

    Attributes:
        state (str): Record text, sent as the state.
        positive (bool): True for Banking77 ``fraud`` or DIFrauD ``scam``.

    Examples:
        ```python
        PublicRecord(state="I did not make this payment", positive=True)
        ```
    """

    state: str
    positive: bool


@dataclass(frozen=True, slots=True)
class PublicWorkload:
    """Records, questions, scored Noul name and baseline of one set.

    Attributes:
        records (tuple[PublicRecord, ...]): Records in loader order.
        questions (dict[str, Noul | Choice]): Questions sent in each call.
        noul (str): Question whose probability parity scores.
        baseline (Baseline | None): Parity reference; ``None`` records only.

    Examples:
        ```python
        work = PublicWorkload((), {}, "is_scam", None)
        ```
    """

    records: tuple[PublicRecord, ...]
    questions: dict[str, Noul | Choice]
    noul: str
    baseline: Baseline | None


def validate_questions(questions: Mapping[str, Noul | Choice]) -> None:
    """Reject a Choice with more options than the native limit.

    Args:
        questions: Questions of one workload.

    Raises:
        ValueError: When a Choice has more than ``MAX_CHOICE_OPTIONS`` options.
    """
    for name, question in questions.items():
        if isinstance(question, Choice) and len(question.criteria) > MAX_CHOICE_OPTIONS:
            count = len(question.criteria)
            msg = (
                f"Choice {name!r} has {count} options; "
                f"the limit is {MAX_CHOICE_OPTIONS}"
            )
            raise ValueError(msg)


def _workload(
    records: Sequence[PublicRecord],
    questions: Mapping[str, Noul | Choice],
    noul: str,
    baseline: Baseline | None,
) -> PublicWorkload:
    validate_questions(questions)
    return PublicWorkload(tuple(records), dict(questions), noul, baseline)


def banking77_workload(
    examples: Sequence[banking77.Banking77Example],
    baseline: Baseline | None = BANKING77_BASELINE,
) -> PublicWorkload:
    """Build the Banking77 workload; ``fraud`` is positive.

    Args:
        examples: Rows from ``banking77.load_test_split``.
        baseline: Parity reference; ``None`` records only.

    Returns:
        Text records, the ``reports_unauthorized`` Noul, the ``fraud_type``
        Choice and the baseline.
    """
    records = [PublicRecord(e.text, e.proxy_label == "fraud") for e in examples]
    return _workload(records, _BANKING77_QUESTIONS, "reports_unauthorized", baseline)


def difraud_workload(
    examples: Sequence[difraud.DIFrauDExample],
    baseline: Baseline | None = DIFRAUD_BASELINE,
) -> PublicWorkload:
    """Build the DIFrauD workload; ``scam`` is positive.

    Args:
        examples: Rows from ``difraud.load_test_split``.
        baseline: Parity reference; ``None`` records only.

    Returns:
        Text records, the ``is_scam`` Noul and the baseline.
    """
    records = [PublicRecord(e.text, e.label == "scam") for e in examples]
    return _workload(records, _DIFRAUD_QUESTIONS, "is_scam", baseline)


def missing_data(directory: Path) -> str | None:
    """Return why the data directory cannot feed a run, or ``None``.

    Args:
        directory: Directory that ``fetch_public_data`` filled.

    Returns:
        A reason that names each absent file and the fetch function.
    """
    absent = [
        name
        for name in (BANKING77_FILE, DIFRAUD_FILE)
        if not (directory / name).is_file()
    ]
    if not absent:
        return None
    return (
        f"public data files not found in {PUBLIC_DATASET_ENV}: {absent}; "
        "run typevet_evals.throughput.public_workload.fetch_public_data first"
    )


def fetch_public_data(
    directory: Path, *, client: httpx.Client | None = None
) -> dict[str, int]:
    """Download the Banking77 test CSV and the DIFrauD SMS test JSONL.

    Args:
        directory: Target directory; it is created when absent.
        client: Optional HTTP client; the loaders open one when ``None``.

    Returns:
        Bytes written per file name.
    """
    directory.mkdir(parents=True, exist_ok=True)
    texts = {
        BANKING77_FILE: banking77.download_test_csv(client=client),
        DIFRAUD_FILE: difraud.download_test_jsonl(client=client),
    }
    for name, text in texts.items():
        (directory / name).write_text(text, encoding="utf-8")
    return {name: len(text.encode()) for name, text in texts.items()}


def load_public_workloads(directory: Path) -> dict[str, PublicWorkload]:
    """Build the three #236 sets from the fetched files, offline.

    Args:
        directory: Directory that ``fetch_public_data`` filled.

    Returns:
        ``banking77_balanced`` (every fraud row and as many other rows),
        ``difraud_sms`` (``DIFRAUD_LIMIT`` rows, ``SEED``) and
        ``banking77_full`` (the whole split, record only).
    """
    csv_text = (directory / BANKING77_FILE).read_text(encoding="utf-8")
    jsonl_text = (directory / DIFRAUD_FILE).read_text(encoding="utf-8")
    balanced = banking77.load_test_split(csv_text=csv_text, balanced=True, seed=SEED)
    sms = difraud.load_test_split(jsonl_text=jsonl_text, limit=DIFRAUD_LIMIT, seed=SEED)
    return {
        "banking77_balanced": banking77_workload(balanced),
        "difraud_sms": difraud_workload(sms),
        "banking77_full": banking77_workload(
            banking77.load_test_split(csv_text=csv_text), None
        ),
    }
