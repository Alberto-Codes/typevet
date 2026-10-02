"""Offline judgment session for ``TYPEVET_BACKEND=fake``.

The fake backend needs no server. ``open_fake_judgment`` reads the JSON file
that ``TYPEVET_FAKE__DISTRIBUTIONS`` names and returns a
``FakeJudgmentSession``. Its port builds a ``ScriptedJudgmentFake`` on each
``judge`` call. The file is an object keyed by question name. A Noul takes a
number, a Choice takes an object of label to weight, and a Score takes an
object of level string to weight. Each question that the file does not name
gets a uniform distribution: 0.5 for a Noul, one equal weight for each Choice
label, and one equal weight for each Score level. When the variable is unset
or empty, every question is uniform. Error messages name the variable, never
the file content, and a parse failure raises with no cause or context. The
fake session has no HTTP client, so a caller that needs ``session.client``
passes the session through ``require_live_session`` first.

Examples:
    ```python
    from typevet.adapters.inbound.fake_backend import open_fake_judgment
    from typevet.domain.judgment_questions import Noul

    session = open_fake_judgment({})
    response = session.port.judge("text", {"billing": Noul()}, session.model)
    assert response.nouls["billing"].noul == 0.5
    # require_live_session(session) raises ValueError for this session.
    ```

See Also:
    - [typevet.adapters.inbound.backend_settings][]: ``open_judgment``
    - [typevet.testing.judgment_fake][]: ``ScriptedJudgmentFake``
    - docs/reference/configuration.md: Environment variable reference
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typevet.domain.judgment_questions import Choice, Score
from typevet.testing import ScriptedJudgmentFake

if TYPE_CHECKING:
    from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
        GemmaNativeVisionSession,
    )
    from typevet.adapters.outbound.vllm.judgment_factory import VllmJudgmentSession
    from typevet.domain.judgment_questions import Question
    from typevet.domain.judgment_response import JudgmentResponse
    from typevet.domain.media import ImageInput
    from typevet.ports.judgment import JudgmentPort
    from typevet.testing.judgment_fake import Distribution

DISTRIBUTIONS_ENV = "TYPEVET_FAKE__DISTRIBUTIONS"
FAKE_MODEL = "fake"
_LEVEL = re.compile(r"-?[0-9]+")

Scripted = float | dict[str, float]
"""One file entry: a Noul number, or label or level-string weights."""


@dataclass(frozen=True, slots=True)
class FakeJudgmentSession:
    """Offline session that ``open_judgment`` yields for the fake backend.

    Attributes:
        port (JudgmentPort): Port that answers from scripted or uniform
            distributions.
        model (str): Model id to pass to ``port.judge``.

    Examples:
        ```python
        from typevet.adapters.inbound.fake_backend import open_fake_judgment

        assert open_fake_judgment({}).model == "fake"
        ```
    """

    port: JudgmentPort
    model: str = FAKE_MODEL


def _number(value: object) -> float | None:
    """Return ``value`` as a float when it is a JSON number and not a bool.

    Args:
        value: Parsed JSON value.

    Returns:
        The float value, or ``None`` for a bool or a non-number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _invalid(reason: str) -> ValueError:
    """Build the error for a bad distributions file.

    Args:
        reason: What is wrong, without file content.

    Returns:
        A ``ValueError`` whose message names the variable.
    """
    return ValueError(f"{DISTRIBUTIONS_ENV} {reason}")


def _check_entry(entry: object) -> Scripted:
    """Check one file entry and return it as a number or a weight mapping.

    Args:
        entry: Parsed JSON value for one question name.

    Returns:
        The number as a float, or a copy of the weights as floats.

    Raises:
        ValueError: When the entry is not a number or an object of numbers.
    """
    number = _number(entry)
    if number is not None:
        return number
    if isinstance(entry, dict):
        weights = {str(key): _number(weight) for key, weight in entry.items()}
        checked = {key: w for key, w in weights.items() if w is not None}
        if len(checked) == len(weights):
            return checked
    raise _invalid("values must be numbers or objects of numeric weights")


def load_distributions(environ: Mapping[str, str] | None = None) -> dict[str, Scripted]:
    """Read the file that ``TYPEVET_FAKE__DISTRIBUTIONS`` names.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.

    Returns:
        Entries by question name, or an empty mapping when the variable is
        unset or empty.

    Raises:
        ValueError: When the file is missing or unreadable, is not JSON, is
            not an object, or holds an entry that is not a number or an
            object of numbers. The message names the variable and holds no
            file content or path.
    """
    source = os.environ if environ is None else environ
    raw = source.get(DISTRIBUTIONS_ENV, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(Path(raw).read_text(encoding="utf-8"))
    except OSError:
        raise _invalid("names a file that cannot be read") from None
    except ValueError:
        raise _invalid("names a file that is not valid JSON") from None
    if not isinstance(payload, dict):
        raise _invalid("must name a JSON object keyed by question name")
    return {str(name): _check_entry(entry) for name, entry in payload.items()}


def _uniform(question: object) -> Distribution:
    """Return the uniform distribution for one question.

    Args:
        question: Typed question, or a raw wire mapping.

    Returns:
        Equal weight per Choice label or Score level, else 0.5.
    """
    if isinstance(question, Choice):
        return dict.fromkeys(question.criteria, 1.0)
    if isinstance(question, Score):
        return dict.fromkeys(range(len(question.criteria)), 1.0)
    return 0.5


def _score_levels(weights: Mapping[str, float]) -> dict[int, float]:
    """Convert level-string keys to integer levels.

    Args:
        weights: Level string to weight, from the file.

    Returns:
        Integer level to weight.

    Raises:
        ValueError: When a key is not an integer string.
    """
    if not all(_LEVEL.fullmatch(key) for key in weights):
        raise _invalid("Score levels must be integer strings")
    return {int(key): weight for key, weight in weights.items()}


def _distribution(entry: Scripted, question: object) -> Distribution:
    """Return the fake distribution for one file entry.

    Args:
        entry: Checked file entry for the question name.
        question: Question the entry answers.

    Returns:
        The entry, with integer levels when the question is a Score.

    Raises:
        ValueError: When a Score entry has a key that is not an integer string.
    """
    if isinstance(question, Score) and isinstance(entry, dict):
        return _score_levels(entry)
    return entry


class FakeJudgmentPort:
    """``JudgmentPort`` that builds a ``ScriptedJudgmentFake`` per call.

    Attributes:
        _entries (dict[str, Scripted]): Checked file entries by question name.

    Examples:
        ```python
        from typevet.adapters.inbound.fake_backend import FakeJudgmentPort
        from typevet.domain.judgment_questions import Noul

        port = FakeJudgmentPort({"billing": 0.9})
        assert port.judge("t", {"billing": Noul()}, "fake").nouls["billing"].noul == 0.9
        ```
    """

    def __init__(self, entries: Mapping[str, Scripted]) -> None:
        """Store the checked file entries.

        Args:
            entries: Entries from ``load_distributions``.
        """
        self._entries = dict(entries)

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Fill a distribution for each question and judge with the fake.

        Args:
            state: Content under evaluation.
            questions: Question names to typed or raw questions.
            model: Model id copied onto the response.
            media: Images; the fake ignores them.
            off_option_threshold: Forwarded to the fake.

        Returns:
            The ``ScriptedJudgmentFake`` response.

        Raises:
            ValueError: When a Score entry has a key that is not an integer
                string.
        """
        distributions: dict[str, Distribution] = {}
        for name, question in questions.items():
            entry = self._entries.get(name)
            distributions[name] = (
                _uniform(question) if entry is None else _distribution(entry, question)
            )
        return ScriptedJudgmentFake(distributions).judge(
            state,
            questions,
            model,
            media=media,
            off_option_threshold=off_option_threshold,
        )


def open_fake_judgment(environ: Mapping[str, str] | None = None) -> FakeJudgmentSession:
    """Build the offline session from ``TYPEVET_FAKE__DISTRIBUTIONS``.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.

    Returns:
        A session whose port answers from the file or uniform distributions.

    Raises:
        ValueError: When the distributions file is missing or invalid.
    """
    return FakeJudgmentSession(port=FakeJudgmentPort(load_distributions(environ)))


def require_live_session(
    session: GemmaNativeVisionSession | VllmJudgmentSession | FakeJudgmentSession,
) -> GemmaNativeVisionSession | VllmJudgmentSession:
    """Return a live session unchanged, and refuse the offline fake session.

    Callers that read ``session.client`` use this to narrow the session that
    ``open_judgment`` yields.

    Args:
        session: Session from ``open_judgment``.

    Returns:
        The same llama.cpp or vLLM session.

    Raises:
        ValueError: When ``session`` is a ``FakeJudgmentSession``. The message
            names ``TYPEVET_BACKEND``.
    """
    if not isinstance(session, FakeJudgmentSession):
        return session
    msg = "TYPEVET_BACKEND=fake gives no HTTP client; use llama_cpp or vllm"
    raise ValueError(msg)
