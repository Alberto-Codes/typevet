"""Contract: every ``examples/*/run.py`` runs offline on the fake backend (#395).

Each example loads by file path into a fresh module and runs ``main()`` in
process with ``TYPEVET_BACKEND=fake``. A ``TYPEVET_FAKE__DISTRIBUTIONS`` file
scripts one question of the example. Each example runs with two scripted
winners, so neither a uniform default nor one fixed label can pass. Each
example has one row in ``EXPECTED`` that names the scripted question, the two
cases and the output line that proves the winner. An example in ``IMAGE_PAIR``
reads two image paths from the environment, so the test writes two tiny PNG
files under ``tmp_path`` and sets the two variables first.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"

# Examples that read two image paths from these variables (#397).
IMAGE_PAIR = frozenset({"face_pair", "signature_pair"})
IMAGE_VARIABLES = ("TYPEVET_EXAMPLE_IMAGE_A", "TYPEVET_EXAMPLE_IMAGE_B")

# One case: (value for the question in the distributions file, expected winner).
Case = tuple[object, str]


def _choice_cases(first: str, second: str) -> tuple[Case, Case]:
    """Return two Choice cases that put all weight on one label each.

    Args:
        first: Winner of the first case.
        second: Winner of the second case.

    Returns:
        Two cases. Both labels must differ from the uniform default winner (the
        first option), so an example that prints one fixed label fails one case.
    """
    return ({first: 1.0}, first), ({second: 1.0}, second)


def _tiny_png(rgb: tuple[int, int, int]) -> bytes:
    """Return a valid 4x4 RGB PNG of one color, made with the standard library.

    Args:
        rgb: The color of every pixel.

    Returns:
        The encoded PNG bytes.
    """

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body)
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + bytes(rgb) * 4 for _ in range(4))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


# Example directory -> (question name, two cases, where to look, expected stdout
# text with ``{winner}``). "last": the last stdout line equals the text.
# "part": some line holds it. Calibrate scripts P(True) 0.9 and 0.6. The Platt
# map (slope 2.0, intercept -1.0) gives 0.967 and 0.453, so the 0.6 case fails
# an example that takes the winner from the raw probability.
EXPECTED: dict[str, tuple[str, tuple[Case, Case], str, str]] = {
    "calibrate": ("fraud", ((0.9, "yes"), (0.6, "no")), "last", "winner: {winner}"),
    "check_register": (
        "verdict",
        _choice_cases("differs", "unreadable"),
        "last",
        "winner: {winner}",
    ),
    "face_pair": (
        "same_person",
        ((0.9, "yes"), (0.1, "no")),
        "last",
        "winner: {winner}",
    ),
    "receipt_claim": (
        "verdict",
        _choice_cases("contradicted", "insufficient_evidence"),
        "last",
        "winner: {winner}",
    ),
    "signature_pair": (
        "same_signer",
        ((0.9, "yes"), (0.1, "no")),
        "last",
        "winner: {winner}",
    ),
    "terminal-demo": (
        "verdict",
        _choice_cases("contradicted", "insufficient_evidence"),
        "part",
        "-> model answer: {winner} (",
    ),
}
NAMES = sorted({p.parent.name for p in EXAMPLES.glob("*/run.py")} | set(EXPECTED))


@pytest.mark.parametrize("case", [0, 1])
@pytest.mark.parametrize("name", NAMES)
def test_example_prints_the_scripted_winner(
    name: str,
    case: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert name in EXPECTED, f"add a row for examples/{name}/ to EXPECTED"
    script = EXAMPLES / name / "run.py"
    assert script.is_file(), f"missing {script.relative_to(REPO)}"
    question, cases, where, template = EXPECTED[name]
    value, winner = cases[case]
    dist = tmp_path / "distributions.json"
    dist.write_text(json.dumps({question: value}))
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("TYPEVET_BACKEND", "fake")
    monkeypatch.setenv("TYPEVET_FAKE__DISTRIBUTIONS", str(dist))
    if name in IMAGE_PAIR:
        for variable, rgb in zip(
            IMAGE_VARIABLES, ((200, 40, 40), (40, 40, 200)), strict=True
        ):
            image = tmp_path / f"{variable.lower()}.png"
            image.write_bytes(_tiny_png(rgb))
            monkeypatch.setenv(variable, str(image))

    spec = importlib.util.spec_from_file_location(f"example_{name}", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "OUT_DIR"):
        monkeypatch.setattr(module, "OUT_DIR", tmp_path / "typevet-receipts")
    module.main()

    lines = capsys.readouterr().out.splitlines()
    text = template.format(winner=winner)
    if where == "last":
        assert lines[-1] == text
    else:
        assert any(text in line for line in lines)
