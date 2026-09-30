"""Unit tests for the seeded live slice: requests and pins (#349).

The live check-match test builds its requests and slice pins with
``check_match_slice``. These tests prove offline that the seed in the
environment reaches both the requests and the ``generator_seed`` pin.
"""

from __future__ import annotations

import pytest

from typevet_evals.check_match import (
    SEED_ENV,
    build_check_match_request,
    check_cases,
    check_match_slice,
    render_check,
)

pytestmark = pytest.mark.unit


def _images(requests: object) -> list[bytes]:
    assert isinstance(requests, tuple)
    return [request.media[0].data for request in requests]


def test_seed_from_the_environment_reaches_requests_and_pins() -> None:
    made = check_match_slice({SEED_ENV: "3"}, rows=1)

    assert made.seed == 3
    expected = [
        build_check_match_request(case, image=render_check(case))
        for case in check_cases(3, 1)
    ]
    assert [r.case_id for r in made.requests] == [r.case_id for r in expected]
    assert _images(made.requests) == [r.media[0].data for r in expected]
    assert made.pins["generator_seed"] == 3
    assert made.pins["dataset"] == "synthetic checks (#315 generator)"
    assert made.pins["register_rows"] == 1
    assert made.pins["variants_per_row"] == 7
    assert isinstance(made.pins["slice_sha256"], str)


def test_another_seed_gives_other_images_and_another_digest() -> None:
    default = check_match_slice({}, rows=1)
    seeded = check_match_slice({SEED_ENV: "3"}, rows=1)

    assert default.seed == 0
    assert default.pins["generator_seed"] == 0
    assert _images(default.requests) != _images(seeded.requests)
    assert default.pins["slice_sha256"] != seeded.pins["slice_sha256"]


def test_same_seed_gives_the_same_digest() -> None:
    first = check_match_slice({SEED_ENV: "3"}, rows=1)
    second = check_match_slice({SEED_ENV: "3"}, rows=1)

    assert first.pins == second.pins


def test_bad_seed_is_refused_before_any_render() -> None:
    with pytest.raises(ValueError, match=SEED_ENV):
        check_match_slice({SEED_ENV: "1_0"}, rows=1)
