"""The studentized range upper tail against frozen high-precision references (D31)."""

from __future__ import annotations

import json
import math
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from inferplot.stats import games_howell, srange_logsf, srange_sf
from tests.oracle_fixtures import load_cases
from tests.stats.fixture_data import between_cases, groups
from tests.tolerances import TOLERANCES, assert_close

REFERENCE = (
    Path(__file__).resolve().parents[1] / "data" / "studentized_range_reference.json"
)
SETS = (
    "grid",
    "df_inf",
    "k2_identity",
    "fixture_pairs",
    "failed_fixture_pairs",
    "added_fixture_pairs",
)
# fixture_pairs (the cases of the first oracle run) and added_fixture_pairs (the
# cases added later) together hold every pair of every completed case.
COMPLETED_SETS = ("fixture_pairs", "added_fixture_pairs")


@cache
def _reference() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _points(name: str) -> list[dict[str, Any]]:
    return list(_reference()["sets"][name]["points"])


def _df(value: object) -> float:
    return math.inf if value == "Inf" else float(value)  # type: ignore[arg-type]


def test_reference_file_has_the_frozen_sets() -> None:
    sizes = {name: len(_points(name)) for name in SETS}
    assert sizes == {
        "grid": 384,
        "df_inf": 72,
        "k2_identity": 224,
        "fixture_pairs": 31,
        "failed_fixture_pairs": 3,
        "added_fixture_pairs": 78,
    }
    assert "studentized_range_mpmath.py" in _reference()["generator"]
    # 78 pairs, 42 distinct (k, df, q) points: several cases share their data.
    distinct = {(p["k"], p["df"], p["q"]) for p in _points("added_fixture_pairs")}
    assert len(distinct) == 42


@pytest.mark.parametrize("name", SETS)
def test_tail_matches_high_precision_reference(name: str) -> None:
    for point in _points(name):
        k, df, q = int(point["k"]), _df(point["df"]), float(point["q"])
        what = f"{name} k={k} df={point['df']} q={q!r}"
        expected = Decimal(point["p"])
        assert_close(
            "studentized_range_tail", srange_sf(q, k, df), float(expected), what=what
        )
        # The log stays accurate below the double underflow range too.
        log_p = srange_logsf(q, k, df)
        rtol = TOLERANCES["studentized_range_tail"].rtol
        assert abs(log_p - float(expected.ln())) <= rtol, what


def test_every_completed_fixture_pair_has_a_reference() -> None:
    # SPEC section 10: every Games-Howell p is checked against a high-precision
    # reference, so the frozen sets must cover every pair of every completed
    # parametric case with three or more groups (and nothing else).
    expected = {
        (case.name, case.analysis_levels[i], case.analysis_levels[j])
        for case in between_cases("parametric", two_groups=False)
        for i in range(len(case.analysis_levels))
        for j in range(i + 1, len(case.analysis_levels))
    }
    frozen = [
        (p["case"], p["pair"][0], p["pair"][1])
        for name in COMPLETED_SETS
        for p in _points(name)
    ]
    assert len(frozen) == len(set(frozen))
    assert set(frozen) == expected
    assert len(expected) == 109


def test_edge_negative_b_c_reference() -> None:
    # SPEC section 5 item 6 and DECISIONS.md D31: the b-c label differs from
    # ggstatsplot's because R's ptukey is off in the far tail; the reference holds
    # the accurate raw p that the differing label rests on.
    point = next(
        p
        for p in _points("added_fixture_pairs")
        if p["case"] == "ggbetweenstats-edge_negative-parametric"
        and p["pair"] == ["b", "c"]
    )
    assert point["p"].startswith("4.28309162932008")
    value = srange_sf(point["q"], point["k"], point["df"])
    assert_close(
        "studentized_range_tail", value, float(Decimal(point["p"])), what="b-c"
    )


@pytest.mark.parametrize("name", [*COMPLETED_SETS, "failed_fixture_pairs"])
def test_fixture_pairs_are_the_games_howell_pairs(name: str) -> None:
    # The frozen pairs are the q and df that games_howell computes on the fixture
    # data, so its raw p is checked against the reference at the very same point.
    cases = {case.name: case for case in load_cases()}
    by_case: dict[str, list[dict[str, Any]]] = {}
    for point in _points(name):
        by_case.setdefault(point["case"], []).append(point)
    assert (
        len(by_case)
        == {
            "fixture_pairs": 7,
            "added_fixture_pairs": 22,
            "failed_fixture_pairs": 1,
        }[name]
    )
    for case_name, points in by_case.items():
        case = cases[case_name]
        levels = case.analysis_levels
        pairs = {(levels[p.i], levels[p.j]): p for p in games_howell(groups(case))}
        for point in points:
            pair = pairs[(point["pair"][0], point["pair"][1])]
            what = f"{name} {point['pair']}"
            assert pair.df is not None
            assert_close("closed_form", abs(pair.statistic), point["q"], what=what)
            assert_close("closed_form", pair.df, point["df"], what=what)
            assert_close(
                "studentized_range_tail",
                pair.p_value,
                float(Decimal(point["p"])),
                what=f"{what} raw p",
            )


def test_iris_far_tail_is_accurate() -> None:
    # DECISIONS.md D31: R's ptukey prints 1.41e-9 (after Holm) for iris
    # setosa-versicolor; the accurate raw tail is 7.45e-15.
    point = next(
        p for p in _points("fixture_pairs") if p["pair"] == ["setosa", "versicolor"]
    )
    value = srange_sf(point["q"], point["k"], point["df"])
    assert value == pytest.approx(7.451651008907960e-15, rel=1e-14)


def test_k2_is_the_two_sided_t_tail() -> None:
    # Q_{2,df} = sqrt(2) |T_df|: for df = 1 (Cauchy) P(|T| > x) = (2 / pi) atan(1 / x),
    # in closed form even where P underflows.
    for q in (0.5, 3.0, 1e3, 1e100, 1e300):
        x = q / math.sqrt(2.0)
        exact = math.log(2.0 / math.pi) + math.log(math.atan(1.0 / x))
        assert srange_logsf(q, 2, 1.0) == pytest.approx(exact, rel=1e-13, abs=1e-14)


def test_edges() -> None:
    assert srange_sf(0.0, 3, 10.0) == 1.0
    assert srange_sf(-1.0, 3, 10.0) == 1.0
    assert srange_sf(math.inf, 3, 10.0) == 0.0
    assert srange_logsf(math.inf, 3, 10.0) == -math.inf
    # Far below the double underflow range the log is still finite and decreasing.
    a, b = srange_logsf(60.0, 3, math.inf), srange_logsf(61.0, 3, math.inf)
    assert -1e4 < b < a < -745.0
    assert srange_sf(60.0, 3, math.inf) == 0.0
    tails = [srange_sf(q, 4, 12.5) for q in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
    assert tails == sorted(tails, reverse=True)
    assert 1.0 - srange_sf(1e-300, 3, 10.0) < 1e-15


@pytest.mark.parametrize(
    ("q", "k", "df"),
    [
        (1.0, 1, 5.0),
        (1.0, 3, 0.0),
        (math.nan, 3, 5.0),
        (1.0, 3, -1.0),
        (1.0, 3, math.nan),
    ],
)
def test_invalid_arguments(q: float, k: int, df: float) -> None:
    with pytest.raises(ValueError, match=r"k must|q must|df must"):
        srange_sf(q, k, df)
