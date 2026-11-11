"""The rank tests rank the values as written (DECISIONS.md D126).

A value computed in binary (0.1 + 0.2) and the same number converted from text
(0.3) are two doubles and one number. Every rank test gives the same result for
both writings: the tie, the order and the zero difference are the data's. R ranks
the doubles (oracle probe-nearzero; tests/test_probe_round3.py pins R's side).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from inferplot.stats.written import as_written, written_values
from tests.timing import allowed

COMPUTED = 0.1 + 0.2
WRITTEN = 0.3


def _two_groups(v: float) -> StatsResult:
    y = [v, 0.5, 0.9, 1.4, 0.2, 1.1, 0.3, 0.75, 1.8, 0.05, 2.2, 1.3]
    frame = pd.DataFrame({"g": ["a"] * 6 + ["b"] * 6, "y": y})
    return ip.analyze.betweenstats(frame, "g", "y", type="nonparametric")


def _three_groups(v: float) -> StatsResult:
    y = [v, 0.5, 0.9, 1.4, 0.3, 1.1, 0.6, 0.75, 1.8, 0.3, 2.2, 1.3]
    frame = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4 + ["c"] * 4, "y": y})
    return ip.analyze.betweenstats(frame, "g", "y", type="nonparametric")


def _paired(v: float) -> StatsResult:
    before = [0.3, 0.5, 1.0, 0.2, 0.9, 0.4, 1.5, 0.6, 2.0, 1.1]
    after = [v, 0.9, 1.4, 0.1, 1.1, 0.75, 1.8, 0.05, 2.2, 1.3]
    frame = pd.DataFrame(
        {
            "id": list(range(10)) * 2,
            "when": ["before"] * 10 + ["after"] * 10,
            "y": before + after,
        }
    )
    return ip.analyze.withinstats(
        frame, "when", "y", subject_id="id", type="nonparametric"
    )


def _three_conditions(v: float) -> StatsResult:
    a = [0.3, 0.5, 1.0, 0.2, 0.9, 0.4]
    b = [v, 0.9, 1.4, 0.1, 1.1, 0.75]
    c = [0.6, 0.2, 1.4, 0.4, 0.8, 0.9]
    frame = pd.DataFrame(
        {
            "id": list(range(6)) * 3,
            "when": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "y": a + b + c,
        }
    )
    return ip.analyze.withinstats(
        frame, "when", "y", subject_id="id", type="nonparametric"
    )


def _correlation(v: float) -> StatsResult:
    frame = pd.DataFrame(
        {
            "x": [v, 0.3, 0.9, 1.4, 0.2, 1.1, 0.6, 0.75],
            "y": [1.0, 2.0, 2.5, 3.5, 0.5, 3.0, 2.2, 2.4],
        }
    )
    return ip.analyze.scatterstats(frame, "x", "y", type="nonparametric")


def _one_sample(v: float) -> StatsResult:
    frame = pd.DataFrame({"y": [v, 0.9, 1.4, 0.2, 1.1, 0.75, 1.8, 0.05, 2.2, 1.3]})
    return ip.analyze.histostats(frame, "y", type="nonparametric", test_value=0.3)


DESIGNS: dict[str, Callable[[float], StatsResult]] = {
    "two groups": _two_groups,
    "three groups": _three_groups,
    "paired": _paired,
    "three conditions": _three_conditions,
    "correlation": _correlation,
    "one sample": _one_sample,
}


@pytest.mark.parametrize("design", list(DESIGNS))
def test_a_computed_value_and_its_decimal_give_one_result(design: str) -> None:
    assert COMPUTED != WRITTEN
    computed, written = DESIGNS[design](COMPUTED), DESIGNS[design](WRITTEN)
    assert computed.text.apa == written.text.apa
    assert computed.test == written.test
    assert computed.effect_size == written.effect_size
    assert computed.pairwise == written.pairwise


def test_the_tie_is_made_not_broken() -> None:
    # In these data the value in question is tied with a literal 0.3 (or is a
    # zero difference); ranking the doubles would break the tie.
    two = _two_groups(COMPUTED)
    assert two.test is not None
    assert two.test.statistic == 14.5  # a half: the two 0.3 share a mid-rank
    paired = _paired(COMPUTED)
    assert paired.test is not None
    assert paired.test.statistic == 35.0  # nine nonzero differences, not ten
    assert "n = 10 pairs" in paired.text.apa


def test_as_written_leaves_text_values_alone() -> None:
    rng = np.random.Generator(np.random.PCG64(11))
    decimals = np.round(rng.normal(50.0, 20.0, 2000), 2)
    assert np.array_equal(as_written(decimals), decimals)
    long_digits = rng.normal(0.0, 1.0, 2000)  # no short decimal reading
    assert np.array_equal(as_written(long_digits), long_digits)
    whole = rng.integers(-1000, 1000, 500).astype(np.float64)
    assert np.array_equal(as_written(whole), whole)
    computed = np.array([0.1 + 0.2, 3 * 0.3, 0.7 - 0.4, 1.1 * 3])
    assert as_written(computed).tolist() == [0.3, 0.9, 0.3, 3.3]
    # At most two units in the last place, and symmetric.
    assert np.all(np.abs(as_written(computed) - computed) <= 2 * np.spacing(computed))
    assert np.array_equal(as_written(-computed), -as_written(computed))


def test_the_statistics_functions_rank_as_written() -> None:
    a_computed = [COMPUTED, 0.5, 0.9, 1.4]
    a_written = [WRITTEN, 0.5, 0.9, 1.4]
    b = [0.3, 0.75, 1.8, 0.05, 2.2]
    assert stats.mann_whitney(a_computed, b) == stats.mann_whitney(a_written, b)
    assert stats.rank_biserial(a_computed, b) == stats.rank_biserial(a_written, b)
    c = [0.6, 0.3, 1.2, 0.8]
    assert stats.kruskal_wallis([a_computed, b, c]) == stats.kruskal_wallis(
        [a_written, b, c]
    )
    assert stats.dunn([a_computed, b, c]) == stats.dunn([a_written, b, c])
    pair = [0.3, 0.1, 1.0, 1.2]
    assert stats.signed_rank(a_computed, pair) == stats.signed_rank(a_written, pair)
    cols_computed = [a_computed, pair, [0.6, 0.2, 1.4, 0.4]]
    cols_written = [a_written, pair, [0.6, 0.2, 1.4, 0.4]]
    assert stats.friedman(cols_computed) == stats.friedman(cols_written)
    x_computed = [COMPUTED, 0.3, 0.9, 1.4, 0.2]
    x_written = [WRITTEN, 0.3, 0.9, 1.4, 0.2]
    y = [1.0, 2.0, 2.5, 3.5, 0.5]
    assert stats.spearman(x_computed, y) == stats.spearman(x_written, y)


# --------------------------------------------------------------------------- #
# Phase 5 review (statistics half): H1, M3, M4, M5, L1 and the planted bugs
# that no test caught
# --------------------------------------------------------------------------- #


def _writings(values: list[float]) -> list[list[float]]:
    """One-decimal values typed, and computed three ways that land on other
    doubles for many of them, each within two units in the last place of the
    decimal (one or two operations on decimals; a writing that cancels leading
    digits, such as (k + 30) / 10 - 3, can be several units off and is a
    different number to this rule, as it is to R)."""
    tenths = [round(v * 10) for v in values]
    return [
        [k / 10 for k in tenths],
        [k * 0.1 for k in tenths],
        [k / 100 * 10 for k in tenths],
        [math.fsum([0.1] * abs(k)) * (1 if k >= 0 else -1) for k in tenths],
    ]


def test_the_writings_differ_in_binary_and_agree_as_written() -> None:
    values = [k / 10 for k in range(-60, 61)]
    typed, *computed = (np.array(w) for w in _writings(values))
    for other in computed:
        assert int((other != typed).sum()) >= 15  # really other doubles
        assert np.array_equal(as_written(other), typed)


def test_one_sample_effect_size_is_the_r_of_its_own_v() -> None:
    # H1: the rank-biserial r ranked the raw doubles while V ranked the values
    # as written, so r contradicted V on a computed column.
    typed = [0.3, -0.3, 1.0, 2.0, -0.5, 0.7]
    computed = [0.3, -(0.1 + 0.2), 1.0, 2.0, -0.5, 0.7]
    results = [
        ip.analyze.histostats(
            pd.DataFrame({"x": x}), "x", type="nonparametric", test_value=0
        )
        for x in (typed, computed)
    ]
    for result in results:
        assert result.test is not None
        assert result.effect_size is not None
        v, total = result.test.statistic, 21.0
        assert v == 16.5
        assert result.effect_size.estimate == pytest.approx((2 * v - total) / total)
    assert results[0].effect_size == results[1].effect_size


@pytest.mark.parametrize("seed", range(12))
def test_every_writing_of_one_decimal_data_gives_one_result(seed: int) -> None:
    # Random one-decimal data in four writings: every rank test, its effect size
    # and interval, the tie count and the checks are the same (the review found
    # the one-sample r differing in about half of such sets, and n_tied and the
    # heavy_ties warning changing with the writing).
    rng = np.random.Generator(np.random.PCG64(seed))
    a = np.round(rng.normal(1.0, 0.6, 14), 1).tolist()
    b = np.round(rng.normal(1.3, 0.6, 14), 1).tolist()
    c = np.round(rng.normal(0.8, 0.6, 14), 1).tolist()
    seen: dict[str, set[str]] = {}

    def record(name: str, result: StatsResult) -> None:
        codes = sorted(issue.code for issue in result.check())
        key = repr(
            (result.test, result.effect_size, result.pairwise, result.n_tied, codes)
        )
        seen.setdefault(name, set()).add(key)

    for wa, wb, wc in zip(_writings(a), _writings(b), _writings(c), strict=True):
        ids = list(range(14))
        record(
            "one sample",
            ip.analyze.histostats(
                pd.DataFrame({"y": wa}), "y", type="nonparametric", test_value=0.3 * 3
            ),
        )
        two = pd.DataFrame({"g": ["a"] * 14 + ["b"] * 14, "y": wa + wb})
        record("two groups", ip.analyze.betweenstats(two, "g", "y", type="np"))
        three = pd.DataFrame(
            {"g": ["a"] * 14 + ["b"] * 14 + ["c"] * 14, "y": wa + wb + wc}
        )
        record("three groups", ip.analyze.betweenstats(three, "g", "y", type="np"))
        paired = pd.DataFrame(
            {"id": ids * 2, "g": ["a"] * 14 + ["b"] * 14, "y": wa + wb}
        )
        record(
            "paired",
            ip.analyze.withinstats(paired, "g", "y", subject_id="id", type="np"),
        )
        within = pd.DataFrame(
            {
                "id": ids * 3,
                "g": ["a"] * 14 + ["b"] * 14 + ["c"] * 14,
                "y": wa + wb + wc,
            }
        )
        record(
            "three conditions",
            ip.analyze.withinstats(within, "g", "y", subject_id="id", type="np"),
        )
        record(
            "correlation",
            ip.analyze.scatterstats(
                pd.DataFrame({"x": wa, "y": wb}), "x", "y", type="np"
            ),
        )
    assert {name: len(keys) for name, keys in seen.items()} == dict.fromkeys(seen, 1)
    assert len(seen) == 6


def test_whole_numbers_are_never_merged() -> None:
    # M3: from 2^51 on, neighbouring whole numbers lie within two units in the
    # last place of a round one; they were read as that round number and tied.
    xs = [5e15 - 2, 5e15 - 1, 5e15, 5e15 + 1, 5e15 + 2, 5e15 + 7, 5e15 + 9, 5e15 + 11]
    assert as_written(np.array(xs)).tolist() == xs
    frame = pd.DataFrame({"x": xs, "y": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    result = ip.analyze.scatterstats(frame, "x", "y", type="nonparametric")
    assert result.effect_size is not None
    assert result.effect_size.estimate == 1.0
    assert result.n_tied == 0
    # Nanosecond timestamps 256 ns apart around a whole second.
    stamps = np.array([1_700_000_000_000_000_000 + 256 * i for i in range(-4, 5)])
    floats = stamps.astype(np.float64)
    assert np.array_equal(as_written(floats), floats)
    assert np.unique(as_written(floats)).size == 9
    # Whole numbers of every size an integer column can hold.
    rng = np.random.Generator(np.random.PCG64(3))
    whole = np.rint(rng.uniform(-1.0, 1.0, 4000) * 10.0 ** rng.integers(0, 20, 4000))
    whole = whole[np.abs(whole) < 2.0**64]
    assert whole.size > 3500
    assert np.array_equal(as_written(whole), whole)
    # Beyond that a whole-valued double is a decimal in scientific notation:
    # 17 x 1e29 and 1.7e30 are one number as written.
    assert 17 * 1e29 != 1.7e30
    assert as_written(np.array([17 * 1e29]))[0] == 1.7e30
    assert as_written(np.array([1e16 - 4, 1e16, 1e16 + 4])).tolist() == [
        1e16 - 4,
        1e16,
        1e16 + 4,
    ]


def test_as_written_agrees_with_the_reading_value_by_value() -> None:
    # The fast path (values recognized as unchanged without rational arithmetic)
    # against the definition, on values of every kind and size, both signs.
    rng = np.random.Generator(np.random.PCG64(5))
    pieces = [
        np.round(rng.normal(0, 50, 3000), 2),
        np.round(rng.normal(0, 50, 3000), 6),
        rng.normal(0, 1, 3000),
        np.arange(-300, 300) * 0.1,
        np.arange(1, 400) * 0.3,
        (np.arange(1, 400) + 0.1) * 2.54,
        np.round(rng.uniform(0, 1, 2000), 9),
        np.round(rng.uniform(0, 1, 2000), 11),
        np.round(rng.uniform(0, 1, 500), 3) * 10.0 ** rng.integers(-40, 40, 500),
        np.array([0.0, -0.0, 5e-324, 2.2e-308, 1e-300, 1.7e308, 0.1 + 0.2, 0.7 - 0.4]),
        np.array([float(f"{v:.17g}") for v in np.nextafter(np.arange(1, 200) / 10, 9)]),
    ]
    values = np.concatenate(pieces)
    values = np.concatenate([values, -values])
    expected = np.array([float(r) for r in written_values(values)])
    got = as_written(values)
    assert np.array_equal(got, expected)
    assert np.array_equal(np.signbit(got), np.signbit(values))
    # Idempotent, and never further than two units in the last place.
    assert np.array_equal(as_written(got), got)
    assert np.all(np.abs(got - values) <= 2 * np.spacing(np.abs(values)))


def test_two_units_in_the_last_place_are_read_and_three_are_not() -> None:
    # The planted bug that narrowed the reading to one unit was not caught: no
    # test had a value between one and two units from its decimal. The distance
    # is measured from the decimal itself (0.3, not the double nearest it).
    between_one_and_two = beyond_two = 0
    for text in ("0.3", "1.7", "12.9", "0.0041", "730.25", "0.07", "5.1"):
        decimal = Fraction(text)
        nearest = float(text)
        for steps in range(-3, 4):
            value = nearest + steps * float(np.spacing(nearest))
            units = abs(Fraction(value) - decimal) / Fraction(math.ulp(value))
            got = float(as_written(np.array([value]))[0])
            assert (got == nearest) is (units <= 2), (text, steps, float(units))
            between_one_and_two += 1 < units <= 2
            beyond_two += units > 2
    assert between_one_and_two >= 10
    assert beyond_two >= 10


def test_a_million_values_are_read_in_under_a_second() -> None:
    # M5: reading a million distinct decimals took 14 s.
    rng = np.random.Generator(np.random.PCG64(9))
    values = np.round(rng.normal(0.0, 1000.0, 1_000_000), 6)
    start = time.perf_counter()
    out = as_written(values)
    seconds = time.perf_counter() - start
    assert np.array_equal(out, values)
    assert seconds < allowed(1.0), f"as_written of 1e6 values took {seconds:.2f} s"


def test_the_second_sample_is_ranked_as_written_too() -> None:
    # Planted bugs that ranked only one of the two samples as written were not
    # caught: the tests perturbed the first sample only.
    a = [0.3, 0.5, 0.9, 1.4]
    b_typed = [0.3, 0.75, 1.8, 0.05, 2.2]
    b_computed = [0.1 + 0.2, 0.75, 1.8, 0.05, 2.2]
    assert stats.mann_whitney(a, b_computed) == stats.mann_whitney(a, b_typed)
    assert stats.rank_biserial(a, b_computed) == stats.rank_biserial(a, b_typed)
    assert stats.signed_rank(a, b_computed[:4]) == stats.signed_rank(a, b_typed[:4])
    x = [1.0, 2.0, 2.5, 3.5, 0.5]
    y_typed = [0.3, 0.3, 0.9, 1.4, 0.2]
    y_computed = [0.1 + 0.2, 0.3, 0.9, 1.4, 0.2]
    assert stats.spearman(x, y_computed) == stats.spearman(x, y_typed)


def test_data_constant_as_written_get_the_plain_message() -> None:
    # L1: [0.3] * 7 + [0.1 + 0.2] is one value as written; the analysis's own
    # message says so, not an internal one about ties.
    frame = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y": [0.3] * 7 + [0.1 + 0.2]})
    with pytest.raises(ValueError, match="Every value of y is the same"):
        ip.analyze.betweenstats(frame, "g", "y", type="nonparametric")
    paired = pd.DataFrame(
        {
            "id": [0, 1, 2, 3] * 2,
            "g": ["a"] * 4 + ["b"] * 4,
            "y": [0.3, 0.5, 0.9, 1.2, 0.1 + 0.2, 0.5, 3 * 0.3, 1.2],
        }
    )
    with pytest.raises(ValueError, match="Every subject has the same value"):
        ip.analyze.withinstats(paired, "g", "y", subject_id="id", type="nonparametric")
