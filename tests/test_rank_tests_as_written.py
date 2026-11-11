"""The rank tests rank the values as written (DECISIONS.md D126).

A value computed in binary (0.1 + 0.2) and the same number converted from text
(0.3) are two doubles and one number. Every rank test gives the same result for
both writings: the tie, the order and the zero difference are the data's. R ranks
the doubles (oracle probe-nearzero; tests/test_probe_round3.py pins R's side).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from inferplot.stats.written import as_written

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
