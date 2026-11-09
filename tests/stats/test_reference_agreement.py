"""inferplot.stats against the independent audit implementations in tools/reference.

Where tools/reference covers a method (welch_anova, omega2_partial_from_f, hedges_j,
jzs_bf10_t, holm, games_howell's q), both are run on every fixture's groups and on
random data; agreement here is a second check next to the R oracle.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest
from scipy import stats
from tools.reference import stats_reference as ref

from inferplot.stats import (
    games_howell,
    hedges_j,
    jzs_bf10_two_sample,
    p_adjust,
    partial_omega_squared,
    welch_anova,
    welch_t,
)
from tests.oracle_fixtures import cases_for
from tests.stats.fixture_data import groups
from tests.tolerances import assert_close


def _fixture_group_sets() -> list[Any]:
    sets = [
        pytest.param(groups(case), id=case.name)
        for case in cases_for("ggbetweenstats")
        if case.error is None and case.type == "parametric"
    ]
    rng = np.random.Generator(np.random.PCG64(99))
    for k in (2, 3, 6):
        sizes = rng.integers(4, 40, size=k).tolist()
        draw = [rng.normal(rng.uniform(-1, 1), rng.uniform(0.3, 3), n) for n in sizes]
        sets.append(pytest.param(draw, id=f"random-k{k}"))
    return sets


@pytest.mark.parametrize("data", _fixture_group_sets())
def test_against_reference(data: list[np.ndarray]) -> None:
    anova = welch_anova(data)
    f, df1, df2, p = ref.welch_anova(data)
    assert_close("closed_form", anova.f, f, what="Welch F")
    assert_close("closed_form", anova.df2, df2, what="Welch df2")
    assert_close("closed_form", anova.p_value, p, what="Welch p")
    assert anova.df1 == df1
    omega = partial_omega_squared(anova.f, anova.df1, anova.df2).estimate
    assert_close(
        "closed_form", omega, ref.omega2_partial_from_f(f, df1, df2), what="omega"
    )
    pairs = games_howell(data)
    reference = ref.games_howell(data)
    for pair, (_, _, q, _) in zip(pairs, reference, strict=True):
        assert_close("closed_form", abs(pair.statistic), q, what="Games-Howell |q|")
    raw = [pair.p_value for pair in pairs]
    assert p_adjust(raw, "holm") == tuple(ref.holm(raw))
    for i, j in itertools.combinations(range(len(data)), 2):
        a, b = data[i], data[j]
        df = welch_t(a, b).df
        assert_close("closed_form", hedges_j(df), ref.hedges_j(df), what="J")
        bf10, _ = jzs_bf10_two_sample(a, b, prior_scale=0.707)
        student = float(stats.ttest_ind(a, b, equal_var=True).statistic)
        n_eff = a.size * b.size / (a.size + b.size)
        expected = ref.jzs_bf10_t(student, n_eff, a.size + b.size - 2.0, r=0.707)
        if math.isfinite(expected):
            assert_close("ttest_bayes_factor", bf10, expected, what="JZS BF10")
