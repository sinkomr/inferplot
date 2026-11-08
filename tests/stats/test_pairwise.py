"""Games-Howell pairwise comparisons followed by p_adjust."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from tools.reference import stats_reference as ref

from inferplot.stats import P_ADJUST_METHODS, games_howell, p_adjust, welch_t
from tests.tolerances import TOLERANCES, assert_close

RNG = np.random.Generator(np.random.PCG64(8))
GROUPS = [RNG.normal(m, s, n) for m, s, n in [(0, 1, 10), (1, 2, 13), (0.3, 0.6, 8)]]


def test_pairs_in_input_order_with_signed_q() -> None:
    pairs = games_howell(GROUPS)
    assert [(p.i, p.j) for p in pairs] == [(0, 1), (0, 2), (1, 2)]
    for pair in pairs:
        a, b = GROUPS[pair.i], GROUPS[pair.j]
        se = math.sqrt((a.var(ddof=1) / a.size + b.var(ddof=1) / b.size) / 2.0)
        assert_close(
            "closed_form", pair.statistic, (b.mean() - a.mean()) / se, what="q"
        )


def test_agrees_with_reference() -> None:
    # tools/reference uses SciPy's studentized range for the raw p; SPEC.md §10's
    # games_howell_p entry covers that cross-check (SciPy's own far-tail error).
    pairs = games_howell(GROUPS)
    reference = ref.games_howell(GROUPS)
    adjusted = ref.holm([p for *_, p in reference])
    for pair, (i, j, q, p), p_holm in zip(pairs, reference, adjusted, strict=True):
        assert (pair.i, pair.j) == (i, j)
        assert_close("closed_form", abs(pair.statistic), q, what="|q|")
        assert_close("games_howell_p", pair.p_value, p, what="raw p")
        assert_close("games_howell_p", pair.p_adjusted, p_holm, what="Holm p")


def test_two_groups_is_welch_t() -> None:
    # Q_{2,df} = sqrt(2) |T_df|, so with k = 2 Games-Howell is Welch's t-test.
    a, b = GROUPS[0], GROUPS[1]
    (pair,) = games_howell([a, b], p_adjust_method="none")
    welch = welch_t(a, b)
    assert pair.df == pytest.approx(welch.df, rel=1e-14)
    assert pair.statistic == pytest.approx(-welch.t * math.sqrt(2.0), rel=1e-14)
    rtol = TOLERANCES["studentized_range_tail"].rtol
    assert pair.p_value == pytest.approx(welch.p_value, rel=rtol)


@pytest.mark.parametrize("method", P_ADJUST_METHODS)
def test_every_adjust_method(method: str) -> None:
    pairs = games_howell(GROUPS, p_adjust_method=method)  # type: ignore[arg-type]
    raw = [p.p_value for p in pairs]
    assert [p.p_adjusted for p in pairs] == list(p_adjust(raw, method))  # type: ignore[arg-type]


def test_far_tail_pair_is_not_zero() -> None:
    groups = [np.linspace(0.0, 1.0, 30) + shift for shift in (0.0, 4.0, 4.1)]
    pairs = games_howell(groups)
    assert 0.0 < pairs[0].p_value < 1e-30


def test_validation() -> None:
    with pytest.raises(ValueError, match="constant"):
        games_howell([[1.0, 1.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="at least 2"):
        games_howell([[1.0], [1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="at least two groups"):
        games_howell([[1.0, 2.0]])
    with pytest.raises(ValueError, match="p_adjust_method"):
        games_howell(GROUPS, p_adjust_method="tukey")  # type: ignore[arg-type]


def test_twenty_groups_of_thirty_take_under_half_a_second() -> None:
    rng = np.random.Generator(np.random.PCG64(20))
    groups = [rng.normal(rng.uniform(0, 2), rng.uniform(0.5, 2), 30) for _ in range(20)]
    pairs = games_howell(groups)  # warm-up (first-call imports and caches)
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        pairs = games_howell(groups)
        best = min(best, time.perf_counter() - start)
    assert len(pairs) == 190
    assert best < 0.5, f"games_howell for k = 20 took {best:.3f} s"
