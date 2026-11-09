"""Wide dynamic range between groups: an exact result or a plain ValueError.

Every method runs on the data times one power of two that brings the largest |value|
of all groups into [1/2, 1) (``common_scale``). A group whose values are small next
to that peak has a scaled variance term v / n that can fall below the smallest
normal double, 2^-1022 (``TINY``), where its subnormal squares carry absolute
rounding errors of 2^-1075: below it the term cannot be trusted. The methods then
raise a plain-language ValueError (``check_variance_term`` for two samples,
``constant_group_message`` for k groups); at or above it the term is accurate to
about 2^-53 relative.

Before this check, [1e160] * 3 against [1, 2, 3] gave Welch's t 4.9e-4 too large,
Welch's ANOVA returned NaN and Games-Howell a q 7.8e-4 off (review of 64e39f0).

Expected values come from exact arithmetic on the raw doubles: means, variances, F
and df as fractions, square roots in 50-digit decimals; never from inferplot.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot.stats import (
    games_howell,
    hedges_g,
    hedges_j,
    jzs_bf10_two_sample,
    student_t,
    welch_anova,
    welch_t,
)
from inferplot.stats.parametric import TINY
from tests.tolerances import assert_close

betweenstats: Any = ip.analyze.betweenstats

TWO_SAMPLE_TOO_LITTLE = (
    "sample a is constant and sample b varies by less than double precision can "
    "resolve next to the largest |value| of both samples"
)
GROUP_2_TOO_LITTLE = "group 2 varies, but by less than double precision"


# ------------------------------------------------------------- exact references


def _fr(values: Sequence[float]) -> list[Fraction]:
    return [Fraction(float(v)) for v in values]


def _mean(x: Sequence[float]) -> Fraction:
    xs = _fr(x)
    return sum(xs, Fraction(0)) / len(xs)


def _var(x: Sequence[float]) -> Fraction:
    m = _mean(x)
    return sum(((v - m) ** 2 for v in _fr(x)), Fraction(0)) / (len(x) - 1)


def _sqrt(value: Fraction) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 50
        return (Decimal(value.numerator) / Decimal(value.denominator)).sqrt()


def _ratio(num: Fraction, den_squared: Fraction) -> float:
    """num / sqrt(den_squared), correctly rounded from 50 digits."""
    with localcontext() as ctx:
        ctx.prec = 50
        return float(
            Decimal(num.numerator) / Decimal(num.denominator) / _sqrt(den_squared)
        )


def exact_welch(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    w_a, w_b = _var(a) / len(a), _var(b) / len(b)
    t = _ratio(_mean(a) - _mean(b), w_a + w_b)
    df = (w_a + w_b) ** 2 / (w_a**2 / (len(a) - 1) + w_b**2 / (len(b) - 1))
    return t, float(df)


def exact_student(a: Sequence[float], b: Sequence[float]) -> float:
    n_a, n_b = len(a), len(b)
    pooled = ((n_a - 1) * _var(a) + (n_b - 1) * _var(b)) / (n_a + n_b - 2)
    return _ratio(_mean(a) - _mean(b), pooled * Fraction(n_a + n_b, n_a * n_b))


def exact_cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """(mean(a) - mean(b)) / sqrt((v_a + v_b) / 2), Hedges' g before J."""
    return _ratio(_mean(a) - _mean(b), (_var(a) + _var(b)) / 2)


def exact_welch_anova(groups: Sequence[Sequence[float]]) -> tuple[float, float]:
    k = len(groups)
    n = [len(g) for g in groups]
    means = [_mean(g) for g in groups]
    weights = [ni / _var(g) for ni, g in zip(n, groups, strict=True)]
    total = sum(weights, Fraction(0))
    centre = sum((w * m for w, m in zip(weights, means, strict=True)), Fraction(0))
    centre /= total
    between = sum(
        (w * (m - centre) ** 2 for w, m in zip(weights, means, strict=True)),
        Fraction(0),
    ) / (k - 1)
    lam = sum(
        ((1 - w / total) ** 2 / (ni - 1) for w, ni in zip(weights, n, strict=True)),
        Fraction(0),
    )
    f = between / (1 + Fraction(2 * (k - 2), k * k - 1) * lam)
    return float(f), float(Fraction(k * k - 1) / (3 * lam))


def exact_games_howell_q(groups: Sequence[Sequence[float]]) -> list[float]:
    """q of every pair i < j, signed as mean(j) - mean(i)."""
    out: list[float] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            w = _var(groups[i]) / len(groups[i]) + _var(groups[j]) / len(groups[j])
            out.append(_ratio(_mean(groups[j]) - _mean(groups[i]), w / 2))
    return out


def assert_two_sample_exact(a: Sequence[float], b: Sequence[float]) -> None:
    """Welch's t and df, Student's t and Hedges' g (and its CI, BF finite) agree
    with exact arithmetic within closed_form."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t, df = exact_welch(a, b)
    welch = welch_t(a_arr, b_arr)
    assert_close("closed_form", welch.t, t, what="Welch t")
    assert_close("closed_form", welch.df, df, what="Welch df")
    assert 0.0 <= welch.p_value <= 1.0
    assert_close(
        "closed_form", student_t(a_arr, b_arr).t, exact_student(a, b), what="t"
    )
    g = hedges_g(a_arr, b_arr)
    expected_g = exact_cohens_d(a, b) * hedges_j(df)
    assert_close("closed_form", g.estimate, expected_g, what="Hedges' g")
    assert math.isfinite(g.conf_low)
    assert math.isfinite(g.conf_high)
    assert g.conf_low <= g.estimate <= g.conf_high
    bf10, log_bf01 = jzs_bf10_two_sample(a_arr, b_arr, prior_scale=0.707)
    assert not math.isnan(bf10)
    assert math.isfinite(log_bf01)


def assert_k_groups_exact(groups: Sequence[Sequence[float]]) -> None:
    arrays = [np.asarray(g, dtype=float) for g in groups]
    f, df2 = exact_welch_anova(groups)
    anova = welch_anova(arrays)
    assert_close("closed_form", anova.f, f, what="Welch F")
    assert_close("closed_form", anova.df2, df2, what="Welch df2")
    assert 0.0 <= anova.p_value <= 1.0
    pairs = games_howell(arrays)
    for pair, q in zip(pairs, exact_games_howell_q(groups), strict=True):
        assert_close("closed_form", pair.statistic, q, what=f"q {pair.i}-{pair.j}")
        assert 0.0 <= pair.p_value <= 1.0
        assert 0.0 <= pair.p_adjusted <= 1.0


def frame(groups: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    rows = [(g, float(v)) for g, values in groups.items() for v in values]
    return pd.DataFrame(rows, columns=["g", "y"])


# ------------------------------------------------- the reproductions of the review


def test_welch_t_raises_where_the_constant_group_dwarfs_the_other() -> None:
    # Was t = 1.7328971554364237e160, 4.9e-4 above the exact 1.7320508075688774e160.
    with pytest.raises(ValueError, match=r"sample a is constant and sample b varies"):
        welch_t(np.array([1e160] * 3), np.array([1.0, 2.0, 3.0]))
    # Was 4.9e-10 off, outside closed_form.
    with pytest.raises(ValueError, match=r"rescaling both samples does not help"):
        welch_t(np.array([1e160] * 3), np.array([1e3, 2e3, 3e3]))
    for method in (student_t, hedges_g):
        with pytest.raises(ValueError, match=r"sample a is constant and sample b"):
            method(np.array([1e160] * 3), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match=r"sample b is constant and sample a"):
        jzs_bf10_two_sample([1.0, 2.0, 3.0], [1e160] * 3, prior_scale=0.707)


def test_welch_t_is_exact_where_the_spread_is_resolved() -> None:
    a = [1e160] * 3
    assert_two_sample_exact(a, [1e10, 2e10, 3e10])
    t, _ = exact_welch(a, [1e10, 2e10, 3e10])
    assert welch_t(np.array(a), np.array([1e10, 2e10, 3e10])).t == pytest.approx(
        t, rel=1e-15
    )


def test_both_constant_keeps_its_message() -> None:
    with pytest.raises(ValueError, match="both samples are constant: Welch's t"):
        welch_t([1e160, 1e160], [1.0, 1.0])
    with pytest.raises(ValueError, match="both samples are constant: Student's t"):
        student_t([1e160, 1e160], [1.0, 1.0])


REVIEW_GROUPS = [[1e160, 2e160, 1.5e160], [1.0, 2.0, 3.0], [4.0, 5.0, 7.0]]


def test_welch_anova_and_games_howell_raise_instead_of_nan() -> None:
    # Welch's ANOVA returned F = df2 = p = nan; Games-Howell q(2, 3) = 4.47563
    # where 4.4721 is right (7.8e-4).
    arrays = [np.array(g) for g in REVIEW_GROUPS]
    with pytest.raises(ValueError, match=GROUP_2_TOO_LITTLE + ".*Welch's ANOVA"):
        welch_anova(arrays)
    with pytest.raises(ValueError, match=GROUP_2_TOO_LITTLE + ".*Games-Howell"):
        games_howell(arrays)


def test_the_analysis_names_the_groups_in_plain_words() -> None:
    # Were "F must be finite; got nan" and "t and ncp must be finite; got 1.73e+160
    # and -inf".
    groups = dict(zip("abc", REVIEW_GROUPS, strict=True))
    with pytest.raises(ValueError, match="double precision") as anova:
        betweenstats(frame(groups), "g", "y")
    assert str(anova.value).startswith(
        "The values of y in the groups 'b' and 'c' of x='g' vary by less than double "
        "precision can resolve next to the largest |y| (a standard deviation below "
        "roughly 1e-154 times it), so Welch's ANOVA and the Games-Howell "
        "comparisons cannot weight those groups by their variances."
    )
    with pytest.raises(ValueError, match="double precision") as two:
        betweenstats(frame({"a": [1e160] * 3, "b": [1.0, 2.0, 3.0]}), "g", "y")
    assert str(two.value) == (
        "The values of y in the group 'b' of x='g' vary by less than double precision "
        "can resolve next to the largest |y| (a standard deviation below roughly "
        "1e-154 times it), and the group 'a' has no variability, so Welch's t-test "
        "has no accurate standard error. Rescaling y does not help, since it "
        "multiplies the spread and the largest |y| alike; check y for entry or unit "
        "errors."
    )
    for message in (str(anova.value), str(two.value)):
        assert "nan" not in message
        assert "inf" not in message


# ------------------------------------------------------------ the two boundaries


def test_a_peak_of_1e150_computes_exactly() -> None:
    assert_two_sample_exact([1e150] * 3, [1.0, 2.0, 3.0])
    groups = [[1e150, 2e150, 1.5e150], [1.0, 2.0, 3.0], [4.0, 5.0, 7.0]]
    assert_k_groups_exact(groups)
    assert welch_anova([np.array(g) for g in groups]).f == pytest.approx(
        15.448851774530, rel=1e-12
    )
    result = betweenstats(frame(dict(zip("abc", groups, strict=True))), "g", "y")
    assert_close(
        "closed_form", result.test.statistic, exact_welch_anova(groups)[0], what="F"
    )


def test_a_peak_of_1e160_raises() -> None:
    with pytest.raises(ValueError, match=TWO_SAMPLE_TOO_LITTLE):
        welch_t(np.array([1e160] * 3), np.array([1.0, 2.0, 3.0]))
    groups = [np.array([1e160, 2e160, 1.5e160]), np.array([1.0, 2.0, 3.0])]
    groups.append(np.array([4.0, 5.0, 7.0]))
    with pytest.raises(ValueError, match=GROUP_2_TOO_LITTLE):
        welch_anova(groups)


def test_the_boundary_is_the_smallest_normal_double() -> None:
    # b = [0, 1, 2] has v = 1, so its scaled term is 2^(-2e) / 3 for a constant
    # group in [2^(e-1), 2^e): e = 510 gives 2^-1020 / 3 >= TINY, e = 511 does not.
    b = [0.0, 1.0, 2.0]
    below = math.nextafter(2.0**510, 0.0)
    assert 2.0**-1020 / 3 >= TINY > 2.0**-1022 / 3
    assert_two_sample_exact([below] * 3, b)
    with pytest.raises(ValueError, match=TWO_SAMPLE_TOO_LITTLE):
        welch_t([2.0**510] * 3, b)
    groups = [[below] * 2 + [below / 2], b, [3.0, 5.0, 4.0]]
    assert_k_groups_exact(groups)
    with pytest.raises(ValueError, match=GROUP_2_TOO_LITTLE):
        welch_anova([[2.0**510] * 2 + [2.0**509], b, [3.0, 5.0, 4.0]])


def test_varying_peak_group_next_to_a_small_one_is_exact() -> None:
    # The group with the largest |value| varies, so its term dominates and the
    # small group's subnormal squares do not matter.
    assert_two_sample_exact([1e160, 2e160, 3e160], [1.0, 2.0, 3.0])
    assert_two_sample_exact([1.0, 2.0, 3.0], [1e300, -2e300, 3e300])
    result = betweenstats(
        frame({"a": [1e160, 2e160, 3e160], "b": [1.0, 2.0, 3.0]}), "g", "y"
    )
    t, df = exact_welch([1e160, 2e160, 3e160], [1.0, 2.0, 3.0])
    assert_close("closed_form", result.test.statistic, t, what="t")
    assert_close("closed_form", result.test.df_error, df, what="df")
    assert result.bayes is not None
    assert math.isfinite(result.bayes.log_bf01)


def test_subnormal_squares_above_the_boundary_are_accurate() -> None:
    # 100 values 0..99 next to a constant group just below 2^512: the scaled
    # deviations are 2^-512 to 2^-506, so the smallest squares are subnormal while
    # v / n stays above TINY.
    b = [float(i) for i in range(100)]
    assert_two_sample_exact([math.nextafter(2.0**512, 0.0)] * 4, b)


# ---------------------------------------------------- |t| up to 2^512 end to end


def test_the_largest_two_sample_t_runs_through_the_whole_analysis() -> None:
    # The guard bounds |t| by 2 / sqrt(TINY) = 2^512 (about 1.34e154); here
    # t = 5.8e153 with every number of the analysis finite.
    peak = math.nextafter(2.0**510, 0.0)
    a, b = [peak] * 3, [0.0, 1.0, 2.0]
    result = betweenstats(frame({"a": a, "b": b}), "g", "y")
    t, df = exact_welch(a, b)
    assert t > 5e153
    assert_close("closed_form", result.test.statistic, t, what="t")
    assert_close("closed_form", result.test.df_error, df, what="df")
    # At df = 2 the two-sided p is 1 - t / sqrt(2 + t^2) = 1 / t^2 (1 + O(t^-2)):
    # 2.97e-308, still a normal double.
    assert_close("closed_form", result.test.p_value, 1.0 / t / t, what="p")
    es = result.effect_size
    for value in (es.estimate, es.conf_low, es.conf_high):
        assert math.isfinite(value)
    assert es.conf_low < es.estimate < es.conf_high
    assert result.bayes is not None
    assert result.bayes.bf10 == math.inf
    # log BF10 follows the large-t limit (tests/stats/test_bayes.py).
    student = exact_student(a, b)
    n_eff, nu, r = 1.5, 4.0, 0.707
    limit = (
        0.5 * (nu - 1.0) * (2.0 * math.log(student) - math.log(nu))
        + math.log(2.0 / (nu - 1.0))
        + 0.5 * math.log(n_eff * r * r)
        - 0.5 * math.log(2.0 * math.pi)
    )
    assert -result.bayes.log_bf01 == pytest.approx(limit, rel=1e-13)
    assert "inf" not in result.text.subtitle.lower()


def test_the_bayes_factor_check_names_bf_message() -> None:
    # Student's term can be below TINY while Welch's is not: a constant group of
    # many values next to one of two values.
    peak = math.nextafter(2.0**510, 0.0)
    data = frame({"a": [peak] * 40, "b": [0.0, 1.0]})
    with pytest.raises(ValueError, match=r"so the Bayes factor \(on Student's t\)"):
        betweenstats(data, "g", "y")
    with pytest.raises(ValueError, match=r"bf_message=False"):
        betweenstats(data, "g", "y")
    result = betweenstats(data, "g", "y", bf_message=False)
    t, _ = exact_welch([peak] * 40, [0.0, 1.0])
    assert_close("closed_form", result.test.statistic, t, what="t")


# --------------------------------------------------------- Welch's ANOVA weights


def test_welch_anova_weights_near_the_largest_double_do_not_overflow() -> None:
    # Six groups with scaled variance terms just above TINY: weights of 3 * 2^1020
    # each, whose sum (18 * 2^1020) is above the largest double. Before, math.fsum
    # raised OverflowError.
    d = 2.0**-10
    peak = [2.0**499, 1.8 * 2.0**499, 1.4 * 2.0**499]
    groups = [peak] + [[c, c + d, c + 2 * d] for c in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    assert_k_groups_exact(groups)


# ------------------------------------------------------- a sweep of dynamic ranges

RANGES = [10.0**e for e in range(100, 301, 5)]


def _configurations(r: float) -> list[list[list[float]]]:
    small, other = [1.0, 2.0, 3.0], [4.0, 5.0, 7.0]
    return [
        [[r] * 3, small],
        [[r, 2 * r, 1.5 * r], small],
        [small, [-r, -r, -r, -r]],
        [[r, 2 * r, 1.5 * r], small, other],
        [[r, r, 2 * r], other, small, [0.5, 0.25, 0.75]],
        [[-r] * 3 + [-r / 2], small, other],
    ]


@pytest.mark.parametrize("r", RANGES)
def test_every_range_is_exact_or_raises_plainly(r: float) -> None:
    for groups in _configurations(r):
        try:
            if len(groups) == 2:
                assert_two_sample_exact(*groups)
            else:
                assert_k_groups_exact(groups)
        except ValueError as error:
            message = str(error)
            assert "double precision" in message, message
            assert "nan" not in message
            assert "inf" not in message
        names = "abcd"[: len(groups)]
        data = frame(dict(zip(names, groups, strict=True)))
        try:
            result = betweenstats(data, "g", "y")
        except ValueError as error:
            message = str(error)
            assert message.startswith("The values of y in the group"), message
            assert "double precision" in message
            continue
        numbers = [result.test.statistic, result.test.df_error, result.test.p_value]
        es = result.effect_size
        numbers += [es.estimate, es.conf_low, es.conf_high]
        for pair in result.pairwise:
            numbers += [pair.statistic, pair.p_value, pair.p_adjusted]
        assert all(math.isfinite(v) for v in numbers), numbers
        if len(groups) == 2:
            t, _ = exact_welch(*groups)
        else:
            t, _ = exact_welch_anova(groups)
        assert_close("closed_form", result.test.statistic, t, what=f"r={r:g}")


def test_the_sweep_covers_both_outcomes() -> None:
    computed = raised = 0
    for r in RANGES:
        try:
            welch_t([r] * 3, [1.0, 2.0, 3.0])
            computed += 1
        except ValueError:
            raised += 1
    # sd 1 next to a constant group: resolved up to about 1.3e154 (2^511 / sqrt 3)
    assert computed == 11  # 1e100 ... 1e150
    assert raised == len(RANGES) - 11
