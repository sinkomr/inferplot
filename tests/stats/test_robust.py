"""The robust layer: definitions, the bootstrap rules, the critical-value table,
invariances and edge inputs (oracle parity is in test_robust_parity.py)."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy import integrate, special, stats
from tools.reference import smm_table
from tools.reference import stats_reference as ref

from inferplot.stats import (
    BOOTSTRAP_RESAMPLES,
    akp_effect,
    explanatory_effect,
    lincon,
    smm_critical_value,
    trimmed_mean,
    trimmed_means_anova,
    winsorized_normal_variance,
    winsorized_variance,
    yuen_t,
)
from inferplot.stats._smm_table import SMM_ROWS, SMM_TABLE_05
from inferplot.stats.robust import (
    _floor_percentile_bounds,  # pyright: ignore[reportPrivateUsage]
    _rounded_percentile_bounds,  # pyright: ignore[reportPrivateUsage]
)
from tests.stats.fixture_data import between_cases, groups
from tests.tolerances import assert_close

FloatArray = NDArray[np.float64]


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _draw(sizes: list[int], seed: int) -> list[FloatArray]:
    rng = _rng(seed)
    return [rng.normal(0.3 * i, 1.0 + 0.4 * i, n) for i, n in enumerate(sizes)]


# ------------------------------------------------------------------ definitions


@pytest.mark.parametrize("n", [5, 10, 11, 23, 60])
@pytest.mark.parametrize("tr", [0.0, 0.1, 0.2, 0.25])
def test_trimmed_mean_and_winsorized_variance_definitions(n: int, tr: float) -> None:
    x = _rng(n).standard_t(3, n)
    assert_close(
        "closed_form",
        trimmed_mean(x, tr=tr),
        float(stats.trim_mean(x, tr)),
        what="trimmed mean",
    )
    g = math.floor(tr * n)
    s = np.sort(x)
    clipped = np.clip(x, s[g], s[n - g - 1])
    assert_close(
        "closed_form",
        winsorized_variance(x, tr=tr),
        float(clipped.var(ddof=1)),
        what="winsorized variance",
    )


def test_winsorized_normal_variance() -> None:
    for tr in (0.05, 0.1, 0.2, 0.3, 0.45):
        c = float(stats.norm.isf(tr))
        inner = integrate.quad(
            lambda z: z * z * stats.norm.pdf(z), -c, c, epsabs=0, epsrel=1e-13
        )[0]
        expected = inner + 2 * tr * c * c
        assert_close(
            "closed_form", winsorized_normal_variance(tr), expected, what=f"tr {tr}"
        )
    assert winsorized_normal_variance(0.0) == 1.0
    # Algina, Keselman & Penfield's constant, and WRS2's approximations of it.
    k = math.sqrt(winsorized_normal_variance(0.2))
    assert round(k, 3) == 0.642
    assert abs(0.4120867 / winsorized_normal_variance(0.2) - 1) == pytest.approx(
        6.48e-8, rel=1e-3
    )


# ------------------------------------------------------------------ bootstrap rules


@pytest.mark.parametrize(
    ("count", "akp", "t1way"),
    [
        # 0-based indices of the bounds. At B = 100, alpha / 2 B is
        # 2.5000000000000022 (alpha = 1 - 0.95 in floating point): t1way's
        # rounding gives the 4th smallest, akp.effect's floor the 3rd; both take
        # the 97th. These are the order statistics that reproduce ggstatsplot's
        # 100-resample bounds exactly in every robust fixture.
        (100, (2, 96), (3, 96)),
        (2000, (50, 1949), (50, 1949)),
        (20000, (500, 19499), (500, 19499)),
    ],
)
def test_percentile_order_statistics(
    count: int, akp: tuple[int, int], t1way: tuple[int, int]
) -> None:
    ordered = np.arange(count, dtype=np.float64)
    assert _floor_percentile_bounds(ordered, 0.95) == akp
    assert _rounded_percentile_bounds(ordered, 0.95) == t1way


def test_bootstrap_is_seeded_and_records_its_seed() -> None:
    a, b, c = _draw([12, 15, 9], 3)
    first = explanatory_effect([a, b, c], seed=5)
    assert first == explanatory_effect([a, b, c], seed=5)
    assert first.seed == 5
    assert first.resamples == BOOTSTRAP_RESAMPLES == 2000
    other = explanatory_effect([a, b, c], seed=6)
    assert (other.conf_low, other.conf_high) != (first.conf_low, first.conf_high)
    # Unequal sizes: the estimate is the median of the resamples, so it is random.
    assert other.estimate != first.estimate
    effect = akp_effect(a, b, seed=7)
    assert effect == akp_effect(a, b, seed=7)
    assert effect.seed == 7
    assert akp_effect(a, b).seed == 0
    assert effect.estimate == akp_effect(a, b, seed=8).estimate


def test_equal_sizes_give_a_fixed_explanatory_estimate() -> None:
    data = _draw([10, 10, 10], 4)
    assert explanatory_effect(data, seed=1).estimate == (
        explanatory_effect(data, seed=2).estimate
    )


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_bad_seeds(seed: object) -> None:
    a, b = _draw([8, 8], 1)
    with pytest.raises(ValueError, match="seed"):
        akp_effect(a, b, seed=seed)  # type: ignore[arg-type]


# ------------------------------------------------------------------ SMM table


def test_table_shape() -> None:
    assert SMM_ROWS == (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 30, 40,
                        60, 200)  # fmt: skip
    assert sorted(SMM_TABLE_05) == list(range(1, 46))
    for row in SMM_TABLE_05.values():
        assert len(row) == len(SMM_ROWS)
        assert all(round(v, 2) == v for v in row)
        assert list(row) == sorted(row, reverse=True)  # falls as df grows
    for df_index in range(len(SMM_ROWS)):
        column = [SMM_TABLE_05[c][df_index] for c in range(1, 46)]
        assert column == sorted(column)  # grows with C


def test_one_comparison_is_student_t_and_infinite_df_is_normal() -> None:
    # Closed forms, independent of the quadrature: C = 1 is |T_df|, and at
    # infinite df P(max |Z| <= c) = (2 Phi(c) - 1)^C.
    for index, df in enumerate(SMM_ROWS[:-1]):
        assert SMM_TABLE_05[1][index] == round(float(stats.t.isf(0.025, df)), 2)
    for c in range(1, 46):
        exact = float(stats.norm.ppf((1 + 0.95 ** (1 / c)) / 2))
        assert SMM_TABLE_05[c][-1] == round(exact, 2), c


@pytest.mark.parametrize(
    ("comparisons", "df"),
    [(2, 4), (3, 5), (3, 9), (6, 60), (10, 5), (10, 7), (15, 12), (28, 3), (45, 40)],
)
def test_entries_are_the_rounded_quantiles(comparisons: int, df: int) -> None:
    exact = smm_table.smm_quantile(comparisons, df)
    entry = SMM_TABLE_05[comparisons][SMM_ROWS.index(df)]
    published = smm_table.PUBLISHED_DIFFERS.get((comparisons, df))
    assert entry == (published if published is not None else round(exact, 2))


@pytest.mark.slow
def test_whole_table_is_regenerated_exactly() -> None:
    assert smm_table.build() == SMM_TABLE_05


def test_c3_df5_is_the_published_value_not_the_rounded_quantile() -> None:
    assert smm_table.smm_quantile(3, 5) == pytest.approx(3.39924, abs=1e-5)
    assert smm_critical_value(3, 5) == 3.39


def test_critical_value_interpolation() -> None:
    for c in (1, 3, 10, 45):
        for index, df in enumerate(SMM_ROWS):
            assert smm_critical_value(c, df) == SMM_TABLE_05[c][index]
    # Linear in 1/df between rows: C = 3 between df 5 (3.39) and 6 (3.19).
    df = 5.161
    expected = 3.39 + (3.19 - 3.39) * (1 / 5 - 1 / df) / (1 / 5 - 1 / 6)
    assert smm_critical_value(3, df) == pytest.approx(expected, rel=1e-15)
    # From df 200 on, the last (infinite-df) row.
    assert smm_critical_value(6, 200) == smm_critical_value(6, 1e6) == 2.63


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {"n_comparisons": 3, "df": 10, "conf_level": 0.9},
            NotImplementedError,
            "0.95",
        ),
        ({"n_comparisons": 46, "df": 10}, NotImplementedError, "45"),
        ({"n_comparisons": 3, "df": 1.5}, NotImplementedError, "below 2"),
        ({"n_comparisons": 0, "df": 10}, ValueError, "at least 1"),
        ({"n_comparisons": 3, "df": 0.0}, ValueError, "positive"),
        ({"n_comparisons": 3, "df": math.nan}, ValueError, "positive"),
        ({"n_comparisons": 3, "df": 10, "conf_level": 1.0}, ValueError, "conf_level"),
    ],
)
def test_critical_value_refusals(
    kwargs: dict[str, float], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        smm_critical_value(**kwargs)  # type: ignore[arg-type]


def test_lincon_refuses_what_the_table_cannot_give() -> None:
    data = _draw([8, 8, 8], 2)
    with pytest.raises(NotImplementedError, match="conf_level"):
        lincon(data, conf_level=0.9)
    with pytest.raises(NotImplementedError, match="45"):
        lincon(_draw([6] * 11, 3))
    with pytest.raises(NotImplementedError, match="below 2"):
        lincon([np.array([0.0, 1.0]), np.array([5.0, 9.0]), np.array([2.0, 2.5])])


# ------------------------------------------------------------------ reference


def _reference_sets() -> list[object]:
    sets: list[object] = [
        pytest.param(groups(case), id=case.name) for case in between_cases("robust")
    ]
    for k in (2, 3, 5):
        sizes = _rng(k).integers(5, 40, size=k).tolist()
        sets.append(pytest.param(_draw(sizes, 10 + k), id=f"random-k{k}"))
    return sets


@pytest.mark.parametrize("data", _reference_sets())
def test_against_reference(data: list[FloatArray]) -> None:
    # Both compute the upper tails directly (DECISIONS.md D43).
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            result = yuen_t(data[i], data[j])
            t, df, p = ref.yuen(data[i], data[j])
            assert_close("closed_form", result.t, abs(t), what="Yuen t")
            assert_close("closed_form", result.df, df, what="Yuen df")
            assert_close("closed_form", result.p_value, p, what="Yuen p")
    if len(data) > 2:
        anova = trimmed_means_anova(data)
        f, df1, df2, p = ref.t1way(data)
        assert_close("closed_form", anova.f, f, what="t1way F")
        assert anova.df1 == df1
        assert_close("closed_form", anova.df2, df2, what="t1way df2")
        assert_close("closed_form", anova.p_value, p, what="t1way p")


def test_yuen_reduces_to_welch_without_trimming() -> None:
    a, b = _draw([9, 14], 8)
    result = yuen_t(a, b, tr=0.0)
    welch = stats.ttest_ind(a, b, equal_var=False)
    assert_close("closed_form", result.t, abs(float(welch.statistic)), what="t")
    assert_close("closed_form", result.df, float(welch.df), what="df")  # pyright: ignore[reportAttributeAccessIssue]


# ------------------------------------------------------------------ invariances


def _everything(data: list[FloatArray]) -> tuple[object, ...]:
    return (
        yuen_t(data[0], data[1]),
        akp_effect(data[0], data[1]),
        trimmed_means_anova(data),
        explanatory_effect(data),
        lincon(data),
    )


@settings(max_examples=25, deadline=None)
@given(power=st.integers(-900, 900), seed=st.integers(0, 2**16))
def test_power_of_two_scaling_is_exact(power: int, seed: int) -> None:
    data = _draw([7, 11, 9], seed)
    factor = math.ldexp(1.0, power)
    base = _everything(data)
    scaled = _everything([d * factor for d in data])
    y0, a0, f0, x0, l0 = base
    y1, a1, f1, x1, l1 = scaled
    assert (y1.t, y1.df, y1.p_value) == (y0.t, y0.df, y0.p_value)  # type: ignore[attr-defined]
    assert y1.difference == y0.difference * factor  # type: ignore[attr-defined]
    assert y1.se == y0.se * factor  # type: ignore[attr-defined]
    assert a1 == a0
    assert f1 == f0
    assert x1 == x0
    for p0, p1 in zip(l0, l1, strict=True):  # type: ignore[arg-type]
        assert (p1.t, p1.df, p1.p_value, p1.critical_value) == (
            p0.t,
            p0.df,
            p0.p_value,
            p0.critical_value,
        )
        assert p1.estimate == p0.estimate * factor
        assert (p1.conf_low, p1.conf_high) == (
            p0.conf_low * factor,
            p0.conf_high * factor,
        )


@settings(max_examples=25, deadline=None)
@given(
    shift=st.floats(-1e3, 1e3),
    scale=st.floats(0.01, 100),
    seed=st.integers(0, 2**16),
)
def test_location_and_scale_equivariance(shift: float, scale: float, seed: int) -> None:
    data = _draw([8, 12, 10], seed)
    moved = [d * scale + shift for d in data]
    y0, y1 = yuen_t(data[0], data[1]), yuen_t(moved[0], moved[1])
    # The data are rounded once by the transformation, so the comparison allows
    # for it: rtol 1e-9 on statistics that are ratios of differences.
    assert y1.t == pytest.approx(y0.t, rel=1e-9)
    assert y1.df == pytest.approx(y0.df, rel=1e-9)
    assert y1.difference == pytest.approx(y0.difference * scale, rel=1e-9, abs=1e-9)
    f0, f1 = trimmed_means_anova(data), trimmed_means_anova(moved)
    assert f1.f == pytest.approx(f0.f, rel=1e-9)
    assert f1.df2 == pytest.approx(f0.df2, rel=1e-9)
    e0, e1 = akp_effect(data[0], data[1]), akp_effect(moved[0], moved[1])
    assert e1.estimate == pytest.approx(e0.estimate, rel=1e-9)
    assert e1.conf_low == pytest.approx(e0.conf_low, rel=1e-9)
    assert e1.conf_high == pytest.approx(e0.conf_high, rel=1e-9)
    x0, x1 = explanatory_effect(data), explanatory_effect(moved)
    assert x1.estimate == pytest.approx(x0.estimate, rel=1e-9)
    assert x1.conf_low == pytest.approx(x0.conf_low, rel=1e-9)
    assert x1.conf_high == pytest.approx(x0.conf_high, rel=1e-9)


def test_sign_flip() -> None:
    a, b = _draw([9, 13], 21)
    assert akp_effect(-a, -b).estimate == -akp_effect(a, b).estimate
    y0, y1 = yuen_t(a, b), yuen_t(-a, -b)
    assert (y1.t, y1.df, y1.p_value) == (y0.t, y0.df, y0.p_value)
    assert y1.difference == -y0.difference
    # Swapping the samples flips the difference, not |t|.
    swapped = yuen_t(b, a)
    assert swapped.t == y0.t
    assert swapped.difference == -y0.difference


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**16))
def test_row_order_does_not_matter(seed: int) -> None:
    data = _draw([9, 6, 12], seed)
    rng = _rng(seed + 1)
    shuffled = [rng.permutation(d) for d in data]
    assert _everything(shuffled) == _everything(data)


def test_group_relabelling() -> None:
    data = _draw([10, 10, 10, 10], 31)
    order = [2, 0, 3, 1]
    permuted = [data[i] for i in order]
    f0, f1 = trimmed_means_anova(data), trimmed_means_anova(permuted)
    assert f1.f == pytest.approx(f0.f, rel=1e-13)
    assert f1.df2 == pytest.approx(f0.df2, rel=1e-13)
    assert f1.p_value == pytest.approx(f0.p_value, rel=1e-12, abs=2**-53)
    x0, x1 = explanatory_effect(data), explanatory_effect(permuted)
    assert x1.estimate == pytest.approx(x0.estimate, rel=1e-13)
    pairs0 = {(p.i, p.j): p for p in lincon(data)}
    for p in lincon(permuted):
        i, j = order[p.i], order[p.j]
        q, sign = (pairs0[(i, j)], 1.0) if i < j else (pairs0[(j, i)], -1.0)
        assert p.estimate == pytest.approx(sign * q.estimate, rel=1e-13)
        assert p.df == pytest.approx(q.df, rel=1e-13)
        assert p.p_adjusted == pytest.approx(q.p_adjusted, rel=1e-12, abs=1e-15)


# ------------------------------------------------------------------ edge inputs


def test_trimming_must_leave_two_values() -> None:
    short = np.array([1.0, 2.0, 3.0])  # tr 0.45: g = 1, one value left
    other = np.array([4.0, 6.0, 9.0, 10.0])
    with pytest.raises(ValueError, match="leaves 1"):
        yuen_t(short, other, tr=0.45)
    with pytest.raises(ValueError, match="leaves 1"):
        trimmed_means_anova([short, other, other], tr=0.45)
    with pytest.raises(ValueError, match="leaves 1"):
        lincon([short, other, other], tr=0.45)
    with pytest.raises(ValueError, match="at least 2"):
        yuen_t(np.array([1.0]), other)
    # Two values per group: nothing is trimmed at tr 0.2, Yuen's test is defined.
    two = yuen_t(np.array([1.0, 2.0]), np.array([4.0, 7.0]))
    assert two.df > 0
    assert math.isfinite(two.t)


@pytest.mark.parametrize("tr", [-0.1, 0.5, 0.7, math.nan])
def test_bad_trim(tr: float) -> None:
    with pytest.raises(ValueError, match="tr must"):
        trimmed_mean(np.arange(5.0), tr=tr)


def test_constant_and_zero_winsorized_variance() -> None:
    constant = np.full(8, 3.0)
    varying = np.array([1.0, 2.0, 4.0, 4.5, 6.0, 7.0, 9.0, 12.0])
    # [1, 1, 1, 1, 9] winsorized at g = 1 is constant, but not the data.
    flat = np.array([1.0, 1.0, 1.0, 1.0, 9.0])
    with pytest.raises(ValueError, match="both samples have a winsorized variance"):
        yuen_t(constant, flat)
    one = yuen_t(constant, varying)
    assert math.isfinite(one.t)
    assert one.df == pytest.approx(5.0, rel=1e-15)  # h - 1 of the varying sample
    with pytest.raises(ValueError, match="group 2 has a winsorized variance of 0"):
        trimmed_means_anova([varying, flat, varying + 1])
    with pytest.raises(ValueError, match="sample a has a winsorized variance of 0"):
        akp_effect(flat, varying)
    assert math.isfinite(akp_effect(varying, constant).estimate)
    with pytest.raises(ValueError, match="groups 1 and 3 both"):
        lincon([constant, varying, flat])
    with pytest.raises(ValueError, match="0 / 0"):
        explanatory_effect([constant, constant, constant])


def test_undefined_resamples_raise() -> None:
    # Most resamples of `a` with eight or more zeros winsorize to all zeros with a
    # trimmed mean of 0, as `b` has: 0 / 0, so no percentile interval exists.
    a = np.array([0.0] * 6 + [1.0] * 4)
    b = np.zeros(10)
    with pytest.raises(ValueError, match="undefined"):
        akp_effect(a, b)


def test_constant_groups_in_the_explanatory_effect() -> None:
    # Constant but different groups: every resample has between-group variation
    # and a pooled winsorized variance > 0.
    data = [np.full(10, 1.0), np.full(10, 2.0), np.full(10, 4.0)]
    effect = explanatory_effect(data)
    assert math.isfinite(effect.estimate)
    assert effect.estimate > 0


def test_large_n_timing() -> None:
    data = _draw([1200, 1300, 1100], 41)  # the size of the edge_large_n fixture
    start = time.perf_counter()
    explanatory_effect(data)
    akp_effect(data[0], data[1])
    trimmed_means_anova(data)
    lincon(data)
    seconds = time.perf_counter() - start
    assert seconds <= 5.0, f"robust statistics at n = 3,600 took {seconds:.2f} s"
    big = _draw([300_000, 300_000, 400_000], 42)
    start = time.perf_counter()
    trimmed_means_anova(big)
    lincon(big)
    seconds = time.perf_counter() - start
    assert seconds <= 3.0, f"t1way and lincon at n = 1e6 took {seconds:.2f} s"


# ------------------------------------------------------------------ tiny p-values


def _iris_petal_length() -> list[FloatArray]:
    path = Path(__file__).resolve().parents[2] / "oracle" / "datasets" / "iris.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    species = sorted({row["Species"] for row in rows})
    return [
        np.array([float(r["Petal.Length"]) for r in rows if r["Species"] == s])
        for s in species
    ]


def test_p_values_are_accurate_where_one_minus_cdf_is_zero() -> None:
    # R's 1 - pf and 2 (1 - pt) give exactly 0 here (ggstatsplot would print
    # 2.23e-308); inferplot reports the upper tail (DECISIONS.md D43). References:
    # mpmath at 50 digits, the regularized incomplete beta at these F, t and df:
    # I_{df2/(df2 + df1 F)}(df2/2, df1/2) and I_{df/(df + t^2)}(df/2, 1/2)
    # (uv run --with mpmath).
    setosa, versicolor, virginica = _iris_petal_length()
    anova = trimmed_means_anova([setosa, versicolor, virginica])
    assert_close("closed_form", anova.f, 1510.6836909956144, what="t1way F")
    assert_close("closed_form", anova.df2, 45.87618043707867, what="t1way df2")
    assert 1.0 - float(special.fdtr(anova.df1, anova.df2, anova.f)) == 0.0
    assert anova.p_value > 0.0
    assert_close("closed_form", anova.p_value, 1.3623605417577488e-42, what="p")
    yuen = yuen_t(setosa, virginica)
    assert_close("closed_form", yuen.t, 45.12516909123536, what="Yuen t")
    assert_close("closed_form", yuen.df, 34.46263814754606, what="Yuen df")
    assert 2.0 * (1.0 - float(special.stdtr(yuen.df, yuen.t))) == 0.0
    assert yuen.p_value > 0.0
    assert_close("closed_form", yuen.p_value, 3.0399821392506091e-32, what="p")
    pairs = lincon([setosa, versicolor, virginica])
    assert pairs[1].p_value == yuen.p_value  # setosa-virginica: the same test
    assert all(pair.p_value > 0.0 for pair in pairs)
