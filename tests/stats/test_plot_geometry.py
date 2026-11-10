"""What the histostats and scatterstats figures draw: the histogram's bins, the OLS
line with its band, the marginal histograms. Against every built layer the
fixtures record (ggplot2 4.0.3 under ggstatsplot 1.1.4) and by definition."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from scipy import stats

from inferplot.stats import (
    MARGINAL_BINS,
    SMOOTH_POINTS,
    histogram_bins,
    histogram_binwidth,
    marginal_bins,
    ols_band,
)
from tests.oracle_fixtures import OracleCase, cases_for, number, require_cases

HISTO = require_cases(cases_for("gghistostats"), "gghistostats fixtures")
SCATTER = require_cases(cases_for("ggscatterstats"), "ggscatterstats fixtures")


def _ids(cases: Any) -> list[Any]:
    return [pytest.param(c, id=c.name) for c in cases]


def _edges(bins: list[Any], low: str, high: str) -> list[float]:
    return [number(bins[0][low])] + [number(b[high]) for b in bins]


def _assert_edges(mine: tuple[float, ...], theirs: list[float]) -> None:
    # The fixtures store 15 significant digits.
    assert len(mine) == len(theirs)
    for a, b in zip(mine, theirs, strict=True):
        assert math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-14), (a, b)


@pytest.mark.parametrize("case", _ids(HISTO))
def test_histogram_bins(case: OracleCase) -> None:
    x = case.analyzed_values("x")
    (params,) = case.layer_params("GeomBar")
    (bins,) = case.layer_built("GeomBar")
    got = histogram_bins(x, binwidth=case.args.get("binwidth"))
    assert math.isclose(got.binwidth, number(params["stat"]["binwidth"]), rel_tol=1e-14)
    _assert_edges(got.edges, _edges(bins, "xmin", "xmax"))
    assert list(got.counts) == [int(b["count"]) for b in bins]
    assert sum(got.counts) == got.n == len(x)
    for share, b in zip(got.proportions, bins, strict=True):
        assert share == int(b["count"]) / len(x)


@pytest.mark.parametrize("case", _ids(SCATTER))
def test_marginal_bins(case: OracleCase) -> None:
    built = 0
    for geom, column, low, high in (
        ("GeomXsidebar", "x", "xmin", "xmax"),
        ("GeomYsidebar", "y", "ymin", "ymax"),
    ):
        layers = [layer["built"] for layer in case.plot_layers if layer["geom"] == geom]
        if not layers:
            assert case.args.get("marginal") is False
            continue
        (side,) = layers
        values = case.analyzed_values(column)  # type: ignore[arg-type]
        got = marginal_bins(values)
        _assert_edges(got.edges, _edges(side, low, high))
        assert list(got.counts) == [int(b["count"]) for b in side]
        built += 1
    assert built == (0 if case.args.get("marginal") is False else 2)


def _smooth_cases() -> list[Any]:
    chosen = [
        c
        for c in SCATTER
        if (c.args.get("smooth.line.args") or {}).get("method", "lm") == "lm"
        and c.layer_built("GeomSmooth")[0]
    ]
    return _ids(require_cases(chosen, "lm smooths"))


@pytest.mark.parametrize("case", _smooth_cases())
def test_ols_band(case: OracleCase) -> None:
    x = case.analyzed_values("x")
    y = case.analyzed_values("y")
    level = float(case.args.get("conf.level", 0.95))
    (built,) = case.layer_built("GeomSmooth")
    band = ols_band(x, y, conf_level=level)
    assert len(built) == SMOOTH_POINTS == len(band.x)
    for i, row in enumerate(built):
        assert band.x[i] == number(row["x"]) or math.isclose(
            band.x[i], number(row["x"]), rel_tol=1e-14
        )
        for key, mine in (
            ("y", band.fit[i]),
            ("se", band.se[i]),
            ("ymin", band.conf_low[i]),
            ("ymax", band.conf_high[i]),
        ):
            assert math.isclose(mine, number(row[key]), rel_tol=1e-8, abs_tol=1e-12), (
                key,
                i,
            )
    assert band.x[0] == min(x)
    assert band.x[-1] == max(x)


def test_constant_x_has_no_line_and_one_marginal_bin() -> None:
    (case,) = [
        c for c in SCATTER if c.name == "ggscatterstats-edge_corr_constant_x-parametric"
    ]
    x = case.analyzed_values("x")
    assert case.layer_built("GeomSmooth") == [[]]
    with pytest.raises(ValueError, match="least-squares line is undefined"):
        ols_band(x, case.analyzed_values("y"))
    got = marginal_bins(x)
    assert got.edges == (4.95, 5.05)
    assert got.counts == (10,)


# ------------------------------------------------------------------ definitions


def test_histogram_definition() -> None:
    x = [4.2, 11.5, 7.3, 5.8, 6.4, 10.0, 11.2, 11.2, 5.2, 7.0]
    width = histogram_binwidth(x)
    assert width == (11.5 - 4.2) / math.sqrt(10)
    got = histogram_bins(x)
    # Edges on the grid (k + 1/2) w, from the bin holding min to the one holding max.
    for edge in got.edges:
        k = edge / width - 0.5
        assert abs(k - round(k)) < 1e-9
    assert got.edges[0] <= min(x) < got.edges[1]
    assert got.edges[-2] < max(x) <= got.edges[-1]
    assert sum(got.counts) == 10
    # Closed on the right: a value on an inner edge counts in the bin on its left.
    on_edge = histogram_bins([0.5, 1.5, 2.5, 2.0], binwidth=1.0)
    assert on_edge.edges == (0.5, 1.5, 2.5)
    assert on_edge.counts == (2, 2)


def test_constant_x_gets_one_bin_a_tenth_wide() -> None:
    # ggstatsplot's default for zero-range data, as edge_one_constant records it
    # (ten values of 5: [4.95, 5.05]; with binwidth = 1, [4.5, 5.5]); 0.1 whatever
    # the magnitude of the value (replaces D89's refusal).
    assert histogram_binwidth([2.0, 2.0]) == 0.1
    for value in (2.0, 0.0, -5.0, 1234.5):
        one = histogram_bins([value] * 3)
        assert (one.counts, one.binwidth) == ((3,), 0.1)
        assert math.isclose(one.edges[0], value - 0.05, rel_tol=1e-13, abs_tol=1e-13)
        assert math.isclose(one.edges[1], value + 0.05, rel_tol=1e-13, abs_tol=1e-13)
    (case,) = [
        c for c in HISTO if c.name == "gghistostats-edge_one_constant-parametric"
    ]
    _assert_edges(histogram_bins(case.analyzed_values("x")).edges, [4.95, 5.05])


def test_histogram_given_width_and_limits() -> None:
    assert histogram_bins([3.0, 3.0, 3.0], binwidth=0.5).counts == (3,)
    # A width below the spacing of doubles at the values' magnitude would merge
    # edges into empty or zero-width bins: refused, with or without binwidth.
    for values, width in (([1e20, 1e20], None), ([1e20, 1e20], 1.0)):
        with pytest.raises(ValueError, match="too small to resolve"):
            histogram_bins(values, binwidth=width)
    with pytest.raises(ValueError, match="too small to resolve"):
        histogram_bins([1e17, 1e17 + 16.0])
    with pytest.raises(ValueError, match="binwidth must be"):
        histogram_bins([1.0, 2.0], binwidth=0.0)
    with pytest.raises(ValueError, match="1,000,000"):
        histogram_bins([0.0, 1.0], binwidth=1e-7)
    with pytest.raises(ValueError, match="finite"):
        histogram_bins([0.0, math.nan])


def test_marginal_definition() -> None:
    x = np.array([1.0, 2.0, 2.5, 7.0])
    got = marginal_bins(x)
    width = 6.0 / (MARGINAL_BINS - 1)
    assert len(got.counts) == MARGINAL_BINS
    assert math.isclose(got.edges[0], 1.0 - width / 2)
    assert math.isclose(got.edges[-1], 7.0 + width / 2)
    assert sum(got.counts) == 4
    with pytest.raises(ValueError, match="bins"):
        marginal_bins(x, bins=1)


def test_ols_band_definition() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=25)
    y = 2.0 * x + rng.normal(size=25)
    band = ols_band(x, y, conf_level=0.9)
    fit = stats.linregress(x, y)
    assert math.isclose(band.slope, float(fit.slope), rel_tol=1e-13)
    assert math.isclose(band.intercept, float(fit.intercept), rel_tol=1e-12)
    q = float(stats.t.ppf(0.95, 23))
    assert math.isclose(band.critical_value, q, rel_tol=1e-12)
    resid = y - (fit.intercept + fit.slope * x)
    s2 = float(resid @ resid) / 23
    sxx = float(((x - x.mean()) ** 2).sum())
    for x0, f, se, lo, hi in zip(
        band.x, band.fit, band.se, band.conf_low, band.conf_high, strict=True
    ):
        assert math.isclose(
            f, fit.intercept + fit.slope * x0, rel_tol=1e-12, abs_tol=1e-13
        )
        expected_se = math.sqrt(s2 * (1 / 25 + (x0 - x.mean()) ** 2 / sxx))
        assert math.isclose(se, expected_se, rel_tol=1e-12)
        assert math.isclose(lo, f - q * se, rel_tol=1e-12, abs_tol=1e-13)
        assert math.isclose(hi, f + q * se, rel_tol=1e-12, abs_tol=1e-13)
    assert band.df == 23.0
    assert band.conf_level == 0.9


def test_ols_band_is_exact_at_extreme_scales() -> None:
    # Timestamps: x near 1.7e9 varying in its fractions; the slope is exact.
    x = 1.7e9 + np.arange(10.0) / 8
    y = 3.0 * (np.arange(10.0) / 8) + 0.5
    band = ols_band(x, y)
    assert band.slope == 3.0
    assert band.se == (0.0,) * SMOOTH_POINTS  # an exact line: no residual
    exact = sum(Fraction(v) for v in y.tolist()) / 10
    assert band.fit[0] == float(
        exact - 3 * (sum(Fraction(v) for v in x.tolist()) / 10 - Fraction(x[0]))
    )
    rng = np.random.default_rng(9)
    u, v = rng.normal(size=12), rng.normal(size=12)
    base = ols_band(u, v)
    for scale in (1e-150, 1e150):
        scaled = ols_band(u * scale, v / scale)
        assert math.isclose(scaled.slope, base.slope / scale**2, rel_tol=1e-14)
        for got, ref in zip(scaled.fit, base.fit, strict=True):
            assert math.isclose(got * scale, ref, rel_tol=1e-13, abs_tol=1e-15)
        for got, ref in zip(scaled.se, base.se, strict=True):
            assert math.isclose(got * scale, ref, rel_tol=1e-13)
    with pytest.raises(ValueError, match="at least 3"):
        ols_band([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="same number"):
        ols_band([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])


def _band_definition(
    x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]
) -> tuple[list[float], list[float]]:
    """fit and se at the 80 grid points from the definition, in exact rationals,
    divided by max |y| (so the reference's own floats neither overflow nor
    underflow)."""
    n = x.size
    xs = [Fraction(v) for v in x.tolist()]
    ys = [Fraction(v) for v in y.tolist()]
    mx = sum(xs, Fraction(0)) / n
    my = sum(ys, Fraction(0)) / n
    sxx = sum(((v - mx) ** 2 for v in xs), Fraction(0))
    sxy = sum(((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)), Fraction(0))
    slope = sxy / sxx
    residuals = [b - my - slope * (a - mx) for a, b in zip(xs, ys, strict=True)]
    s2 = sum((r * r for r in residuals), Fraction(0)) / (n - 2)
    lo, hi = float(x.min()), float(x.max())
    step = (hi - lo) / 79
    grid = [lo + i * step for i in range(80)]
    grid[-1] = hi
    scale = Fraction(float(np.max(np.abs(y))))
    fits = [float((my + slope * (Fraction(g) - mx)) / scale) for g in grid]
    ses = [
        math.sqrt(
            float(s2 * (Fraction(1, n) + (Fraction(g) - mx) ** 2 / sxx) / scale**2)
        )
        for g in grid
    ]
    return fits, ses


@pytest.mark.parametrize(
    "exponent", [-300, -200, -165, -162, -160, 0, 154, 155, 200, 300]
)
@pytest.mark.parametrize("axis", ["x", "y"])
def test_ols_band_at_any_scale(exponent: int, axis: str) -> None:
    # Review Z4: the band's variance overflowed for y of order 1e155 (an
    # OverflowError) and underflowed below 1e-162 (a band of no width).
    rng = np.random.default_rng(3)
    x = rng.normal(size=15)
    y = 0.5 * x + rng.normal(size=15)
    if axis == "x":
        x = x * 10.0**exponent
    else:
        y = y * 10.0**exponent
    band = ols_band(x, y)
    fits, ses = _band_definition(x, y)
    scale = float(np.max(np.abs(y)))
    for got, want in zip(band.se, ses, strict=True):
        assert got / scale == pytest.approx(want, rel=1e-13)
    for got, want in zip(band.fit, fits, strict=True):
        assert got / scale == pytest.approx(want, rel=1e-13, abs=1e-15)
    width = band.conf_high[40] - band.conf_low[40]
    assert width == pytest.approx(2 * band.critical_value * band.se[40], rel=1e-12)
    assert width > 0.0


def test_ols_band_raises_where_a_double_cannot_hold_it() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=15) * 1e-300
    y = rng.normal(size=15) * 1e300  # a slope of order 1e600
    with pytest.raises(ValueError, match="the slope of the least-squares line"):
        ols_band(x, y)


@pytest.mark.parametrize(
    ("values", "binwidth", "message"),
    [
        ([1.0, 2.0, 3.0], 5e-324, "more bins"),
        ([1.7e308, -1.7e308, 0.0], 0.1, "range of x exceeds"),
    ],
)
def test_histogram_overflow_guard(
    values: list[float], binwidth: float, message: str
) -> None:
    # Review Z6 (7): OverflowError "cannot convert float infinity to integer".
    with pytest.raises(ValueError, match=message):
        histogram_bins(np.array(values), binwidth=binwidth)


def test_marginal_bins_of_a_subnormal_range() -> None:
    # Review Z6 (7): a ZeroDivisionError from a bin width that underflows to 0.
    with pytest.raises(ValueError, match="too small for doubles to separate 30"):
        marginal_bins(np.array([0.0, 5e-324, 1e-323]))


def test_a_value_within_rounding_of_a_break_counts_on_its_left() -> None:
    # Planted bug G05 (no fuzz at the breaks) survived the suite: stat_bin moves
    # every inner break right by 1e-8 of the bin width, so a value one unit in the
    # last place above the break 1.5 still counts in (0.5, 1.5].
    above = float(np.nextafter(1.5, 2.0))
    bins = histogram_bins(np.array([0.0, above, 3.0]), binwidth=1.0)
    assert bins.edges == (-0.5, 0.5, 1.5, 2.5, 3.5)
    assert bins.counts == (1, 1, 0, 1)
    # Beyond the fuzz it counts on the right.
    beyond = histogram_bins(np.array([0.0, 1.5 + 1e-6, 3.0]), binwidth=1.0)
    assert beyond.counts == (1, 0, 1, 1)
