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


def test_histogram_given_width_and_limits() -> None:
    assert histogram_bins([3.0, 3.0, 3.0], binwidth=0.5).counts == (3,)
    with pytest.raises(ValueError, match="binwidth="):
        histogram_bins([2.0, 2.0])
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
