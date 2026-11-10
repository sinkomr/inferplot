"""MAP centrality: Sheather-Jones bandwidth, binned kernel density, grid maximum.

ggstatsplot's centrality label for ``type = "bayes"`` is
``bayestestR::map_estimate()`` of each group's values. The fixtures record the
values (the label layer's y column); every one sits exactly on the 1,024-point grid
from the group's minimum to its maximum.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from inferplot.stats import MapEstimate, bw_sj, map_estimate
from inferplot.stats import density as density_module
from inferplot.stats.descriptive import FloatArray
from inferplot.text import format_value
from tests.oracle_fixtures import (
    OracleCase,
    cases_for,
    number,
    probe,
    r_double,
    require_cases,
)
from tests.stats.fixture_data import groups
from tests.tolerances import MAP_CENTRALITY, assert_close, assert_within_grid

GRID_POINTS = 1024


def _bayes_cases() -> list[OracleCase]:
    cases = [c for c in cases_for("ggbetweenstats") if c.type == "bayes"]
    return list(require_cases(cases, "ggbetweenstats type bayes"))


def _map_rows(case: OracleCase) -> dict[str, tuple[float, str]]:
    """Level -> (MAP value, label text) from the centrality label layer."""
    layers = case.layer_data("GeomLabelRepel")
    assert len(layers) == 1, f"{case.name}: one centrality label layer expected"
    x, y = case.args["x"], case.args["y"]
    out: dict[str, tuple[float, str]] = {}
    for row in layers[0]:
        expression = str(row["expression"])
        assert "widehat(mu)[MAP]" in expression, expression
        out[str(row[x])] = (number(row[y]), expression)
    return out


BAYES_CASES = _bayes_cases()


@pytest.mark.parametrize("case", BAYES_CASES, ids=lambda c: c.name)
def test_map_matches_oracle(case: OracleCase) -> None:
    rows = _map_rows(case)
    assert set(rows) == set(case.analysis_levels)
    for level, values in zip(case.analysis_levels, groups(case), strict=True):
        expected, expression = rows[level]
        got = map_estimate(values)
        step = (float(values.max()) - float(values.min())) / (GRID_POINTS - 1)
        assert got.grid_step == step
        assert_within_grid(
            got.estimate,
            expected,
            origin=float(values.min()),
            step=step,
            what=f"{case.name} {level} MAP",
        )
        # The label prints the MAP to `digits` decimals (ggstatsplot's default 2).
        digits = int(case.args.get("digits", 2))
        assert f'"{format_value(got.estimate, digits)}"' in expression, (
            f"{case.name} {level}: label {expression} vs {got.estimate!r}"
        )


def test_map_tolerance_is_the_same_grid_point() -> None:
    assert MAP_CENTRALITY.steps == 0


def test_fixture_maps_lie_on_the_grid() -> None:
    # The premise of the grid tolerance: bayestestR evaluates the density on
    # 1,024 equally spaced points from min(x) to max(x) (estimate_density's
    # default extend = FALSE) and returns one of them.
    for case in BAYES_CASES:
        rows = _map_rows(case)
        for level, values in zip(case.analysis_levels, groups(case), strict=True):
            lo, hi = float(values.min()), float(values.max())
            index = (rows[level][0] - lo) / ((hi - lo) / (GRID_POINTS - 1))
            assert abs(index - round(index)) < 1e-9, (case.name, level, index)


def test_result_is_plain_python() -> None:
    got = map_estimate(np.array([1.0, 2.0, 2.5, 3.0, 7.0]))
    assert isinstance(got, MapEstimate)
    assert type(got.estimate) is float
    assert type(got.bandwidth) is float
    assert type(got.grid_step) is float
    assert type(got.index) is int
    assert got.estimate == 1.0 + got.index * got.grid_step or got.index == 1023


def test_grid_ends_are_the_data_range() -> None:
    x = np.array([0.5, 1.0, 1.1, 1.2, 1.3, 1.25, 9.0])
    got = map_estimate(x)
    assert 0.5 <= got.estimate <= 9.0
    assert got.grid_step == (9.0 - 0.5) / 1023


def test_bandwidth_is_positive_and_scale_equivariant() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    h = bw_sj(x)
    assert h > 0
    # The bin anchor (zero, or min(x) by max|x| / width, which a power of two
    # leaves unchanged) makes scaling by a power of two exact.
    assert bw_sj(x * 8.0) == h * 8.0
    assert map_estimate(x * 8.0).estimate == map_estimate(x).estimate * 8.0


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([3.0], id="n=1"),
        pytest.param([], id="empty"),
        pytest.param([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0], id="zero IQR"),
    ],
)
def test_no_density_raises_value_error(values: list[float]) -> None:
    with pytest.raises(ValueError, match=r"MAP|bandwidth|density|value"):
        map_estimate(np.array(values, dtype=np.float64))


def test_constant_data_give_the_value() -> None:
    # bayestestR::map_estimate returns the value itself where bw.SJ fails on
    # constant data (probe-map "constant"; D55).
    got = map_estimate(np.array([2.5, 2.5, 2.5, 2.5]))
    assert got == MapEstimate(estimate=2.5, index=0, grid_step=0.0, bandwidth=0.0)
    with pytest.raises(ValueError, match="interquartile range is 0"):
        bw_sj(np.array([2.5, 2.5, 2.5, 2.5]))


# probe-map vectors where inferplot's MAP or bandwidth is not R's, with why.
# normal_n1e5: R's densities at its grid points 512 and 513 (1-based) differ by
# 6.2e-15 relative, below the FFT's rounding, and R's which.max takes the later;
# inferplot's tie rule (map_estimate, 1e-12) takes the first: one grid step.
MAP_DIFFERS = {"normal_n1e5": 1}
# shifted_large, timestamps and anchor_26027507 (values near 1e8, 1.7e9 and 2.6e7;
# 2.4e10, 7.5e11 and 1.00000066 x 2^31 bin widths from zero): R's bw.SJ anchors the
# pair-distance bins at min(x) there and at zero everywhere else, as inferplot does
# (density._pair_counts); with the zero anchor their bandwidths would differ by
# 9.6e-4, 4.9e-4 and 1.04e-3 relative.
MIN_ANCHORED = {
    "shifted_large": 9.6e-4,
    "timestamps": 4.9e-4,
    "anchor_26027507": 1.04e-3,
}
# The probe-map vector closest to 2^31 bin widths from below (0.999998 x 2^31):
# R still anchors at zero there; anchored at min(x) its bandwidth would differ by
# 1.31e-3 relative.
ZERO_ANCHORED_NEAREST = ("anchor_26027440", 1.31e-3)


def _probe_map_rows() -> list[Any]:
    rows = probe("probe-map").rows("rows")
    return [pytest.param(row, id=str(row["name"])) for row in rows]


@pytest.mark.parametrize("row", _probe_map_rows())
def test_map_and_bandwidth_against_probe_map(row: Any) -> None:
    # Every probe-map vector: the MAP on R's grid point (MAP_CENTRALITY) and the
    # bandwidth bw.SJ gives, or both failing where R's bw.SJ fails.
    name = str(row["name"])
    x = np.array([r_double(str(h)) for h in row["x_hex"]], dtype=np.float64)
    r_map = r_double(str(row["map_estimate_hex"]))
    if row["bw_sj"]["error"] is not None:
        with pytest.raises(ValueError, match="no Sheather-Jones bandwidth"):
            bw_sj(x)
        if r_map is None:  # zero IQR: R's MAP is NA, ggstatsplot's label empty
            assert row["centrality_label"] == ['list(widehat(mu)[MAP] == "")']
            with pytest.raises(ValueError, match="no Sheather-Jones bandwidth"):
                map_estimate(x)
        else:  # constant data: the value itself
            assert map_estimate(x).estimate == r_map
        return
    assert r_map is not None
    got = map_estimate(x)
    r_bw = r_double(str(row["bw_sj"]["hex"]))
    assert r_bw is not None
    assert_close("closed_form", got.bandwidth, r_bw, what=f"{name} bw")
    if name in MIN_ANCHORED:
        assert got.bandwidth == pytest.approx(r_bw, rel=1e-14)
    lo, step = float(x.min()), got.grid_step
    if name in MAP_DIFFERS:
        r_index = (r_map - lo) / step
        assert round(r_index) - got.index == MAP_DIFFERS[name]
        return
    if got.estimate == r_map:
        # The same double. (Far from zero the grid points themselves are rounded:
        # timestamps' sit 1.3e-5 steps off lo + i x step, beyond on_grid.)
        return
    assert_within_grid(got.estimate, r_map, origin=lo, step=step, what=f"{name} MAP")


def test_probe_map_differences_are_all_listed() -> None:
    names = {str(row["name"]) for row in probe("probe-map").rows("rows")}
    assert set(MAP_DIFFERS) | set(MIN_ANCHORED) <= names
    assert len(names) == 56


def _zero_anchored_bandwidth(x: FloatArray) -> float:
    """bw_sj with the bins anchored at zero whatever max|x| / width is."""
    original = density_module._ANCHOR_AT_MIN  # pyright: ignore[reportPrivateUsage]
    try:
        density_module._ANCHOR_AT_MIN = math.inf  # pyright: ignore[reportPrivateUsage]
        return bw_sj(x)
    finally:
        density_module._ANCHOR_AT_MIN = original  # pyright: ignore[reportPrivateUsage]


def _binned_pairs(bins: FloatArray, size: int) -> list[float]:
    """Pair counts by bin distance, by brute force."""
    index = bins.astype(np.int64)
    diffs = np.abs(index[:, None] - index[None, :])[np.triu_indices(index.size, 1)]
    return np.bincount(diffs, minlength=size).astype(np.float64).tolist()


def test_bins_anchor_at_min_exactly_where_r_does() -> None:
    # probe-map: R anchors its bins at min(x) on the three vectors at or beyond 2^31
    # bin widths from zero (max|x| / width 1.00000066 x 2^31, 2.4e10 and 7.5e11) and
    # at zero on the 52 others with spread, the farthest 0.999998 x 2^31
    # (anchor_26027440). The zero anchor misses R's bandwidth on the first three by
    # the amounts in MIN_ANCHORED, the min(x) anchor the farthest zero-anchored one
    # by ZERO_ANCHORED_NEAREST's: the threshold is 2^31 to within 2e-6.
    ratios: dict[str, float] = {}
    for row in probe("probe-map").rows("rows"):
        x = np.array([r_double(str(h)) for h in row["x_hex"]], dtype=np.float64)
        lo, hi = float(x.min()), float(x.max())
        if hi == lo:
            continue
        name = str(row["name"])
        ratios[name] = max(abs(lo), abs(hi)) / (1.01 * (hi - lo) / 1000)
        if name in MIN_ANCHORED:
            r_bw = r_double(str(row["bw_sj"]["hex"]))
            assert r_bw is not None
            assert abs(_zero_anchored_bandwidth(x) / r_bw - 1) == pytest.approx(
                MIN_ANCHORED[name], rel=0.01
            )
    far = {name for name, ratio in ratios.items() if ratio >= 2.0**31}
    assert far == set(MIN_ANCHORED)
    nearest, miss = ZERO_ANCHORED_NEAREST
    near_ratio, near_name = max(
        (r, name) for name, r in ratios.items() if name not in far
    )
    assert near_name == nearest
    assert 0.999998 * 2.0**31 < near_ratio < 2.0**31
    assert min(ratios[name] for name in far) < 1.000001 * 2.0**31
    row = next(r for r in probe("probe-map").rows("rows") if r["name"] == nearest)
    x = np.array([r_double(str(h)) for h in row["x_hex"]], dtype=np.float64)
    r_bw = r_double(str(row["bw_sj"]["hex"]))
    assert r_bw is not None
    assert bw_sj(x) == pytest.approx(r_bw, rel=1e-14)
    original = density_module._ANCHOR_AT_MIN  # pyright: ignore[reportPrivateUsage]
    try:
        density_module._ANCHOR_AT_MIN = 0.0  # pyright: ignore[reportPrivateUsage]
        at_min = bw_sj(x)
    finally:
        density_module._ANCHOR_AT_MIN = original  # pyright: ignore[reportPrivateUsage]
    assert abs(at_min / r_bw - 1) == pytest.approx(miss, rel=0.01)


def test_values_far_from_zero_have_a_map() -> None:
    # 1.7e9 plus small dyadic offsets: 1.4e15 bin widths from zero, beyond 2^53
    # with the zero anchor (which raised); the min anchor bins them exactly.
    offsets = np.array([0, 1, 2, 2, 3, 3, 3, 4, 5, 7, 9, 12], dtype=np.float64)
    x = 1.7e9 + offsets * 20 * 2.0**-22
    got = map_estimate(x)
    assert got.index == 224
    assert got.estimate == 1700000000.0000126
    counts, width = density_module._pair_counts(x)  # pyright: ignore[reportPrivateUsage]
    assert counts.tolist() == _binned_pairs(
        np.floor((x - x.min()) / width), counts.size
    )


@pytest.mark.parametrize(
    ("ratio", "at_min"), [(2.0**31 * 0.999, False), (2.0**31 * 1.001, True)]
)
def test_anchor_threshold(ratio: float, at_min: bool) -> None:
    # A spread of 1 placed so that max|x| / width is just below or just above 2^31.
    spread = np.array([0.0, 0.1, 0.25, 0.3, 0.55, 0.6, 0.62, 0.9, 1.0])
    x = ratio * (1.01 / 1000) - 1.0 + spread
    lo, hi = float(x.min()), float(x.max())
    assert (max(abs(lo), abs(hi)) / (1.01 * (hi - lo) / 1000) >= 2.0**31) is at_min
    counts, width = density_module._pair_counts(x)  # pyright: ignore[reportPrivateUsage]
    bins = np.floor((x - lo) / width) if at_min else np.trunc(x / width)
    assert counts.tolist() == _binned_pairs(bins, counts.size)


def test_rejects_non_finite_and_two_dimensional() -> None:
    with pytest.raises(ValueError, match="finite"):
        map_estimate(np.array([1.0, 2.0, math.nan, 4.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        map_estimate(np.ones((3, 3)))


def test_two_values_tie_at_the_ends() -> None:
    # Two values: a symmetric density with equal peaks at both data points (the
    # bandwidth is small against their distance). Exactly tied grid points give the
    # first, as R's which.max would without rounding, whatever the FFT's last bits.
    got = map_estimate(np.array([0.0, 1.0]))
    assert got.bandwidth < 0.5
    assert got.index == 0
    assert got.estimate == 0.0
    assert map_estimate(np.array([-1.0, 0.0])).estimate == -1.0


def test_symmetric_sample_takes_the_first_tied_point() -> None:
    # 2 lies halfway between grid points 511 and 512, which tie.
    got = map_estimate(np.array([1.0, 2.0, 2.0, 3.0]))
    assert got.index == 511


def test_huge_n_is_linear_time() -> None:
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(0.0, 1.0, 1_500_000), rng.normal(6.0, 0.5, 500_000)])
    got = map_estimate(x)
    # The taller mode is the N(0, 1) one: density 0.75 / sqrt(2 pi) = 0.299 at 0
    # against 0.25 / (0.5 sqrt(2 pi)) = 0.199 at 6.
    assert abs(got.estimate) < 0.05
    assert got.bandwidth > 0


def test_pair_counts_match_brute_force() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(10.0, 3.0, 300)
    counts, width = density_module._pair_counts(x)  # pyright: ignore[reportPrivateUsage]
    bins = np.trunc(x / width).astype(np.int64)
    diffs = np.abs(bins[:, None] - bins[None, :])[np.triu_indices(x.size, 1)]
    expected = np.bincount(diffs, minlength=counts.size)
    assert counts.tolist() == expected.astype(np.float64).tolist()
