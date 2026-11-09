"""MAP centrality: Sheather-Jones bandwidth, binned kernel density, grid maximum.

ggstatsplot's centrality label for ``type = "bayes"`` is
``bayestestR::map_estimate()`` of each group's values. The fixtures record the
values (the label layer's y column); every one sits exactly on the 1,024-point grid
from the group's minimum to its maximum.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from inferplot.stats import MapEstimate, bw_sj, map_estimate
from inferplot.stats import density as density_module
from inferplot.text import format_value
from tests.oracle_fixtures import OracleCase, cases_for, number, require_cases
from tests.stats.fixture_data import groups
from tests.tolerances import MAP_CENTRALITY, assert_within_grid

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
        # The label prints the MAP to two decimals (ggstatsplot's default digits).
        assert f'"{format_value(got.estimate, 2)}"' in expression, (
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
    # Bins are anchored at zero, so scaling by a power of two is exact.
    assert bw_sj(x * 8.0) == h * 8.0
    assert map_estimate(x * 8.0).estimate == map_estimate(x).estimate * 8.0


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([3.0], id="n=1"),
        pytest.param([], id="empty"),
        pytest.param([2.0, 2.0, 2.0, 2.0], id="all tied"),
        pytest.param([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0], id="zero IQR"),
    ],
)
def test_no_density_raises_value_error(values: list[float]) -> None:
    with pytest.raises(ValueError, match=r"MAP|bandwidth|density|value"):
        map_estimate(np.array(values, dtype=np.float64))


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
