"""Every value has a mark: violins of heavy-tailed data, outlier markers without
points, and a subsample that keeps every level and its extremes."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib.patches import Polygon
from scipy.special import ndtr

from inferplot.render import betweenstats, comparison
from inferplot.render.comparison import KDE_GRID, VIOLIN_HALF_WIDTH, _bandwidth
from tests.render._helpers import by_gid
from tests.render._results import frame_result


def _exact_cell_density(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """The Gaussian KDE of every value, averaged over each grid cell (unbinned)."""
    cell = float(grid[1] - grid[0])
    out = np.zeros(grid.size)
    for chunk in np.array_split(x, max(1, x.size // 2000)):
        hi = (grid[:, None] + cell / 2 - chunk[None, :]) / bw
        lo = (grid[:, None] - cell / 2 - chunk[None, :]) / bw
        out += (ndtr(hi) - ndtr(lo)).sum(axis=1)
    return out / cell


def _heavy() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [
        rng.lognormal(0.0, 3.0, size=20_000),
        np.append(rng.normal(0.0, 1.0, size=3_000), 1e6),
        rng.normal(0.0, 1.0, size=500),
    ]


@pytest.mark.parametrize("layers", [("violin", "box"), ("box", "points")])
def test_a_far_value_keeps_the_violin(layers: tuple[str, ...]) -> None:
    groups = _heavy()
    result = frame_result(["lognormal", "one far", "plain"], groups)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no NaN or division warnings
        fig = betweenstats(result, layers=layers)
    added = fig.result.warnings[len(result.warnings) :]
    assert not [w for w in added if "not drawn" in w]
    prefix = "violin" if "violin" in layers else "density"
    for i, x in enumerate(groups):
        patch = by_gid(fig, f"{prefix}-{i}")
        assert isinstance(patch, Polygon)
        xy = np.asarray(patch.get_xy())
        assert np.isfinite(xy).all()
        half = np.abs(xy[:, 0] - i)
        assert half.max() == pytest.approx(VIOLIN_HALF_WIDTH)
        if i < 2:
            # The drawn outline against the exact density averaged over the same
            # cells, from every value without binning: within 1% of the peak.
            grid = np.linspace(x.min(), x.max(), KDE_GRID)
            exact = _exact_cell_density(x, grid, _bandwidth(x))
            exact = VIOLIN_HALF_WIDTH * exact / exact.max()
            drawn = half[:KDE_GRID]
            assert np.abs(drawn - exact).max() < 0.01 * VIOLIN_HALF_WIDTH
    assert "data_outside_view" not in [i.code for i in fig.check()]


def test_a_density_over_a_range_wider_than_a_double_is_refused() -> None:
    reason = comparison._density_outline(np.array([-1.5e308, 1.5e308, 0.0]))  # pyright: ignore[reportPrivateUsage]
    assert isinstance(reason, str)
    assert "range" in reason


def test_a_violin_that_cannot_be_drawn_is_said_in_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(5)
    groups = [rng.normal(0, 1, 3000), rng.normal(1, 1, 3000)]
    result = frame_result(["first", "second"], groups)
    real = comparison._density_outline  # pyright: ignore[reportPrivateUsage]

    def fails_for_first(x: np.ndarray) -> object:
        return (
            "its density is not finite"
            if x.size == 3000 and x[0] == groups[0][0]
            else real(x)
        )

    monkeypatch.setattr(comparison, "_density_outline", fails_for_first)
    fig = betweenstats(result)
    assert by_gid(fig, "violin-0") is None
    assert by_gid(fig, "violin-1") is not None
    added = fig.result.warnings[len(result.warnings) :]
    assert any(
        "'first'" in w and "not drawn" in w and "not finite" in w for w in added
    ), added
    # The subsample's note does not claim the missing violin shows all values.
    (note,) = [w for w in added if "subsample" in w]
    assert "'first'" in note
    (info,) = [i for i in fig.check() if i.code == "points_subsampled"]
    assert "'first'" in info.message


def _drawn(fig: object, i: int) -> np.ndarray:
    points = by_gid(fig, f"points-{i}")  # pyright: ignore[reportArgumentType]
    assert points is not None
    return np.asarray(points.get_offsets())[:, 1]  # pyright: ignore[reportAttributeAccessIssue]


def test_the_subsample_keeps_every_level_and_its_extremes() -> None:
    rng = np.random.default_rng(8)
    groups = [rng.normal(0, 1, 60_000), rng.normal(3, 1, 8), rng.normal(1, 1, 700)]
    result = frame_result(["big", "eight", "mid"], groups)
    total = sum(g.size for g in groups)
    fig = betweenstats(result)
    want = [max(int(2000 * g.size / total + 0.5), min(g.size, 50)) for g in groups]
    assert want == [1977, 8, 50]
    for i, g in enumerate(groups):
        ys = _drawn(fig, i)
        assert ys.size == want[i]
        assert g.min() in ys
        assert g.max() in ys
        assert set(ys.tolist()) <= set(g.tolist())
    sub = fig.subsample
    assert sub is not None
    assert (sub.shown, sub.total) == (sum(want), total)
    added = fig.result.warnings[len(result.warnings) :]
    (note,) = [w for w in added if "subsample" in w]
    assert f"subsample of {sum(want):,}" in note
    assert "smallest and largest" in note
    (info,) = [i for i in fig.check() if i.code == "points_subsampled"]
    assert "smallest and largest" in info.message
    again = betweenstats(result)
    assert np.array_equal(_drawn(again, 0), _drawn(fig, 0))  # seeded


def test_within_subsample_keeps_every_conditions_extremes() -> None:
    from inferplot.render import withinstats
    from tests.render._within import many_subjects

    result = many_subjects(3_000, 3)
    fig = withinstats(result)
    for j, values in enumerate(result.values):
        ys = _drawn(fig, j)
        assert ys.size == 2000 // 3
        assert min(values) in ys
        assert max(values) in ys
    added = fig.result.warnings[len(result.warnings) :]
    (note,) = [w for w in added if "subsample" in w]
    assert "smallest and largest" in note


def _with_outliers() -> tuple[list[np.ndarray], object]:
    rng = np.random.default_rng(1)
    groups = [np.append(rng.normal(m, 1, 30), m + 6) for m in (5, 6, 7.5)]
    return groups, frame_result(["a", "b", "c"], groups)


def _beyond_whiskers(x: np.ndarray) -> list[float]:
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    low = x[x >= q1 - 1.5 * iqr].min()
    high = x[x <= q3 + 1.5 * iqr].max()
    return sorted(x[(x < low) | (x > high)].tolist())


@pytest.mark.parametrize("layers", [("box",), ("violin", "box")])
def test_without_points_a_box_marks_its_outliers(layers: tuple[str, ...]) -> None:
    groups, result = _with_outliers()
    fig = betweenstats(result, layers=layers)  # pyright: ignore[reportArgumentType]
    for i, x in enumerate(groups):
        expected = _beyond_whiskers(x)
        assert expected  # each group has its far value
        marks = by_gid(fig, f"outliers-{i}")
        assert marks is not None
        xy = np.asarray(marks.get_offsets())  # pyright: ignore[reportAttributeAccessIssue]
        assert sorted(xy[:, 1].tolist()) == expected
        assert np.all(xy[:, 0] == i)


def test_with_points_the_points_show_the_outliers() -> None:
    _, result = _with_outliers()
    fig = betweenstats(result)  # pyright: ignore[reportArgumentType]
    assert by_gid(fig, "outliers-0") is None
