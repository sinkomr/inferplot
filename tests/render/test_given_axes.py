"""Drawing into a given Axes: refused kinds of Axes, shared axes, other panels."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from inferplot.render import StatsFigure, betweenstats
from inferplot.result import StatsResult
from tests.render._results import frame_result


def _three() -> StatsResult:
    rng = np.random.default_rng(1)
    groups = [rng.normal(m, 1, 20) for m in (5, 6, 7.5)]
    return frame_result(["a", "b", "c"], groups, pairwise_display="all")


@pytest.mark.parametrize("projection", ["polar", "3d", "aitoff"])
def test_an_axes_that_is_not_rectangular_2d_is_refused(projection: str) -> None:
    figure = Figure()
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(projection=projection)
    before = list(ax.get_children())
    with pytest.raises(ValueError, match="rectangular") as info:
        betweenstats(_three(), ax=ax)
    assert projection in str(info.value)
    assert list(ax.get_children()) == before


def _two() -> tuple[StatsResult, StatsResult]:
    """Two panels' results: a wide range and a narrow one, three levels each."""
    rng = np.random.default_rng(4)
    wide = frame_result(
        ["a", "b", "c"],
        [rng.normal(m, 3, 40) for m in (5, 9, 14)],
        y="outcome 1",
        pairwise_display="all",
    )
    narrow = frame_result(
        ["a", "b", "c"],
        [rng.normal(m, 0.15, 40) for m in (5, 5.2, 5.5)],
        y="outcome 2",
        pairwise_display="all",
    )
    return wide, narrow


def _panels(*, sharex: bool = False, sharey: bool = False) -> tuple[Figure, list[Axes]]:
    figure = Figure(figsize=(20, 6.5), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharex=sharex, sharey=sharey, squeeze=False)[0]
    return figure, list(axes)


@pytest.mark.parametrize("how", ["sharex", "twinx"])
def test_a_shared_x_axis_is_refused_before_drawing(how: str) -> None:
    if how == "sharex":
        _, axes = _panels(sharex=True)
        ax = axes[0]
    else:
        figure = Figure()
        FigureCanvasAgg(figure)
        ax = figure.add_subplot()
        ax.twinx()
    before = list(ax.get_children())
    with pytest.raises(ValueError, match="shares its x axis") as info:
        betweenstats(_three(), ax=ax)
    assert "sharex=False" in str(info.value)
    assert "n" in str(info.value)
    assert list(ax.get_children()) == before


def _outside(fig: StatsFigure) -> int:
    ax = fig.axes["main"]
    lo, hi = sorted(ax.get_ylim())
    values = np.concatenate([np.asarray(v) for v in fig.result.values])
    return int(((values < lo) | (values > hi)).sum())


def _brackets_inside(fig: StatsFigure) -> bool:
    lo, hi = sorted(fig.axes["main"].get_ylim())
    ys = [np.asarray(line.get_ydata(), dtype=np.float64) for _, line, _ in fig.brackets]  # pyright: ignore[reportUnknownArgumentType]
    return all(lo <= float(y.min()) and float(y.max()) <= hi for y in ys)


@pytest.mark.parametrize("wide_first", [True, False])
def test_a_shared_y_axis_shows_every_panel_whole(wide_first: bool) -> None:
    wide, narrow = _two()
    figure, axes = _panels(sharey=True)
    first, second = (wide, narrow) if wide_first else (narrow, wide)
    one = betweenstats(first, ax=axes[0])
    two = betweenstats(second, ax=axes[1])
    figure.canvas.draw()
    for fig in (one, two):
        assert _outside(fig) == 0
        assert fig.brackets
        assert _brackets_inside(fig)
        codes = [i.code for i in fig.check()]
        assert "data_outside_view" not in codes
        assert "label_overlap" not in codes, [i.message for i in fig.check()]
    assert axes[0].get_ylim() == axes[1].get_ylim()


def test_a_shared_y_axis_keeps_the_users_own_panel_in_view() -> None:
    _, narrow = _two()
    _figure, axes = _panels(sharey=True)
    axes[0].plot([0, 1], [-50.0, 80.0])
    fig = betweenstats(narrow, ax=axes[1])
    lo, hi = axes[0].get_ylim()
    assert lo <= -50.0
    assert hi >= 80.0
    assert _outside(fig) == 0


@pytest.mark.parametrize("own", [True, False])
def test_limits_changed_after_drawing_are_reported(own: bool) -> None:
    wide, _ = _two()
    if own:
        fig = betweenstats(wide)
    else:
        _, axes = _panels()
        fig = betweenstats(wide, ax=axes[0])
    lo, hi = fig.axes["main"].get_ylim()
    assert not [i for i in fig.check() if i.code == "data_outside_view"]
    fig.axes["main"].set_ylim(4.0, 9.0)
    (issue,) = [i for i in fig.check() if i.code == "data_outside_view"]
    assert f"{_outside(fig)} of 120 values" in issue.message
    assert "bracket" in issue.message
    assert f"ax.set_ylim({lo:.6g}, {hi:.6g})" in issue.fix


def test_a_later_panel_that_cuts_an_earlier_one_is_reported() -> None:
    # The user plots into the other panel after inferplot drew; the shared limits
    # follow their data and cut inferplot's panel.
    wide, _ = _two()
    _, axes = _panels(sharey=True)
    fig = betweenstats(wide, ax=axes[0])
    axes[1].set_ylim(5.0, 6.0)
    (issue,) = [i for i in fig.check() if i.code == "data_outside_view"]
    assert "values" in issue.message


def _stacked(hspace: float) -> tuple[Figure, list[Axes]]:
    figure = Figure(figsize=(7, 9), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, squeeze=False, gridspec_kw={"hspace": hspace})[:, 0]
    return figure, list(axes)


def test_a_text_over_another_panels_title_is_reported() -> None:
    _, axes = _stacked(0.05)
    axes[1].set_title("Second experiment")
    fig = betweenstats(_three(), ax=axes[0], xlab="the first x label")
    issues = [i for i in fig.check() if i.code == "label_overlap"]
    assert issues, [i.message for i in fig.check()]
    message = issues[0].message
    assert "another panel" in message
    assert "'Second experiment'" in message


def test_a_text_over_a_suptitle_is_reported() -> None:
    figure, axes = _stacked(0.6)
    figure.subplots_adjust(top=0.99)
    figure.suptitle("All experiments", y=0.995)
    fig = betweenstats(_three(), ax=axes[0], title="First experiment")
    issues = [i for i in fig.check() if i.code == "label_overlap"]
    assert issues
    assert "'All experiments'" in issues[0].message


def test_panels_with_room_report_no_overlap() -> None:
    _, axes = _stacked(0.6)
    axes[1].set_title("Second experiment")
    fig = betweenstats(_three(), ax=axes[0])
    assert not [i for i in fig.check() if i.code == "label_overlap"]
