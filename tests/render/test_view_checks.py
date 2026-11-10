"""fig.check() on changes made to a figure after inferplot drew it: inverted or
non-linear axes, limits that leave data out of the view."""

from __future__ import annotations

import numpy as np
import pytest

from inferplot.render import StatsFigure, betweenstats
from inferplot.result import StatsResult
from tests.render._results import frame_result


def _three() -> StatsResult:
    rng = np.random.default_rng(1)
    groups = [rng.normal(m, 1, 20) for m in (5, 6, 7.5)]
    return frame_result(["a", "b", "c"], groups, pairwise_display="all")


def _issue(fig: StatsFigure, code: str) -> str | None:
    found = [i for i in fig.check() if i.code == code]
    assert len(found) <= 1
    return None if not found else found[0].message + " " + found[0].fix


def test_upright_figure_reports_no_axis_change() -> None:
    assert _issue(betweenstats(_three()), "axis_changed") is None


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ("invert_yaxis", "y axis is inverted"),
        ("invert_xaxis", "x axis is inverted"),
        ("logy", "y axis has a 'log' scale"),
    ],
)
def test_inverted_or_rescaled_axis_is_reported(change: str, words: str) -> None:
    fig = betweenstats(_three())
    ax = fig.axes["main"]
    if change == "logy":
        ax.set_yscale("log")
    else:
        getattr(ax, change)()
    message = _issue(fig, "axis_changed")
    assert message is not None
    assert words in message
    assert "ax." in message  # says how to undo it


def test_no_check_message_quotes_mathtext_source() -> None:
    # Two renders into one Axes: the second figure's check names the first's texts.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6, 5))
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    ax.text(0.5, 0.5, r"user's $\alpha$ note", transform=ax.transAxes)
    betweenstats(_three(), ax=ax)
    second = betweenstats(_three(), ax=ax)
    issues = [i for i in second.check() if i.code in ("label_overlap", "text_clipped")]
    assert issues
    for issue in issues:
        assert "$" not in issue.message, issue.message
        assert "\\" not in issue.message, issue.message
    words = " ".join(i.message for i in issues)
    assert "centrality label of 'a'" in words  # the first render's, in plain words
