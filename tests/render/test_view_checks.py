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


def _scaled(scale: float) -> StatsResult:
    rng = np.random.default_rng(1)
    groups = [rng.normal(m, 1, 20) * scale for m in (5, 6, 7.5)]
    return frame_result(["a", "b", "c"], groups, pairwise_display="all")


@pytest.mark.parametrize("scale", [1e200, 1e-200, 1e300, 1e-250])
def test_any_scale_the_statistics_accept_draws(scale: float) -> None:
    import warnings

    result = _scaled(scale)
    with warnings.catch_warnings():
        # RuntimeWarning only (overflow, division): a blanket "error" would also
        # undo the configured ignore of matplotlib 3.8's pyparsing deprecation.
        warnings.simplefilter("error", RuntimeWarning)
        fig = betweenstats(result)
    reference = betweenstats(_scaled(1.0))
    # The same picture: the limits scale with the values.
    lo, hi = fig.axes["main"].get_ylim()
    ref_lo, ref_hi = reference.axes["main"].get_ylim()
    assert lo / scale == pytest.approx(ref_lo, rel=1e-9)
    assert hi / scale == pytest.approx(ref_hi, rel=1e-9)
    # (The figure may be taller: matplotlib writes the axis' power of ten above it.)
    codes = [i.code for i in fig.check()]
    assert codes == [i.code for i in reference.check()]


@pytest.mark.parametrize(("scale", "words"), [(1e-300, "multiply y by 1e300")])
def test_values_matplotlib_cannot_draw_raise_a_plain_error(
    scale: float, words: str
) -> None:
    with pytest.raises(ValueError, match="too small for matplotlib to draw") as info:
        betweenstats(_scaled(scale))
    assert words in str(info.value)


def test_a_flat_line_is_reported() -> None:
    fig = betweenstats(_three())
    assert _issue(fig, "flat_data") is None
    fig.axes["main"].set_ylim(-1e6, 1e6)
    message = _issue(fig, "flat_data")
    assert message is not None
    assert "flat line" in message
    assert "ax.set_ylim(" in message
