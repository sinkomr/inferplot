"""Contrast: inferplot's ink on a dark given Axes, colormap palettes away from
their ends, and the low_contrast check."""

from __future__ import annotations

import matplotlib
import matplotlib.style
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure

from inferplot.render import StatsFigure, betweenstats
from inferplot.render.theme import INK, LIGHT_INK, contrast
from inferplot.result import StatsResult
from tests.render._helpers import by_gid
from tests.render._results import frame_result


def _three() -> StatsResult:
    rng = np.random.default_rng(1)
    groups = [rng.normal(m, 1, 20) for m in (5, 6, 7.5)]
    return frame_result(["a", "b", "c"], groups, pairwise_display="all")


def _low(fig: StatsFigure) -> str | None:
    found = [i for i in fig.check() if i.code == "low_contrast"]
    return found[0].message if found else None


def test_contrast_ratio_matches_wcag() -> None:
    assert contrast("black", "white") == pytest.approx(21.0)
    assert contrast("white", "white") == pytest.approx(1.0)
    assert contrast("#777777", "white") == pytest.approx(4.48, abs=0.01)


def test_a_dark_given_axes_gets_light_ink() -> None:
    with matplotlib.style.context("dark_background"):
        figure = Figure(figsize=(8, 6), dpi=100)
        FigureCanvasAgg(figure)
        ax = figure.add_subplot()
        fig = betweenstats(_three(), ax=ax, title="Dark")
        figure.canvas.draw()
    light = to_rgba(LIGHT_INK)
    for gid in ("subtitle", "caption-0", "bracket-label-0", "bracket-0"):
        artist = by_gid(fig, gid)
        assert artist is not None
        assert to_rgba(artist.get_color()) == light, gid  # pyright: ignore[reportAttributeAccessIssue]
    texts = [
        ax.xaxis.label,
        ax.yaxis.label,
        ax.title if ax.get_title() else ax._left_title,  # pyright: ignore[reportAttributeAccessIssue]
        *ax.get_xticklabels(),
        *ax.get_yticklabels(),
    ]
    for text in texts:
        if text.get_text():
            assert contrast(text.get_color(), "black") > 10, text.get_text()
    assert _low(fig) is None


def test_a_light_given_axes_keeps_the_usual_ink() -> None:
    figure = Figure(figsize=(8, 6), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    fig = betweenstats(_three(), ax=ax)
    subtitle = by_gid(fig, "subtitle")
    assert subtitle is not None
    assert to_hex(subtitle.get_color()) == INK.lower()  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("name", ["Blues", "Greys", "viridis"])
def test_a_continuous_colormap_is_sampled_away_from_its_ends(name: str) -> None:
    fig = betweenstats(_three(), palette=name)
    cmap = matplotlib.colormaps[name]
    for i in range(3):
        violin = by_gid(fig, f"violin-{i}")
        assert violin is not None
        assert to_rgba(violin.get_edgecolor()) == pytest.approx(cmap((i + 0.5) / 3))  # pyright: ignore[reportAttributeAccessIssue]
    assert _low(fig) is None


def test_a_listed_colormap_keeps_its_colours_in_order() -> None:
    fig = betweenstats(_three(), palette="Set2")
    cmap = matplotlib.colormaps["Set2"]
    for i in range(3):
        violin = by_gid(fig, f"violin-{i}")
        assert violin is not None
        assert to_rgba(violin.get_edgecolor()) == pytest.approx(cmap(i))  # pyright: ignore[reportAttributeAccessIssue]


def test_a_white_palette_is_reported() -> None:
    fig = betweenstats(_three(), palette=["white", "#fefefe", "tab:blue"])
    message = _low(fig)
    assert message is not None
    assert "'a'" in message
    assert "'b'" in message
    assert "'c'" not in message


def test_the_default_palette_is_not_reported() -> None:
    assert _low(betweenstats(_three())) is None
    rng = np.random.default_rng(2)
    eight = frame_result(
        [f"g{i}" for i in range(8)], [rng.normal(0, 1, 10) for _ in range(8)]
    )
    assert _low(betweenstats(eight)) is None


def test_a_later_change_of_background_is_reported() -> None:
    fig = betweenstats(_three())
    fig.axes["main"].set_facecolor("black")
    fig.figure.set_facecolor("black")
    message = _low(fig)
    assert message is not None
    assert "subtitle" in message or "tick label" in message
    assert "3:1" in message
