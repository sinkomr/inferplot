"""A cell label moved beside its bar carries a key of its segment's colour
(review B13): a small square at the label's left, filled with the segment's
colour as drawn and edged in inferplot's ink, between the joining line's end
and the label; with several labels beside one bar, each names its level by
colour as the legend does."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

import inferplot as ip
from inferplot.render import StatsFigure
from tests.render._bars import table_result


def _tiny() -> ip.StatsResult:
    return table_result(
        [[120, 0, 40], [1, 30, 2], [60, 50, 1], [3, 0, 57]],
        ["often", "sometimes", "rarely", "never"],
    )


def _by_gid(fig: StatsFigure, gid: str) -> object | None:
    found = [a for a in fig.figure.findobj() if a.get_gid() == gid]
    return found[0] if found else None


@pytest.mark.parametrize("given", [False, True])
def test_every_moved_label_has_a_key_of_its_segments_colour(given: bool) -> None:
    ax = Figure(figsize=(9, 6)).add_subplot() if given else None
    fig = ip.render.barstats(_tiny(), ax=ax)
    canvas = fig.figure.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        canvas = FigureCanvasAgg(fig.figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    moved = [c for c in fig.cell_labels if c.place == "outside"]
    assert len(moved) >= 3
    for cell in moved:
        gid = str(cell.text.get_gid()).replace("cell-label-", "key-")
        key = _by_gid(fig, gid)
        assert isinstance(key, Line2D), gid
        assert key.get_marker() == "s"
        face = to_rgba(key.get_markerfacecolor())
        assert np.allclose(face, cell.segment.get_facecolor())
        # Between the joining line's end and the label, clear of the label.
        leader = _by_gid(fig, gid.replace("key-", "leader-"))
        assert isinstance(leader, Line2D)
        ax_ = fig.axes["main"]
        key_xy = np.asarray(key.get_xydata(), dtype=float)[0]
        end_xy = np.asarray(leader.get_xydata(), dtype=float)[1]
        key_x = float(ax_.transData.transform(key_xy)[0])
        end_x = float(ax_.transData.transform(end_xy)[0])
        assert isinstance(cell.text, Text)
        box = cell.text.get_window_extent(renderer)
        assert end_x < key_x < box.x0
        half = float(key.get_markersize()) / 2 * fig.figure.dpi / 72
        assert key_x + half <= box.x0 + 0.5
    assert [i.code for i in fig.check() if i.severity == "warning"] == [
        i.code for i in fig.check() if i.code == "small_expected"
    ]


def test_labels_on_their_segment_have_no_key() -> None:
    fig = ip.render.barstats(_tiny())
    inside = [c for c in fig.cell_labels if c.place == "inside"]
    assert inside
    for cell in inside:
        gid = str(cell.text.get_gid()).replace("cell-label-", "key-")
        assert _by_gid(fig, gid) is None
    assert all(
        isinstance(p, Rectangle) for p in fig.axes["main"].patches
    )  # the keys are markers, not patches: the segments stay the only patches
