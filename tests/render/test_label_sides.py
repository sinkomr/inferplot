"""A centrality label reads as its own level's (review of phase 3, finding 1):
it stays on its point's usual side, takes the other side only within its own
half of the spacing, never lies mostly beyond a neighbouring point or over the
centrality path between two other conditions; inferplot's own figure widens its
levels when no place fits; and fig.check() reports a label that reads as another
level's (``centrality_label_ambiguous``), wherever it ended up."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import inferplot as ip
from inferplot.render import StatsFigure
from inferplot.render.labels import Ambiguity, ambiguities
from tests.render._helpers import by_gid


def _frame(seed: int, n: int, k: int) -> pd.DataFrame:
    """The review's design: subjects differ a lot, conditions a little, so the
    centrality points lie close together and the labels compete for room."""
    rng = np.random.default_rng(seed)
    m = rng.normal(50, 10, (n, 1)) + rng.normal(0, 3, (n, k))
    return pd.DataFrame(
        [(f"s{i:03d}", f"c{j + 1}", m[i, j]) for i in range(n) for j in range(k)],
        columns=["id", "cond", "y"],
    )


def _boxes(fig: StatsFigure) -> dict[int, Bbox]:
    fig.draw()
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    out: dict[int, Bbox] = {}
    for label in fig.labels:
        patch = label.text.get_bbox_patch()
        assert patch is not None
        out[label.index] = Bbox.union(
            [label.text.get_window_extent(renderer), patch.get_window_extent(renderer)]
        )
    return out


def _point_x(fig: StatsFigure) -> dict[int, float]:
    ax = fig.axes["main"]
    return {
        i: float(ax.transData.transform((i, 0.0))[0])
        for i, d in enumerate(fig.result.descriptives)
        if d.value is not None
    }


def _left_of_middle(fig: StatsFigure) -> list[int]:
    """Labels of levels with a neighbour on the left drawn left of their point
    and reaching past the middle toward that neighbour (the review's measure)."""
    boxes, xs = _boxes(fig), _point_x(fig)
    return [
        i
        for i, box in boxes.items()
        if i - 1 in xs and box.x0 < (xs[i] + xs[i - 1]) / 2 - 0.5
    ]


def _codes(fig: StatsFigure) -> list[str]:
    return [c.code for c in fig.check()]


# --------------------------------------------------------------------------- #
# the geometry
# --------------------------------------------------------------------------- #


def test_a_right_label_reaching_across_the_gap_reads_as_its_own() -> None:
    points = {0: 100.0, 1: 210.0, 2: 320.0}
    # Box from 20 px right of its point to 2 px past the next point: the usual
    # place of a wide label, read as its own as every label sits the same way.
    box = Bbox.from_extents(120.0, 50.0, 212.0, 70.0)
    assert ambiguities(0, box, "right", points) == []
    # Mostly beyond the next point: reads as that point's.
    box = Bbox.from_extents(190.0, 50.0, 290.0, 70.0)
    assert ambiguities(0, box, "right", points) == [Ambiguity("beyond", 1)]


def test_a_label_on_the_other_side_must_stay_in_its_own_half() -> None:
    points = {0: 100.0, 1: 210.0, 2: 320.0}
    inside = Bbox.from_extents(160.0, 50.0, 190.0, 70.0)  # middle is at 155
    assert ambiguities(1, inside, "right", points) == []
    past = Bbox.from_extents(105.0, 50.0, 190.0, 70.0)
    assert ambiguities(1, past, "right", points) == [Ambiguity("beside", 0)]
    # The first of two within conditions goes left by default: there the right
    # is the other side.
    two = {0: 100.0, 1: 210.0}
    right = Bbox.from_extents(120.0, 50.0, 200.0, 70.0)
    assert ambiguities(0, right, "left", two) == [Ambiguity("beside", 1)]


def test_a_label_over_the_path_between_two_other_conditions() -> None:
    points = {0: 100.0, 1: 210.0, 2: 320.0, 3: 430.0}
    path = np.array(
        [[[100.0, 60.0], [210.0, 60.0]],
         [[210.0, 60.0], [320.0, 60.0]],
         [[320.0, 60.0], [430.0, 60.0]]]
    )  # fmt: skip
    over = Bbox.from_extents(330.0, 50.0, 380.0, 70.0)  # over the piece 2 to 3
    assert ambiguities(0, over, "right", points, path) == [
        Ambiguity("beyond", 1),
        Ambiguity("path", 2),
    ]
    # Over its own pieces it is not reported here (path_hidden says so).
    own = Bbox.from_extents(120.0, 50.0, 200.0, 70.0)
    assert ambiguities(0, own, "right", points, path) == []


# --------------------------------------------------------------------------- #
# the drawing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type_", ["parametric", "robust"])
@pytest.mark.parametrize("within", [True, False])
def test_the_reviews_figure_keeps_every_label_beside_its_point(
    type_: str, within: bool
) -> None:
    # 20 subjects at 5 conditions with centres within half a unit: before the
    # fix the label of c2 went left, beside the point of c1 (both functions).
    data = _frame(1, 20, 5)
    if within:
        fig = ip.withinstats(data, "cond", "y", subject_id="id", type=type_)
    else:
        fig = ip.betweenstats(data, "cond", "y", type=type_)
    assert _left_of_middle(fig) == []
    boxes, xs = _boxes(fig), _point_x(fig)
    for label in fig.labels:
        assert ambiguities(label.index, boxes[label.index], label.usual, xs) == []
    codes = _codes(fig)
    assert "centrality_label_ambiguous" not in codes
    assert "centrality_label_overlap" not in codes
    assert "label_overlap" not in codes
    # Every label on its usual side (right of its point).
    for label in fig.labels:
        assert label.usual == "right"
        assert label.text.get_horizontalalignment() == "left"


@pytest.mark.parametrize("seed", range(8))
def test_own_figures_with_close_centres_read_unambiguously(seed: int) -> None:
    rng = np.random.default_rng(9000 + seed)
    k = int(rng.integers(4, 7))
    n = int(rng.integers(8, 40))
    fig = ip.withinstats(
        _frame(9000 + seed, n, k), "cond", "y", subject_id="id",
        pairwise_display="none",
    )  # fmt: skip
    assert _left_of_middle(fig) == []
    codes = _codes(fig)
    assert "centrality_label_ambiguous" not in codes
    assert "centrality_label_overlap" not in codes


def test_a_figure_that_fits_keeps_the_usual_places() -> None:
    # Labels that fit beside their points stay at the drawing's own choice.
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "cond": np.repeat(["c1", "c2", "c3"], 12),
            "y": np.concatenate([rng.normal(m, 1.0, 12) for m in (10, 20, 30)]),
        }
    )
    fig = ip.betweenstats(data, "cond", "y")
    assert [label.text.get_verticalalignment() for label in fig.labels] == [
        "center"
    ] * 3


def test_a_narrow_axes_reports_what_it_cannot_place() -> None:
    # ax= fixes the room: no label goes beside another point; what does not fit
    # is reported, not hidden.
    figure = Figure(figsize=(4.0, 4.0), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = ip.withinstats(_frame(1, 20, 5), "cond", "y", subject_id="id", ax=ax)
    assert _left_of_middle(fig) == []
    issues = {c.code: c for c in fig.check()}
    assert {"centrality_label_ambiguous", "centrality_label_overlap"} & set(issues)
    if "centrality_label_ambiguous" in issues:
        fix = issues["centrality_label_ambiguous"].fix
        assert "make this Axes wider" in fix
        assert "leave ax out" not in fix


def test_a_label_moved_beside_another_point_is_reported() -> None:
    fig = ip.withinstats(_frame(1, 20, 5), "cond", "y", subject_id="id")
    text = by_gid(fig, "centrality-label-2")
    assert isinstance(text, Text)
    value = fig.result.descriptives[2].value
    assert value is not None
    # Right-aligned at 1.3: left of its point, past the middle toward c2's.
    text.set_horizontalalignment("right")
    text.set_x(1.75)
    (issue,) = [c for c in fig.check() if c.code == "centrality_label_ambiguous"]
    assert issue.severity == "warning"
    assert (
        "the centrality label of 'c3' sits beside the centrality point of 'c2'"
        in issue.message
    )
    assert "If you moved a label, move it back" in issue.fix
    assert "leave ax out" not in issue.fix
    assert "centrality_plotting=False" in issue.fix
    assert "digits=" in issue.fix


def test_a_label_moved_over_the_path_between_two_other_conditions() -> None:
    fig = ip.withinstats(_frame(1, 20, 5), "cond", "y", subject_id="id")
    text = by_gid(fig, "centrality-label-0")
    assert isinstance(text, Text)
    a, b = (fig.result.descriptives[j].value for j in (2, 3))
    assert a is not None
    assert b is not None
    text.set_transform(fig.axes["main"].transData)
    text.set_verticalalignment("center")
    text.set_x(2.3)
    text.set_y((a + b) / 2)
    (issue,) = [c for c in fig.check() if c.code == "centrality_label_ambiguous"]
    assert (
        "the centrality label of 'c1' lies over the centrality path between "
        "'c3' and 'c4'" in issue.message
    )
    assert "lies mostly beyond the centrality point of 'c2'" in issue.message


def _link(fig: StatsFigure, i: int) -> Line2D:
    link = by_gid(fig, f"centrality-link-{i}")
    assert isinstance(link, Line2D)
    return link


def test_links_along_the_path_have_a_casing() -> None:
    # Three or more conditions: links leave their point along the path and are
    # drawn over a casing of the background colour; with two conditions the
    # links point away from the path and are drawn plain.
    fig = ip.withinstats(_frame(1, 20, 4), "cond", "y", subject_id="id")
    cased = [bool(_link(fig, i).get_path_effects()) for i in range(4)]
    assert cased == [True] * 3 + [False]
    two = ip.withinstats(_frame(1, 20, 2), "cond", "y", subject_id="id")
    assert not any(_link(two, i).get_path_effects() for i in range(2))
    between = ip.betweenstats(_frame(1, 20, 4), "cond", "y")
    assert not any(_link(between, i).get_path_effects() for i in range(4))
