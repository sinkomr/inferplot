"""Centrality labels never cover a centrality point (ggstatsplot repels its
labels): each label takes the first free place of an ordered set of candidates,
and fig.check() reports a point still at least half covered."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from inferplot.render import StatsFigure, betweenstats
from inferplot.render.comparison import CENTRALITY_SIZE, LabelPlace, label_candidates
from inferplot.result import StatsResult
from tests.render._helpers import by_gid, case_named
from tests.render._results import fixture_result, frame_result


def _label_boxes(fig: StatsFigure) -> dict[int, Bbox]:
    fig.draw()
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    out: dict[int, Bbox] = {}
    for text in fig.axes["main"].texts:
        gid = str(text.get_gid() or "")
        if gid.startswith("centrality-label-"):
            patch = text.get_bbox_patch()
            assert patch is not None
            out[int(gid.rsplit("-", 1)[1])] = patch.get_window_extent(renderer)
    return out


def covered(fig: StatsFigure) -> list[tuple[str, str, float]]:
    """(level whose point is covered, by whose label, share of the point's square);
    the review's measure, independent of the renderer's own geometry."""
    labels = _label_boxes(fig)
    ax = fig.axes["main"]
    half = np.sqrt(CENTRALITY_SIZE) / 2 * fig.figure.dpi / 72
    out: list[tuple[str, str, float]] = []
    for i, d in enumerate(fig.result.descriptives):
        if d.value is None:
            continue
        cx, cy = ax.transData.transform((i, d.value))
        for j, box in labels.items():
            w = min(box.x1, cx + half) - max(box.x0, cx - half)
            h = min(box.y1, cy + half) - max(box.y0, cy - half)
            if w > 0 and h > 0:
                share = float(w * h / (4 * half * half))
                out.append((fig.result.levels[i], fig.result.levels[j], share))
    return out


def _codes(fig: StatsFigure) -> list[str]:
    return [i.code for i in fig.check()]


def test_candidates_start_with_the_drawings_choice() -> None:
    assert label_candidates(LabelPlace()) == [
        LabelPlace("right", "center"),
        LabelPlace("right", "above"),
        LabelPlace("right", "below"),
        LabelPlace("left", "center"),
        LabelPlace("left", "above"),
        LabelPlace("left", "below"),
    ]
    assert label_candidates(LabelPlace("left", "below"))[:3] == [
        LabelPlace("left", "below"),
        LabelPlace("left", "center"),
        LabelPlace("left", "above"),
    ]


@pytest.mark.parametrize("name", ["InsectSprays", "edge_k5"])
def test_no_label_covers_a_centrality_point(name: str) -> None:
    fig = betweenstats(fixture_result(case_named(name)))
    assert covered(fig) == []
    assert "centrality_label_overlap" not in _codes(fig)
    assert "label_overlap" not in _codes(fig)


def test_a_figure_without_collisions_keeps_the_usual_places() -> None:
    fig = betweenstats(fixture_result(case_named("ToothGrowth")))
    for i in range(2):
        text = by_gid(fig, f"centrality-label-{i}")
        assert isinstance(text, Text)
        assert text.get_horizontalalignment() == "left"
        assert text.get_verticalalignment() == "center"


def _random(seed: int) -> StatsResult:
    """The review's generator: k groups of 25, sd 1, means within one sd."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 7))
    means = 50 + rng.uniform(0, 1.0, size=k)
    groups = [rng.normal(m, 1.0, size=25) for m in means]
    return frame_result([f"g{i}" for i in range(k)], groups)


@pytest.mark.parametrize("seed", range(12))
def test_close_centres_get_labels_off_every_point(seed: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = betweenstats(_random(seed))
    assert not [c for c in covered(fig) if c[2] >= 0.5]
    codes = _codes(fig)
    assert "centrality_label_overlap" not in codes
    assert "label_overlap" not in codes
    assert "text_clipped" not in codes


def test_a_label_moved_onto_a_point_is_reported() -> None:
    fig = betweenstats(fixture_result(case_named("ToothGrowth")))
    text = by_gid(fig, "centrality-label-0")
    assert isinstance(text, Text)
    value = fig.result.descriptives[1].value
    assert value is not None
    text.set_position((0.95, value))
    (issue,) = [i for i in fig.check() if i.code == "centrality_label_overlap"]
    assert f"centrality point of {fig.result.levels[1]!r}" in issue.message
    assert "centrality label of" in issue.message
