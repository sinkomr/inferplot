"""render.withinstats: layers, subject and centrality paths, labels, subsampling,
checks, ax=, artist arguments and saving, on hand-built within results with 2, 3
and 5 conditions (tests/render/_within.py)."""

from __future__ import annotations

import dataclasses
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from inferplot.render import StatsFigure, withinstats
from inferplot.render.comparison import (
    JITTER_HALF_WIDTH,
    LABEL_OFFSET,
    MAX_POINTS,
    SUBSAMPLE_SIZE,
)
from inferplot.result import StatsResult
from inferplot.text import figure_texts
from tests.render._helpers import (
    SVG_NS,
    by_gid,
    gids,
    numbered_count,
    numbered_texts,
    pdf_info_string,
    svg_description,
    text_of,
)
from tests.render._within import (
    five_conditions,
    many_subjects,
    three_conditions,
    two_conditions,
    within_result,
)

FIGURE_CODES = frozenset(
    {
        "label_overlap",
        "text_clipped",
        "path_hidden",
        "paths_crowded",
        "figure_oversized",
    }
)


def _codes(fig: StatsFigure) -> list[str]:
    return [issue.code for issue in fig.check() if issue.code in FIGURE_CODES]


def _points(fig: StatsFigure, i: int) -> np.ndarray[Any, Any]:
    artist = by_gid(fig, f"points-{i}")
    assert isinstance(artist, PathCollection)
    return np.asarray(artist.get_offsets(), dtype=np.float64)


def _subject_paths(fig: StatsFigure) -> list[np.ndarray[Any, Any]]:
    artist = by_gid(fig, "subject-paths")
    assert isinstance(artist, LineCollection)
    return [np.asarray(s, dtype=np.float64) for s in artist.get_segments()]


def _centrality_path(fig: StatsFigure) -> np.ndarray[Any, Any]:
    artist = by_gid(fig, "centrality-path")
    assert isinstance(artist, Line2D)
    return np.asarray(artist.get_xydata(), dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType]


def _new_warnings(fig: StatsFigure, result: StatsResult) -> tuple[str, ...]:
    assert fig.result.warnings[: len(result.warnings)] == result.warnings
    return fig.result.warnings[len(result.warnings) :]


def _user_axes(width: float, height: float, dpi: int = 100) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, height), dpi=dpi)
    FigureCanvasAgg(figure)
    return figure, figure.add_subplot()


def path_reach(fig: StatsFigure) -> float:
    """How deep the drawn paths run into any text's rendered box, in pixels.

    An independent measure for ``check()``: the centrality path and every subject
    path are sampled every 0.05 px in pixel space; a text's box is its rendered
    extent with its label box. Returns the deepest sample's distance to the
    nearest edge of a box it is inside (0.0 when none is).
    """
    fig.draw()
    figure = fig.figure
    renderer = figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    ax = fig.axes["main"]
    boxes = []
    for text in figure.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        patch = text.get_bbox_patch()
        if patch is not None:
            box = Bbox.union([box, patch.get_window_extent(renderer)])
        boxes.append(box)
    lines: list[np.ndarray[Any, Any]] = []
    if by_gid(fig, "centrality-path") is not None:
        lines.append(_centrality_path(fig))
    if by_gid(fig, "subject-paths") is not None:
        lines += _subject_paths(fig)
    deepest = 0.0
    for line in lines:
        xy = ax.transData.transform(line)
        for (x0, y0), (x1, y1) in itertools.pairwise(xy):
            if not np.isfinite([x0, y0, x1, y1]).all():
                continue
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 0.05))
            t = np.linspace(0.0, 1.0, n)
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for box in boxes:
                inside = (px > box.x0) & (px < box.x1) & (py > box.y0) & (py < box.y1)
                if inside.any():
                    depth = np.minimum.reduce(
                        [px - box.x0, box.x1 - px, py - box.y0, box.y1 - py]
                    )[inside]
                    deepest = max(deepest, float(depth.max()))
    return deepest


# --------------------------------------------------------------------------- #
# layers and texts
# --------------------------------------------------------------------------- #


def test_two_conditions_draw_every_layer_and_both_paths() -> None:
    result = two_conditions()
    fig = withinstats(result)
    present = gids(fig)
    for i in range(2):
        assert {f"violin-{i}", f"box-{i}", f"points-{i}"} <= present
        assert {f"centrality-{i}", f"centrality-label-{i}"} <= present
    assert {"subject-paths", "centrality-path"} <= present
    texts = figure_texts(result)
    assert text_of(fig, "subtitle") == texts.subtitle
    assert numbered_texts(fig, "caption") == list(texts.caption_lines)
    assert numbered_texts(fig, "centrality-label") == list(texts.centrality_labels)
    assert [t.get_text() for t in fig.axes["main"].get_xticklabels()] == list(
        texts.tick_labels
    )
    assert numbered_count(fig, "bracket") == 0  # no pairs with two conditions
    assert len(_subject_paths(fig)) == len(result.subjects) == 20
    centers = [d.value for d in result.descriptives]
    assert _centrality_path(fig).tolist() == [[0.0, centers[0]], [1.0, centers[1]]]
    assert fig.result.warnings == result.warnings
    assert fig.paths is not None
    assert fig.paths.levels == result.levels
    assert _codes(fig) == []
    assert path_reach(fig) == 0.0


@pytest.mark.parametrize(
    "make",
    [
        lambda: three_conditions("all"),
        lambda: three_conditions("significant"),
        lambda: five_conditions("significant"),
        lambda: five_conditions("all"),
    ],
    ids=["three-all", "three-significant", "five-significant", "five-all"],
)
def test_three_or_more_conditions_draw_no_subject_paths(make: Any) -> None:
    result: StatsResult = make()
    k = len(result.levels)
    fig = withinstats(result)
    present = gids(fig)
    assert "subject-paths" not in present
    assert all(f"points-{i}" in present for i in range(k))
    path = _centrality_path(fig)
    assert path[:, 0].tolist() == list(range(k))
    assert path[:, 1].tolist() == [d.value for d in result.descriptives]
    shown = [p for p in result.pairwise if p.displayed]
    assert len(fig.brackets) == len(shown) > 0
    texts = figure_texts(result)
    labels = [
        t for n, t in enumerate(texts.pairwise_labels) if result.pairwise[n].displayed
    ]
    assert numbered_texts(fig, "bracket-label") == labels
    assert numbered_texts(fig, "caption") == list(texts.caption_lines)
    assert fig.result.warnings == result.warnings  # the default point_path adds none
    assert _codes(fig) == []
    assert path_reach(fig) == 0.0


# --------------------------------------------------------------------------- #
# subject paths and jitter
# --------------------------------------------------------------------------- #


def test_each_path_ends_on_its_subjects_points() -> None:
    result = two_conditions()
    fig = withinstats(result)
    before, after = _points(fig, 0), _points(fig, 1)
    paths = np.stack(_subject_paths(fig))  # (subjects, levels, 2)
    # Exactly the drawn points, subject by subject, in the result's subject order.
    assert np.array_equal(paths[:, 0, :], before)
    assert np.array_equal(paths[:, 1, :], after)
    assert before[:, 1].tolist() == list(result.values[0])
    assert after[:, 1].tolist() == list(result.values[1])
    # One offset per subject, shared by its points at both levels.
    offset = before[:, 0]
    assert np.allclose(after[:, 0] - 1.0, offset, rtol=0.0, atol=1e-12)
    assert np.all(np.abs(offset) <= JITTER_HALF_WIDTH)
    assert len(set(offset.tolist())) == offset.size


def test_jitter_follows_the_seed() -> None:
    result = two_conditions()
    a = _points(withinstats(result), 0)
    b = _points(withinstats(result), 0)
    assert np.array_equal(a, b)
    other = dataclasses.replace(
        result, settings=dataclasses.replace(result.settings, seed=7)
    )
    assert not np.array_equal(a, _points(withinstats(other), 0))


def test_point_path_false_draws_no_subject_paths() -> None:
    result = two_conditions()
    fig = withinstats(result, point_path=False)
    present = gids(fig)
    assert "subject-paths" not in present
    assert {"points-0", "points-1", "centrality-path"} <= present
    assert _new_warnings(fig, result) == ()
    assert _codes(fig) == []


def test_point_path_false_with_three_conditions_has_no_effect_and_says_so() -> None:
    result = three_conditions()
    fig = withinstats(result, point_path=False)
    assert _new_warnings(fig, result) == (
        "point_path=False has no effect: subject paths are drawn only with two "
        "conditions (as in ggwithinstats), and x='dose' has 3.",
    )
    # An explicit True is told from the default (None) and says so too.
    assert _new_warnings(withinstats(result, point_path=True), result) == (
        "point_path=True has no effect: subject paths are drawn only with two "
        "conditions (as in ggwithinstats), and x='dose' has 3.",
    )
    assert _new_warnings(withinstats(result), result) == ()


def test_layers_without_points_draw_no_subject_paths() -> None:
    result = two_conditions()
    fig = withinstats(result, layers=("violin", "box"))
    present = gids(fig)
    assert not any(g.startswith(("points-", "subject-paths")) for g in present)
    assert "centrality-path" in present
    assert _new_warnings(fig, result) == ()  # the default point_path adds nothing
    fig = withinstats(result, layers=("violin", "box"), point_path=False)
    assert _new_warnings(fig, result) == (
        "point_path=False has no effect: subject paths join the points, and "
        "layers=('violin', 'box') draws no points.",
    )
    assert _codes(fig) == []


# --------------------------------------------------------------------------- #
# centrality path and labels
# --------------------------------------------------------------------------- #


def test_centrality_path_false() -> None:
    result = two_conditions()
    fig = withinstats(result, centrality_path=False)
    present = gids(fig)
    assert "centrality-path" not in present
    assert {"subject-paths", "centrality-0", "centrality-label-1"} <= present
    assert _new_warnings(fig, result) == ()
    assert _codes(fig) == []


def test_centrality_plotting_false_draws_no_centrality_path() -> None:
    # As in ggwithinstats, the centrality path belongs to the centrality points.
    result = two_conditions()
    fig = withinstats(result, centrality_plotting=False)
    present = gids(fig)
    assert not any(g.startswith("centrality") for g in present)
    assert "subject-paths" in present
    assert fig.paths is not None
    assert fig.paths.centrality is None
    assert _new_warnings(fig, result) == ()  # the default centrality_path (None)
    fig = withinstats(result, centrality_plotting=False, centrality_path=True)
    assert _new_warnings(fig, result) == (
        "centrality_path=True has no effect: the centrality path joins the "
        "centrality points, which centrality_plotting=False leaves out.",
    )
    fig = withinstats(result, centrality_plotting=False, centrality_path=False)
    assert _new_warnings(fig, result) == (
        "centrality_path=False has no effect: the centrality path joins the "
        "centrality points, which centrality_plotting=False leaves out.",
    )


def _label(fig: StatsFigure, i: int) -> Text:
    text = by_gid(fig, f"centrality-label-{i}")
    assert isinstance(text, Text)
    return text


def test_labels_leave_the_paths_free() -> None:
    # Two conditions: the first label left of its point, the second right of it.
    fig = withinstats(two_conditions())
    first, second = _label(fig, 0), _label(fig, 1)
    assert (first.get_horizontalalignment(), first.get_position()[0]) == (
        "right",
        -LABEL_OFFSET,
    )
    assert (second.get_horizontalalignment(), second.get_position()[0]) == (
        "left",
        1 + LABEL_OFFSET,
    )
    # Three conditions rising then falling (means 3.5, 5.8, 4.6): the first label
    # below its point's height, the middle one above, the last one centered.
    fig = withinstats(three_conditions())
    a, b, c = (d.value for d in three_conditions().descriptives)
    assert a is not None
    assert b is not None
    assert c is not None
    assert a < b > c
    assert [_label(fig, i).get_verticalalignment() for i in range(3)] == [
        "top",
        "bottom",
        "center",
    ]
    assert all(_label(fig, i).get_horizontalalignment() == "left" for i in range(3))
    assert path_reach(fig) == 0.0
    # Without paths the labels sit as in betweenstats: right of the point, centered.
    fig = withinstats(two_conditions(), point_path=False, centrality_path=False)
    for i in range(2):
        label = _label(fig, i)
        assert label.get_horizontalalignment() == "left"
        assert label.get_verticalalignment() == "center"


def test_a_level_without_a_centrality_value_breaks_the_path() -> None:
    # A MAP without a density (D55) has no value: no point, no label, and the
    # centrality path stops on either side of it.
    built = three_conditions()
    descriptives = list(built.descriptives)
    descriptives[1] = dataclasses.replace(descriptives[1], value=None)
    result = dataclasses.replace(built, descriptives=tuple(descriptives))
    fig = withinstats(result)
    path = _centrality_path(fig)
    assert np.isnan(path[1, 1])
    assert path[[0, 2], 1].tolist() == [built.descriptives[0].value,
                                        built.descriptives[2].value]  # fmt: skip
    assert "centrality-label-1" not in gids(fig)
    assert _label(fig, 0).get_verticalalignment() == "center"
    assert _codes(fig) == []


# --------------------------------------------------------------------------- #
# many values: whole subjects are subsampled
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("n", "k"), [(3_000, 2), (2_000, 3), (1_100, 5)])
def test_many_values_draw_whole_subjects(n: int, k: int) -> None:
    result = many_subjects(n, k)
    total = n * k
    assert total > MAX_POINTS
    fig = withinstats(result)
    m = SUBSAMPLE_SIZE // k
    sub = fig.subsample
    assert sub is not None
    assert (sub.shown, sub.total, sub.subjects, sub.seed) == (m * k, total, (m, n), 0)
    drawn = [_points(fig, j) for j in range(k)]
    assert all(points.shape == (m, 2) for points in drawn)
    # Each drawn subject is one of the result's, with all its values.
    index = {value: i for i, value in enumerate(result.values[0])}
    rows = [index[y] for y in drawn[0][:, 1].tolist()]
    assert rows == sorted(set(rows))
    for j in range(k):
        assert drawn[j][:, 1].tolist() == [result.values[j][i] for i in rows]
    if k == 2:
        paths = np.stack(_subject_paths(fig))
        assert paths.shape == (m, 2, 2)
        assert np.array_equal(paths[:, 0, :], drawn[0])
        assert np.array_equal(paths[:, 1, :], drawn[1])
    what = "points and subject paths" if k == 2 else "points"
    assert _new_warnings(fig, result) == (
        f"The plot has {total:,} values, more than 5,000, so the {what} show every "
        f"value of a random subsample of {m:,} of the {n:,} subjects ({m * k:,} "
        "values, seed 0), always with the subjects that hold each condition's "
        "smallest and largest value; the violins and boxes show all values.",
    )
    issues = {issue.code: issue for issue in fig.check()}
    assert issues["points_subsampled"].message == (
        f"The points show a random subsample of {m * k:,} of the {total:,} values: "
        f"every value of {m:,} of the {n:,} subjects, with those that hold each "
        "condition's smallest and largest value (seed 0); the violins and boxes "
        "are drawn from all values."
    )
    assert {"label_overlap", "text_clipped", "path_hidden"}.isdisjoint(issues)


def test_many_values_without_points_are_not_subsampled() -> None:
    result = many_subjects(3_000)
    fig = withinstats(result, layers=("violin", "box"))
    assert fig.subsample is None
    assert _new_warnings(fig, result) == ()


def test_many_values_without_violins_draw_density_outlines() -> None:
    fig = withinstats(many_subjects(3_000), layers=("box", "points"))
    assert {"density-0", "density-1"} <= gids(fig)
    assert "density outlines and boxes show all values" in fig.result.warnings[-1]


# --------------------------------------------------------------------------- #
# check(): crowded and hidden paths
# --------------------------------------------------------------------------- #


def test_crowded_subject_paths_are_reported() -> None:
    fig = withinstats(many_subjects(300))
    (issue,) = [i for i in fig.check() if i.code == "paths_crowded"]
    assert issue.severity == "warning"
    assert issue.message.startswith("The 300 subject paths are at least half opaque")
    assert "point_path=False" in issue.fix
    assert "paths_crowded" not in _codes(
        withinstats(many_subjects(300), point_path=False)
    )
    assert _codes(withinstats(many_subjects(100))) == []


def test_a_text_over_the_paths_is_reported() -> None:
    # Wide enough for the hand-built subtitle (a Welch t with n_pairs, 85
    # characters), so that the only issue is the user's note.
    _, ax = _user_axes(8.0, 6.0)
    fig = withinstats(two_conditions(), ax=ax)
    assert _codes(fig) == []
    # A boxed note of the user's, in the middle of the plot between the conditions.
    ax.text(
        0.5,
        float(np.mean(two_conditions().values)),
        "a note over the paths",
        ha="center",
        va="center",
        bbox={"facecolor": "white"},
        zorder=10,
    )
    (issue,) = [i for i in fig.check() if i.code == "path_hidden"]
    assert issue.severity == "warning"
    assert issue.message.startswith(
        "A text is drawn over a path: the centrality path between 'before' and "
        "'after' runs under the text 'a note over the paths'; "
    )
    assert "of the 20 subject paths run under the text 'a note over the paths'" in (
        issue.message
    )
    assert path_reach(fig) > 1.0


# --------------------------------------------------------------------------- #
# ax=, artist arguments, errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("make", [two_conditions, three_conditions, five_conditions])
def test_draw_into_a_given_axes(make: Any) -> None:
    result = make()
    figure, ax = _user_axes(9.0, 7.0)
    fig = withinstats(result, ax=ax, title="Within subjects")
    assert fig.figure is figure
    assert fig.axes["main"] is ax
    subtitle = by_gid(fig, "subtitle")
    assert isinstance(subtitle, Text)
    assert subtitle in ax.texts
    assert ax.get_title(loc="left") == "Within subjects"
    assert figure.get_size_inches().tolist() == [9.0, 7.0]
    assert _codes(fig) == []
    assert path_reach(fig) == 0.0


def test_artist_args_style_the_points_not_the_paths() -> None:
    fig = withinstats(
        two_conditions(),
        point_args={"alpha": 0.2, "sizes": [30.0]},
        boxplot_args={"linewidth": 2.0},
        violin_args={"alpha": 0.1},
    )
    points = by_gid(fig, "points-0")
    assert isinstance(points, PathCollection)
    assert points.get_alpha() == 0.2
    paths = by_gid(fig, "subject-paths")
    assert isinstance(paths, LineCollection)
    assert paths.get_alpha() == pytest.approx(0.35)


def _state(ax: Axes) -> tuple[object, ...]:
    return (
        len(ax.patches),
        len(ax.collections),
        len(ax.lines),
        len(ax.texts),
        ax.get_xlim(),
        ax.get_ylim(),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"point_args": {"bogus": 1}},
        {"layers": ("box",), "point_args": {"alpha": 0.5}},
        {"point_args": {"linewidth": float("nan")}},
        {"title": r"$\frac$"},
        {"point_path": "yes"},
        {"palette": ["red"]},
    ],
    ids=lambda kwargs: "-".join(kwargs),
)
def test_a_failed_call_leaves_the_given_axes_unchanged(
    kwargs: dict[str, object],
) -> None:
    _, ax = _user_axes(6.0, 4.0)
    ax.plot([0.0, 1.0], [0.0, 1.0])
    before = _state(ax)
    with pytest.raises((TypeError, ValueError)):
        withinstats(two_conditions(), ax=ax, **kwargs)  # pyright: ignore[reportArgumentType]
    assert _state(ax) == before


def test_inconsistent_results_raise() -> None:
    result = two_conditions()
    short = dataclasses.replace(result, values=(result.values[0], result.values[1][:5]))
    with pytest.raises(ValueError, match="one value per subject at every level"):
        withinstats(short)
    with pytest.raises(ValueError, match="one value per subject at every level"):
        withinstats(dataclasses.replace(result, subjects=()))
    with pytest.raises(ValueError, match="inconsistent"):
        withinstats(dataclasses.replace(result, values=result.values[:1]))


def test_a_subtitle_needs_results_subtitle_false() -> None:
    with pytest.raises(ValueError, match="results_subtitle=False"):
        withinstats(two_conditions(), subtitle="mine")
    rows = [[1.0, 2.0], [2.0, 2.5], [3.0, 5.0], [4.0, 4.5], [2.5, 3.5]]
    quiet = within_result(["a", "b"], rows, results_subtitle=False)
    fig = withinstats(quiet, subtitle="Five subjects", caption="Made-up data.")
    assert text_of(fig, "subtitle") == "Five subjects"
    assert numbered_texts(fig, "caption")[-1] == "Made-up data."
    assert _codes(fig) == []


# --------------------------------------------------------------------------- #
# saving
# --------------------------------------------------------------------------- #


def _svg_group(path: Path, gid: str) -> ET.Element:
    root = ET.parse(path).getroot()
    (group,) = [el for el in root.iter(f"{SVG_NS}g") if el.get("id") == gid]
    return group


def test_svg_keeps_paths_as_editable_vectors(tmp_path: Path) -> None:
    result = two_conditions()
    fig = withinstats(result)
    path = tmp_path / "f.svg"
    fig.save(path)
    # One group of one <path> per subject, and one centrality path.
    group = _svg_group(path, "subject-paths")
    assert len(group.findall(f"{SVG_NS}path")) == len(result.subjects)
    assert len(_svg_group(path, "centrality-path").findall(f"{SVG_NS}path")) == 1
    embedded = json.loads(svg_description(path))
    assert embedded["analysis"] == "withinstats"
    assert embedded["values"] == embedded["subjects"] == []  # the user's data (D36)
    back = StatsResult.from_json(svg_description(path))
    with pytest.raises(ValueError, match=r"ip\.analyze\.withinstats\(data, x, y"):
        withinstats(back)


def test_pdf_and_identical_bytes(tmp_path: Path) -> None:
    fig = withinstats(two_conditions())
    fig.save(tmp_path / "a.pdf")
    data = (tmp_path / "a.pdf").read_bytes()
    assert json.loads(pdf_info_string(data, "Subject"))["analysis"] == "withinstats"
    assert b"/Type3" not in data
    for suffix in (".svg", ".pdf", ".png"):
        fig.save(tmp_path / f"b{suffix}", dpi=72)
        withinstats(two_conditions()).save(tmp_path / f"c{suffix}", dpi=72)
        assert (tmp_path / f"b{suffix}").read_bytes() == (
            tmp_path / f"c{suffix}"
        ).read_bytes()


def test_a_result_read_back_from_json_redraws_the_same_figure(tmp_path: Path) -> None:
    result = three_conditions()
    back = StatsResult.from_json(result.to_json())
    assert back == result
    withinstats(result).save(tmp_path / "a.svg")
    withinstats(back).save(tmp_path / "b.svg")
    assert (tmp_path / "a.svg").read_bytes() == (tmp_path / "b.svg").read_bytes()
