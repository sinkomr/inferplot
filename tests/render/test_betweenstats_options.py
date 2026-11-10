"""Renderer options: many levels, many points, ax=, palettes, user texts, errors."""

from __future__ import annotations

import dataclasses
import io
import re

import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from inferplot.render import StatsFigure, betweenstats
from inferplot.render.betweenstats import MAX_POINTS, SUBSAMPLE_SIZE
from inferplot.render.checks import figure_issues
from inferplot.render.theme import OKABE_ITO
from inferplot.result import StatsResult
from inferplot.text import figure_texts
from tests.render._helpers import (
    by_gid,
    case_named,
    figure_codes,
    gids,
    numbered_texts,
    stroke_reach,
    text_of,
)
from tests.render._results import (
    fixture_result,
    frame_result,
    many_levels_result,
    many_points_result,
)


def _plant() -> StatsResult:
    return fixture_result(case_named("PlantGrowth-display-all"))


def _tooth() -> StatsResult:
    return fixture_result(case_named("ToothGrowth"))


def _points(fig: StatsFigure, i: int) -> PathCollection:
    artist = by_gid(fig, f"points-{i}")
    assert isinstance(artist, PathCollection)
    return artist


def _rgba(artist: PathCollection) -> tuple[float, ...]:
    return tuple(np.asarray(artist.get_facecolor(), dtype=float)[0].tolist())


# --------------------------------------------------------------------------- #
# colors
# --------------------------------------------------------------------------- #


def test_default_palette_is_okabe_ito() -> None:
    fig = betweenstats(_plant())
    for i in range(3):
        want = to_rgba(OKABE_ITO[i], alpha=0.6)
        assert _rgba(_points(fig, i)) == pytest.approx(want)


def test_more_than_eight_levels_use_tab20_and_warn() -> None:
    result = many_levels_result(12)
    fig = betweenstats(result)
    tab20 = matplotlib.colormaps["tab20"]
    for i in range(12):
        assert _rgba(_points(fig, i))[:3] == pytest.approx(tab20(i)[:3])
    (warning,) = fig.result.warnings[len(result.warnings) :]
    assert "12 levels" in warning
    assert "tab20" in warning
    assert "palette=" in warning
    # The input result is unchanged; only warnings differ.
    assert result.warnings == fig.result.warnings[: len(result.warnings)]
    assert dataclasses.replace(fig.result, warnings=result.warnings) == result
    assert figure_codes(fig) == []
    assert [t.get_text() for t in fig.axes["main"].get_xticklabels()] == list(
        figure_texts(result).tick_labels
    )


def test_eight_levels_do_not_warn() -> None:
    result = many_levels_result(8)
    fig = betweenstats(result)
    assert fig.result.warnings == result.warnings


def test_palette_list_and_colormap_names() -> None:
    result = _plant()
    fig = betweenstats(result, palette=["red", "#00ff00", (0.0, 0.0, 1.0)])
    assert _rgba(_points(fig, 1))[:3] == pytest.approx((0.0, 1.0, 0.0))
    fig = betweenstats(result, palette="Set2")
    set2 = matplotlib.colormaps["Set2"]
    assert _rgba(_points(fig, 2))[:3] == pytest.approx(set2(2)[:3])
    fig = betweenstats(result, palette="viridis")
    viridis = matplotlib.colormaps["viridis"]
    # A continuous colormap is sampled at the centres of k bins, (i + 0.5) / k.
    assert _rgba(_points(fig, 2))[:3] == pytest.approx(viridis(2.5 / 3)[:3])
    assert fig.result.warnings == result.warnings


@pytest.mark.parametrize(
    ("palette", "error", "match"),
    [
        ("not_a_colormap", ValueError, "colormap"),
        ("viridiss", ValueError, "Did you mean 'viridis'"),
        (["red", "blue"], ValueError, "2 colors but the plot has 3 levels"),
        (["red", "notacolor", "blue"], ValueError, "not a matplotlib color"),
        (5, TypeError, "palette"),
    ],
)
def test_bad_palettes(palette: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        betweenstats(_plant(), palette=palette)  # pyright: ignore[reportArgumentType]


def test_qualitative_colormap_too_short() -> None:
    with pytest.raises(ValueError, match="'Set1' has 9 colors but the plot has 12"):
        betweenstats(many_levels_result(12), palette="Set1")


# --------------------------------------------------------------------------- #
# many points
# --------------------------------------------------------------------------- #


def test_many_points_are_subsampled_with_density() -> None:
    result = many_points_result()
    total = sum(len(v) for v in result.values)
    assert total > MAX_POINTS
    fig = betweenstats(result)
    shown = [np.asarray(_points(fig, i).get_offsets()).shape[0] for i in range(3)]
    assert sum(shown) == SUBSAMPLE_SIZE
    present = gids(fig)
    assert {"violin-0", "violin-1", "violin-2"} <= present
    (warning,) = fig.result.warnings[len(result.warnings) :]
    assert f"{total:,}" in warning
    assert "2,000" in warning
    assert "seed 0" in warning
    issues = {i.code: i for i in fig.check()}
    assert issues["points_subsampled"].severity == "info"
    assert "2,000 of the 6,600" in issues["points_subsampled"].message
    assert "label_overlap" not in issues
    assert "text_clipped" not in issues
    # Every point drawn is one of the values of its level.
    for i in range(3):
        ys = np.asarray(_points(fig, i).get_offsets())[:, 1]
        assert set(ys.tolist()) <= set(result.values[i])


def test_many_points_without_violin_draw_density_outlines() -> None:
    fig = betweenstats(many_points_result(), layers=("box", "points"))
    present = gids(fig)
    assert {"density-0", "density-1", "density-2"} <= present
    assert not any(g.startswith("violin-") for g in present)
    assert "density outlines" in fig.result.warnings[-1]
    density = by_gid(fig, "density-0")
    assert isinstance(density, Polygon)
    assert not density.get_fill()


def test_many_values_without_points_are_not_subsampled() -> None:
    result = many_points_result()
    fig = betweenstats(result, layers=("violin", "box"))
    assert fig.subsample is None
    assert fig.result.warnings == result.warnings
    assert "points_subsampled" not in {i.code for i in fig.check()}


def test_subsample_and_jitter_follow_the_seed() -> None:
    result = many_points_result()
    a = np.asarray(_points(betweenstats(result), 0).get_offsets())
    b = np.asarray(_points(betweenstats(result), 0).get_offsets())
    assert np.array_equal(a, b)
    other = dataclasses.replace(
        result, settings=dataclasses.replace(result.settings, seed=7)
    )
    c = np.asarray(_points(betweenstats(other), 0).get_offsets())
    assert not np.array_equal(a, c)


def test_jitter_is_seeded_and_points_are_the_values() -> None:
    result = _tooth()
    a = np.asarray(_points(betweenstats(result), 0).get_offsets())
    b = np.asarray(_points(betweenstats(result), 0).get_offsets())
    assert np.array_equal(a, b)
    assert a[:, 1].tolist() == list(result.values[0])
    assert np.all(np.abs(a[:, 0]) <= 0.12)


# --------------------------------------------------------------------------- #
# ax=
# --------------------------------------------------------------------------- #


def _user_axes(width: float, height: float, n: int = 1) -> tuple[Figure, list[Axes]]:
    figure = Figure(figsize=(width, height), dpi=100)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, n, squeeze=False)[0]
    return figure, list(axes)


def test_draw_into_a_given_axes() -> None:
    figure, (ax,) = _user_axes(7.5, 7.0)
    fig = betweenstats(_plant(), ax=ax, title="Plant growth")
    assert fig.figure is figure
    assert fig.axes["main"] is ax
    texts = figure_texts(_plant())
    subtitle = by_gid(fig, "subtitle")
    assert isinstance(subtitle, Text)
    assert subtitle.get_text() == texts.subtitle
    assert subtitle in fig.axes["main"].texts  # inside the Axes
    assert numbered_texts(fig, "caption") == list(texts.caption_lines)
    assert fig.axes["main"].get_title(loc="left") == "Plant growth"
    assert figure_codes(fig) == []
    assert figure.get_size_inches().tolist() == [7.5, 7.0]


def test_given_axes_too_small_is_reported() -> None:
    _, (ax,) = _user_axes(2.5, 2.0)
    fig = betweenstats(_plant(), ax=ax)
    codes = figure_codes(fig)
    assert "text_clipped" in codes or "label_overlap" in codes
    message = " ".join(i.message for i in fig.check() if i.code in codes)
    assert "subtitle" in message


def test_two_panels_report_a_subtitle_wider_than_its_axes() -> None:
    figure, axes = _user_axes(9.0, 6.0, n=2)
    left = betweenstats(_plant(), ax=axes[0])
    betweenstats(_tooth(), ax=axes[1])
    clipped = [i for i in left.check() if i.code == "text_clipped"]
    assert clipped
    assert "outside it" in clipped[0].message
    assert len(figure.axes) == 2


def _three_groups() -> StatsResult:
    """The review's repro: three groups of 12, every pair shown."""
    rng = np.random.default_rng(0)
    groups = [rng.normal(i, 1, 12) for i in range(3)]
    return frame_result(["a", "b", "c"], groups, x="g", y="y", pairwise_display="all")


def _given_axes(kind: str) -> Axes:
    # 1.6 and 1.7 in high: too short for the three levels at full spacing, so the
    # levels are pushed closer together. (The review's 10 x 4 and 9 x 3.5 in
    # figures now fit them: the levels are spaced in pixels, not by 7% of the
    # data range.)
    if kind == "add_axes":
        figure = Figure(figsize=(10, 2.6))
        return figure.add_axes((0.08, 0.3, 0.9, 0.62))
    figure = Figure(figsize=(9, 2.4), layout="constrained")
    return figure.add_subplot()


@pytest.mark.parametrize("kind", ["add_axes", "constrained"])
def test_bracket_label_drawn_across_another_bracket_is_reported(kind: str) -> None:
    # A short Axes: the label of b-c sits on the bar of a-c. (The label of a-b
    # may cross that bar too; a pixel decides.)
    fig = betweenstats(_three_groups(), ax=_given_axes(kind))
    (issue,) = [i for i in fig.check() if i.code == "label_overlap"]
    assert issue.severity == "warning"
    assert re.match(
        r"[12] bracket labels? cross(es)? the lines? of (another bracket|other "
        r"brackets): ",
        issue.message,
    ), issue.message
    assert (
        "bracket label of 'b' and 'c' 'p_Holm-adj. = 0.02' crosses the bracket of "
        "'a' and 'c'"
    ) in issue.message
    assert "pairwise_display" in issue.fix
    # inferplot's own figure makes room for it.
    assert figure_codes(betweenstats(_three_groups())) == []


def _two_brackets(label_x: float, label_y: float) -> StatsFigure:
    """Brackets a-b (legs 5 to 6) and a-c (legs 8 to 9) in a 320-pixel-high Axes
    showing y from 0 to 10, with the label of a-b at (label_x, label_y)."""
    figure = Figure(figsize=(6.0, 4.0), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.0, 10.0)
    inner = ax.add_line(Line2D([0, 0, 1, 1], [5, 6, 6, 5]))
    outer = ax.add_line(Line2D([0, 0, 2, 2], [8, 9, 9, 8]))
    inner_label = ax.text(label_x, label_y, "p = 0.5", ha="center", va="bottom")
    outer_label = ax.text(1.0, 9.2, "p = 0.01", ha="center", va="bottom")
    return StatsFigure(
        figure=figure,
        axes={"main": ax},
        result=_plant(),
        own=False,
        roles={id(inner_label): "label a-b", id(outer_label): "label a-c"},
        inside=(),
        subsample=None,
        brackets=(
            (inner_label, inner, "bracket a-b"),
            (outer_label, outer, "bracket a-c"),
        ),
    )


@pytest.mark.parametrize(
    ("x", "y", "in_box", "crosses"),
    [
        (0.5, 6.2, False, False),  # above its own bar, below the other bracket
        (0.5, 8.1, True, False),  # inside a-c's box: between its legs, below its bar
        (1.0, 8.75, True, True),  # across a-c's bar
        (0.0, 8.2, True, True),  # across a-c's left leg
        (2.0, 8.2, True, True),  # across a-c's right leg
    ],
)
def test_bracket_crossing_needs_a_line_not_its_box(
    x: float, y: float, in_box: bool, crosses: bool
) -> None:
    fig = _two_brackets(x, y)
    found = [i.message for i in figure_issues(fig) if i.code == "label_overlap"]
    (label, _, _), (_, outer, _) = fig.brackets
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    box = label.get_window_extent(renderer)
    assert bool(box.overlaps(outer.get_window_extent(renderer))) is in_box
    if crosses:
        assert found == [
            "1 bracket label crosses the line of another bracket: label a-b crosses "
            "bracket a-c."
        ]
    else:
        assert found == []


def _move_box(fig: StatsFigure, label: Text, **edges: float) -> Bbox:
    """Move ``label`` so its drawn box has the given edges (pixels): x0, x1, cx
    for the horizontal and y0, y1 for the vertical; returns the drawn box."""
    ax = fig.axes["main"]
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    box = label.get_window_extent(renderer)
    dx = dy = 0.0
    if "x0" in edges:
        dx = edges["x0"] - box.x0
    if "x1" in edges:
        dx = edges["x1"] - box.x1
    if "cx" in edges:
        dx = edges["cx"] - (box.x0 + box.x1) / 2
    if "y0" in edges:
        dy = edges["y0"] - box.y0
    if "y1" in edges:
        dy = edges["y1"] - box.y1
    x, y = ax.transData.transform(label.get_position())
    label.set_position(tuple(ax.transData.inverted().transform((x + dx, y + dy))))
    fig.draw()
    return label.get_window_extent(renderer)


def _leg_stroke(fig: StatsFigure, line: Line2D, end: int) -> Bbox:
    """The drawn stroke of a bracket's left (end 0) or right (end 1) leg, caps in."""
    xy = line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float))
    half = float(line.get_linewidth()) * fig.figure.dpi / 72 / 2
    (xa, ya), (xb, yb) = (xy[0], xy[1]) if end == 0 else (xy[3], xy[2])
    return Bbox.from_extents(
        min(xa, xb) - half, min(ya, yb) - half, max(xa, xb) + half, max(ya, yb) + half
    )


@pytest.mark.parametrize("x", [0.0, 2.0])  # a-c's left and right leg
@pytest.mark.parametrize(("depth", "crosses"), [(0.3, False), (1.0, True)])
def test_bracket_crossing_allows_subpixel_contact(
    x: float, depth: float, crosses: bool
) -> None:
    # The box of label a-b, centered on a leg of a-c, reaches ``depth`` pixels
    # into the leg's projecting cap from below.
    fig = _two_brackets(x, 7.0)
    (label, _, _), (_, outer, _) = fig.brackets
    leg = _leg_stroke(fig, outer, 0 if x == 0.0 else 1)
    box = _move_box(fig, label, cx=(leg.x0 + leg.x1) / 2, y1=leg.y0 + depth)
    assert box.y1 - leg.y0 == pytest.approx(depth, abs=0.02)
    found = [i.message for i in figure_issues(fig) if i.code == "label_overlap"]
    reach, _ = stroke_reach(fig)
    assert bool(found) is crosses is (reach > 0.5)


@pytest.mark.parametrize("dpi", [72, 100, 200])
@pytest.mark.parametrize(
    ("where", "depth", "crosses"),
    [
        ("tip", 0.3, False),  # box top 0.3 px into the leg's cap: contact
        ("tip", 1.0, True),  # one pixel into the leg: drawn across it
        ("tip", 3.0, True),
        ("side", 0.3, False),  # box edge 0.3 px into the leg's side
        ("side", 1.0, True),  # one pixel into the leg's side
    ],
)
def test_a_given_axes_reports_a_label_across_a_leg(
    dpi: int, where: str, depth: float, crosses: bool
) -> None:
    # A tall Axes: nothing crosses. Then the label of b-c (lowest level) is moved
    # onto the right leg of a-c (the level above), as deep as given, and check()
    # must agree with the dense sampling of the drawn line.
    figure = Figure(figsize=(10.0, 7.0), dpi=dpi)
    FigureCanvasAgg(figure)
    fig = betweenstats(_three_groups(), ax=figure.add_subplot())
    assert figure_codes(fig) == []
    assert len(fig.brackets) == 3
    (label,) = [t for t, _, n in fig.brackets if n == "the bracket of 'b' and 'c'"]
    (outer,) = [ln for _, ln, n in fig.brackets if n == "the bracket of 'a' and 'c'"]
    leg = _leg_stroke(fig, outer, 1)
    if where == "tip":
        box = _move_box(fig, label, cx=(leg.x0 + leg.x1) / 2, y1=leg.y0 + depth)
        assert box.y1 - leg.y0 == pytest.approx(depth, abs=0.02)
    else:
        # Just below the bar of a-c, the box's right edge in the leg's left side.
        box = _move_box(fig, label, x1=leg.x0 + depth, y1=leg.y1 - leg.width - 1.0)
        assert box.x1 - leg.x0 == pytest.approx(depth, abs=0.02)
    reach, _ = stroke_reach(fig)
    found = [i.message for i in fig.check() if i.code == "label_overlap"]
    assert bool(found) is crosses is (reach > 0.5)
    if crosses:
        assert len(found) == 1
        assert "label of 'b' and 'c'" in found[0]
        assert "crosses the bracket of 'a' and 'c'" in found[0]


def test_a_tall_given_axes_fits_the_brackets() -> None:
    _, (ax,) = _user_axes(10.0, 7.0)
    fig = betweenstats(_three_groups(), ax=ax)
    assert len(fig.brackets) == 3
    assert figure_codes(fig) == []
    deepest, clearance = stroke_reach(fig)
    assert deepest == 0.0
    assert clearance >= 3.0  # a label pad and half a line width at 100 dpi


# --------------------------------------------------------------------------- #
# user texts and artist arguments
# --------------------------------------------------------------------------- #


def test_user_texts() -> None:
    result = fixture_result(case_named("PlantGrowth-nosubtitle"))
    fig = betweenstats(
        result,
        title="Plant weight",
        subtitle="Dried weight, three conditions",
        caption="Source: PlantGrowth (R datasets).",
        xlab="Condition",
        ylab="Weight (g)",
    )
    assert text_of(fig, "title") == "Plant weight"
    assert text_of(fig, "subtitle") == "Dried weight, three conditions"
    captions = numbered_texts(fig, "caption")
    assert captions == [
        *figure_texts(result).caption_lines,
        "Source: PlantGrowth (R datasets).",
    ]
    assert fig.axes["main"].get_xlabel() == "Condition"
    assert fig.axes["main"].get_ylabel() == "Weight (g)"
    assert figure_codes(fig) == []


def test_long_subtitle_widens_the_figure() -> None:
    result = fixture_result(case_named("ToothGrowth-nosubtitle"))
    narrow = betweenstats(result)
    wide = betweenstats(result, subtitle="A very long subtitle " * 8)
    assert wide.figure.get_size_inches()[0] > narrow.figure.get_size_inches()[0] + 3
    assert figure_codes(wide) == []


def test_every_text_uses_dejavu_sans() -> None:
    fig = betweenstats(_plant(), title="T", caption="C")
    fig.check()
    texts = [t for t in fig.figure.findobj(Text) if t.get_text()]
    assert len(texts) > 15
    for text in texts:
        assert text.get_fontfamily() == ["DejaVu Sans"], text.get_text()
        assert text.get_math_fontfamily() == "dejavusans", text.get_text()


def test_artist_args_are_applied() -> None:
    fig = betweenstats(
        _plant(),
        point_args={"alpha": 0.2, "sizes": [30.0]},
        boxplot_args={"facecolor": "lightgray", "linewidth": 2.0},
        violin_args={"alpha": 0.1},
    )
    assert _points(fig, 0).get_alpha() == 0.2
    box = by_gid(fig, "box-1")
    assert isinstance(box, PathPatch)
    assert box.get_linewidth() == 2.0
    violin = by_gid(fig, "violin-2")
    assert isinstance(violin, Polygon)
    assert violin.get_alpha() == 0.1


@pytest.mark.parametrize("name", ["point_args", "boxplot_args", "violin_args"])
def test_unknown_artist_property_raises(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        betweenstats(_plant(), **{name: {"not_a_property": 1}})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("name", "layers", "missing"),
    [
        ("point_args", ("box",), "points"),
        ("point_args", ("violin", "box"), "points"),
        ("boxplot_args", ("points",), "box"),
        ("boxplot_args", ("violin", "points"), "box"),
        ("violin_args", ("box", "points"), "violin"),
    ],
)
def test_artist_args_for_a_layer_not_drawn_raise(
    name: str, layers: tuple[str, ...], missing: str
) -> None:
    with pytest.raises(ValueError, match=f"^{name} styles the {missing} layer") as info:
        betweenstats(_plant(), layers=layers, **{name: {"alpha": 0.5}})  # pyright: ignore[reportArgumentType]
    message = str(info.value)
    assert f"layers={layers!r}" in message
    assert f"add {missing!r} to layers" in message
    # An empty dict styles nothing and is accepted.
    betweenstats(_plant(), layers=layers, **{name: {}})  # pyright: ignore[reportArgumentType]


def test_violin_args_do_not_style_density_outlines() -> None:
    # With many points and no violin layer, density outlines are drawn instead of
    # violins; violin_args has nothing to style.
    with pytest.raises(ValueError, match="violin_args styles the violin layer"):
        betweenstats(
            many_points_result(),
            layers=("box", "points"),
            violin_args={"alpha": 0.5},
        )


@pytest.mark.parametrize(
    ("name", "layers"),
    [
        ("point_args", ("points",)),
        ("boxplot_args", ("box",)),
        ("violin_args", ("violin",)),
    ],
)
def test_artist_args_are_checked_before_drawing(
    name: str, layers: tuple[str, ...]
) -> None:
    with pytest.raises(TypeError, match=f"{name} has a property"):
        betweenstats(_plant(), layers=layers, **{name: {"bogus": 1}})  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=f"{name} has a value matplotlib rejects"):
        betweenstats(_plant(), layers=layers, **{name: {"facecolor": "notacolor"}})  # pyright: ignore[reportArgumentType]


# Values that Artist.set() stores without checking; matplotlib rejects them only
# when it draws (Agg) or writes a vector file (PDF). Found by trying bad values for
# every settable property of the three artist classes.
_REJECTED_WHEN_DRAWN = [
    ("point_args", {"linewidth": "bad"}),
    ("point_args", {"linewidths": "bad"}),
    ("point_args", {"lw": (1, "a")}),
    ("point_args", {"linewidths": 1e9}),
    ("point_args", {"linewidth": float("nan")}),
    ("point_args", {"linewidths": 1.0, "linestyle": {}}),
    ("boxplot_args", {"linewidth": float("nan")}),
    ("violin_args", {"lw": float("nan")}),
]


@pytest.mark.parametrize("own", [True, False], ids=["own-figure", "ax"])
@pytest.mark.parametrize(
    ("name", "args"),
    _REJECTED_WHEN_DRAWN,
    ids=[f"{n}-{'-'.join(a)}" for n, a in _REJECTED_WHEN_DRAWN],
)
def test_a_value_rejected_only_when_drawn_raises_at_the_call(
    name: str, args: dict[str, object], own: bool
) -> None:
    figure, (ax,) = _user_axes(7.5, 7.0)
    ax.plot([0.0, 1.0], [0.0, 1.0])
    before = _state(ax)
    with pytest.raises(ValueError, match=f"^{name} has a value matplotlib rejects"):
        betweenstats(_plant(), ax=None if own else ax, **{name: args})  # pyright: ignore[reportArgumentType]
    assert _state(ax) == before
    figure.savefig(io.BytesIO(), format="pdf")  # the user's figure still saves


@pytest.mark.parametrize("name", ["point_args", "boxplot_args", "violin_args"])
def test_a_wrongly_typed_value_of_a_known_property_is_a_rejected_value(
    name: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{name} has a value matplotlib rejects"):
        betweenstats(_plant(), **{name: {"alpha": "x"}})  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=f"^{name} has a property") as info:
        betweenstats(_plant(), **{name: {"alpah": 0.5}})  # pyright: ignore[reportArgumentType]
    assert "'alpah'" in str(info.value)
    assert "Did you mean 'alpha'?" in str(info.value)


def test_artist_args_that_draw_are_accepted_with_ax() -> None:
    figure, (ax,) = _user_axes(7.5, 7.0)
    fig = betweenstats(
        _plant(),
        ax=ax,
        point_args={"linewidths": 0.5, "edgecolor": "black", "sizes": [20.0]},
        boxplot_args={"linewidth": 1.5, "linestyle": "--", "hatch": "//"},
        violin_args={"alpha": 0.4, "lw": 2.0},
    )
    assert np.asarray(_points(fig, 0).get_linewidth()).tolist() == [0.5]
    assert figure_codes(fig) == []
    figure.savefig(io.BytesIO(), format="svg")


def test_texts_the_user_drew_before_keep_their_font() -> None:
    figure, (ax,) = _user_axes(7.5, 7.0)
    user_title = ax.set_title("My panel", fontfamily="serif", loc="right")
    note = ax.annotate(
        "note", (0.02, 0.5), xycoords="axes fraction", fontfamily="monospace"
    )
    fig = betweenstats(_plant(), ax=ax, title="Plant growth")
    assert user_title.get_fontfamily() == ["serif"]
    assert note.get_fontfamily() == ["monospace"]
    assert user_title.get_text() == "My panel"
    # Every text inferplot drew is DejaVu Sans, and check() measures them.
    canvas = figure.canvas
    assert isinstance(canvas, FigureCanvasAgg)
    canvas.draw()
    own = [
        *ax.xaxis.findobj(Text),
        *ax.yaxis.findobj(Text),
        *(t for t in ax.texts if t is not note),
    ]
    title = by_gid(fig, "title")  # the title inferplot set (left)
    assert isinstance(title, Text)
    assert title.get_text() == "Plant growth"
    own = [t for t in [*own, title] if t.get_text()]
    assert len(own) > 10
    for text in own:
        assert text.get_fontfamily() == ["DejaVu Sans"], text.get_text()
        assert text.get_math_fontfamily() == "dejavusans", text.get_text()
    assert figure_codes(fig) == []


def test_check_still_reports_inferplots_texts_beside_user_texts() -> None:
    _, (ax,) = _user_axes(2.5, 2.0)
    ax.set_title("Mine", fontfamily="serif", loc="right")
    fig = betweenstats(_plant(), ax=ax)
    codes = figure_codes(fig)
    assert "text_clipped" in codes or "label_overlap" in codes
    message = " ".join(i.message for i in fig.check() if i.code in codes)
    assert "subtitle" in message


def _state(ax: Axes) -> tuple[object, ...]:
    return (
        len(ax.patches),
        len(ax.collections),
        len(ax.lines),
        len(ax.texts),
        ax.get_xlim(),
        ax.get_ylim(),
        ax.get_title(loc="left"),
        ax.get_title(),
        ax.get_xlabel(),
        ax.get_ylabel(),
        tuple(ax.get_xticks()),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"point_args": {"bogus": 1}},
        {"boxplot_args": {"bogus": 1}},
        {"violin_args": {"bogus": 1}},
        {"violin_args": {"facecolor": "notacolor"}},
        {"layers": ("box",), "point_args": {"alpha": 0.5}},
        {"title": r"$\frac$"},
        {"xlab": r"$\frac$"},
        {"caption": r"$\notacommand$"},
    ],
    ids=lambda kwargs: "-".join(kwargs),
)
def test_a_failed_call_leaves_the_given_axes_unchanged(
    kwargs: dict[str, object],
) -> None:
    _, (ax,) = _user_axes(6.0, 4.0)
    ax.plot([0.0, 1.0], [0.0, 1.0])
    before = _state(ax)
    with pytest.raises((TypeError, ValueError)):
        betweenstats(_plant(), ax=ax, **kwargs)  # pyright: ignore[reportArgumentType]
    assert _state(ax) == before


@pytest.mark.parametrize("own", [True, False], ids=["own-figure", "ax"])
@pytest.mark.parametrize("name", ["title", "subtitle", "caption", "xlab", "ylab"])
def test_invalid_mathtext_in_user_text_raises_at_the_call(name: str, own: bool) -> None:
    result = fixture_result(case_named("PlantGrowth-nosubtitle"))
    ax = None if own else _user_axes(7.5, 7.0)[1][0]
    with pytest.raises(
        ValueError, match=f"^{name} is drawn as matplotlib mathtext"
    ) as info:
        betweenstats(result, ax=ax, **{name: r"Mass $\frac$ in g"})  # pyright: ignore[reportArgumentType]
    message = str(info.value)
    assert "\\$" in message  # how to write a literal dollar sign


@pytest.mark.parametrize("own", [True, False], ids=["own-figure", "ax"])
def test_user_texts_are_mathtext(own: bool) -> None:
    result = fixture_result(case_named("PlantGrowth-nosubtitle"))
    ax = None if own else _user_axes(7.5, 7.0)[1][0]
    fig = betweenstats(
        result,
        ax=ax,
        title=r"Revenue in \$ by region (in \$1000s)",
        ylab=r"Weight $\mu$g",
        caption=r"$R^2$ in \$",
    )
    assert text_of(fig, "title") == r"Revenue in \$ by region (in \$1000s)"
    assert figure_codes(fig) == []


def test_constant_level_skips_its_violin_with_a_warning() -> None:
    # A level whose values are all equal has no density to draw. (The statistics
    # need a variance, so the values are swapped in after the analysis.)
    built = frame_result(
        ["a", "b", "c"],
        [np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.5, 4.0]), np.array([1.0, 1.5])],
    )
    flat = dataclasses.replace(
        built, values=((1.0, 2.0, 3.0), (2.0, 2.0, 2.0), (1.0, 1.5))
    )
    fig = betweenstats(flat)
    present = gids(fig)
    assert {"violin-0", "violin-2", "box-1", "points-1"} <= present
    assert "violin-1" not in present
    (warning,) = fig.result.warnings[len(flat.warnings) :]
    assert "violin of level 'b' is not drawn" in warning


# --------------------------------------------------------------------------- #
# argument errors
# --------------------------------------------------------------------------- #


def test_unknown_keyword_suggests_the_closest() -> None:
    with pytest.raises(TypeError, match="point_args"):
        betweenstats(_plant(), pointargs={})  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError, match="snake_case"):
        betweenstats(_plant(), **{"centrality.plotting": False})  # pyright: ignore[reportArgumentType]


def test_extra_positional_argument() -> None:
    with pytest.raises(TypeError, match="keyword-only"):
        betweenstats(_plant(), False)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    ("layers", "error"),
    [
        (("violin", "boxes"), ValueError),
        ((), ValueError),
        ("box", TypeError),
        (("box", "box"), ValueError),
    ],
)
def test_bad_layers(layers: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="layers"):
        betweenstats(_plant(), layers=layers)  # pyright: ignore[reportArgumentType]


def test_result_of_another_analysis() -> None:
    other = dataclasses.replace(_plant(), analysis="withinstats")
    with pytest.raises(TypeError, match=r"ip\.render\.withinstats"):
        betweenstats(other)
    with pytest.raises(TypeError, match=r"analyze\.betweenstats"):
        betweenstats({"analysis": "betweenstats"})  # pyright: ignore[reportArgumentType]


def test_subtitle_with_results_subtitle_raises() -> None:
    with pytest.raises(ValueError, match="results_subtitle=False"):
        betweenstats(_plant(), subtitle="mine")


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"centrality_plotting": 1}, TypeError),
        ({"xlab": 5}, TypeError),
        ({"title": ["a"]}, TypeError),
        ({"caption": 3.0}, TypeError),
        ({"point_args": [("alpha", 1)]}, TypeError),
        ({"ax": "axes"}, TypeError),
    ],
)
def test_bad_values(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        betweenstats(_plant(), **kwargs)  # pyright: ignore[reportArgumentType]


def test_inconsistent_result() -> None:
    broken = dataclasses.replace(_plant(), values=((1.0, 2.0),))
    with pytest.raises(ValueError, match="inconsistent"):
        betweenstats(broken)


def test_repr() -> None:
    assert "weight by group" in repr(betweenstats(_plant()))
