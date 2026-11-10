# NumPy's stubs type several functions taking array-likes as partly Unknown, which
# strict mode reports on every call; the arguments we pass are all float arrays.
# pyright: reportUnknownMemberType=false
"""Figure-level problems for ``fig.check()`` (SPEC section 8).

Measured on the drawn figure from the rendered bounding boxes of its texts:

* ``label_overlap`` (warning): two texts' boxes intersect, or a bracket label's
  box reaches into the line (bar or leg) of another bracket by more than
  anti-aliasing slack;
* ``text_clipped`` (warning): a text extends outside the figure, or a text drawn
  inside the Axes (labels, and the subtitle and caption when ``ax=`` was given)
  extends outside that Axes;
* ``points_subsampled`` (info): the points show a subsample of a large panel;
* ``path_hidden`` (warning, withinstats): a text is drawn over the centrality path
  or a subject path for more than a pixel of its length;
* ``paths_crowded`` (warning, withinstats): the subject paths are at least half
  opaque over more than 15% of the area between the two conditions, measured on
  the paths alone rendered as drawn, so they hide the boxes and violins beneath
  them;
* ``line_label_ambiguous`` (warning, histostats): the label of the centrality
  line or of the test value line lies nearer to the other line than to its
  own, or the other line runs through it;
* ``figure_oversized`` (warning): inferplot's own figure is far larger than the
  plotting area it was laid out around (the panels fill under 25% of it) or
  larger than any of its layouts needs (30 in a side, plus 3 in a level and 0.5
  in a bracket for the comparison figures); the message names the long texts
  that widen it, or says that the layout went wrong;
* ``centrality_label_ambiguous`` (warning): a centrality label that reads as
  another level's (:func:`inferplot.render.labels.ambiguities`): mostly beyond
  the neighbouring point on its usual side, past the middle toward the
  neighbouring point on the other side, or over the centrality path between two
  other conditions.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, Transform
from numpy.typing import NDArray

from inferplot.check import Issue
from inferplot.render.figure import (
    describe_text,
    drawn_before,
    plain_text,
    root_figure,
    shows_all,
)
from inferplot.render.labels import ambiguities, covered
from inferplot.render.view_checks import view_issues

if TYPE_CHECKING:
    from matplotlib.backend_bases import RendererBase
    from matplotlib.collections import LineCollection

    from inferplot.render.figure import DrawnPaths, PointLabels, StatsFigure

__all__ = ["figure_issues", "text_extents"]

# Pixels a text may cross a border, or reach into a bracket's line, before it
# counts (anti-aliasing slack).
_CLIP_TOLERANCE: Final = 0.5
_MAX_LISTED: Final = 5
# Pixels of a path's length a text may cover before it counts (an end of the path
# touching a text's box by anti-aliasing slack).
_HIDDEN_TOLERANCE: Final = 1.0
# The subject paths are crowded when they are at least _CROWDED_OPACITY opaque over
# more than _CROWDED_SHARE of the area between the two conditions (calibrated by
# eye on figures of 10 to 2,500 subjects: 200 subjects cover 10% and still read,
# 300 cover 18% and hide the boxes and the centrality path).
_CROWDED_OPACITY: Final = 0.5
_CROWDED_SHARE: Final = 0.15


def _renderer(fig: StatsFigure) -> RendererBase | None:
    get = getattr(root_figure(fig.figure).canvas, "get_renderer", None)
    return cast("RendererBase", get()) if callable(get) else None


def _hidden_tick_labels(axis: Axis) -> set[int]:
    """ids of tick labels that are not drawn (tick outside the view or hidden)."""
    lo, hi = sorted(float(v) for v in axis.get_view_interval())
    eps = (hi - lo) * 1e-10
    hidden: set[int] = set()
    for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
        loc = float(tick.get_loc())
        if not (tick.get_visible() and lo - eps <= loc <= hi + eps):
            hidden.update((id(tick.label1), id(tick.label2)))
    return hidden


def _axis_roles(ax: Axes) -> dict[int, str]:
    roles: dict[int, str] = {}
    for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        roles[id(axis.label)] = f"{name}-axis label"
        for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
            roles[id(tick.label1)] = f"{name} tick label"
            roles[id(tick.label2)] = f"{name} tick label"
    return roles


def text_extents(fig: StatsFigure) -> list[tuple[Text, str, Bbox]]:
    """Every drawn, non-empty text with its description and rendered box (pixels).

    Call after drawing. With inferplot's own figure every text of the figure
    counts; with ``ax=`` only the texts of that Axes.
    """
    ax = fig.axes["main"]
    owner = root_figure(fig.figure) if fig.own else ax
    renderer = _renderer(fig)
    hidden: set[int] = set()
    roles: dict[int, str] = {}
    panels = [ax] if not fig.own else list(root_figure(fig.figure).axes)
    # With the secondary axes of each (histostats' percent axis).
    panels += [child for panel in panels for child in _children(panel)]
    for axes in panels:
        if not axes.axison:  # set_axis_off(): no axis text is drawn
            hidden |= {
                id(t) for t in [*axes.xaxis.findobj(Text), *axes.yaxis.findobj(Text)]
            }
            roles |= _axis_roles(axes)
            continue
        if axes.xaxis.get_visible():
            hidden |= _hidden_tick_labels(axes.xaxis)
        else:
            hidden |= {id(t) for t in axes.xaxis.findobj(Text)}
        if axes.yaxis.get_visible():
            hidden |= _hidden_tick_labels(axes.yaxis)
        else:
            hidden |= {id(t) for t in axes.yaxis.findobj(Text)}
        roles |= _axis_roles(axes)
    out: list[tuple[Text, str, Bbox]] = []
    for item in owner.findobj(Text):
        text = item
        if id(text) in hidden:
            continue
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        patch = text.get_bbox_patch()
        if patch is not None:
            box = Bbox.union([box, patch.get_window_extent(renderer)])
        described = fig.describe(text) or describe_text(
            roles.get(id(text), "text"), plain_text(text)
        )
        out.append((text, described, box))
    return out


def _children(axes: Axes) -> list[Axes]:
    """The child Axes of ``axes`` (secondary axes), visible or not."""
    return list(cast("list[Axes]", getattr(axes, "child_axes", [])))


def _axes_hidden(axes: Axes) -> set[int]:
    """ids of an Axes' axis texts that are not drawn."""
    if not axes.axison:
        return {id(t) for t in [*axes.xaxis.findobj(Text), *axes.yaxis.findobj(Text)]}
    hidden: set[int] = set()
    for axis in (axes.xaxis, axes.yaxis):
        if axis.get_visible():
            hidden |= _hidden_tick_labels(axis)
        else:
            hidden |= {id(t) for t in axis.findobj(Text)}
    return hidden


def _title_roles(axes: Axes) -> dict[int, str]:
    roles: dict[int, str] = {}
    for name in ("title", "_left_title", "_right_title"):
        text = getattr(axes, name, None)
        if isinstance(text, Text):
            roles[id(text)] = "title"
    return roles


def _other_extents(fig: StatsFigure) -> list[tuple[str, Bbox]]:
    """With ``ax=``: the plotting area and drawn texts of every other Axes of the
    figure, and the figure's own texts (a suptitle), in words, with their boxes.

    An Axes at the main Axes' position (a twin) has no plotting area of its own
    there; its texts still count.
    """
    ax = fig.axes["main"]
    root = root_figure(fig.figure)
    renderer = _renderer(fig)
    out: list[tuple[str, Bbox]] = []
    in_axes: set[int] = set()
    main_frame = ax.get_window_extent(renderer)
    for other in root.get_axes():
        texts = other.findobj(Text)
        in_axes |= {id(t) for t in texts}
        if other is ax or not other.get_visible():
            continue
        frame = other.get_window_extent(renderer)
        if not np.allclose(frame.extents, main_frame.extents, atol=0.5):
            out.append(("the plotting area of another panel", frame))
        hidden = _axes_hidden(other)
        roles = _axis_roles(other) | _title_roles(other)
        for child in _children(other):
            hidden |= _axes_hidden(child)
            roles |= _axis_roles(child)
        for text in texts:
            if id(text) in hidden or not text.get_visible():
                continue
            if not text.get_text().strip():
                continue
            # A text another inferplot drawing made is named with its plain
            # wording (a subtitle reads as text, not as squeezed mathtext).
            described = drawn_before(text) or describe_text(
                roles.get(id(text), "text"), plain_text(text)
            )
            out.append(
                (f"{described} of another panel", text.get_window_extent(renderer))
            )
    suptitle = getattr(root, "_suptitle", None)
    for text in root.findobj(Text):
        if id(text) in in_axes or not text.get_visible():
            continue
        if not text.get_text().strip() or fig.describe(text) is not None:
            continue
        role = "figure title" if text is suptitle else "figure text"
        out.append(
            (describe_text(role, plain_text(text)), text.get_window_extent(renderer))
        )
    return out


def _panel_overlaps(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> list[str]:
    """With ``ax=``: this panel's texts that overlap another panel or its texts,
    or a figure text."""
    if fig.own:
        return []
    others = _other_extents(fig)
    found: list[str] = []
    for _, described, box in extents:
        for other, frame in others:
            width = min(box.x1, frame.x1) - max(box.x0, frame.x0)
            height = min(box.y1, frame.y1) - max(box.y0, frame.y0)
            if width > _CLIP_TOLERANCE and height > _CLIP_TOLERANCE:
                found.append(f"{described} and {other}")
    return found


def _layout_cause(fig: StatsFigure) -> str:
    """With ``ax=`` in a figure under constrained or tight layout: the advice that
    names the layout engine as the cause; empty otherwise."""
    if fig.own:
        return ""
    engine = type(root_figure(fig.figure).get_layout_engine()).__name__
    names = {
        "ConstrainedLayoutEngine": ("constrained", "layout='constrained'"),
        "TightLayoutEngine": ("tight", "layout='tight' or tight_layout()"),
    }
    if engine not in names:
        return ""
    name, how = names[engine]
    return (
        f" The figure uses matplotlib's {name} layout, which counts the texts "
        "inferplot draws inside the Axes (subtitle, caption, labels) as its "
        "decorations and shrinks the Axes to make room for any wider than it, "
        "down to nothing: make the figure larger, or make it without "
        f"{how} and place the Axes yourself."
    )


def _listing(items: list[str]) -> str:
    listed = "; ".join(items[:_MAX_LISTED])
    if len(items) > _MAX_LISTED:
        listed += f"; and {len(items) - _MAX_LISTED} more"
    return listed


def _overlaps(
    extents: list[tuple[Text, str, Bbox]], apart: frozenset[int] = frozenset()
) -> list[str]:
    """Pairs of texts whose boxes intersect, except pairs of texts both in
    ``apart`` (point labels, which ``point_labels_crowded`` reports)."""
    found: list[str] = []
    for a in range(len(extents)):
        ta, da, ba = extents[a]
        for b in range(a + 1, len(extents)):
            tb, db, bb = extents[b]
            if id(ta) in apart and id(tb) in apart:
                continue
            width = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            height = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if width > 0 and height > 0:
                found.append(f"{da} and {db}")
    return found


def _reaches_into(box: BboxBase, stroke: BboxBase) -> bool:
    """Whether a text box reaches into a stroke by more than anti-aliasing slack.

    In each direction the overlap must exceed _CLIP_TOLERANCE, or half the
    stroke's extent in that direction when the stroke is thinner than that (a
    0.8-point line at low dpi), so that a box drawn across a line of any width
    counts and a box touching its edge or the end of its cap by a fraction of a
    pixel does not.
    """
    width = min(box.x1, stroke.x1) - max(box.x0, stroke.x0)
    height = min(box.y1, stroke.y1) - max(box.y0, stroke.y0)
    return width > min(_CLIP_TOLERANCE, stroke.width / 2) and height > min(
        _CLIP_TOLERANCE, stroke.height / 2
    )


def _segments(line: Line2D, dpi: float) -> list[Bbox]:
    """Each straight piece of a drawn line, as a box in pixels as thick as the line.

    The box reaches half the line width past each end: bracket lines have
    matplotlib's default projecting cap style.
    """
    # matplotlib's stubs leave these partly Unknown; both are float arrays.
    data = np.asarray(cast("Any", line).get_xydata(), dtype=np.float64)
    pixels = np.asarray(cast("Any", line.get_transform()).transform(data))
    points = [(float(x), float(y)) for x, y in pixels]
    half = float(line.get_linewidth()) * dpi / 72.0 / 2.0
    return [
        Bbox.from_extents(
            min(x0, x1) - half,
            min(y0, y1) - half,
            max(x0, x1) + half,
            max(y0, y1) + half,
        )
        for (x0, y0), (x1, y1) in itertools.pairwise(points)
    ]


def _crossings(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> list[tuple[int, str]]:
    """Bracket labels whose box crosses the bar or a leg of another bracket.

    One entry per label and bracket crossed: the label's position among the
    brackets and the crossing in words. A label may sit inside the area an
    enclosing bracket spans (between its legs, below its bar); only the drawn
    lines count, and only when the label's box reaches into one by more than
    anti-aliasing slack (:func:`_reaches_into`).
    """
    boxes = {id(text): (described, box) for text, described, box in extents}
    dpi = float(root_figure(fig.figure).dpi)
    lines = [
        _segments(line, dpi) if line.get_visible() and line.axes is not None else []
        for _, line, _ in fig.brackets
    ]
    found: list[tuple[int, str]] = []
    for n, (label, _, _) in enumerate(fig.brackets):
        drawn = boxes.get(id(label))
        if drawn is None:
            continue
        described, box = drawn
        for m, (_, _, bracket) in enumerate(fig.brackets):
            if m != n and any(_reaches_into(box, piece) for piece in lines[m]):
                found.append((n, f"{described} crosses {bracket}"))
    return found


def _outside(box: BboxBase, frame: BboxBase) -> bool:
    tol = _CLIP_TOLERANCE
    return (
        box.x0 < frame.x0 - tol
        or box.y0 < frame.y0 - tol
        or box.x1 > frame.x1 + tol
        or box.y1 > frame.y1 + tol
    )


def _clipped(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> tuple[list[str], list[str]]:
    frame = root_figure(fig.figure).bbox
    axes_frame = fig.axes["main"].bbox
    inside = {id(t) for t in fig.inside}
    out_of_figure: list[str] = []
    out_of_axes: list[str] = []
    for text, described, box in extents:
        if _outside(box, frame):
            out_of_figure.append(described)
        elif id(text) in inside and _outside(box, axes_frame):
            out_of_axes.append(described)
    return out_of_figure, out_of_axes


FloatArray = NDArray[np.float64]


def _pixels(ax: Axes, points: FloatArray) -> FloatArray:
    return np.asarray(cast("Any", ax.transData).transform(points), dtype=np.float64)


def _hidden(
    fig: StatsFigure, paths: DrawnPaths, extents: list[tuple[Text, str, Bbox]]
) -> list[str]:
    """The paths a text is drawn over, in words."""
    ax = fig.axes["main"]
    names = [described for _, described, _ in extents]
    boxes = [box for _, _, box in extents]
    found: list[str] = []
    line = paths.centrality
    if line is not None and line.get_visible() and line.axes is not None:
        xy = _pixels(ax, np.asarray(cast("Any", line).get_xydata(), dtype=np.float64))
        lengths = covered(np.stack([xy[:-1], xy[1:]], axis=1), boxes)
        for j in range(lengths.shape[0]):
            texts = [names[b] for b in np.flatnonzero(lengths[j] > _HIDDEN_TOLERANCE)]
            if texts:
                found.append(
                    f"the centrality path between {paths.levels[j]!r} and "
                    f"{paths.levels[j + 1]!r} runs under the {', the '.join(texts)}"
                )
    subjects = paths.subjects
    if subjects is not None and subjects.get_visible() and subjects.axes is not None:
        segments = [np.asarray(s, dtype=np.float64) for s in subjects.get_segments()]
        if segments:
            xy = np.stack([_pixels(ax, s) for s in segments])  # (paths, levels, 2)
            pieces = np.stack([xy[:, :-1], xy[:, 1:]], axis=2).reshape(-1, 2, 2)
            lengths = covered(pieces, boxes)
            per_path = lengths.reshape(xy.shape[0], -1, len(boxes)).sum(axis=1)
            under = per_path > _HIDDEN_TOLERANCE
            if under.any():
                texts = [names[b] for b in np.flatnonzero(under.any(axis=0))]
                count = int(under.any(axis=1).sum())
                found.append(
                    f"{count} of the {xy.shape[0]} subject paths "
                    f"{'runs' if count == 1 else 'run'} under the "
                    f"{', the '.join(texts)}"
                )
    return found


def crowding(fig: StatsFigure, subjects: LineCollection) -> float:
    """Share of the area between the two conditions where the subject paths, drawn
    alone as they are drawn in the figure, are at least _CROWDED_OPACITY opaque.

    The area spans the two levels' positions across and the drawn paths' range of
    y, within the Axes. Call after drawing.
    """
    segments = [np.asarray(s, dtype=np.float64) for s in subjects.get_segments()]
    if not segments:
        return 0.0
    figure = root_figure(fig.figure)
    ax = fig.axes["main"]
    width, height = (int(np.ceil(float(size))) for size in figure.bbox.size)
    renderer = RendererAgg(width, height, figure.dpi)
    subjects.draw(renderer)
    # The buffer's first row is the top of the figure.
    opacity = np.asarray(renderer.buffer_rgba())[::-1, :, 3] / 255.0
    ys = _pixels(ax, np.concatenate(segments))[:, 1]
    xs = _pixels(ax, np.array([[0.0, 0.0], [1.0, 0.0]]))[:, 0]
    frame = ax.bbox
    x0 = max(int(np.floor(float(xs.min()))), int(np.floor(frame.x0)), 0)
    x1 = min(int(np.ceil(float(xs.max()))), int(np.ceil(frame.x1)), width)
    y0 = max(int(np.floor(float(ys.min()))), int(np.floor(frame.y0)), 0)
    y1 = min(int(np.ceil(float(ys.max()))), int(np.ceil(frame.y1)), height)
    region = opacity[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float((region >= _CROWDED_OPACITY).mean())


def _path_issues(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> list[Issue]:
    """path_hidden and paths_crowded (withinstats)."""
    paths = fig.paths
    if paths is None:
        return []
    issues: list[Issue] = []
    hidden = _hidden(fig, paths, extents)
    if hidden:
        issues.append(
            Issue(
                code="path_hidden",
                severity="warning",
                message=f"A text is drawn over a path: {'; '.join(hidden)}.",
                fix=(
                    "Give the plot more room (a larger Axes passed as ax=, or "
                    "inferplot's own figure by leaving ax out), shorten the texts, "
                    "or leave the path out with point_path=False or "
                    "centrality_path=False."
                ),
            )
        )
    subjects = paths.subjects
    if subjects is not None and subjects.get_visible() and subjects.axes is not None:
        share = crowding(fig, subjects)
        if share > _CROWDED_SHARE:
            n = len(subjects.get_segments())
            issues.append(
                Issue(
                    code="paths_crowded",
                    severity="warning",
                    message=(
                        f"The {n:,} subject paths are at least half opaque over "
                        f"{share:.0%} of the area between the two conditions, so "
                        "they hide the boxes and violins beneath them and single "
                        "subjects cannot be followed."
                    ),
                    fix=(
                        "Pass point_path=False to draw the points without paths "
                        "(the centrality path still shows how the centre changes), "
                        "or draw into a larger Axes with ax=."
                    ),
                )
            )
    return issues


def _ambiguous_labels(fig: StatsFigure) -> Issue | None:
    """Centrality labels that read as another level's, wherever they are now."""
    renderer = _renderer(fig)
    ax = fig.axes["main"]
    shown = [
        label
        for label in fig.labels
        if label.text.get_visible()
        and label.text.axes is not None
        and label.point.get_visible()
        and label.point.axes is not None
    ]
    if len(shown) < 2:
        return None
    points: dict[int, float] = {}
    for label in fig.labels:
        if label.point.get_visible() and label.point.axes is not None:
            offsets = np.asarray(
                cast("Any", label.point).get_offsets(), dtype=np.float64
            )
            transform = label.point.get_offset_transform()
            points[label.index] = float(transform.transform(offsets)[0][0])
    path: FloatArray | None = None
    line = fig.paths.centrality if fig.paths is not None else None
    if line is not None and line.get_visible() and line.axes is not None:
        xy = _pixels(ax, np.asarray(cast("Any", line).get_xydata(), dtype=np.float64))
        path = np.stack([xy[:-1], xy[1:]], axis=1)
    levels = fig.result.levels
    found: list[str] = []
    for label in shown:
        if label.index not in points:
            continue
        box = label.text.get_window_extent(renderer)
        patch = label.text.get_bbox_patch()
        if patch is not None:
            box = Bbox.union([box, patch.get_window_extent(renderer)])
        name = f"the centrality label of {label.level!r}"
        for why in ambiguities(label.index, box, label.usual, points, path):
            other = levels[why.other]
            if why.kind == "beyond":
                found.append(
                    f"{name} lies mostly beyond the centrality point of {other!r}"
                )
            elif why.kind == "beside":
                found.append(
                    f"{name} sits beside the centrality point of {other!r}, past "
                    "the middle between the two points"
                )
            else:
                found.append(
                    f"{name} lies over the centrality path between {other!r} and "
                    f"{levels[why.other + 1]!r}"
                )
    if not found:
        return None
    if fig.own:
        room = (
            "draw it into a wider Axes of your own with ax= (made with a larger "
            "figsize), where each label fits between its point and the next"
        )
    else:
        room = (
            "make this Axes wider (a larger figsize when you create the figure, or "
            "fewer panels side by side), so that each label fits between its point "
            "and the next"
        )
    return Issue(
        code="centrality_label_ambiguous",
        severity="warning",
        message=(
            f"A centrality label can be read as another level's: {_listing(found)}."
        ),
        fix=(
            f"To keep the labels, {room}, or show fewer digits with the analysis' "
            "digits= (for example digits=1). Or leave the labels out with "
            "centrality_plotting=False and report the values from "
            "fig.result.descriptives. If you moved a label, move it back beside "
            "its own point."
        ),
    )


def _distance(box: BboxBase, piece: FloatArray) -> float:
    """Pixels between a box and a straight piece of line ((2, 2): its ends),
    sampled along the piece; 0 where the line runs through the box."""
    t = np.linspace(0.0, 1.0, 65)[:, None]
    points = piece[0] + t * (piece[1] - piece[0])
    dx = np.maximum(np.maximum(box.x0 - points[:, 0], points[:, 0] - box.x1), 0.0)
    dy = np.maximum(np.maximum(box.y0 - points[:, 1], points[:, 1] - box.y1), 0.0)
    return float(np.hypot(dx, dy).min())


def _line_distance(box: BboxBase, line: Line2D) -> float:
    data = np.asarray(cast("Any", line).get_xydata(), dtype=np.float64)
    xy = np.asarray(cast("Any", line.get_transform()).transform(data), dtype=np.float64)
    return min(
        (_distance(box, np.stack([a, b])) for a, b in itertools.pairwise(xy)),
        default=float("inf"),
    )


def _ambiguous_line_labels(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> Issue | None:
    """``line_label_ambiguous``: a label of one of histostats' lines that lies
    nearer to another line than to its own, or that another line runs through."""
    boxes = {id(text): (described, box) for text, described, box in extents}
    shown = [
        label
        for label in fig.line_labels
        if id(label.text) in boxes
        and label.line.get_visible()
        and label.line.axes is not None
    ]
    if len(shown) < 2:
        return None
    dpi = float(root_figure(fig.figure).dpi)
    to_pt = 72.0 / dpi
    found: list[str] = []
    for label in shown:
        described, box = boxes[id(label.text)]
        own = _line_distance(box, label.line)
        for other in shown:
            if other is label:
                continue
            near = _line_distance(box, other.line)
            if near == 0.0:
                found.append(f"{other.name} runs through the {described}")
            elif near + _CLIP_TOLERANCE < own:
                found.append(
                    f"the {described} lies nearer to {other.name} ({near * to_pt:.0f} "
                    f"pt) than to its own, {label.name} ({own * to_pt:.0f} pt)"
                )
    if not found:
        return None
    return Issue(
        code="line_label_ambiguous",
        severity="warning",
        message=f"A line's label can be read as another line's: {_listing(found)}.",
        fix=(
            "Give the plot more room (a wider Axes passed as ax=, made with a larger "
            "figsize, or inferplot's own figure by leaving ax out) so that each "
            "label fits beside its own line; if you moved a label, move it back "
            "beside its line. Or leave the centrality line out with "
            "centrality_plotting=False and report the value from "
            "fig.result.descriptives."
        ),
    )


def _crossed_labels(
    fig: StatsFigure,
    labels: PointLabels,
    boxes: dict[int, tuple[str, Bbox]],
) -> tuple[list[str], list[str]]:
    """The point labels (in words) that the least-squares line, and that the
    link of another label, crosses as drawn."""
    from inferplot.render.point_labels import crossing
    from inferplot.render.scatterstats import line_strokes

    dpi = float(root_figure(fig.figure).dpi)
    shown = [(t, boxes[id(t)]) for t in labels.texts if id(t) in boxes]
    if not shown:
        return [], []
    extents = np.array([[b.x0, b.y0, b.x1, b.y1] for _, (_, b) in shown])
    by_line: list[str] = []
    strokes = line_strokes(labels.line, dpi)
    if strokes.shape[0]:
        hit = crossing(strokes, extents).any(axis=0)
        by_line = [d for (_, (d, _)), h in zip(shown, hit, strict=True) if h]
    links: list[FloatArray] = []
    owners: list[int] = []
    for link, owner in labels.links:
        if not link.get_visible() or link.axes is None:
            continue
        ends: list[FloatArray] = []
        # ConnectionPatch keeps its two ends and their coordinates as attributes.
        drawn = cast("Any", link)
        for xy, coords in ((drawn.xy1, drawn.coords1), (drawn.xy2, drawn.coords2)):
            if not isinstance(coords, Transform):
                break
            ends.append(np.asarray(coords.transform(xy), dtype=np.float64))
        if len(ends) != 2:
            continue
        start, end = ends
        length = float(np.hypot(*(end - start)))
        if length == 0.0:
            continue
        shrink = float(drawn.shrinkA) * dpi / 72.0
        start = start + (end - start) * min(1.0, shrink / length)
        half = float(link.get_linewidth()) * dpi / 72.0 / 2.0
        links.append(np.array([*start, *end, half]))
        owners.append(id(owner))
    by_link: list[str] = []
    if links:
        hit = crossing(np.array(links), extents)
        for j, (text, (described, _)) in enumerate(shown):
            if any(hit[k, j] and owners[k] != id(text) for k in range(len(links))):
                by_link.append(described)
    return by_line, by_link


def _crowded_labels(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> Issue | None:
    """scatterstats' point labels that overlap each other, or that were too many
    to place. (Labels on points of a dense cloud are not reported: the placement
    covers as few points as it can, and some labels must lie over points.)"""
    labels = fig.point_labels
    if labels is None:
        return None
    boxes = {id(text): (described, box) for text, described, box in extents}
    shown = [boxes[id(t)] for t in labels.texts if id(t) in boxes]
    found: list[str] = []
    pairs = [
        f"{da} and {db}"
        for (da, ba), (db, bb) in itertools.combinations(shown, 2)
        if min(ba.x1, bb.x1) - max(ba.x0, bb.x0) > 0
        and min(ba.y1, bb.y1) - max(ba.y0, bb.y0) > 0
    ]
    if pairs:
        noun = "pair of point labels overlaps" if len(pairs) == 1 else (
            "pairs of point labels overlap"
        )  # fmt: skip
        found.append(f"{len(pairs)} {noun}: {_listing(pairs)}")
    by_line, by_link = _crossed_labels(fig, labels, boxes)
    if by_line:
        noun = "point label lies" if len(by_line) == 1 else "point labels lie"
        found.append(
            f"{len(by_line)} {noun} across the least-squares line: {_listing(by_line)}"
        )
    if by_link:
        noun = "point label is" if len(by_link) == 1 else "point labels are"
        found.append(
            f"{len(by_link)} {noun} crossed by the line joining another label "
            f"to its point: {_listing(by_link)}"
        )
    if labels.total > labels.limit:
        found.append(
            f"{labels.total:,} points are labelled, more than the {labels.limit:,} "
            "inferplot places clear of each other and of the points, so the labels "
            "are drawn beside their points without avoiding anything"
        )
    if not found:
        return None
    return Issue(
        code="point_labels_crowded",
        severity="warning",
        message="The point labels are crowded: " + "; ".join(found) + ".",
        fix=(
            "Label fewer points: set the label column (label_var) to None or '' "
            "for the points you do not need to name, as ggstatsplot's "
            "label.expression selects them; or draw into a larger Axes with ax= "
            "(made with a larger figsize, marginal=False). A label across the "
            "least-squares line also clears it with smooth=False, which leaves "
            "the line out."
        ),
    )


# inferplot's own figure: the plotting area (every panel) fills at least this
# share of the figure, and the figure is at most _OVERSIZED_IN on a side (plus,
# for the comparison figures, room for their levels and brackets). Measured on
# every fixture figure (2026-11-10): the share is 0.55 to 0.70 for all four
# functions (betweenstats 0.55 to 0.70, withinstats 0.56 to 0.67, histostats
# 0.55 to 0.61, scatterstats 0.55 to 0.68, also over 300 random data sets from
# 1e-6 to 1e9), the largest figure 12.2 x 4.6 in (nine levels). The bug these
# catch made 80 to 19,000 in wide histograms whose Axes filled under 5%.
_MIN_PLOT_SHARE: Final = 0.25
_OVERSIZED_IN: Final = 30.0
_PER_LEVEL_IN: Final = 3.0  # more than the widest level spacing seen (1.4 in)
_PER_BRACKET_IN: Final = 0.5  # more than one bracket level (about 0.3 in)
# A text longer than this (inches) is named as the cause; inferplot's own
# texts (subtitles with large numbers) stay under 8 in.
_LONG_TEXT_IN: Final = 12.0


def _oversized(fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]) -> Issue | None:
    """``figure_oversized``: inferplot's own figure much larger than the plotting
    area it was laid out around, or larger than any of its layouts needs."""
    if not fig.own:
        return None
    root = root_figure(fig.figure)
    renderer = _renderer(fig)
    dpi = float(root.dpi)
    frame = Bbox.union([a.get_window_extent(renderer) for a in fig.axes.values()])
    width, height = (float(v) for v in root.get_size_inches())
    share = frame.width * frame.height / (width * height * dpi * dpi)
    max_w = max_h = _OVERSIZED_IN
    if not fig.data:  # betweenstats, withinstats: room for levels and brackets
        max_w += _PER_LEVEL_IN * len(fig.result.levels)
        max_h += _PER_BRACKET_IN * len(fig.brackets)
    if share >= _MIN_PLOT_SHARE and width <= max_w and height <= max_h:
        return None
    # Texts longer than any inferplot draws itself, which the figure grows to
    # hold (a title or caption line is not wrapped).
    long = sorted(
        (
            (max(box.width, box.height) / dpi, described)
            for _, described, box in extents
            if max(box.width, box.height) / dpi > _LONG_TEXT_IN
        ),
        key=lambda r: -r[0],
    )
    if long:
        why = "the figure grows to hold " + _listing(
            [f"the {d} ({r:.1f} in long)" for r, d in long]
        )
        fix = (
            "Shorten those texts (title=, caption=, xlab=, ylab=, or the column "
            "names), or draw into an Axes of your own with ax=, where the figure "
            "keeps its size."
        )
    else:
        why = (
            "no text needs that room, so inferplot's layout went wrong; please "
            "report it as an inferplot bug, with the call that drew this figure"
        )
        fix = (
            "Draw into an Axes of your own with ax= (for example "
            "Figure(figsize=(7, 5)).add_subplot()), where the figure keeps its "
            "size, and report the call."
        )
    return Issue(
        code="figure_oversized",
        severity="warning",
        message=(
            f"inferplot's own figure is {width:.1f} x {height:.1f} in, and its "
            f"plotting area fills {share:.0%} of it (at least "
            f"{_MIN_PLOT_SHARE:.0%} expected, at most {max_w:.0f} x {max_h:.0f} "
            f"in): {why}."
        ),
        fix=fix,
    )


def figure_issues(fig: StatsFigure) -> tuple[Issue, ...]:
    """Rendering issues: label_overlap, text_clipped, points_subsampled, and for
    withinstats path_hidden and paths_crowded."""
    fig.draw()
    extents = text_extents(fig)
    issues: list[Issue] = []
    labels = fig.point_labels
    apart: frozenset[int] = frozenset(
        id(t) for t in (labels.texts if labels is not None else ())
    )
    overlaps = _overlaps(extents, apart)
    crossings = _crossings(fig, extents)
    room = (
        "Give the plot more room (a larger Axes passed as ax=, or inferplot's own "
        "figure by leaving ax out), shorten the texts, or draw fewer brackets "
        "with pairwise_display='significant' or 'none'."
    ) + _layout_cause(fig)
    found: list[str] = []
    if overlaps:
        noun = (
            "pair of texts overlaps" if len(overlaps) == 1 else "pairs of texts overlap"
        )
        found.append(f"{len(overlaps)} {noun}: {_listing(overlaps)}")
    if crossings:
        labels = len({n for n, _ in crossings})
        noun = (
            "bracket label crosses the line of another bracket"
            if labels == 1
            else "bracket labels cross the lines of other brackets"
        )
        listed = _listing([words for _, words in crossings])
        found.append(f"{labels} {noun}: {listed}")
    across = _panel_overlaps(fig, extents)
    if across:
        noun = (
            "text of this panel overlaps"
            if len(across) == 1
            else "texts of this panel overlap"
        )
        found.append(
            f"{len(across)} {noun} another panel or a figure text: {_listing(across)}"
        )
    if found:
        issues.append(
            Issue(
                code="label_overlap",
                severity="warning",
                message=". ".join(found) + ".",
                fix=room,
            )
        )
    out_of_figure, out_of_axes = _clipped(fig, extents)
    parts: list[str] = []
    if out_of_figure:
        noun = "text extends" if len(out_of_figure) == 1 else "texts extend"
        parts.append(
            f"{len(out_of_figure)} {noun} outside the figure: {_listing(out_of_figure)}"
        )
    if out_of_axes:
        noun = (
            "text drawn inside the Axes extends"
            if len(out_of_axes) == 1
            else ("texts drawn inside the Axes extend")
        )
        parts.append(f"{len(out_of_axes)} {noun} outside it: {_listing(out_of_axes)}")
    if parts:
        issues.append(
            Issue(
                code="text_clipped",
                severity="warning",
                message=". ".join(parts) + ".",
                fix=room,
            )
        )
    crowded = _crowded_labels(fig, extents)
    if crowded is not None:
        issues.append(crowded)
    oversized = _oversized(fig, extents)
    if oversized is not None:
        issues.append(oversized)
    issues += _path_issues(fig, extents)
    ambiguous = _ambiguous_labels(fig)
    if ambiguous is not None:
        issues.append(ambiguous)
    ambiguous = _ambiguous_line_labels(fig, extents)
    if ambiguous is not None:
        issues.append(ambiguous)
    issues += view_issues(fig, extents)
    sub = fig.subsample
    if sub is not None and sub.message is not None:
        issues.append(
            Issue(
                code="points_subsampled",
                severity="info",
                message=sub.message,
                fix=(
                    "Nothing needs fixing. Pass another seed to the analysis to draw "
                    "a different subsample."
                ),
            )
        )
    elif sub is not None:
        whole = (
            ", with each level's smallest and largest value"
            if sub.subjects is None
            else (
                f": every value of {sub.subjects[0]:,} of the {sub.subjects[1]:,} "
                "subjects, with those that hold each condition's smallest and "
                "largest value"
            )
        )
        issues.append(
            Issue(
                code="points_subsampled",
                severity="info",
                message=(
                    f"The points show a random subsample of {sub.shown:,} of the "
                    f"{sub.total:,} values{whole} (seed {sub.seed}); "
                    f"{shows_all(sub, 'are drawn from')}."
                ),
                fix=(
                    "Nothing needs fixing. Pass another seed to the analysis to draw "
                    "a different subsample, or leave 'points' out of layers."
                ),
            )
        )
    return tuple(issues)
