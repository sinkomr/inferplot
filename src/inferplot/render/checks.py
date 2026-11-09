"""Figure-level problems for ``fig.check()`` (SPEC section 8).

Measured on the drawn figure from the rendered bounding boxes of its texts:

* ``label_overlap`` (warning): two texts' boxes intersect, or a bracket label's
  box crosses the line (bar or leg) of another bracket;
* ``text_clipped`` (warning): a text extends outside the figure, or a text drawn
  inside the Axes (labels, and the subtitle and caption when ``ax=`` was given)
  extends outside that Axes;
* ``points_subsampled`` (info): the points show a subsample of a large panel.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase

from inferplot.check import Issue
from inferplot.render.figure import describe_text, root_figure

if TYPE_CHECKING:
    from matplotlib.backend_bases import RendererBase

    from inferplot.render.figure import StatsFigure

__all__ = ["figure_issues", "text_extents"]

# Pixels a text may cross a border by before it counts (anti-aliasing slack).
_CLIP_TOLERANCE: Final = 0.5
_MAX_LISTED: Final = 5


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
    for axes in [ax] if not fig.own else list(root_figure(fig.figure).axes):
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
            roles.get(id(text), "text"), text.get_text()
        )
        out.append((text, described, box))
    return out


def _listing(items: list[str]) -> str:
    listed = "; ".join(items[:_MAX_LISTED])
    if len(items) > _MAX_LISTED:
        listed += f"; and {len(items) - _MAX_LISTED} more"
    return listed


def _overlaps(extents: list[tuple[Text, str, Bbox]]) -> list[str]:
    found: list[str] = []
    for a in range(len(extents)):
        _, da, ba = extents[a]
        for b in range(a + 1, len(extents)):
            _, db, bb = extents[b]
            width = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            height = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if width > 0 and height > 0:
                found.append(f"{da} and {db}")
    return found


def _intersect(a: BboxBase, b: BboxBase) -> bool:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    return width > 0 and height > 0


def _segments(line: Line2D, dpi: float) -> list[Bbox]:
    """Each straight piece of a drawn line, as a box in pixels as thick as the line."""
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
    lines count.
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
            if m != n and any(_intersect(box, piece) for piece in lines[m]):
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


def figure_issues(fig: StatsFigure) -> tuple[Issue, ...]:
    """Rendering issues: label_overlap, text_clipped, points_subsampled."""
    fig.draw()
    extents = text_extents(fig)
    issues: list[Issue] = []
    overlaps = _overlaps(extents)
    crossings = _crossings(fig, extents)
    room = (
        "Give the plot more room (a larger Axes passed as ax=, or inferplot's own "
        "figure by leaving ax out), shorten the texts, or draw fewer brackets "
        "with pairwise_display='significant' or 'none'."
    )
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
    sub = fig.subsample
    if sub is not None:
        issues.append(
            Issue(
                code="points_subsampled",
                severity="info",
                message=(
                    f"The points show a random subsample of {sub.shown:,} of the "
                    f"{sub.total:,} values (seed {sub.seed}); the {sub.density} and "
                    "boxes are drawn from all values."
                ),
                fix=(
                    "Nothing needs fixing. Pass another seed to the analysis to draw "
                    "a different subsample, or leave 'points' out of layers."
                ),
            )
        )
    return tuple(issues)
