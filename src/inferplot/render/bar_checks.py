"""barstats' figure checks (SPEC section 8): ``cell_labels_hidden`` and
``level_colors_alike`` (:func:`level_color_issue`).

Every cell with observations has a label (its percentage, count or both). The
drawing writes it on its segment, or beside its bar where the segment is too
small, joined to it by a short line; with ``ax=`` a label for which the column
beside its bar has no room is left out. Measured on the drawn figure, after
anything the user changed, a label counts as hidden when it is:

* left out, or not drawn (made invisible or removed);
* on its segment, but with its rendered box reaching more than 0.5 px beyond
  the segment's (it then reads over a neighbouring segment or the background);
* beside its bar, but with its box over a bar segment (its own bar's or
  another's), where it reads as that segment's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from matplotlib.colors import to_hex

from inferplot.check import Issue
from inferplot.render import theme
from inferplot.render.figure import axes_background, root_figure

if TYPE_CHECKING:
    from matplotlib.backend_bases import RendererBase
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    from inferplot.render.figure import CellLabel, StatsFigure

__all__ = ["COLOR_DIFFERENCE", "cell_label_issue", "level_color_issue"]

# Pixels a label may reach beyond its segment, or into another one, before it
# counts (anti-aliasing slack).
_TOLERANCE: Final = 0.5
_MAX_LISTED: Final = 5


def _drawn(label: CellLabel) -> bool:
    text = label.text
    return text.get_visible() and text.axes is not None


def _beyond(box: Bbox, frame: Bbox) -> bool:
    return (
        box.x0 < frame.x0 - _TOLERANCE
        or box.x1 > frame.x1 + _TOLERANCE
        or box.y0 < frame.y0 - _TOLERANCE
        or box.y1 > frame.y1 + _TOLERANCE
    )


def _over(box: Bbox, frame: Bbox) -> bool:
    width = min(box.x1, frame.x1) - max(box.x0, frame.x0)
    height = min(box.y1, frame.y1) - max(box.y0, frame.y0)
    return width > _TOLERANCE and height > _TOLERANCE


def cell_label_issue(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> Issue | None:
    """Cell labels that are left out or do not read as their cell's; None for
    other figures and when every label reads as its own."""
    labels = fig.cell_labels
    if not labels:
        return None
    getter = getattr(root_figure(fig.figure).canvas, "get_renderer", None)
    get = cast("RendererBase", getter()) if callable(getter) else None
    boxes = {id(text): box for text, _, box in extents}
    segments = [
        (label, label.segment.get_window_extent(get))
        for label in labels
        if label.segment.get_visible() and label.segment.axes is not None
    ]
    found: list[str] = []
    for label in labels:
        what = f"the label of {label.name} ({label.words!r})".replace("\\n", " ")
        if label.place == "omitted":
            found.append(f"{what} is left out for lack of room")
            continue
        box = boxes.get(id(label.text))
        if not _drawn(label) or box is None:
            found.append(f"{what} is not drawn")
            continue
        own = label.segment.get_window_extent(get)
        if label.place == "inside":
            if _beyond(box, own):
                found.append(f"{what} reaches beyond its segment")
            continue
        under = [other.name for other, frame in segments if _over(box, frame)]
        if under:
            found.append(f"{what}, beside its bar, lies over the segment of {under[0]}")
    if not found:
        return None
    listed = "; ".join(found[:_MAX_LISTED])
    if len(found) > _MAX_LISTED:
        listed += f"; and {len(found) - _MAX_LISTED} more"
    noun = "cell label is" if len(found) == 1 else "cell labels are"
    return Issue(
        code="cell_labels_hidden",
        severity="warning",
        message=f"{len(found)} {noun} not shown where they can be read: {listed}.",
        fix=(
            "Give the bars more room: a larger Axes passed as ax= (taller for small "
            "cells, wider for the labels beside the bars), or inferplot's own "
            "figure by leaving ax out, which makes room for every label. Every "
            "count and percentage is in fig.result.cells."
        ),
    )


# CIE76 colour difference (Delta E*ab) below which two levels' colours count as
# the same: about 2.3 is just noticeable between large patches side by side, and
# the legend keys are small. Matplotlib's colormaps sampled at 20 levels come
# near it ('Blues' 2.9, 'Greys' 2.1: twenty shades a legend cannot key); the
# default Okabe-Ito colours differ by at least 26, tab20's by 17. A luminance
# contrast cannot be used: Okabe-Ito's orange and sky blue have 1.02:1.
COLOR_DIFFERENCE: Final = 3.0


def level_color_issue(fig: StatsFigure) -> Issue | None:
    """``level_colors_alike`` (barstats): two levels of x whose segments are
    drawn, as measured on the figure (over the Axes background, with their
    alpha), in colours that cannot be told apart; the colour is the only key to
    the legend. None for the other figures."""
    if not fig.segments:
        return None
    ax = fig.axes["main"]
    background = axes_background(ax)
    drawn: dict[int, tuple[float, float, float]] = {}
    for i, segment in fig.segments:
        if i in drawn or not segment.get_visible() or segment.axes is None:
            continue
        if not segment.get_fill():
            continue
        drawn[i] = theme.over(segment.get_facecolor(), background)
    levels = fig.result.levels
    found: list[str] = []
    order = sorted(drawn)
    for n, i in enumerate(order):
        for j in order[n + 1 :]:
            difference = theme.color_difference(drawn[i], drawn[j])
            if difference < COLOR_DIFFERENCE:
                found.append(
                    f"{levels[i]!r} and {levels[j]!r} ({to_hex(drawn[i])} and "
                    f"{to_hex(drawn[j])}, a colour difference of {difference:.1f})"
                )
    if not found:
        return None
    listed = "; ".join(found[:_MAX_LISTED])
    if len(found) > _MAX_LISTED:
        listed += f"; and {len(found) - _MAX_LISTED} more pairs"
    return Issue(
        code="level_colors_alike",
        severity="warning",
        message=(
            f"Levels of {fig.result.x!r} are drawn in colours that cannot be told "
            f"apart (CIELAB difference below {COLOR_DIFFERENCE:g}), and the colour "
            f"is the only key to the legend: {listed}."
        ),
        fix=(
            "Pass palette= with clearly different colours, one per level of x (the "
            "default palette, or a qualitative colormap such as 'tab10' or "
            "'Set2'), or merge levels so fewer colours are needed."
        ),
    )
