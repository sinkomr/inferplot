# matplotlib's and NumPy's stubs type many ``**kwargs`` and overloads as Unknown,
# which strict mode reports on every call; the arguments we pass are all typed.
# pyright: reportUnknownMemberType=false
"""The drawing shared by betweenstats and withinstats: violins, boxes, points,
centrality labels, brackets, subtitle and caption.

Every number and text comes from the :class:`~inferplot.result.StatsResult` and
:func:`inferplot.text.figure_texts`; the renderer computes no statistics. It does
compute drawing aids: kernel density outlines for the violins, quartiles and
whiskers for the boxes, and seeded jitter.

When inferplot creates the figure it sizes it from the rendered sizes of its texts,
so nothing is clipped and no two texts overlap (a long subtitle widens the figure).
With ``ax=`` the subtitle and caption go inside that Axes and ``fig.check()``
reports anything that does not fit.

Every argument is checked before anything is drawn, so a call that raises leaves a
given Axes as it was: each artist dict styles a scratch artist made as the drawing
makes it, which is drawn on a scratch figure and written as SVG and PDF (matplotlib
checks some values only then), and the user's texts are laid out once as mathtext.
With ``ax=`` only the texts inferplot draws get its font; the user's own texts in
that Axes keep theirs.

:class:`Drawing` draws a betweenstats result; ``render.withinstats`` subclasses it
and overrides the hooks marked as such (where the points go, the paths, where the
centrality labels go).
"""

from __future__ import annotations

import dataclasses
import io
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
from matplotlib import cbook
from matplotlib.artist import ArtistInspector
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox, offset_copy
from numpy.typing import NDArray
from scipy.special import ndtr

from inferplot import _validate as v
from inferplot.render import theme
from inferplot.render.brackets import Bracket, VerticalLayout, stack, vertical_layout
from inferplot.render.figure import (
    DrawnPaths,
    DrawnView,
    StatsFigure,
    Subsample,
    axes_background,
    describe_text,
    figure_background,
    plain_text,
    remember,
    root_figure,
    shows_all,
)
from inferplot.result import StatsResult
from inferplot.text import FigureTexts, figure_texts

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.backend_bases import RendererBase

__all__ = [
    "MAX_POINTS",
    "MIN_PER_LEVEL",
    "SUBSAMPLE_SIZE",
    "TICK_LINES",
    "Drawing",
    "LabelPlace",
    "Options",
    "apply_args",
    "check_options",
    "checked_result",
    "draw",
    "points_artist",
    "wrap_tick_label",
]

FloatArray = NDArray[np.float64]

# A panel with more values than this draws a subsample of SUBSAMPLE_SIZE points.
MAX_POINTS: Final = 5_000
SUBSAMPLE_SIZE: Final = 2_000
MIN_PER_LEVEL: Final = 50  # points drawn of each level (all of a smaller one)

# Geometry in x data units (one level per unit).
VIOLIN_HALF_WIDTH: Final = 0.4
BOX_HALF_WIDTH: Final = 0.15
JITTER_HALF_WIDTH: Final = 0.12
LABEL_OFFSET: Final = 0.22  # centrality label starts this far right of its level

# Geometry in points.
MARGIN_PT: Final = 10.0
GAP_PT: Final = 5.0
LABEL_PAD_PT: Final = 3.0  # between a bracket and its label, and around labels
LABEL_BOX_PAD: Final = 0.3  # centrality label box padding, in font sizes
BRACKET_TIP_PT: Final = 4.0  # length of a bracket's legs
BRACKET_LINEWIDTH_PT: Final = 0.8
MIN_LEVEL_SPACING_IN: Final = 1.1
MIN_AXES_HEIGHT_IN: Final = 3.2
MIN_DATA_HEIGHT_IN: Final = 2.5  # the data region; bracket levels go on top
KDE_GRID: Final = 256
KDE_BINS: Final = 2_048


@dataclasses.dataclass(frozen=True, slots=True)
class Options:
    """The checked drawing arguments."""

    centrality: bool
    layers: tuple[str, ...]
    xlab: str | None
    ylab: str | None
    title: str | None
    subtitle: str | None
    caption: str | None
    point_args: dict[str, object]
    boxplot_args: dict[str, object]
    violin_args: dict[str, object]
    point_path: bool = False  # withinstats only
    centrality_path: bool = False  # withinstats only


CENTRALITY_SIZE: Final = 40.0  # centrality point marker area, points squared


@dataclasses.dataclass(frozen=True, slots=True)
class LabelPlace:
    """Where a centrality label goes relative to its point.

    ``side``: right or left of the point, ``LABEL_OFFSET`` away, joined to it by a
    short line; ``rise``: centered on the point's height, or with its box's bottom
    edge just above it ("above") or its top edge just below it ("below").
    """

    side: Literal["right", "left"] = "right"
    rise: Literal["center", "above", "below"] = "center"


_RISES: Final[tuple[Literal["center", "above", "below"], ...]] = (
    "center",
    "above",
    "below",
)


def label_candidates(first: LabelPlace) -> list[LabelPlace]:
    """Where a centrality label may go, in order of preference: ``first`` (the
    drawing's own choice), then on the same side above and below the point's
    height (the link line, at the point's height, leads to the label's corner),
    then the same three on the other side."""
    other: Literal["right", "left"] = "left" if first.side == "right" else "right"
    rises: list[Literal["center", "above", "below"]] = [
        first.rise,
        *(r for r in _RISES if r != first.rise),
    ]
    return [
        *(LabelPlace(side=first.side, rise=r) for r in rises),
        *(LabelPlace(side=other, rise=r) for r in _RISES),
    ]


@dataclasses.dataclass(slots=True)
class _Label:
    """A drawn centrality label with its link line and point, and where it is."""

    i: int
    value: float
    link: Line2D
    point: PathCollection
    text: Text
    place: LabelPlace


def _overlap(a: Bbox, b: Bbox) -> float:
    """Area two boxes share, counting only overlaps of more than half a pixel in
    both directions (anti-aliasing slack)."""
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0.5 or height <= 0.5:
        return 0.0
    return width * height


def checked_result(result: StatsResult, analysis: str, call: str) -> int:
    """Check that ``result`` is a drawable result of ``analysis``; its level count.

    ``call`` is how to recompute it, for the message about a result without
    observations.
    """
    if not isinstance(result, StatsResult):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"render.{analysis}() draws the result of ip.analyze.{analysis}(); "
            f"got {type(result).__name__}."
        )
    if result.analysis != analysis:
        raise TypeError(
            f"render.{analysis}() draws results of ip.analyze.{analysis}(); this "
            f"result comes from ip.analyze.{result.analysis}(). Draw it with "
            f"ip.render.{result.analysis}(result)."
        )
    k = len(result.levels)
    if k >= 1 and not result.values:
        raise ValueError(
            "This result holds no observations to draw: the copy of a result that "
            "save() embeds in a figure file leaves them out. Recompute it with "
            f"{call} and draw that."
        )
    if not (len(result.values) == len(result.descriptives) == k and k >= 1):
        raise ValueError(
            "result is inconsistent: it needs one entry of values and descriptives "
            f"per level; got {k} levels, {len(result.values)} value groups and "
            f"{len(result.descriptives)} descriptives."
        )
    return k


def check_options(result: StatsResult, options: Options, ax: Axes | None) -> None:
    """Checks of the drawing arguments against the result and each other."""
    if options.subtitle is not None and result.settings.results_subtitle:
        raise ValueError(
            "subtitle= would replace the results subtitle, which this result draws "
            "(it was computed with results_subtitle=True). To use your own "
            "subtitle, pass results_subtitle=False to the analysis; to add text, "
            "use title= or caption=."
        )
    if ax is not None and not isinstance(ax, Axes):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"ax must be a matplotlib Axes or None; got {type(ax).__name__}."
        )
    if ax is not None:
        check_axes(ax)
    _check_scale(result)


# The largest |y| inferplot draws must lie within these bounds: matplotlib treats
# axis limits whose size is below about 1e21 times the smallest normal double
# (2.2e-287) as zero and widens them to +-0.05, and above about 1e306 the room
# for brackets above the data comes close to the largest double.
_SMALLEST_DRAWN: Final = 1e-280
_LARGEST_DRAWN: Final = 1e306


def _check_scale(result: StatsResult) -> None:
    """Values matplotlib can draw on an axis; raised before anything is drawn."""
    sizes = [abs(float(v)) for level in result.values for v in level]
    sizes += [abs(d.value) for d in result.descriptives if d.value is not None]
    largest = max(sizes, default=0.0)
    if largest == 0.0 or _SMALLEST_DRAWN <= largest <= _LARGEST_DRAWN:
        return
    exponent = math.floor(math.log10(largest))
    small = largest < _SMALLEST_DRAWN
    if small:
        why = (
            f"the largest is {largest:.3g} in size, and matplotlib treats axis "
            "limits below about 1e-287 as zero"
        )
        how = f"multiply y by 1e{-exponent}"
    else:
        why = (
            f"the largest is {largest:.3g}, and the axis, with room for the "
            "brackets above the data, would pass the largest double (1.8e308)"
        )
        how = f"divide y by 1e{exponent}"
    raise ValueError(
        f"The values of {result.y!r} are too {'small' if small else 'large'} "
        f"for matplotlib to draw: {why}. The statistics do not depend on the "
        f"units: {how} (change its units) before the analysis and draw that."
    )


def check_axes(ax: Axes) -> None:
    """A given Axes is a plain rectangular 2D one whose x axis it does not share;
    checked before drawing."""
    name = str(getattr(ax, "name", ""))
    if name != "rectilinear":
        raise ValueError(
            f"ax is a {name!r} Axes; inferplot draws only into a plain rectangular "
            "2D Axes, where the levels lie along x and the values along y. Pass an "
            "Axes made without projection=, for example figure.add_subplot()."
        )
    if len(_siblings(ax, "x")) > 1:
        raise ValueError(
            "ax shares its x axis with another Axes (sharex=True, or twinx()), and "
            "Axes that share an x axis share its tick labels; inferplot's tick "
            "labels name each level with that panel's n, so one panel would show "
            "the other's. Make the panels with sharex=False (the default), for "
            "example figure.subplots(1, 2, sharey=True)."
        )


def _siblings(ax: Axes, which: Literal["x", "y"]) -> list[Axes]:
    """The Axes that share ``ax``'s x or y axis, ``ax`` included."""
    group = cast(
        "Any", ax.get_shared_x_axes() if which == "x" else ax.get_shared_y_axes()
    )
    return list(cast("list[Axes]", group.get_siblings(ax)))


def draw(
    drawing: type[Drawing],
    result: StatsResult,
    options: Options,
    palette: object,
    ax: Axes | None,
) -> StatsFigure:
    """Resolve the palette, check the artist dicts and user texts, then draw."""
    colors, palette_warning = theme.resolve_palette(palette, len(result.levels))
    fallbacks = theme.user_fallbacks()  # before context() resets the rcParams
    with theme.context(), theme.fallback_fonts(fallbacks):
        invisible = _check_artist_args(options)
        _check_mathtext(options)
        made = drawing(result, options, colors, palette_warning, ax)
        made.warnings.extend(invisible)
        return made.run()


def apply_args(artist: Artist, args: Mapping[str, object], name: str) -> None:
    """``artist.set(**args)``, with a message naming the argument on failure.

    A name the artist has no setter for is an unknown property (``TypeError``);
    any error from setting a known property is a rejected value (``ValueError``),
    whatever exception type matplotlib uses for it.
    """
    if not args:
        return
    kind = type(artist).__name__
    for key in args:
        if not callable(getattr(artist, f"set_{key}", None)):
            close = v.closest_name(key, ArtistInspector(artist).get_setters())
            hint = (
                f" Did you mean {close!r}?"
                if close is not None
                else f" matplotlib's documentation of {kind} lists its properties."
            )
            raise TypeError(
                f"{name} has a property matplotlib's {kind} does not have: "
                f"{key!r}.{hint}"
            )
    try:
        artist.set(**args)
    except Exception as error:
        raise ValueError(f"{name} has a value matplotlib rejects: {error}") from error


# The layer each artist dict styles (SPEC.md section 2).
_ARTIST_LAYERS: Final = (
    ("point_args", "points"),
    ("boxplot_args", "box"),
    ("violin_args", "violin"),
)

# The vector formats a scratch drawing is also written to: matplotlib checks some
# values only there (a NaN line width passes Agg and fails in PDF).
_SCRATCH_FORMATS: Final = ("svg", "pdf")
_SCRATCH_VALUES: Final = (0.0, 1.0, 2.0, 3.0, 4.0)


def points_artist(
    ax: Axes, x: FloatArray, y: FloatArray, color: theme.Color, i: int
) -> PathCollection:
    """One level's points, before ``point_args``."""
    return ax.scatter(
        x,
        y,
        s=14.0,
        color=color,
        alpha=0.6,
        linewidths=0.0,
        zorder=3.0,
        gid=f"points-{i}",
    )


def _box_artist(x: FloatArray, center: float, i: int) -> PathPatch:
    """One level's box, before ``boxplot_args``."""
    return PathPatch(
        _box_path(x, center),
        facecolor=(1.0, 1.0, 1.0, 0.85),
        edgecolor=theme.SOFT_INK,
        linewidth=0.9,
        zorder=2.0,
        gid=f"box-{i}",
    )


def _violin_artist(outline: FloatArray, color: theme.Color, i: int) -> Polygon:
    """One level's violin, before ``violin_args``."""
    return Polygon(
        outline,
        closed=True,
        facecolor=(*color[:3], 0.25),
        edgecolor=color,
        linewidth=0.8,
        zorder=1.0,
        gid=f"violin-{i}",
    )


def _scratch(layer: str) -> tuple[Figure, Artist]:
    """A scratch figure holding one artist of ``layer``, made as the drawing makes
    it; nothing of it reaches the user."""
    figure = Figure(figsize=(2.0, 2.0), dpi=100)
    FigureCanvasAgg(figure)
    figure.patch.set_visible(False)  # so that only the artist leaves pixels
    ax = figure.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_axis_off()  # no texts, so writing it embeds no fonts; no background
    ax.set_xlim(-0.5, 0.5)  # the whole artist in view
    ax.set_ylim(-0.5, 4.5)
    values = np.array(_SCRATCH_VALUES)
    color: theme.Color = (0.0, 0.0, 0.0, 1.0)
    if layer == "points":
        return figure, points_artist(ax, np.zeros_like(values), values, color, 0)
    if layer == "box":
        artist: Artist = _box_artist(values, 0.0, 0)
    else:
        outline = np.array([[-0.4, 0.0], [0.4, 0.0], [0.0, 4.0]])
        artist = _violin_artist(outline, color, 0)
    ax.add_artist(artist)
    return figure, artist


def _draw_error(figure: Figure) -> Exception | None:
    """The error matplotlib raises when it draws ``figure`` (Agg) or writes it as
    SVG or PDF; None when it draws. Warnings are left to the real drawing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            figure.canvas.draw()
            for fmt in _SCRATCH_FORMATS:
                figure.savefig(io.BytesIO(), format=fmt)
        # The same scratch figure draws without the user's properties, so whatever
        # fails here is one of their values.
        except Exception as error:
            return error
    return None


def _check_drawn(name: str, layer: str, args: Mapping[str, object]) -> bool:
    """Apply ``args`` to a scratch artist of ``layer`` and draw it; whether the
    artist is then invisible (it leaves no pixel).

    ``Artist.set`` stores some values without checking them (a line width of
    "bad" for points); matplotlib rejects them only when drawing, which with ``ax=``
    would be the user's later ``savefig``.
    """
    figure, artist = _scratch(layer)
    apply_args(artist, args, name)
    error = _draw_error(figure)
    if error is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figure.canvas.draw()
        return _invisible(figure)
    culprits: list[str] = []
    for key, value in args.items():
        alone, single = _scratch(layer)
        try:
            single.set(**{key: value})
        except Exception:  # this one value is then a culprit
            culprits.append(key)
            continue
        if _draw_error(alone) is not None:
            culprits.append(key)
    shown = ", ".join(
        f"{key}={v.describe_value(args[key])}" for key in culprits or args
    )
    raise ValueError(
        f"{name} has a value matplotlib rejects when it draws the {layer}: {shown} "
        f"({type(error).__name__}: {error}). Use a value of the kind matplotlib "
        f"documents for that property, or remove it from {name}."
    ) from error


# Properties an artist dict may not set (SPEC.md section 2), with why: they carry
# or move the data, or detach the artist from the figure. Found by going through
# the setters of the points' PathCollection, the box's PathPatch and the violin's
# Polygon; names that are not properties of these artists (``data``, ``xdata``,
# ``verts``, ...) are listed too, so they get this message rather than a
# suggestion of a similar property.
_DATA: Final = (
    "would replace the positions or shape of the {layer}, which inferplot draws "
    "from the result's values"
)
_MOVED: Final = (
    "would draw the {layer} in other coordinates than the axes' data "
    "coordinates, so it would no longer line up with the axis, the other layers "
    "and the brackets"
)
_DETACHED: Final = "would detach the {layer} from the Axes inferplot draws into"
_CLIPPED: Final = (
    "would clip the {layer} to another region than the Axes, hiding values"
)
_REFUSED: Final[Mapping[str, str]] = {
    "offsets": _DATA,
    "paths": _DATA,
    "path": _DATA,
    "xy": _DATA,
    "verts": _DATA,
    "segments": _DATA,
    "data": _DATA,
    "xdata": _DATA,
    "ydata": _DATA,
    "transform": _MOVED,
    "offset_transform": _MOVED,
    "figure": _DETACHED,
    "axes": _DETACHED,
    "clip_path": _CLIPPED,
    "clip_box": _CLIPPED,
    "animated": (
        "would leave the {layer} out of every ordinary drawing and saved file "
        "(matplotlib draws animated artists only for blitting)"
    ),
}


def _refuse_moving_args(name: str, layer: str, args: Mapping[str, object]) -> None:
    """Refuse a property that carries or moves the data or detaches the artist.

    Aliases are read as the property they stand for.
    """
    figure, artist = _scratch(layer)
    del figure
    for key in args:
        prop = key
        if key not in _REFUSED:
            try:
                (prop,) = cbook.normalize_kwargs({key: None}, artist)
            except TypeError:
                continue  # a property of two names at once; Artist.set reports it
        why = _REFUSED.get(prop)
        if why is not None:
            raise ValueError(
                f"{name} sets {key!r}, which {why.format(layer=_LAYER_NOUN[layer])}. "
                f"Remove {key!r} from {name}"
                + (
                    "; to draw other values, change the data before the analysis."
                    if why is _DATA
                    else "."
                )
            )


_LAYER_NOUN: Final = {"points": "points", "box": "boxes", "violin": "violins"}


def _invisible(figure: Figure) -> bool:
    """Whether a scratch figure (transparent but for its one artist) drew nothing:
    every pixel of the Agg buffer is fully transparent."""
    canvas = cast("FigureCanvasAgg", figure.canvas)
    alpha = np.asarray(canvas.buffer_rgba())[..., 3]
    return not bool(alpha.any())


def _invisible_warning(name: str, layer: str, args: Mapping[str, object]) -> str:
    noun = _LAYER_NOUN[layer]
    shown = ", ".join(f"{key}={v.describe_value(value)}" for key, value in args.items())
    return (
        f"{name}={{{shown}}} makes the {noun} invisible: nothing of them is drawn, "
        f"although layers= asks for them. To leave the {noun} out, remove "
        f"{layer!r} from layers= instead; otherwise choose a visible style."
    )


def _check_artist_args(options: Options) -> list[str]:
    """Each non-empty artist dict styles a layer that is drawn, sets no property
    that moves or replaces the data, and an artist of that layer styled with it
    draws; checked before anything is drawn.

    Returns a warning for each dict that leaves its layer invisible (D39).
    """
    found: list[str] = []
    for name, layer in _ARTIST_LAYERS:
        args: dict[str, object] = getattr(options, name)
        if not args:
            continue
        if layer not in options.layers:
            with_layer = tuple(
                item for item in v.LAYERS if item in options.layers or item == layer
            )
            raise ValueError(
                f"{name} styles the {layer} layer, which layers={options.layers!r} "
                f"does not draw. To style it, add {layer!r} to layers, for example "
                f"layers={with_layer!r}; otherwise remove {name}=."
            )
        _refuse_moving_args(name, layer, args)
        if _check_drawn(name, layer, args):
            found.append(_invisible_warning(name, layer, args))
    return found


_USER_TEXTS: Final = ("title", "subtitle", "caption", "xlab", "ylab")


def _check_mathtext(options: Options) -> None:
    """Lay out each user text once, so an expression matplotlib cannot read raises
    here, before anything is drawn (with ``ax=`` it would otherwise surface only
    when the figure is drawn)."""
    given = [(name, getattr(options, name)) for name in _USER_TEXTS]
    given = [(name, value) for name, value in given if value is not None]
    if not given:
        return
    figure = Figure()
    renderer = FigureCanvasAgg(figure).get_renderer()
    for name, value in given:
        text = figure.text(0.0, 0.0, value, fontproperties=theme.font(theme.SIZE_LABEL))
        try:
            text.get_window_extent(renderer)
        except ValueError as error:
            lines = [line.strip() for line in str(error).splitlines() if line.strip()]
            detail = lines[-1] if lines else type(error).__name__
            raise ValueError(
                f"{name} is drawn as matplotlib mathtext, where text between two $ "
                f"signs is a math expression, and matplotlib cannot read it: "
                f"{detail}. Fix the expression, or write \\$ for a literal dollar "
                f"sign; got {name}={value!r}."
            ) from error


# --------------------------------------------------------------------------- #
# drawing aids
# --------------------------------------------------------------------------- #


def _bandwidth(x: FloatArray) -> float:
    """Silverman's rule of thumb (1986, eq. 3.31), as R's ``bw.nrd0``."""
    with np.errstate(over="ignore", invalid="ignore"):
        sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd):
        # Squares of values near the largest double overflow; scale them first.
        scale = float(np.abs(x).max())
        sd = scale * float(np.std(x / scale, ddof=1))
    q75, q25 = (float(q) for q in np.percentile(x, [75.0, 25.0]))
    lo = min(sd, (q75 - q25) / 1.34)
    if lo <= 0.0:
        lo = sd if sd > 0.0 else 1.0
    return 0.9 * lo * x.size ** (-0.2)


def _ndtr(z: FloatArray) -> FloatArray:
    """The standard normal CDF, elementwise."""
    return np.asarray(ndtr(z), dtype=np.float64)


def _density_outline(x: FloatArray) -> tuple[FloatArray, FloatArray] | str | None:
    """Gaussian kernel density on [min, max], scaled to the violin's half width.

    None when the values do not span a range (fewer than two different values);
    the reason in words when the density cannot be computed. Large samples are
    binned first so drawing stays fast.

    The density is evaluated at KDE_GRID points. When the bandwidth is narrower
    than the spacing of those points (heavy tails, or one value far from the
    rest), the density at the points would miss most of the values between them
    (and with binning be 0 everywhere: 0/0), so each point gets the density
    averaged over its cell instead: the kernel's mass that falls in the cell,
    divided by its width (Gaussian CDF differences). Elsewhere the two agree to
    drawing precision, and the point values are kept.
    """
    if x.size < 2 or float(x.min()) == float(x.max()):
        return None
    lo, hi = float(x.min()), float(x.max())
    if not np.isfinite(hi - lo):
        return "its values span a range wider than the largest double"
    bw = _bandwidth(x)
    grid = np.linspace(lo, hi, KDE_GRID)
    cell = (hi - lo) / (KDE_GRID - 1)
    if x.size > KDE_BINS:
        counts, edges = np.histogram(x, bins=KDE_BINS)
        points = (edges[:-1] + edges[1:]) / 2
        weights = counts.astype(np.float64)
    else:
        points, weights = x, np.ones(x.size)
    if bw >= cell:
        z = (grid[:, None] - points[None, :]) / bw
        if x.size > KDE_BINS:
            # Integer counts as doubles: the same products as before.
            density = (np.exp(-0.5 * z * z) * weights[None, :]).sum(axis=1)
        else:
            density = np.exp(-0.5 * z * z).sum(axis=1)
    else:
        width = max(bw, cell * 1e-12)
        upper = _ndtr((grid[:, None] + cell / 2 - points[None, :]) / width)
        lower = _ndtr((grid[:, None] - cell / 2 - points[None, :]) / width)
        density = ((upper - lower) * weights[None, :]).sum(axis=1) / cell
    peak = float(density.max())
    if not (np.isfinite(peak) and peak > 0.0):
        return "its density could not be computed"
    half = np.asarray(VIOLIN_HALF_WIDTH * density / peak, dtype=np.float64)
    return np.asarray(grid, dtype=np.float64), half


def _whiskers(x: FloatArray) -> tuple[float, float, float, float, float]:
    """Quartiles (R's type 7), median and Tukey whisker ends: the most extreme
    values within 1.5 IQR of the box."""
    q1, med, q3 = (float(q) for q in np.percentile(x, [25.0, 50.0, 75.0]))
    iqr = q3 - q1
    low = float(x[x >= q1 - 1.5 * iqr].min())
    high = float(x[x <= q3 + 1.5 * iqr].max())
    return q1, med, q3, low, high


def outliers(x: FloatArray) -> FloatArray:
    """The values beyond the whiskers, in data order."""
    _, _, _, low, high = _whiskers(x)
    return x[(x < low) | (x > high)]


def _box_path(x: FloatArray, center: float) -> Path:
    """Box (quartiles, R's type 7), median line and Tukey whiskers as one path."""
    q1, med, q3, low, high = _whiskers(x)
    a, b = center - BOX_HALF_WIDTH, center + BOX_HALF_WIDTH
    vertices = [
        (a, q1), (b, q1), (b, q3), (a, q3), (a, q1),
        (a, med), (b, med),
        (center, q3), (center, high),
        (center, q1), (center, low),
    ]  # fmt: skip
    codes = [
        Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
        Path.MOVETO, Path.LINETO,
        Path.MOVETO, Path.LINETO,
        Path.MOVETO, Path.LINETO,
    ]  # fmt: skip
    return Path(vertices, codes)


def with_extremes(
    n: int, size: int, forced: NDArray[np.intp], rng: np.random.Generator
) -> NDArray[np.intp]:
    """Sorted indices of ``size`` of ``n`` items: every one of ``forced`` and a
    seeded uniform random choice of the others."""
    forced = np.unique(forced)
    rest = np.setdiff1d(np.arange(n), forced)
    pick = rng.choice(rest, size=max(0, size - forced.size), replace=False)
    return np.sort(np.concatenate([forced, pick]))


def _split_subsample(
    values: Sequence[FloatArray], rng: np.random.Generator
) -> list[FloatArray]:
    """A seeded subsample of about SUBSAMPLE_SIZE values, stratified by level.

    Level g of n_g values (N in all) gets max(round(SUBSAMPLE_SIZE n_g / N),
    min(n_g, MIN_PER_LEVEL)) of them, its smallest and largest value always
    among them and the rest drawn uniformly at random, so every level has marks
    and every level's extremes are drawn.
    """
    total = sum(x.size for x in values)
    out: list[FloatArray] = []
    for x in values:
        n = x.size
        share = int(SUBSAMPLE_SIZE * n / total + 0.5)
        size = min(n, max(share, min(n, MIN_PER_LEVEL)))
        if size >= n:
            out.append(x)
            continue
        extremes = np.array([np.argmin(x), np.argmax(x)], dtype=np.intp)
        out.append(x[with_extremes(n, size, extremes, rng)])
    return out


# --------------------------------------------------------------------------- #
# the drawing
# --------------------------------------------------------------------------- #


# Tick labels: a level name wraps on spaces into at most TICK_LINES lines, each
# as wide as the narrowest level spacing of inferplot's own figure allows.
TICK_LINES: Final = 3
_ELLIPSIS: Final = "\u2026"  # as the alt text shortens names


def _shortened(text: str, fits: Callable[[str], bool]) -> str:
    """``text`` cut at its end, with an ellipsis, to the longest prefix that
    fits; ``text`` itself when it fits."""
    if fits(text):
        return text
    lo, hi = 0, len(text)  # the longest fitting prefix lies in [lo, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(text[:mid].rstrip() + _ELLIPSIS):
            lo = mid
        else:
            hi = mid
    return text[:lo].rstrip() + _ELLIPSIS


def wrap_tick_label(label: str, fits: Callable[[str], bool]) -> str:
    """A tick label ``"<level>\n(n = <n>)"`` as drawn.

    The level name is wrapped on spaces into lines that ``fits`` accepts, at
    most TICK_LINES of them; a word too long for a line, and whatever does not
    fit in the last line, is shortened with an ellipsis. The result keeps the
    full name; a name that fits on one line is drawn as it is.
    """
    name, sep, count = label.rpartition("\n")
    if not sep:
        name, count = label, ""
    lines: list[str] = []
    for paragraph in name.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            joined = f"{current} {word}" if current else word
            if fits(joined) or not current:
                current = joined
            else:
                lines.append(current)
                current = word
        lines.append(current)
    if len(lines) > TICK_LINES:
        rest = " ".join(lines[TICK_LINES - 1 :])
        lines = [*lines[: TICK_LINES - 1], _shortened(rest, fits)]
    wrapped = "\n".join(_shortened(line, fits) for line in lines)
    return f"{wrapped}\n{count}" if sep else wrapped


def px(points: float, dpi: float) -> float:
    return points * dpi / 72.0


_EMPTY: Final = np.empty(0, dtype=np.float64)
# The attribute of a given Axes that holds its drawing, so that a later drawing
# into an Axes sharing its y axis can lay it out again at the shared scale.
_PANEL_ATTR: Final = "_inferplot_drawing"


class Drawing:
    """One rendering of a result; holds the artists while laying them out.

    Draws a betweenstats result. The hooks ``place_points``, ``draw_paths``,
    ``label_place`` and ``drawn_paths`` let a subclass change where the points go,
    add paths, and move the centrality labels.
    """

    def __init__(
        self,
        result: StatsResult,
        options: Options,
        colors: list[theme.Color],
        palette_warning: str | None,
        ax: Axes | None,
    ) -> None:
        self.result = result
        self.options = options
        self.colors = colors
        self.texts: FigureTexts = figure_texts(result)
        self.own = ax is None
        if ax is None:
            self.figure = Figure(figsize=(6.0, 4.5), dpi=100)
            FigureCanvasAgg(self.figure)
            self.ax = self.figure.add_axes((0.15, 0.2, 0.7, 0.6))
        else:
            self.figure = root_figure(ax.figure)
            self.ax = ax
        self.dpi = float(self.figure.dpi)
        self.values = [np.asarray(x, dtype=np.float64) for x in result.values]
        self.warnings: list[str] = [] if palette_warning is None else [palette_warning]
        self.roles: dict[int, str] = {}
        self.inside: list[Text] = []
        self.subsample: Subsample | None = None
        self.centrality_labels: list[Text] = []
        self.left_labels: list[Text] = []  # centrality labels left of their point
        self.raised: list[tuple[Text, float]] = []  # labels off-center, offset (px)
        self.bracket_lines: list[Line2D] = []
        self.bracket_labels: list[Text] = []
        self.bracket_pairs: list[tuple[int, int]] = []
        self.bracket_names: list[str] = []
        self.header: list[Text] = []  # title, subtitle (own figure)
        self.captions: list[Text] = []
        self.view = DrawnView(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        self.undrawn: list[str] = []  # levels whose violin or outline is not drawn
        self.outlier_marks: list[PathCollection] = []
        self.tick_texts: list[str] | None = None
        self.label_entries: list[_Label] = []
        # The y layout's inputs: bracket levels used, room above and below (px).
        self.y_room: tuple[int, float, float] = (0, 0.0, 0.0)
        self.levels_of_brackets: list[int] = []

    # -- helpers ---------------------------------------------------------------

    def renderer(self) -> RendererBase | None:
        get = getattr(self.figure.canvas, "get_renderer", None)
        return cast("RendererBase", get()) if callable(get) else None

    def extent(self, text: Text) -> Bbox:
        """Rendered box of ``text`` in pixels, its label box included."""
        renderer = self.renderer()
        box = text.get_window_extent(renderer)
        patch = text.get_bbox_patch()
        if patch is not None:
            pad = 2 * px(LABEL_BOX_PAD * float(text.get_fontsize()), self.dpi)
            box = Bbox.from_bounds(box.x0 - pad / 2, box.y0 - pad / 2,
                                   box.width + pad, box.height + pad)  # fmt: skip
        return box

    def tick_labels(self) -> list[str]:
        """The x tick labels as drawn: :func:`wrap_tick_label` with lines as wide
        as the narrowest level spacing of inferplot's own figure leaves (measured
        in the tick font), so that one long name does not widen every level."""
        if self.tick_texts is not None:
            return self.tick_texts
        room = MIN_LEVEL_SPACING_IN * self.dpi - 2 * px(GAP_PT, self.dpi)
        probe = self.figure.text(
            0.0, 0.0, "", fontproperties=theme.font(theme.SIZE_TICK), parse_math=False
        )
        renderer = self.renderer()

        def fits(line: str) -> bool:
            probe.set_text(line)
            return float(probe.get_window_extent(renderer).width) <= room

        try:
            self.tick_texts = [wrap_tick_label(t, fits) for t in self.texts.tick_labels]
        finally:
            probe.remove()
        return self.tick_texts

    def role(self, text: Text, name: str, plain: str | None = None) -> Text:
        """Record what ``text`` is (and its plain wording) for check() messages."""
        described = describe_text(name, plain_text(text) if plain is None else plain)
        self.roles[id(text)] = described
        remember(text, described)
        return text

    # -- hooks -------------------------------------------------------------------

    def place_points(self, rng: np.random.Generator) -> list[tuple[FloatArray, ...]]:
        """Hook: each level's drawn points as (x, y), subsampling large panels.

        Between groups: a subsample stratified by level (:func:`_split_subsample`)
        when there are more than MAX_POINTS values, and independent seeded jitter
        for every point.
        """
        layers = self.options.layers
        seed = self.result.settings.seed
        shown = self.values
        total = sum(x.size for x in self.values)
        if "points" in layers and total > MAX_POINTS:
            shown = _split_subsample(self.values, rng)
            density = "violins" if "violin" in layers else "density outlines"
            self.subsample = Subsample(
                shown=sum(y.size for y in shown),
                total=total,
                seed=seed,
                density=density,
            )
        out: list[tuple[FloatArray, ...]] = []
        for i, y in enumerate(shown):
            if "points" in layers and y.size > 0:
                jitter = rng.uniform(-JITTER_HALF_WIDTH, JITTER_HALF_WIDTH, size=y.size)
                out.append((i + jitter, y))
            else:
                out.append((_EMPTY, _EMPTY))
        return out

    def subsample_warning(self, sub: Subsample) -> str:
        """Hook: the ``warnings`` entry for a subsample of the points."""
        return (
            f"The plot has {sub.total:,} values, more than {MAX_POINTS:,}, so the "
            f"points show a random subsample of {sub.shown:,} (seed {sub.seed}): "
            f"about {SUBSAMPLE_SIZE:,} shared among the levels by their size, at "
            f"least {MIN_PER_LEVEL} of each level (all of a smaller one), always "
            "with each level's smallest and largest value; "
            f"{shows_all(sub, 'show')}."
        )

    def draw_paths(self, points: list[tuple[FloatArray, ...]]) -> None:
        """Hook: lines joining the drawn points (none between groups)."""
        del points

    def label_place(self, i: int) -> LabelPlace:
        """Hook: where the centrality label of level ``i`` goes."""
        del i
        return LabelPlace()

    def drawn_paths(self) -> DrawnPaths | None:
        """Hook: the paths drawn, for ``fig.check()``."""
        return None

    # -- main ------------------------------------------------------------------

    def run(self) -> StatsFigure:
        self.draw_data()
        self.draw_ticks_and_labels()
        self.draw_text_blocks()
        self.layout()
        for last in (False, False, True):
            if not self.choose_label_places():
                break
            self.layout()  # the labels moved: room above, below or left
            if last:
                break
        if not self.own:
            self.finish_ink()
        self.finish_fonts()
        result = self.result
        if self.warnings:
            result = dataclasses.replace(
                result, warnings=result.warnings + tuple(self.warnings)
            )
        return StatsFigure(
            figure=self.figure,
            axes={"main": self.ax},
            result=result,
            own=self.own,
            roles=self.roles,
            inside=tuple(self.inside),
            subsample=self.subsample,
            brackets=tuple(
                zip(
                    self.bracket_labels,
                    self.bracket_lines,
                    self.bracket_names,
                    strict=True,
                )
            ),
            paths=self.drawn_paths(),
            view=self.view,
            colors=tuple(self.colors),
            centers=tuple(
                (self.result.levels[e.i], e.point) for e in self.label_entries
            ),
        )

    def draw_data(self) -> None:
        ax, result, options = self.ax, self.result, self.options
        layers = options.layers
        rng = np.random.Generator(np.random.PCG64(result.settings.seed))
        points = self.place_points(rng)
        outline = "violin" in layers or self.subsample is not None
        for i, (level, x) in enumerate(zip(result.levels, self.values, strict=True)):
            color = self.colors[i]
            if outline:
                self.draw_density(i, level, x, color)
            if "box" in layers and x.size > 0:
                box = _box_artist(x, float(i), i)
                ax.add_patch(box)
                apply_args(box, options.boxplot_args, "boxplot_args")
                if "points" not in layers:
                    self.draw_outliers(i, x)
            px_, py_ = points[i]
            if "points" in layers and py_.size > 0:
                drawn = points_artist(ax, px_, py_, color, i)
                apply_args(drawn, options.point_args, "point_args")
        if self.subsample is not None:
            # Said after the densities, which may leave a level without one.
            self.subsample = dataclasses.replace(
                self.subsample, undrawn=tuple(self.undrawn)
            )
            self.warnings.append(self.subsample_warning(self.subsample))
        self.draw_paths(points)
        if options.centrality:
            self.draw_centrality()
        self.make_bracket_labels()

    def draw_outliers(self, i: int, x: FloatArray) -> None:
        """Without a points layer, mark the values beyond the box's whiskers
        (with one, the points show them): small open circles at the level."""
        far = outliers(x)
        if far.size == 0:
            return
        marks = self.ax.scatter(
            np.full(far.size, float(i)),
            far,
            s=14.0,
            facecolors="none",
            edgecolors=theme.SOFT_INK,
            linewidths=0.8,
            zorder=2.5,
            gid=f"outliers-{i}",
        )
        self.outlier_marks.append(marks)

    def draw_density(
        self, i: int, level: str, x: FloatArray, color: theme.Color
    ) -> None:
        drawn = _density_outline(x)
        what = "violin" if "violin" in self.options.layers else "density outline"
        if drawn is None or isinstance(drawn, str):
            why = "it needs at least two different values" if drawn is None else drawn
            self.warnings.append(f"The {what} of level {level!r} is not drawn: {why}.")
            self.undrawn.append(level)
            return
        grid, half = drawn
        xs = np.concatenate([i - half, (i + half)[::-1]])
        ys = np.concatenate([grid, grid[::-1]])
        if what == "violin":
            patch = _violin_artist(np.column_stack([xs, ys]), color, i)
            self.ax.add_patch(patch)
            apply_args(patch, self.options.violin_args, "violin_args")
        else:
            patch = Polygon(
                np.column_stack([xs, ys]),
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                zorder=1.0,
                gid=f"density-{i}",
            )
            self.ax.add_patch(patch)

    def draw_centrality(self) -> None:
        ax = self.ax
        for i, (d, label) in enumerate(
            zip(self.result.descriptives, self.texts.centrality_labels, strict=True)
        ):
            if d.value is None:
                continue  # a MAP without a density: nothing to mark (D55)
            place = self.label_place(i)
            end = i + LABEL_OFFSET if place.side == "right" else i - LABEL_OFFSET
            link = Line2D(
                [i, end],
                [d.value, d.value],
                color=theme.CENTRALITY_COLOR,
                linewidth=0.7,
                zorder=4.0,
                gid=f"centrality-link-{i}",
            )
            ax.add_line(link)
            point = ax.scatter(
                [i],
                [d.value],
                s=CENTRALITY_SIZE,
                color=theme.CENTRALITY_COLOR,
                zorder=5.0,
                gid=f"centrality-{i}",
            )
            where: dict[str, Any] = {}
            va = "center"
            if place.rise != "center":
                # The box's edge clears the point's height by a point.
                rise = LABEL_BOX_PAD * theme.SIZE_LABEL + 1.0
                up = place.rise == "above"
                where["transform"] = offset_copy(
                    ax.transData,
                    fig=self.figure,
                    y=rise if up else -rise,
                    units="points",
                )
                va = "bottom" if up else "top"
            text = ax.text(
                end,
                d.value,
                label,
                ha="left" if place.side == "right" else "right",
                va=va,
                fontproperties=theme.font(theme.SIZE_LABEL),
                bbox={
                    "boxstyle": f"round,pad={LABEL_BOX_PAD}",
                    "facecolor": "white",
                    "edgecolor": theme.CENTRALITY_COLOR,
                    "linewidth": 0.7,
                },
                zorder=6.0,
                gid=f"centrality-label-{i}",
                **where,
            )
            self.label_entries.append(
                _Label(
                    i=i, value=d.value, link=link, point=point, text=text, place=place
                )
            )
            if place.side == "left":
                self.left_labels.append(text)
            if place.rise != "center":
                self.raised.append(
                    (text, px(LABEL_BOX_PAD * theme.SIZE_LABEL + 1.0, self.dpi))
                )
            self.centrality_labels.append(
                self.role(
                    text,
                    f"centrality label of {d.level!r}",
                    self.result.text.centrality_labels[i],
                )
            )
            self.inside.append(text)

    # -- centrality label places ------------------------------------------------

    def layout(self) -> None:
        if self.own:
            self.layout_own()
        else:
            self.layout_in_axes()

    def place_label(self, entry: _Label, place: LabelPlace) -> None:
        """Move a drawn centrality label (and its link line) to ``place``."""
        ax, i = self.ax, entry.i
        end = i + LABEL_OFFSET if place.side == "right" else i - LABEL_OFFSET
        entry.link.set_xdata([i, end])
        text = entry.text
        text.set_x(end)
        text.set_horizontalalignment("left" if place.side == "right" else "right")
        if place.rise == "center":
            text.set_transform(ax.transData)
            text.set_verticalalignment("center")
        else:
            rise = LABEL_BOX_PAD * theme.SIZE_LABEL + 1.0
            up = place.rise == "above"
            text.set_transform(
                offset_copy(
                    ax.transData, fig=self.figure, y=rise if up else -rise,
                    units="points",
                )
            )  # fmt: skip
            text.set_verticalalignment("bottom" if up else "top")
        entry.place = place
        self.left_labels = [
            e.text for e in self.label_entries if e.place.side == "left"
        ]
        lift = px(LABEL_BOX_PAD * theme.SIZE_LABEL + 1.0, self.dpi)
        self.raised = [
            (e.text, lift) for e in self.label_entries if e.place.rise != "center"
        ]

    def point_boxes(self) -> list[Bbox]:
        """Each centrality point's square in pixels (its marker's extent)."""
        half = px(np.sqrt(CENTRALITY_SIZE) / 2, self.dpi)
        boxes: list[Bbox] = []
        for e in self.label_entries:
            x, y = (float(v) for v in self.ax.transData.transform((e.i, e.value)))
            boxes.append(Bbox.from_extents(x - half, y - half, x + half, y + half))
        return boxes

    def choose_label_places(self) -> bool:
        """Put each centrality label where it covers no centrality point and no
        other centrality label: the first free place of :func:`label_candidates`
        (the drawing's own choice first, so a figure where nothing collides is
        unchanged), else the one that covers least. Whether a label moved.

        Measured in pixels on the current layout; ``fig.check()`` reports a point
        still half covered (``centrality_label_overlap``).
        """
        entries = self.label_entries
        if len(entries) < 2:
            return False
        points = self.point_boxes()
        area = max(points[0].width * points[0].height, 1.0)
        moved = False
        for entry in entries:
            others = [self.extent(e.text) for e in entries if e is not entry]
            start = entry.place
            best: tuple[float, LabelPlace] | None = None
            for place in label_candidates(self.label_place(entry.i)):
                self.place_label(entry, place)
                box = self.extent(entry.text)
                bad = sum(_overlap(box, p) / area for p in points) + sum(
                    _overlap(box, o) / max(o.width * o.height, 1.0) for o in others
                )
                if best is None or bad < best[0]:
                    best = (bad, place)
                if bad == 0.0:
                    break
            assert best is not None
            self.place_label(entry, best[1])
            moved |= best[1] != start
        return moved

    def make_bracket_labels(self) -> None:
        levels = self.result.levels
        for n, pair in enumerate(self.result.pairwise):
            if not pair.displayed:
                continue
            m = len(self.bracket_labels)
            i, j = sorted((levels.index(pair.group1), levels.index(pair.group2)))
            text = self.ax.text(
                (i + j) / 2,
                0.0,
                self.texts.pairwise_labels[n],
                ha="center",
                va="bottom",
                fontproperties=theme.font(theme.SIZE_LABEL),
                zorder=6.0,
                gid=f"bracket-label-{m}",
            )
            self.role(
                text,
                f"bracket label of {pair.group1!r} and {pair.group2!r}",
                self.result.text.pairwise_labels[n],
            )
            self.inside.append(text)
            self.bracket_labels.append(text)
            self.bracket_pairs.append((i, j))
            self.bracket_names.append(
                f"the bracket of {pair.group1!r} and {pair.group2!r}"
            )

    def draw_ticks_and_labels(self) -> None:
        ax, result, options = self.ax, self.result, self.options
        k = len(result.levels)
        ax.set_xticks(
            list(range(k)),
            labels=self.tick_labels(),
            parse_math=False,
            fontproperties=theme.font(theme.SIZE_TICK),
        )
        ax.tick_params(axis="x", length=0.0, pad=6.0)
        ax.tick_params(axis="y", labelsize=theme.SIZE_TICK)
        ax.set_xlabel(
            result.x if options.xlab is None else options.xlab,
            fontproperties=theme.font(theme.SIZE_AXIS_LABEL),
            parse_math=options.xlab is not None,
        )
        ax.set_ylabel(
            result.y if options.ylab is None else options.ylab,
            fontproperties=theme.font(theme.SIZE_AXIS_LABEL),
            parse_math=options.ylab is not None,
        )

    def draw_text_blocks(self) -> None:
        """Title, subtitle and caption lines (positions set by the layout).

        With inferplot's own figure they are figure texts; with ``ax=`` the title
        is the Axes title and the subtitle and caption go inside the Axes.
        """
        options, result = self.options, self.result
        title = options.title
        if options.subtitle is None:
            subtitle, plain_subtitle = self.texts.subtitle, result.text.subtitle
        else:
            subtitle, plain_subtitle = options.subtitle, None
        plain_captions = [] if result.text.caption is None else (
            result.text.caption.split("\n")
        )  # fmt: skip
        lines: list[tuple[str, bool, str | None]] = [
            (line, line.startswith("$") and line.endswith("$"), plain)
            for line, plain in zip(
                self.texts.caption_lines, plain_captions, strict=True
            )
        ]
        if options.caption is not None:
            lines.append((options.caption, True, None))
        ax = self.ax
        place = self.figure if self.own else ax
        where: dict[str, Any] = {} if self.own else {"transform": ax.transAxes}
        if title is not None:
            title_font = theme.font(theme.SIZE_TITLE, weight="bold")
            if self.own:
                text = self.figure.text(
                    0.0, 1.0, title, ha="left", va="top", fontproperties=title_font
                )
                self.header.append(text)
            else:
                text = ax.set_title(title, loc="left", fontproperties=title_font)
            text.set_gid("title")
            self.role(text, "title")
        if subtitle is not None:
            text = place.text(
                0.0,
                1.0,
                subtitle,
                ha="left",
                va="top",
                fontproperties=theme.font(theme.SIZE_SUBTITLE),
                gid="subtitle",
                **where,
            )
            self.header.append(self.role(text, "subtitle", plain_subtitle))
            if not self.own:
                self.inside.append(text)
        for n, (line, math_, plain) in enumerate(lines):
            text = place.text(
                0.0,
                0.0,
                line,
                ha="left",
                va="top" if self.own else "bottom",
                fontproperties=theme.font(theme.SIZE_CAPTION),
                parse_math=math_,
                gid=f"caption-{n}",
                **where,
            )
            self.captions.append(self.role(text, f"caption line {n + 1}", plain))
            if not self.own:
                self.inside.append(text)

    # -- layout ----------------------------------------------------------------

    def x_needs(self) -> tuple[float, float, float]:
        """Widest centrality label right and left of its point, and widest tick
        label, in pixels."""
        left = {id(t) for t in self.left_labels}
        label_w = max(
            (self.extent(t).width for t in self.centrality_labels if id(t) not in left),
            default=0.0,
        )
        left_w = max((self.extent(t).width for t in self.left_labels), default=0.0)
        probe = [
            self.figure.text(
                0.0,
                0.0,
                s,
                fontproperties=theme.font(theme.SIZE_TICK),
                parse_math=False,
            )
            for s in self.tick_labels()
        ]
        tick_w = max(self.extent(t).width for t in probe)
        for t in probe:
            t.remove()
        return label_w, left_w, tick_w

    def x_layout(self, spacing: float | None) -> tuple[float, float, float]:
        """x-limits and pixels per level; ``spacing`` None fits the Axes width."""
        k = len(self.result.levels)
        gap = px(GAP_PT, self.dpi)
        label_w, left_w, tick_w = self.x_needs()
        right_labels = len(self.centrality_labels) > len(self.left_labels)
        reach = LABEL_OFFSET if right_labels else 0.0
        left_reach = LABEL_OFFSET if self.left_labels else 0.0
        if spacing is None:
            width = float(self.ax.bbox.width)
            share = (label_w + gap) / width if right_labels else 0.0
            span = float(k)
            if share < 1.0:
                span = max(span, (k - 0.5 + reach) / (1.0 - share))
            if self.left_labels:
                # The same bound for labels on the left, and for both sides at once.
                left_share = (left_w + gap) / width
                if left_share < 1.0:
                    span = max(span, (k - 0.5 + left_reach) / (1.0 - left_share))
                if share + left_share < 1.0:
                    both = k - 1 + reach + left_reach
                    span = max(span, both / (1.0 - share - left_share))
            spacing = width / span
        else:
            spacing = max(spacing, label_w + 2 * gap, tick_w + 2 * gap)
            if self.left_labels:
                spacing = max(spacing, left_w + 2 * gap)
        left, right = -0.5, k - 0.5
        if right_labels:
            right = max(right, (k - 1) + reach + (label_w + gap) / spacing)
        if self.left_labels:
            left = min(left, -left_reach - (left_w + gap) / spacing)
        return left, right, spacing

    def place_brackets(
        self, left: float, right: float, spacing: float
    ) -> tuple[float, float, int, list[int]]:
        gap = max(0.08, px(2 * GAP_PT, self.dpi) / spacing)
        specs = [
            Bracket(
                i=i,
                j=j,
                width=(self.extent(t).width + px(2 * LABEL_PAD_PT, self.dpi)) / spacing,
            )
            for (i, j), t in zip(self.bracket_pairs, self.bracket_labels, strict=True)
        ]
        levels = list(stack(specs, gap=gap))
        for spec in specs:
            lo, hi = spec.extent()
            left = min(left, lo - 0.05)
            right = max(right, hi + 0.05)
        return left, right, (max(levels) + 1 if levels else 0), levels

    def center_half_px(self) -> float:
        """How far a centrality label reaches above or below its point, in pixels."""
        raised = {id(t): rise for t, rise in self.raised}
        return max(
            (
                self.extent(t).height + raised[id(t)]
                if id(t) in raised
                else self.extent(t).height / 2
                for t in self.centrality_labels
            ),
            default=0.0,
        )

    def y_layout(
        self,
        n_levels: int,
        *,
        top_px: float = 0.0,
        bottom_px: float = 0.0,
        height_px: float | None = None,
        ppu: float | None = None,
    ) -> VerticalLayout:
        data = [x for x in self.values if x.size > 0]
        data_min = min(float(x.min()) for x in data)
        data_max = max(float(x.max()) for x in data)
        centers = (
            [d.value for d in self.result.descriptives if d.value is not None]
            if self.centrality_labels
            else []
        )
        center_half = self.center_half_px()
        label_h = max((self.extent(t).height for t in self.bracket_labels), default=0.0)
        fixed = height_px is not None
        return vertical_layout(
            data_min=data_min,
            data_max=data_max,
            centers=centers,
            n_levels=n_levels,
            center_half_px=center_half,
            label_px=label_h,
            pad_px=px(LABEL_PAD_PT, self.dpi),
            tip_px=px(BRACKET_TIP_PT, self.dpi),
            line_px=px(BRACKET_LINEWIDTH_PT, self.dpi),
            top_px=top_px,
            bottom_px=bottom_px,
            min_height_px=None if fixed else MIN_AXES_HEIGHT_IN * self.dpi,
            data_height_px=MIN_DATA_HEIGHT_IN * self.dpi,
            height_px=height_px,
            ppu=ppu,
        )

    def apply_limits(
        self, left: float, right: float, layout: VerticalLayout, levels: list[int]
    ) -> None:
        ax = self.ax
        ax.set_xlim(left, right)
        ax.set_ylim(layout.bottom, layout.top)
        self.view.xlim = (left, right)
        self.view.ylim = (layout.bottom, layout.top)
        if self.bracket_lines:  # laid out before: move the lines
            self.levels_of_brackets = levels
            self.place_bracket_levels(layout)
            return
        for m, ((i, j), level) in enumerate(
            zip(self.bracket_pairs, levels, strict=True)
        ):
            y = layout.levels[level]
            line = Line2D(
                [i, i, j, j],
                [y - layout.tip, y, y, y - layout.tip],
                color=theme.INK,
                linewidth=BRACKET_LINEWIDTH_PT,
                zorder=4.0,
                gid=f"bracket-{m}",
            )
            ax.add_line(line)
            self.bracket_lines.append(line)
        self.levels_of_brackets = levels
        self.place_bracket_levels(layout)

    def place_bracket_levels(self, layout: VerticalLayout) -> None:
        """Put each bracket and its label at its level of ``layout``."""
        for (i, j), line, text, level in zip(
            self.bracket_pairs,
            self.bracket_lines,
            self.bracket_labels,
            self.levels_of_brackets,
            strict=True,
        ):
            y = layout.levels[level]
            line.set_data([i, i, j, j], [y - layout.tip, y, y, y - layout.tip])
            text.set_position(((i + j) / 2, y + layout.label_pad))

    def layout_own(self) -> None:
        dpi = self.dpi
        left, right, spacing = self.x_layout(MIN_LEVEL_SPACING_IN * dpi)
        left, right, n_levels, levels = self.place_brackets(left, right, spacing)
        layout = self.y_layout(n_levels)
        self.apply_limits(left, right, layout, levels)
        axes_w = spacing * (right - left)
        axes_h = layout.height_px
        margin, gap = px(MARGIN_PT, dpi), px(GAP_PT, dpi)
        header = [self.extent(t) for t in self.header]
        captions = [self.extent(t) for t in self.captions]
        widths = [b.width for b in header + captions]
        top_block = sum(b.height + gap for b in header)
        bottom_block = (
            sum(b.height for b in captions) + gap * len(captions) if captions else 0.0
        )
        # Room the Axes decorations need around the plotting area (measured).
        extra = [0.8 * dpi, 0.1 * dpi, 0.7 * dpi, 0.1 * dpi]  # left right bottom top
        canvas = self.figure.canvas
        for _ in range(5):
            le, re, be, te = extra
            content_w = max([le + axes_w + re, *widths])
            ax_w = content_w - le - re
            width = 2 * margin + content_w
            height = 2 * margin + top_block + te + axes_h + be + bottom_block
            self.figure.set_size_inches(width / dpi, height / dpi)
            x0, y0 = margin + le, margin + bottom_block + be
            self.ax.set_position(
                (x0 / width, y0 / height, ax_w / width, axes_h / height)
            )
            top = height - margin
            for text, box in zip(self.header, header, strict=True):
                text.set_position((margin / width, top / height))
                top -= box.height + gap
            top = margin + bottom_block - gap
            for text, box in zip(self.captions, captions, strict=True):
                text.set_position((margin / width, top / height))
                top -= box.height + gap
            canvas.draw()
            measured = self.decoration_extra()
            if all(abs(a - b) <= 0.5 for a, b in zip(measured, extra, strict=True)):
                break
            extra = measured

    def decoration_extra(self) -> list[float]:
        """Pixels the Axes' texts reach beyond its plotting area on each side."""
        ax = self.ax
        renderer = self.renderer()
        frame = ax.get_window_extent(renderer)
        boxes = [ax.get_tightbbox(renderer)]
        boxes += [self.extent(t) for t in self.centrality_labels + self.bracket_labels]
        union = Bbox.union([b for b in boxes if b is not None])
        pad = 1.0
        return [
            max(0.0, frame.x0 - union.x0) + pad,
            max(0.0, union.x1 - frame.x1) + pad,
            max(0.0, frame.y0 - union.y0) + pad,
            max(0.0, union.y1 - frame.y1) + pad,
        ]

    def layout_in_axes(self) -> None:
        dpi = self.dpi
        pad, gap = px(LABEL_PAD_PT, dpi), px(GAP_PT, dpi)
        header = [self.extent(t) for t in self.header]
        captions = [self.extent(t) for t in self.captions]
        top_px = sum(b.height for b in header) + 2 * pad if header else 0.0
        bottom_px = (
            sum(b.height for b in captions) + gap * (len(captions) - 1) + 2 * pad
            if captions
            else 0.0
        )
        to_pt = 72.0 / dpi
        for text in self.header:
            text.set_transform(
                offset_copy(
                    self.ax.transAxes, fig=self.figure, x=pad * to_pt, y=-pad * to_pt,
                    units="points",
                )
            )  # fmt: skip
        rise = pad
        for text, box in zip(reversed(self.captions), reversed(captions), strict=True):
            text.set_transform(
                offset_copy(
                    self.ax.transAxes, fig=self.figure, x=pad * to_pt, y=rise * to_pt,
                    units="points",
                )
            )  # fmt: skip
            rise += box.height + gap
        left, right, spacing = self.x_layout(None)
        left, right, n_levels, levels = self.place_brackets(left, right, spacing)
        layout = self.y_layout(
            n_levels,
            top_px=top_px,
            bottom_px=bottom_px,
            height_px=float(self.ax.bbox.height),
        )
        self.y_room = (n_levels, top_px, bottom_px)
        self.apply_limits(left, right, layout, levels)
        setattr(self.ax, _PANEL_ATTR, self)
        self.share_y()

    def layout_at(self, ppu: float) -> VerticalLayout:
        """This panel's y layout at ``ppu`` pixels per data unit, full spacing."""
        n_levels, top_px, bottom_px = self.y_room
        return self.y_layout(n_levels, top_px=top_px, bottom_px=bottom_px, ppu=ppu)

    def share_y(self) -> None:
        """With a y axis shared with other Axes: the limits that every panel
        sharing it needs, at one scale, so a later drawing never cuts an earlier
        panel's data or brackets.

        The panels inferplot drew are laid out again at the shared scale (their
        brackets move); another Axes keeps its data in view. The scale is found
        by iterating span -> span needed at that span, which converges while the
        texts and brackets take less than each Axes' height.
        """
        ax = self.ax
        siblings = [a for a in _siblings(ax, "y") if a is not ax]
        if not siblings:
            return
        panels: list[Drawing] = [self]
        fixed: list[float] = []
        for other in siblings:
            drawn = getattr(other, _PANEL_ATTR, None)
            if not other.has_data():
                continue
            if isinstance(drawn, Drawing) and drawn.ax is other:
                panels.append(drawn)
            else:
                ys = np.asarray(other.dataLim.intervaly, dtype=np.float64)
                fixed += [float(y) for y in ys[np.isfinite(ys)]]
        if len(panels) == 1 and not fixed:
            return
        lo, hi = self.view.ylim
        span = hi - lo
        if fixed:
            span = max(span, max(fixed) - min(fixed))
        layouts: list[VerticalLayout] = []
        for _ in range(100):
            layouts = [p.layout_at(float(p.ax.bbox.height) / span) for p in panels]
            lo = min([lay.bottom for lay in layouts] + fixed)
            hi = max([lay.top for lay in layouts] + fixed)
            if abs((hi - lo) - span) <= 1e-9 * span:
                break
            span = hi - lo
        ax.set_ylim(lo, hi)
        for panel, lay in zip(panels, layouts, strict=True):
            panel.place_bracket_levels(lay)
            panel.view.ylim = (lo, hi)

    def finish_ink(self) -> None:
        """With ``ax=``: inferplot's ink in contrast with what it is drawn on.

        Texts and lines drawn inside the Axes (subtitle, caption, bracket lines
        and labels) take :func:`theme.ink_for` of the Axes background, and the
        axis texts and title outside it that of the figure background: the usual
        near-black on a light background, a near-white on a dark one; so do the
        outlier markers. The centrality labels keep theirs, on their own white
        boxes. On a light
        background nothing is changed.
        """
        inside = axes_background(self.ax)
        outside = figure_background(self.figure)
        ink_in, ink_out = theme.ink_for(inside), theme.ink_for(outside)
        if ink_in != theme.INK:
            for text in [*self.header, *self.captions, *self.bracket_labels]:
                text.set_color(ink_in)
            for line in self.bracket_lines:
                line.set_color(ink_in)
            for marks in self.outlier_marks:
                marks.set_edgecolor(ink_in)
        if ink_out != theme.INK:
            ax = self.ax
            ax.xaxis.label.set_color(ink_out)
            ax.yaxis.label.set_color(ink_out)
            ax.tick_params(axis="both", labelcolor=ink_out)
            for text in ax.findobj(Text):
                if self.roles.get(id(text), "").startswith("title"):
                    text.set_color(ink_out)

    def finish_fonts(self) -> None:
        """DejaVu Sans, also for mathtext, on every text inferplot draws.

        With inferplot's own figure that is every text of the figure. With ``ax=``
        it is the axis texts (labels and tick labels, which inferplot sets) and the
        texts this drawing created; texts the user placed before the call, such as
        their own title or annotations, keep their font.
        """
        # The own figure was drawn by the layout, so its tick labels exist; tick
        # labels made later copy their font from the existing ones.
        if self.own:
            texts = self.figure.findobj(Text)
        else:
            texts = [
                *self.ax.xaxis.findobj(Text),
                *self.ax.yaxis.findobj(Text),
                *(t for t in self.ax.findobj(Text) if id(t) in self.roles),
            ]
        for item in texts:
            item.set_fontfamily(theme.families())
            item.set_math_fontfamily(theme.MATH_FONT_FAMILY)
