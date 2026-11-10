# matplotlib's and NumPy's stubs type many ``**kwargs`` and overloads as Unknown,
# which strict mode reports on every call; the arguments we pass are all typed.
# pyright: reportUnknownMemberType=false
"""The drawing machinery every inferplot figure shares.

* :func:`draw`: the palette, the theme and fonts, and the checks made before
  anything is drawn: each artist dict styles a scratch artist made as the drawing
  makes it, which is drawn on a scratch figure and written as SVG and PDF
  (matplotlib checks some values only then), and the user's texts are laid out
  once as mathtext, so a call that raises leaves a given Axes as it was;
* :class:`BaseDrawing`: the figure or the given Axes, the header (title and
  subtitle) and caption blocks, inferplot's own figure sized from the rendered
  extents of its texts (:meth:`BaseDrawing.fit_figure`), the blocks placed inside
  a given Axes, ink that contrasts with a dark background, and inferplot's font on
  the texts it draws;
* :func:`check_axes`, :func:`apply_args` and the refusal of artist properties that
  carry or move the data.

``render.comparison`` (betweenstats, withinstats), ``render.histostats`` and
``render.scatterstats`` build on it.
"""

from __future__ import annotations

import dataclasses
import io
import math
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

import numpy as np
from matplotlib import cbook
from matplotlib.artist import ArtistInspector
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox, offset_copy

from inferplot import _validate as v
from inferplot.render import theme
from inferplot.render.figure import (
    axes_background,
    describe_text,
    figure_background,
    plain_text,
    remember,
    reported_glyphs,
    root_figure,
)
from inferplot.result import StatsResult

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.backend_bases import RendererBase

    from inferplot.render.figure import StatsFigure

__all__ = [
    "GAP_PT",
    "LABEL_BOX_PAD",
    "LABEL_PAD_PT",
    "MARGIN_PT",
    "BaseDrawing",
    "LayerArgs",
    "UserTexts",
    "apply_args",
    "check_artist_args",
    "check_axes",
    "check_drawable",
    "check_given_axes",
    "check_result_type",
    "check_scale",
    "check_subtitle",
    "check_user_texts",
    "draw",
    "px",
    "shared_limits",
    "siblings",
]

# Geometry in points.
MARGIN_PT: Final = 10.0
GAP_PT: Final = 5.0
LABEL_PAD_PT: Final = 3.0  # between a bracket and its label, and around labels
LABEL_BOX_PAD: Final = 0.3  # label box padding, in font sizes


def px(points: float, dpi: float) -> float:
    return points * dpi / 72.0


class UserTexts(Protocol):
    """The user's texts of a drawing's checked options."""

    @property
    def xlab(self) -> str | None: ...
    @property
    def ylab(self) -> str | None: ...
    @property
    def title(self) -> str | None: ...
    @property
    def subtitle(self) -> str | None: ...
    @property
    def caption(self) -> str | None: ...


# --------------------------------------------------------------------------- #
# checks made before anything is drawn
# --------------------------------------------------------------------------- #


def apply_args(
    artist: Artist, args: Mapping[str, object], name: str, *, levels: bool = True
) -> None:
    """``artist.set(**args)``, with a message naming the argument on failure.

    A name the artist has no setter for is an unknown property (``TypeError``);
    any error from setting a known property is a rejected value (``ValueError``),
    whatever exception type matplotlib uses for it. ``levels``: the figure
    colours its points by level (betweenstats, withinstats), for the messages.
    """
    if not args:
        return
    kind = type(artist).__name__
    for key in args:
        if not callable(getattr(artist, f"set_{key}", None)):
            close = v.closest_name(key, ArtistInspector(artist).get_setters())
            shorthands = _SCATTER_SHORTHANDS if levels else _ONE_COLOR_SHORTHANDS
            shorthand = shorthands.get(key) if kind == "PathCollection" else None
            if shorthand is not None:
                hint = f" {shorthand}"
            elif close is not None:
                hint = f" Did you mean {close!r}?"
            else:
                hint = f" matplotlib's documentation of {kind} lists its properties."
            raise TypeError(
                f"{name} has a property matplotlib's {kind} does not have: "
                f"{key!r}.{hint}"
            )
    try:
        artist.set(**args)
    except Exception as error:
        raise ValueError(f"{name} has a value matplotlib rejects: {error}") from error


# scatter()'s argument names, which the points' PathCollection spells otherwise.
_SCATTER_SHORTHANDS: Final = {
    "s": (
        "That is scatter()'s name for the marker size; the points' property is "
        "'sizes', a list of areas in points squared, for example "
        "point_args={'sizes': [30]}."
    ),
    "c": (
        "That is scatter()'s name for the colour; the points' property is "
        "'color' (one colour for every level, for example "
        "point_args={'color': 'black'}), and palette= colours them by level."
    ),
}
# The same for points of one colour (scatterstats).
_ONE_COLOR_SHORTHANDS: Final = {
    **_SCATTER_SHORTHANDS,
    "c": (
        "That is scatter()'s name for the colour; the points' property is "
        "'color', for example point_args={'color': 'black'} (palette= sets it "
        "too, with the marginal histograms)."
    ),
}


@dataclasses.dataclass(frozen=True, slots=True)
class LayerArgs:
    """An artist dict and the layer it styles, for the checks before drawing.

    ``name`` is the argument (``point_args``), ``layer`` the layer in messages
    (``points``), ``noun`` what its artists are called (``boxes``), ``make``
    makes one artist of the layer in a scratch Axes as the drawing makes it,
    ``pronoun`` refers to the noun ("them", "it"), ``asks`` says what asks for the
    layer (", although layers= asks for them") and ``advice`` how to leave it out
    instead of hiding it.
    """

    name: str
    layer: str
    noun: str
    make: Callable[[Axes], Artist]
    pronoun: str = "them"
    asks: str = ""
    advice: str = "Choose a visible style."
    levels: bool = True  # the layer is coloured by level


# The vector formats a scratch drawing is also written to: matplotlib checks some
# values only there (a NaN line width passes Agg and fails in PDF).
_SCRATCH_FORMATS: Final = ("svg", "pdf")


def _scratch(spec: LayerArgs) -> tuple[Figure, Artist]:
    """A scratch figure holding one artist of ``spec``'s layer, made as the drawing
    makes it; nothing of it reaches the user."""
    figure = Figure(figsize=(2.0, 2.0), dpi=100)
    FigureCanvasAgg(figure)
    figure.patch.set_visible(False)  # so that only the artist leaves pixels
    ax = figure.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_axis_off()  # no texts, so writing it embeds no fonts; no background
    ax.set_xlim(-0.5, 0.5)  # the whole artist in view
    ax.set_ylim(-0.5, 4.5)
    return figure, spec.make(ax)


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


def _check_drawn(
    spec: LayerArgs, args: Mapping[str, object], background: theme.Color | None
) -> str | None:
    """Apply ``args`` to a scratch artist of ``spec``'s layer and draw it; why
    the artist is then invisible (it leaves no pixel, or none that can be told
    from ``background``), or None.

    ``Artist.set`` stores some values without checking them (a line width of
    "bad" for points); matplotlib rejects them only when drawing, which with ``ax=``
    would be the user's later ``savefig``.
    """
    name, layer = spec.name, spec.layer
    figure, artist = _scratch(spec)
    apply_args(artist, args, name, levels=spec.levels)
    error = _draw_error(figure)
    if error is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figure.canvas.draw()
        return _invisible(figure, background)
    culprits: list[str] = []
    for key, value in args.items():
        alone, single = _scratch(spec)
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
# Properties of the rectangles of a histogram that carry its bins, refused like
# the positions of the other layers.
_BAR_DATA: Final = frozenset({"x", "y", "width", "height", "bounds"})


def _refuse_moving_args(spec: LayerArgs, args: Mapping[str, object]) -> None:
    """Refuse a property that carries or moves the data or detaches the artist.

    Aliases are read as the property they stand for.
    """
    name = spec.name
    figure, artist = _scratch(spec)
    del figure
    for key in args:
        prop = key
        if key not in _REFUSED:
            try:
                (prop,) = cbook.normalize_kwargs({key: None}, artist)
            except TypeError:
                continue  # a property of two names at once; Artist.set reports it
        why = _REFUSED.get(prop)
        if why is None and type(artist).__name__ == "Rectangle" and prop in _BAR_DATA:
            why = _DATA
        if why is not None:
            raise ValueError(
                f"{name} sets {key!r}, which {why.format(layer=spec.noun)}. "
                f"Remove {key!r} from {name}"
                + (
                    "; to draw other values, change the data before the analysis."
                    if why is _DATA
                    else "."
                )
            )


# A layer whose every pixel has less contrast than this with the background
# cannot be seen (view_checks' COLOR_CONTRAST: white on white is 1:1).
_SEEN_CONTRAST: Final = 1.1


def _invisible(figure: Figure, background: theme.Color | None) -> str | None:
    """Why a scratch figure (transparent but for its one artist) shows nothing
    on ``background``: "nothing of them is drawn" (every pixel of the Agg buffer
    is fully transparent), or the colour it is drawn in where no pixel, laid
    over ``background``, has 1.1:1 contrast with it (white on white); None
    where something can be seen."""
    canvas = cast("FigureCanvasAgg", figure.canvas)
    rgba = np.asarray(canvas.buffer_rgba()).reshape(-1, 4)
    drawn = rgba[rgba[:, 3] > 0]
    if drawn.shape[0] == 0:
        return "nothing of {pronoun} is drawn"
    if background is None:
        return None
    under = theme.over(background, "white")
    best, shown = 0.0, under
    for pixel in np.unique(drawn, axis=0):
        color = theme.over(tuple(float(c) / 255.0 for c in pixel), under)
        ratio = theme.contrast(color, under)
        if ratio > best:
            best, shown = ratio, color
        if ratio >= _SEEN_CONTRAST:
            return None
    return (
        f"{{subject}} would be drawn in {to_hex(shown)} on the background "
        f"{to_hex(under)} ({best:.2f}:1 contrast), which cannot be told apart"
    )


_SUBJECT: Final = {"them": "they"}  # the pronoun as a sentence's subject


def _invisible_warning(spec: LayerArgs, args: Mapping[str, object], why: str) -> str:
    shown = ", ".join(f"{key}={v.describe_value(value)}" for key, value in args.items())
    subject = _SUBJECT.get(spec.pronoun, spec.pronoun)
    reason = why.format(pronoun=spec.pronoun, subject=subject)
    return (
        f"{spec.name}={{{shown}}} makes the {spec.noun} invisible: {reason}"
        f"{spec.asks}. {spec.advice}"
    )


def check_artist_args(
    given: Sequence[tuple[LayerArgs, Mapping[str, object], str | None]],
    *,
    ax: Axes | None = None,
    colors: bool = False,
) -> list[str]:
    """Each non-empty artist dict styles a layer that is drawn (``refusal``, the
    third item, is the message raised when it is not), sets no property that
    moves or replaces the data, and an artist of that layer styled with it draws;
    checked before anything is drawn.

    Returns a warning for each dict that leaves its layer invisible (D39): no
    pixel drawn, or with ``colors`` (histostats, scatterstats, whose layers lie
    on the background) none that can be told from the background (that of
    ``ax``, white for inferplot's own figure).
    """
    background: theme.Color = (
        (1.0, 1.0, 1.0, 1.0) if ax is None else (*axes_background(ax), 1.0)
    )
    found: list[str] = []
    for spec, args, refusal in given:
        if not args:
            continue
        if refusal is not None:
            raise ValueError(refusal)
        _refuse_moving_args(spec, args)
        why = _check_drawn(spec, args, background if colors else None)
        if why is not None:
            found.append(_invisible_warning(spec, args, why))
    return found


_USER_TEXTS: Final = ("title", "subtitle", "caption", "xlab", "ylab")


def check_user_texts(options: UserTexts) -> None:
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


def check_axes(ax: Axes, *, shared_x: bool = False) -> None:
    """A given Axes is a plain rectangular 2D one; checked before drawing.

    Unless ``shared_x``, it may not share its x axis: the comparison figures'
    tick labels name each level with that panel's n.
    """
    name = str(getattr(ax, "name", ""))
    if name != "rectilinear":
        where = (
            "the levels lie along x and the values along y"
            if not shared_x
            else "the values lie along x and y"
        )
        raise ValueError(
            f"ax is a {name!r} Axes; inferplot draws only into a plain rectangular "
            f"2D Axes, where {where}. Pass an "
            "Axes made without projection=, for example figure.add_subplot()."
        )
    if not shared_x and len(siblings(ax, "x")) > 1:
        raise ValueError(
            "ax shares its x axis with another Axes (sharex=True, or twinx()), and "
            "Axes that share an x axis share its tick labels; inferplot's tick "
            "labels name each level with that panel's n, so one panel would show "
            "the other's. Make the panels with sharex=False (the default), for "
            "example figure.subplots(1, 2, sharey=True)."
        )


def check_result_type(result: StatsResult, analysis: str) -> None:
    """A :class:`StatsResult` of ``analysis``, or ``TypeError`` naming the
    function that draws it."""
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


def check_drawable(result: StatsResult, analysis: str, call: str) -> None:
    """A result of ``analysis`` that still holds its observations (histostats,
    scatterstats); ``call`` is how to recompute it."""
    check_result_type(result, analysis)
    if not result.values or not all(result.values):
        raise ValueError(
            "This result holds no observations to draw: the copy of a result that "
            "save() embeds in a figure file leaves them out. Recompute it with "
            f"{call} and draw that."
        )


def check_one_call_subtitle(subtitle: object, results_subtitle: object) -> None:
    """In the one-call form (``ip.histostats``, ``ip.scatterstats``): a user
    subtitle only with ``results_subtitle=False`` in the same call."""
    if subtitle is not None and results_subtitle is not False:
        raise ValueError(
            "subtitle= would replace the results subtitle, which this call draws "
            "(results_subtitle=True, the default). To use your own subtitle, pass "
            "results_subtitle=False in this call as well; to add text, use title= "
            "or caption=."
        )


def check_subtitle(result: StatsResult, subtitle: str | None) -> None:
    """A user subtitle only where the result draws none of its own."""
    if subtitle is not None and result.settings.results_subtitle:
        raise ValueError(
            "subtitle= would replace the results subtitle, which this result draws "
            "(it was computed with results_subtitle=True). To use your own "
            "subtitle, pass results_subtitle=False to the analysis; to add text, "
            "use title= or caption=."
        )


def check_given_axes(ax: object) -> None:
    """``ax`` is None or a plain rectangular 2D Axes (histostats, scatterstats:
    shared axes are allowed, and laid out to fit every panel that shares them)."""
    if ax is None:
        return
    if not isinstance(ax, Axes):
        raise TypeError(
            f"ax must be a matplotlib Axes or None; got {type(ax).__name__}."
        )
    check_axes(ax, shared_x=True)
    # A scale or direction set before the call is refused, not drawn over: the
    # labels, bins and line are placed for linear upright axes, and inferplot's
    # limits would silently turn an inverted axis back.
    found: list[str] = []
    undo: list[str] = []
    for letter, scale, inverted in (
        ("x", ax.get_xscale(), ax.xaxis_inverted()),
        ("y", ax.get_yscale(), ax.yaxis_inverted()),
    ):
        if scale != "linear":
            found.append(f"a {scale!r} {letter} scale")
            undo.append(f"ax.set_{letter}scale('linear')")
        if inverted:
            found.append(f"an inverted {letter} axis")
            undo.append(f"ax.invert_{letter}axis()")
    if found:
        raise ValueError(
            f"ax has {' and '.join(found)}; inferplot draws into an Axes whose axes "
            "are linear and run up and to the right, as it places its labels, "
            "bars and line for them. Undo it before the call with "
            f"{' and '.join(undo)}. To show the values on another scale, "
            "transform the column before the analysis."
        )


# The largest |value| histostats and scatterstats draw: matplotlib treats axis
# limits whose size is below about 1e21 times the smallest normal double as zero,
# and the padding around the data must stay below the largest double.
_SMALLEST_DRAWN: Final = 1e-280
_LARGEST_DRAWN: Final = 1e306


def check_scale(
    values: Iterable[float],
    column: str | None,
    letter: str,
    *,
    room: str = "with room around the data",
) -> None:
    """Values matplotlib can draw on an axis; raised before anything is drawn.

    ``room`` says what the axis holds besides the data, in the message."""
    largest = max((abs(float(x)) for x in values), default=0.0)
    if largest == 0.0 or _SMALLEST_DRAWN <= largest <= _LARGEST_DRAWN:
        return
    exponent = math.floor(math.log10(largest))
    small = largest < _SMALLEST_DRAWN
    if small:
        why = (
            f"the largest is {largest:.3g} in size, and matplotlib treats axis "
            "limits below about 1e-287 as zero"
        )
        how = f"multiply {letter} by 1e{-exponent}"
    else:
        why = (
            f"the largest is {largest:.3g}, and the axis, {room}, would pass "
            "the largest double (1.8e308)"
        )
        how = f"divide {letter} by 1e{exponent}"
    raise ValueError(
        f"The values of {column!r} are too {'small' if small else 'large'} "
        f"for matplotlib to draw: {why}. The statistics do not depend on the "
        f"units: {how} (change its units) before the analysis and draw that."
    )


_MATH_SPAN: Final = re.compile(r"(?<!\\)\$.*?(?<!\\)\$")


def wrap_words(text: str, fits: Callable[[str], bool]) -> str:
    """``text`` broken into lines at spaces (never inside a mathtext span
    between two dollar signs), each line as long as ``fits`` accepts; a word
    that does not fit alone is a line of its own. Existing line breaks stay."""
    out: list[str] = []
    for paragraph in text.split("\n"):
        # Spaces inside $...$ are protected so that the split keeps the span.
        protected = _MATH_SPAN.sub(lambda m: m.group(0).replace(" ", "\0"), paragraph)
        current = ""
        for word in protected.split(" "):
            joined = f"{current} {word}" if current else word
            if not current or fits(joined.replace("\0", " ")):
                current = joined
            else:
                out.append(current)
                current = word
        out.append(current)
    return "\n".join(out).replace("\0", " ")


def shared_limits(
    ax: Axes, which: Literal["x", "y"], low: float, high: float
) -> tuple[float, float]:
    """The limits ``low``, ``high`` widened to hold the view of every other Axes
    that shares that axis and draws something, so that drawing into one panel
    never cuts another's data."""
    for other in siblings(ax, which):
        if other is ax or not other.has_data():
            continue
        a, b = sorted(other.get_xlim() if which == "x" else other.get_ylim())
        low, high = min(low, float(a)), max(high, float(b))
    return low, high


def siblings(ax: Axes, which: Literal["x", "y"]) -> list[Axes]:
    """The Axes that share ``ax``'s x or y axis, ``ax`` included."""
    group = cast(
        "Any", ax.get_shared_x_axes() if which == "x" else ax.get_shared_y_axes()
    )
    return list(cast("list[Axes]", group.get_siblings(ax)))


def draw(
    make: Callable[[list[theme.Color], str | None], BaseDrawing],
    *,
    palette: object,
    k: int,
    options: UserTexts,
    artist_args: Callable[[], list[str]],
    uses: str | None = None,
) -> StatsFigure:
    """Resolve ``k`` colours of the palette (``uses``: what a figure without
    levels colours, for the messages), check the artist dicts (``artist_args``,
    which returns the warnings of invisible layers) and user texts, then make
    the drawing and run it, inside inferplot's theme and fonts."""
    colors, palette_warning = theme.resolve_palette(palette, k, uses)
    fallbacks = theme.user_fallbacks()  # before context() resets the rcParams
    with reported_glyphs(), theme.context(), theme.fallback_fonts(fallbacks):
        invisible = artist_args()
        check_user_texts(options)
        made = make(colors, palette_warning)
        made.warnings.extend(invisible)
        # The artist dicts whose layer is invisible ("bar_args={...} makes the
        # bars invisible"): the alt text leaves those layers out.
        made.hidden = {warning.split("=", 1)[0] for warning in invisible}
        return made.run()


# --------------------------------------------------------------------------- #
# the drawing
# --------------------------------------------------------------------------- #


class BaseDrawing:
    """One rendering of a result: the figure (inferplot's own, or the one holding
    a given Axes), the header and caption blocks and their layout, ink and fonts.

    A subclass draws its layers and implements :meth:`run`; the hooks
    :meth:`generated_texts`, :meth:`place_axes`, :meth:`decoration_texts`,
    :meth:`ink_inside` and :meth:`font_axes` let it add to the shared parts.
    """

    def __init__(
        self,
        result: StatsResult,
        options: UserTexts,
        colors: list[theme.Color],
        palette_warning: str | None,
        ax: Axes | None,
    ) -> None:
        self.result = result
        self.base_options = options
        self.colors = colors
        self.own = ax is None
        if ax is None:
            self.figure = Figure(figsize=(6.0, 4.5), dpi=100)
            FigureCanvasAgg(self.figure)
            self.ax = self.figure.add_axes((0.15, 0.2, 0.7, 0.6))
        else:
            self.figure = root_figure(ax.figure)
            self.ax = ax
        self.dpi = float(self.figure.dpi)
        self.warnings: list[str] = [] if palette_warning is None else [palette_warning]
        self.roles: dict[int, str] = {}
        self.inside: list[Text] = []
        self.header: list[Text] = []  # title, subtitle (own figure)
        self.captions: list[Text] = []
        # Axis labels as set, before an offset text is folded into them.
        self.unfolded: dict[int, str] = {}
        # The artist dicts (names) that leave their layer invisible.
        self.hidden: set[str] = set()

    def run(self) -> StatsFigure:
        raise NotImplementedError

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

    def role(self, text: Text, name: str, plain: str | None = None) -> Text:
        """Record what ``text`` is (and its plain wording) for check() messages."""
        described = describe_text(name, plain_text(text) if plain is None else plain)
        self.roles[id(text)] = described
        remember(text, described)
        return text

    # -- hooks -------------------------------------------------------------------

    def generated_texts(self) -> tuple[str | None, str | None, Sequence[str]]:
        """Hook: the generated subtitle (mathtext, or None), its plain wording,
        and the generated caption lines (a line wholly between dollar signs is
        mathtext) whose plain wordings are ``result.text.caption``'s lines."""
        return None, None, ()

    def decoration_texts(self) -> list[Text]:
        """Hook: texts drawn inside the Axes that may reach beyond its frame."""
        return []

    def place_axes(
        self, x0: float, y0: float, width: float, height: float, fig_w: float,
        fig_h: float,
    ) -> None:  # fmt: skip
        """Hook: put the plotting area at (x0, y0) with this size, in pixels of
        a figure of ``fig_w`` by ``fig_h`` pixels."""
        self.ax.set_position((x0 / fig_w, y0 / fig_h, width / fig_w, height / fig_h))

    def tight_boxes(self) -> list[Bbox]:
        """Hook: the boxes of the Axes and their decorations, for the layout."""
        box = self.ax.get_tightbbox(self.renderer())
        return [] if box is None else [box]

    def frame(self) -> Bbox:
        """Hook: the plotting area the decorations surround, in pixels."""
        return self.ax.get_window_extent(self.renderer())

    def ink_inside(self, ink: str) -> None:
        """Hook: give ``ink`` to what is drawn inside the Axes (with ``ax=`` on a
        dark background)."""
        for text in [*self.header, *self.captions]:
            text.set_color(ink)

    def font_axes(self) -> list[Axes]:
        """Hook: the Axes whose axis texts get inferplot's font with ``ax=``."""
        return [self.ax]

    # -- header and caption ------------------------------------------------------

    def draw_text_blocks(self) -> None:
        """Title, subtitle and caption lines (positions set by the layout).

        With inferplot's own figure they are figure texts; with ``ax=`` the title
        is the Axes title and the subtitle and caption go inside the Axes.
        """
        options, result = self.base_options, self.result
        generated, plain_generated, caption_lines = self.generated_texts()
        title = options.title
        if options.subtitle is None:
            subtitle, plain_subtitle = generated, plain_generated
        else:
            subtitle, plain_subtitle = options.subtitle, None
        plain_captions = [] if result.text.caption is None else (
            result.text.caption.split("\n")
        )  # fmt: skip
        lines: list[tuple[str, bool, str | None]] = [
            (line, line.startswith("$") and line.endswith("$"), plain)
            for line, plain in zip(caption_lines, plain_captions, strict=True)
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

    def fit_figure(self, axes_w: float, axes_h: float) -> None:
        """Size inferplot's own figure around a plotting area of ``axes_w`` by
        ``axes_h`` pixels (wider when the header or a caption line is wider), with
        the header above, the caption lines below and room for the Axes'
        decorations, measured from the rendered figure until it holds."""
        dpi = self.dpi
        self.wrap_axis_labels(*self.label_room(axes_w, axes_h))
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
            self.place_axes(x0, y0, ax_w, axes_h, width, height)
            top = height - margin
            for text, box in zip(self.header, header, strict=True):
                text.set_position((margin / width, top / height))
                top -= box.height + gap
            top = margin + bottom_block - gap
            for text, box in zip(self.captions, captions, strict=True):
                text.set_position((margin / width, top / height))
                top -= box.height + gap
            self.fold_offsets()
            canvas.draw()
            measured = self.decoration_extra()
            if all(abs(a - b) <= 0.5 for a, b in zip(measured, extra, strict=True)):
                break
            extra = measured

    def fold_offsets(self) -> None:
        """inferplot's own figure draws no axis offset text ("1e12", "+1.7e9"
        above or right of an axis, where it lands on a marginal histogram or the
        header): where an axis has one, it is hidden and its text follows the
        axis label in parentheses, "mpg (1e12)", read as matplotlib prints it
        (a multiplier, then an offset added to the tick values)."""
        for axis in (self.ax.xaxis, self.ax.yaxis):
            formatter = axis.get_major_formatter()
            if not isinstance(formatter, ScalarFormatter):
                continue
            label = axis.label
            plain = self.unfolded.setdefault(id(axis), label.get_text())
            formatter.set_locs(list(axis.get_majorticklocs()))
            offset = formatter.get_offset()
            axis.offsetText.set_visible(not offset)
            shown = f"{plain} ({offset})" if offset else plain
            if label.get_text() != shown:
                label.set_text(shown)

    def label_room(self, axes_w: float, axes_h: float) -> tuple[float, float]:
        """Hook: the width of the x-axis label's line and the height of the
        y-axis label's line, in pixels, for a plotting area of ``axes_w`` by
        ``axes_h``: the main Axes' sides."""
        return axes_w, axes_h

    def wrap_axis_labels(self, width: float, height: float) -> None:
        """Wrap an axis label longer than its side of the plotting area on its
        spaces (outside mathtext), so that a long name does not widen the figure
        to its length; a label that fits is left as it is. A word longer than the
        side stays whole."""
        renderer = self.renderer()
        for label, room in (
            (self.ax.xaxis.label, width),
            (self.ax.yaxis.label, height),
        ):
            shown = label.get_text()
            if " " not in shown:
                continue
            probe = self.figure.text(
                0.0, 0.0, "", fontproperties=label.get_fontproperties(),
                parse_math=bool(label.get_parse_math()),
            )  # fmt: skip

            def fits(
                line: str, probe: Text = probe, room: float = room,
                renderer: RendererBase | None = renderer,
            ) -> bool:  # fmt: skip
                probe.set_text(line)
                return float(probe.get_window_extent(renderer).width) <= room

            try:
                if not fits(shown):
                    label.set_text(wrap_words(shown, fits))
            finally:
                probe.remove()

    def decoration_extra(self) -> list[float]:
        """Pixels the Axes' texts reach beyond its plotting area on each side."""
        frame = self.frame()
        boxes = self.tight_boxes()
        boxes += [self.extent(t) for t in self.decoration_texts()]
        union = Bbox.union(boxes)
        pad = 1.0
        return [
            max(0.0, frame.x0 - union.x0) + pad,
            max(0.0, union.x1 - frame.x1) + pad,
            max(0.0, frame.y0 - union.y0) + pad,
            max(0.0, union.y1 - frame.y1) + pad,
        ]

    def place_blocks_in_axes(self) -> tuple[float, float]:
        """With ``ax=``: the subtitle at the top and the caption lines at the
        bottom inside the Axes, padded; the pixels they take at the top and at the
        bottom."""
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
        return top_px, bottom_px

    # -- finishing -------------------------------------------------------------

    def finish_ink(self) -> None:
        """With ``ax=``: inferplot's ink in contrast with what it is drawn on.

        What is drawn inside the Axes (subtitle, caption; :meth:`ink_inside`)
        takes :func:`theme.ink_for` of the Axes background, and the axis texts and
        title outside it that of the figure background: the usual near-black on a
        light background, a near-white on a dark one. On a light background
        nothing is changed.
        """
        inside = axes_background(self.ax)
        outside = figure_background(self.figure)
        ink_in, ink_out = theme.ink_for(inside), theme.ink_for(outside)
        if ink_in != theme.INK:
            self.ink_inside(ink_in)
        if ink_out != theme.INK:
            for ax in self.font_axes():
                ax.xaxis.label.set_color(ink_out)
                ax.yaxis.label.set_color(ink_out)
                ax.tick_params(axis="both", labelcolor=ink_out)
            for text in self.ax.findobj(Text):
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
                *(t for ax in self.font_axes() for t in ax.xaxis.findobj(Text)),
                *(t for ax in self.font_axes() for t in ax.yaxis.findobj(Text)),
                *(t for t in self.ax.findobj(Text) if id(t) in self.roles),
            ]
        for item in texts:
            item.set_fontfamily(theme.families())
            item.set_math_fontfamily(theme.MATH_FONT_FAMILY)
