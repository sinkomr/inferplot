"""StatsFigure: a drawn figure together with the result it shows."""

from __future__ import annotations

import contextlib
import os
import re
import unicodedata
import warnings
import weakref
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
from numpy.typing import NDArray

from inferplot.render import theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch

    from inferplot.check import Issue
    from inferplot.result import StatsResult

__all__ = [
    "CenterLabel",
    "DrawnData",
    "DrawnPaths",
    "DrawnView",
    "LineLabel",
    "PointLabels",
    "StatsFigure",
    "Subsample",
    "axes_background",
    "describe_text",
    "drawn_before",
    "figure_background",
    "plain_text",
    "remember",
    "reported_glyphs",
    "root_figure",
    "shows_all",
]


@dataclass(frozen=True, slots=True)
class Subsample:
    """Points drawn when a panel has more values than are drawn as points."""

    shown: int
    total: int
    seed: int
    density: str  # the layer that shows all values: "violins" or "density outlines"
    # withinstats: (subjects drawn, subjects in all); whole subjects are sampled.
    subjects: tuple[int, int] | None = None
    # Levels whose violin or density outline is not drawn.
    undrawn: tuple[str, ...] = ()
    # histostats and scatterstats: the check's message in full (they have no
    # levels to speak of).
    message: str | None = None


def shows_all(sub: Subsample, verb: str) -> str:
    """What shows all values beside the subsampled points, in words: "the violins
    and boxes show all values", naming the levels without a violin."""
    if not sub.undrawn:
        return f"the {sub.density} and boxes {verb} all values"
    names = ", ".join(repr(level) for level in sub.undrawn)
    has = "has" if len(sub.undrawn) == 1 else "have"
    return (
        f"the boxes {verb} all values, and the {sub.density} those of every level "
        f"but {names}, which {has} none"
    )


@dataclass(frozen=True, slots=True)
class DrawnPaths:
    """The paths a withinstats figure draws, for ``check()``.

    ``subjects`` joins each drawn subject's points (one path per subject), or None
    when no subject paths are drawn; ``centrality`` runs through the centrality
    points in level order (NaN at a level without one), or None; ``levels`` names
    the levels, in x order.
    """

    subjects: LineCollection | None
    centrality: Line2D | None
    levels: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DrawnData:
    """Marks a histostats or scatterstats panel draws from the data, for the view
    checks (``data_outside_view``, ``flat_data``).

    ``panel`` is a key of ``StatsFigure.axes``; ``what`` names the marks in
    messages ("values", "bars"); ``x`` and ``y`` are (marks, points) arrays of
    each mark's points in data coordinates, which must all lie in view for the
    mark to be shown whole, or None where the mark spans that axis (a value of a
    histogram lies along x at any height, a vertical line spans the height);
    ``spread`` names the axes ("x", "y", "xy" or "") along which the marks'
    range must not be drawn within ``flat_data``'s two pixels.
    """

    panel: str
    what: str
    x: NDArray[np.float64] | None
    y: NDArray[np.float64] | None
    spread: str = ""


@dataclass(frozen=True, slots=True, eq=False)
class PointLabels:
    """The point labels of a scatterstats figure, for ``check()``: the texts, how
    many there are and the most inferplot places clear of each other
    (``limit``); the least-squares line (None where not drawn) and each link
    line with the label it joins to its point, which no other label may lie
    across."""

    texts: tuple[Text, ...]
    total: int
    limit: int
    line: Line2D | None = None
    links: tuple[tuple[ConnectionPatch, Text], ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class LineLabel:
    """A label of a vertical line (histostats' centrality and test value lines),
    for ``check()``: its text, its own line, and what the line is in words
    ("the centrality line")."""

    text: Text
    line: Line2D
    name: str


@dataclass(frozen=True, slots=True)
class CenterLabel:
    """A drawn centrality label, for ``check()``: its level (position ``index``
    on the x axis), its text and point, and the side of its point it goes on
    unless something is in the way (``usual``)."""

    index: int
    level: str
    text: Text
    point: PathCollection
    usual: Literal["right", "left"]


@dataclass(slots=True)
class DrawnView:
    """The axis limits inferplot set for a panel (updated when a later drawing
    into a panel that shares its y axis widens them), for ``check()``'s advice."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]


class StatsFigure:
    """A figure drawn from a :class:`~inferplot.result.StatsResult`.

    ``figure`` is the matplotlib Figure (inferplot's own, or the one holding the
    ``ax`` you passed), ``axes`` maps panel names to Axes (``"main"``), and
    ``result`` is the result drawn, with any rendering warnings added to its
    ``warnings``. ``check()`` lists statistical and rendering problems; ``save()``
    writes .svg, .pdf or .png with the result's JSON embedded.
    """

    __slots__ = (
        "_axes",
        "_brackets",
        "_centers",
        "_color_names",
        "_colors",
        "_data",
        "_figure",
        "_inside",
        "_labels",
        "_line_labels",
        "_own",
        "_paths",
        "_point_labels",
        "_result",
        "_roles",
        "_subsample",
        "_view",
    )

    def __init__(
        self,
        *,
        figure: Figure,
        axes: Mapping[str, Axes],
        result: StatsResult,
        own: bool,
        roles: Mapping[int, str],
        inside: tuple[Text, ...],
        subsample: Subsample | None,
        brackets: tuple[tuple[Text, Line2D, str], ...] = (),
        paths: DrawnPaths | None = None,
        view: DrawnView | None = None,
        colors: tuple[theme.Color, ...] = (),
        centers: tuple[tuple[str, PathCollection], ...] = (),
        labels: tuple[CenterLabel, ...] = (),
        data: tuple[DrawnData, ...] = (),
        color_names: tuple[str, ...] = (),
        point_labels: PointLabels | None = None,
        line_labels: tuple[LineLabel, ...] = (),
    ) -> None:
        self._figure = figure
        self._axes = MappingProxyType(dict(axes))
        self._result = result
        self._own = own
        self._roles = MappingProxyType(dict(roles))
        self._inside = inside
        self._subsample = subsample
        self._brackets = brackets
        self._paths = paths
        self._view = view
        self._colors = colors
        self._centers = centers
        self._labels = labels
        self._data = data
        self._color_names = color_names
        self._point_labels = point_labels
        self._line_labels = line_labels

    @property
    def figure(self) -> Figure:
        """The matplotlib Figure."""
        return self._figure

    @property
    def axes(self) -> Mapping[str, Axes]:
        """The panels by name: ``"main"``, and for scatterstats' marginal
        histograms ``"xside"`` (above) and ``"yside"`` (right)."""
        return self._axes

    @property
    def result(self) -> StatsResult:
        """The result drawn, including rendering warnings."""
        return self._result

    @property
    def subsample(self) -> Subsample | None:
        """How the points were subsampled, or None when every value is drawn."""
        return self._subsample

    def check(self) -> tuple[Issue, ...]:
        """Statistical and rendering problems; an empty tuple when there are none.

        Rendering problems are measured on the drawn figure: overlapping texts, or a
        bracket label drawn across another bracket's line (``label_overlap``),
        texts outside the figure or, for texts drawn inside an Axes, outside that
        Axes (``text_clipped``), and subsampled points (``points_subsampled``);
        for withinstats also a text drawn over a path (``path_hidden``) and
        subject paths so many that they hide what lies beneath (``paths_crowded``).
        Also: a centrality point at least half under a text
        (``centrality_label_overlap``), a centrality label that reads as another
        level's (``centrality_label_ambiguous``), a level colour or text that can
        hardly be told from its background (``low_contrast``), characters no font
        can draw (``glyph_missing``), and, whatever changed the figure after drawing, an
        inverted axis or a non-linear scale (``axis_changed``), values or
        brackets outside the axis limits (``data_outside_view``) and values
        drawn as a flat line (``flat_data``). With ``ax=``, overlaps with other
        panels' texts and a figure title count as ``label_overlap``. inferplot's
        own figure far larger than its plotting area, or than any of its layouts
        needs, is ``figure_oversized``.
        """
        from inferplot.render.checks import figure_issues

        return self._result.check() + figure_issues(self)

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        dpi: float = 300,
        transparent: bool = False,
    ) -> None:
        """Write the figure to ``path`` (.svg, .pdf or .png) with the result embedded.

        Text stays editable (SVG text elements, PDF TrueType fonts) and the result's
        JSON goes into the file's metadata: the PNG text chunk and SVG metadata
        "Description", the PDF "Subject". ``dpi`` is at most 2,400, and a PNG at
        least 10 dpi and at most 100 million pixels.

        The figure is not pyplot's current figure, so saving with pyplot after an
        inferplot call writes pyplot's own, empty figure. Save with
        ``fig.save(path)``; the
        figure's own ``fig.figure.savefig(path)`` also works, but writes text as
        paths and embeds no result.
        """
        from inferplot.export import save

        save(self, path, dpi=dpi, transparent=transparent)

    def _repr_png_(self) -> bytes:
        from inferplot.export import png_bytes

        return png_bytes(self, dpi=100)

    def __repr__(self) -> str:
        r = self._result
        if r.design == "one-sample":
            what = f"{r.x}, {r.n_obs:,} values"
        elif r.design == "correlation":
            what = f"{r.y} and {r.x}, {r.n_obs:,} pairs"
        else:
            what = r.x if r.y is None else f"{r.y} by {r.x}"
            what = f"{what}, {len(r.levels)} levels"
        return (
            f"<StatsFigure {r.analysis}: {what}; "
            "fig.check() lists problems, fig.save(path) writes it>"
        )

    # -- internal -------------------------------------------------------------

    @property
    def own(self) -> bool:
        """True when inferplot created the figure (no ``ax`` was passed)."""
        return self._own

    def describe(self, text: Text) -> str | None:
        """What a text inferplot drew is, in words; None for other texts.

        A text an earlier inferplot drawing left in the same figure is named as
        such.
        """
        own = self._roles.get(id(text))
        if own is not None:
            return own
        before = drawn_before(text)
        return None if before is None else f"{before} of another inferplot drawing"

    @property
    def inside(self) -> tuple[Text, ...]:
        """Texts that must stay inside the main Axes."""
        return self._inside

    @property
    def brackets(self) -> tuple[tuple[Text, Line2D, str], ...]:
        """Each significance bracket: its label, its line, and what it is in words."""
        return self._brackets

    @property
    def paths(self) -> DrawnPaths | None:
        """The subject and centrality paths (withinstats), or None."""
        return self._paths

    @property
    def centers(self) -> tuple[tuple[str, PathCollection], ...]:
        """Each drawn centrality point, with its level."""
        return self._centers

    @property
    def labels(self) -> tuple[CenterLabel, ...]:
        """Each drawn centrality label, in level order."""
        return self._labels

    @property
    def colors(self) -> tuple[theme.Color, ...]:
        """Each level's colour from the palette, in level order."""
        return self._colors

    @property
    def data(self) -> tuple[DrawnData, ...]:
        """What histostats and scatterstats draw from the data, for the view
        checks; empty for the other figures."""
        return self._data

    @property
    def color_names(self) -> tuple[str, ...]:
        """What each of ``colors`` paints, in words ("the bars"); empty where the
        colours follow the levels."""
        return self._color_names

    @property
    def line_labels(self) -> tuple[LineLabel, ...]:
        """histostats' labels of its vertical lines, with their lines."""
        return self._line_labels

    @property
    def point_labels(self) -> PointLabels | None:
        """scatterstats' point labels, or None."""
        return self._point_labels

    @property
    def view(self) -> DrawnView | None:
        """The axis limits inferplot set, or None."""
        return self._view

    def draw(self) -> None:
        """Draw the figure so text extents are current."""
        root = root_figure(self._figure)
        with reported_glyphs():
            if self._own:
                with theme.context():
                    _draw(root)
            else:
                _draw(root)


# What each text inferplot drew is, in words, for every drawing: a figure's check
# names the texts an earlier drawing into the same Axes left there. Weak keys, so
# a text removed from its figure is forgotten.
_DRAWN: weakref.WeakKeyDictionary[Text, str] = weakref.WeakKeyDictionary()


def remember(text: Text, described: str) -> None:
    """Record what a text inferplot drew is, for later drawings' checks."""
    _DRAWN[text] = described


def drawn_before(text: Text) -> str | None:
    """What an inferplot text of any drawing is, in words; None for other texts."""
    return _DRAWN.get(text)


_DOLLAR: Final = re.compile(r"(?<!\\)\$")


def plain_text(text: Text) -> str:
    """A text as it reads on the figure, without mathtext source.

    Math segments (between two unescaped dollar signs) become the characters
    mathtext draws for them, left to right and without accents
    (``$\\hat{\\mu}_{a}$`` reads "μa"); the rest is kept, with ``\\$`` read as a
    dollar sign.
    """
    shown = text.get_text()
    marks = [m.start() for m in _DOLLAR.finditer(shown)]
    if not text.get_parse_math() or len(marks) < 2:
        return shown.replace("\\$", "$")
    from matplotlib.mathtext import MathTextParser

    parser = MathTextParser("path")
    out: list[str] = []
    start = 0
    for a, b in zip(marks[0::2], marks[1::2], strict=False):
        out.append(shown[start:a].replace("\\$", "$"))
        try:
            parsed = parser.parse(shown[a : b + 1], 72, text.get_fontproperties())
            # (font, size, code point, [glyph index from matplotlib 3.9,] x, y):
            # x and y are the last two fields. Accents are left out.
            glyphs = sorted(parsed.glyphs, key=lambda g: (float(g[-2]), -float(g[-1])))
            chars = (chr(int(g[2])) for g in glyphs)
            out.append("".join(c for c in chars if not unicodedata.combining(c)))
        except ValueError:
            out.append("(a math expression)")
        start = b + 1
    out.append(shown[start:].replace("\\$", "$"))
    return "".join(out)


def describe_text(role: str, shown: str, *, limit: int = 40) -> str:
    """``role 'text'`` with the text on one line, shortened to ``limit`` characters."""
    flat = " ".join(shown.split())
    if len(flat) > limit:
        flat = flat[: limit - 3] + "..."
    return f"{role} {flat!r}"


def figure_background(figure: object) -> tuple[float, float, float]:
    """The colour the figure's patch paints, over white where it is not opaque
    (or not drawn)."""
    root = root_figure(figure)
    if not root.patch.get_visible():
        return (1.0, 1.0, 1.0)
    return theme.over(root.get_facecolor(), "white")


def axes_background(ax: Axes) -> tuple[float, float, float]:
    """The colour behind what is drawn inside ``ax``: its patch over the figure's
    background."""
    below = figure_background(ax.figure)
    if not (ax.patch.get_visible() and ax.axison and ax.get_frame_on()):
        return below
    return theme.over(ax.get_facecolor(), below)


def root_figure(figure: object) -> Figure:
    """The top-level Figure of a Figure or SubFigure."""
    from matplotlib.figure import Figure

    current = figure
    for _ in range(100):
        if isinstance(current, Figure):
            return current
        current = getattr(current, "figure", None)
    raise TypeError("could not find the matplotlib Figure that holds this Axes")


# matplotlib's warning for a character no font has: "Glyph 26085 (...) missing
# from font(s) DejaVu Sans." (3.9 and later), "... missing from current font."
# (3.8).
_GLYPH_WARNING: Final = re.compile(r"^Glyph (\d+) .*missing from")


def _glyph_reported(code: int) -> bool:
    """Whether ``fig.check()``'s glyph_missing reports the character ``code``
    when a text has it: every character but controls, format characters and
    spaces (Unicode categories C and Z)."""
    return unicodedata.category(chr(code))[0] not in "CZ"


@contextlib.contextmanager
def reported_glyphs() -> Generator[None, None, None]:
    """Hold back matplotlib's "Glyph ... missing" UserWarning for characters that
    ``fig.check()`` reports as ``glyph_missing``, around inferplot's own drawing
    and saving; every other warning, that one for other characters included,
    reaches the caller as it was raised."""
    with warnings.catch_warnings(record=True) as caught:
        # Recorded, not raised, whatever the caller's filters say; decided below.
        warnings.filterwarnings("always", message="Glyph ", category=UserWarning)
        yield
    for item in caught:
        found = _GLYPH_WARNING.match(str(item.message))
        if (
            issubclass(item.category, UserWarning)
            and found is not None
            and _glyph_reported(int(found.group(1)))
        ):
            continue
        warnings.warn_explicit(
            item.message, item.category, item.filename, item.lineno, source=item.source
        )


def _draw(root: Figure) -> None:
    if isinstance(root.canvas, FigureCanvasAgg):
        root.canvas.draw()
    else:
        root.draw_without_rendering()
