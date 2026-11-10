"""StatsFigure: a drawn figure together with the result it shows."""

from __future__ import annotations

import os
import re
import unicodedata
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text

from inferplot.render import theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from inferplot.check import Issue
    from inferplot.result import StatsResult

__all__ = [
    "DrawnPaths",
    "StatsFigure",
    "Subsample",
    "describe_text",
    "drawn_before",
    "plain_text",
    "remember",
    "root_figure",
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
        "_figure",
        "_inside",
        "_own",
        "_paths",
        "_result",
        "_roles",
        "_subsample",
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

    @property
    def figure(self) -> Figure:
        """The matplotlib Figure."""
        return self._figure

    @property
    def axes(self) -> Mapping[str, Axes]:
        """The panels by name: ``"main"``."""
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
        Changes made after drawing are measured too: an inverted axis or a
        non-linear scale (``axis_changed``).
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
        most 100 million pixels.

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
        return (
            f"<StatsFigure {r.analysis}: {r.y} by {r.x}, {len(r.levels)} levels; "
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

    def draw(self) -> None:
        """Draw the figure so text extents are current."""
        root = root_figure(self._figure)
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
            # (font, size, code point, glyph index, x, y); accents are left out.
            glyphs = sorted(parsed.glyphs, key=lambda g: (float(g[4]), -float(g[5])))
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


def root_figure(figure: object) -> Figure:
    """The top-level Figure of a Figure or SubFigure."""
    from matplotlib.figure import Figure

    current = figure
    for _ in range(100):
        if isinstance(current, Figure):
            return current
        current = getattr(current, "figure", None)
    raise TypeError("could not find the matplotlib Figure that holds this Axes")


def _draw(root: Figure) -> None:
    if isinstance(root.canvas, FigureCanvasAgg):
        root.canvas.draw()
    else:
        root.draw_without_rendering()
