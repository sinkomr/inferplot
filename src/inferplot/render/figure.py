"""StatsFigure: a drawn figure together with the result it shows."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from matplotlib.backends.backend_agg import FigureCanvasAgg

from inferplot.render import theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from inferplot.check import Issue
    from inferplot.result import StatsResult

__all__ = ["DrawnPaths", "StatsFigure", "Subsample", "describe_text", "root_figure"]


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
        "Description", the PDF "Subject".
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
        """What a text inferplot drew is, in words; None for other texts."""
        return self._roles.get(id(text))

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
