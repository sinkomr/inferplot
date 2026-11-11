# matplotlib's and NumPy's stubs type many ``**kwargs`` and overloads as Unknown,
# which strict mode reports on every call; the arguments we pass are all typed.
# pyright: reportUnknownMemberType=false
"""Draw a barstats result: ggstatsplot's ``ggbarstats()`` layers, our look.

* bars: one per level of y that holds observations (one bar without y), each
  stacked to 100% by the levels of x in level order, the first level on top (as
  ggplot2's ``position_fill`` stacks them), one palette colour per level of x;
  a zero cell draws nothing;
* cell labels (``result.settings.label``: the percentage, the count or both),
  written on their segment in the ink that contrasts with it. A label that does
  not fit inside its segment (its height, or with ``ax=`` its width) is moved
  beside its bar, in a column right of the bar, stacked clear of the other
  moved labels of that bar and joined to its segment by a short line; with
  ``ax=``, a moved label for which that column has no room is left out.
  inferplot's own figure is made tall enough for every column, so it leaves no
  label out; ``fig.check()`` reports every label that is left out or does not
  read as its cell's (``cell_labels_hidden``);
* the proportion test of each bar above it (``p = 0.09``, ggstatsplot's
  label) when the result asks for them (``settings.proportion_test``, two or
  more bars); a bar without a test has no label;
* tick labels "<level of y>\\n(n = <n>)" (one-way: "(n = <n>)"), the name of y
  as the x axis title (none one-way), percent ticks 0% to 100% by 10, no y axis
  title, a legend of the levels of x (title ``legend_title``, default the name
  of x), first level on top as in the bars; no grid lines, which would strike
  through the labels beside the bars;
* the header (title, subtitle) and caption lines, as in the other figures.

inferplot's own figure gives every bar the same slot, at least MIN_SPACING_IN
wide and wider for a wide label (the cell labels, the column of moved labels,
the bar's test label, the tick label's lines; level names wrap on spaces into
at most three lines TICK_LINE_IN wide, shortened with "…" beyond): the figure
widens with the number of bars, as betweenstats' does with its levels. 0% to
100% is BARS_HEIGHT_IN high, taller where the legend or a column of moved
labels needs it. The legend is right of the plot; with ``ax=`` it goes inside
the Axes, right of the bars, and the bars share what is left of its width. The
texts are ``figure_texts``'; the renderer computes no statistics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from matplotlib.transforms import offset_copy

from inferplot import _validate as v
from inferplot.render import base, theme
from inferplot.render.base import GAP_PT, LABEL_PAD_PT, LayerArgs, px
from inferplot.render.comparison import wrap_names
from inferplot.render.figure import (
    CellLabel,
    DrawnData,
    DrawnView,
    StatsFigure,
    axes_background,
)
from inferplot.result import StatsResult
from inferplot.text import drawable, figure_alt_text, figure_texts

if TYPE_CHECKING:
    from matplotlib.artist import Artist

__all__ = ["barstats", "check_axes", "check_drawing"]

# inferplot's own figure, in inches: the least distance between the centres of
# two bars, the least width of a bar, and the height of 0% to 100% (taller when
# a bar's moved labels need it).
MIN_SPACING_IN: Final = 0.8
BAR_WIDTH_IN: Final = 0.55
BARS_HEIGHT_IN: Final = 3.0
# A bar's share of its slot (with ax=, and in the own figure where no label is
# moved beside a bar); the least share with ax= where labels are moved.
BAR_SHARE: Final = 0.7
MIN_BAR_SHARE: Final = 0.3
# In points: a label inside its segment keeps CELL_PAD_PT from the segment's
# edges; a moved label starts LEAD_PT right of its bar (the joining line spans
# that gap) and keeps GAP_PT from the next bar; moved labels of one bar keep
# STACK_GAP_PT between them.
CELL_PAD_PT: Final = 1.5
LEAD_PT: Final = 7.0
STACK_GAP_PT: Final = 1.5
LEADER_WIDTH_PT: Final = 0.6
SEGMENT_EDGE_PT: Final = 0.8
# bar_args' bounds (B2): a wider edge covers small segments (1% of a 3 in bar is
# about 2 points) and the legend keys; below this alpha the default palette's
# yellow has under 1.1:1 contrast with white (low_contrast's floor for a level's
# colour), and the Okabe-Ito colours keep a CIELAB difference of at least 10.
MAX_SEGMENT_EDGE_PT: Final = 2.0
MIN_SEGMENT_ALPHA: Final = 0.4
# The bound of inferplot's own figure (review B11): its plotting area is at most
# MAX_PLOT_IN wide and tall, so with the y axis, the legend and the margins the
# figure stays within figure_oversized's 30 in (a long title or caption, which
# figure_oversized reports, aside). Up to MAX_PLOT_IN / MIN_SPACING_IN bars keep
# a slot of MIN_SPACING_IN; more share MAX_PLOT_IN down to MIN_SLOT_IN a bar
# (the bar narrows with its slot, and level names wrap to the slot; 0.7 in
# holds an n label such as "(n = 123)" with its gaps); beyond
# MAX_BARS bars, or where the labels need wider slots or a taller legend than
# the bound allows, the call is refused before a figure is returned.
MAX_PLOT_IN: Final = 26.0
MIN_SLOT_IN: Final = 0.7
MAX_BARS: Final = int(MAX_PLOT_IN // MIN_SLOT_IN)
# Level names in the tick labels and the legend (and its default title) wrap on
# spaces into lines at most this wide; a slot widens for the tick labels' lines.
TICK_LINE_IN: Final = 1.0
LEGEND_LINE_IN: Final = 1.6
PERCENT_TICKS: Final = tuple(float(t) for t in range(0, 101, 10))


@dataclass(frozen=True, slots=True)
class Options:
    """The checked drawing arguments."""

    xlab: str | None
    ylab: str | None
    title: str | None
    subtitle: str | None
    caption: str | None
    legend_title: str | None
    bar_args: dict[str, object]


def check_drawing(
    *,
    xlab: str | None,
    ylab: str | None,
    title: str | None,
    subtitle: str | None,
    caption: str | None,
    legend_title: str | None,
    bar_args: Mapping[str, object] | None,
) -> Options:
    """The checks of the drawing arguments that need no figure: ``ip.barstats``
    runs them before the analysis, so a bad one is reported first."""
    return Options(
        xlab=v.check_optional_text(xlab, "xlab"),
        ylab=v.check_optional_text(ylab, "ylab"),
        title=v.check_optional_text(title, "title"),
        subtitle=v.check_optional_text(subtitle, "subtitle"),
        caption=v.check_optional_text(caption, "caption"),
        legend_title=v.check_optional_text(legend_title, "legend_title"),
        bar_args=v.check_artist_args(bar_args, "bar_args") or {},
    )


@v.checked
def barstats(
    result: StatsResult,
    *,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    legend_title: str | None = None,
    palette: str | Sequence[object] | None = None,
    bar_args: Mapping[str, object] | None = None,
    ax: Axes | None = None,
) -> StatsFigure:
    """Draw a result of ``ip.analyze.barstats``: one stacked bar per level of y
    (one bar without y), filled by the shares of the levels of x.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw (also the
            copy embedded in a saved figure, which keeps the counts).
        xlab, ylab, title, subtitle, caption: texts, as for
            ``ip.render.betweenstats``; ``xlab`` replaces the name of y under
            the bars, ``ylab`` gives the percent axis a title (it has none).
        legend_title: the title of the legend of the levels of x (by default the
            name of x).
        palette: a matplotlib colormap name or a list of colors, one per level of
            x.
        bar_args: matplotlib properties applied with ``Artist.set`` to every bar
            segment (a ``Rectangle``), for example ``{"alpha": 0.8}``;
            properties that set the segments' place and size are refused.
        ax: draw into this Axes instead of a new figure; the subtitle, caption
            and legend then go inside it. Its x axis may not be shared (each
            tick label carries its bar's n).

    Each cell is labelled with its percentage, count or both (the analysis'
    ``label=``) on its segment, or beside its bar where the segment is too
    small; each bar's proportion test is written above it. ``fig.check()``
    reports what does not fit.

    Returns:
        A :class:`~inferplot.render.figure.StatsFigure`; its ``result`` adds any
        rendering warnings to ``result.warnings``.

    Example:
        >>> import pandas as pd
        >>> import inferplot as ip
        >>> df = pd.DataFrame({
        ...     "answer": ["yes", "no", "yes", "no"],
        ...     "group": ["a", "a", "b", "b"],
        ...     "n": [12, 8, 5, 15],
        ... })
        >>> result = ip.analyze.barstats(df, "answer", "group", counts="n")
        >>> fig = ip.render.barstats(result, legend_title="Answer")
        >>> fig.result.text.bar_labels
        ('p = 0.37', 'p = 0.03')
        >>> fig.check()
        ()
    """
    base.check_result_type(result, "barstats")
    options = check_drawing(
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        legend_title=legend_title,
        bar_args=bar_args,
    )
    base.check_subtitle(result, options.subtitle)
    _check_legend_title(options.legend_title)
    check_axes(ax)
    _check_contents(result)
    if ax is None and len(result.bars) > MAX_BARS:
        y = result.y
        raise ValueError(
            f"{len(result.bars)} bars do not fit inferplot's own figure, whose "
            f"plotting area is at most {MAX_PLOT_IN:g} in wide with at least "
            f"{MIN_SLOT_IN:g} in a bar: at most {MAX_BARS} bars fit. Draw fewer "
            f"levels of {y!r} (filter the data or merge levels), or draw into an "
            "Axes of a figure of your own size with ax=, for example "
            "ax=Figure(figsize=(40, 6)).add_subplot() (from matplotlib.figure "
            "import Figure); fig.check() then reports what does not fit."
        )
    return base.draw(
        lambda colors, warning: _BarDrawing(result, options, colors, warning, ax),
        palette=palette,
        k=len(result.levels),
        options=options,
        # The segments lie on the background: a colour that cannot be told
        # from it leaves them invisible, as for histostats' bars.
        artist_args=lambda: base.check_artist_args(
            [(_BAR_ARGS, options.bar_args, None)], ax=ax, colors=True
        ),
    )


def check_axes(ax: object) -> None:
    """``ax`` is None or a plain rectangular 2D Axes with linear, upright axes
    (``base.check_given_axes``) whose x axis is not shared: the tick labels
    carry each bar's n, as betweenstats' carry each level's."""
    base.check_given_axes(ax)
    if isinstance(ax, Axes):
        base.check_axes(ax)


def _check_legend_title(title: str | None) -> None:
    """Lay out ``legend_title`` once, as the other user texts are
    (``base.check_user_texts``), so an expression matplotlib cannot read raises
    before anything is drawn."""
    if title is None:
        return
    figure = Figure()
    renderer = FigureCanvasAgg(figure).get_renderer()
    text = figure.text(0.0, 0.0, title, fontproperties=theme.font(theme.SIZE_LABEL))
    try:
        text.get_window_extent(renderer)
    except ValueError as error:
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        detail = lines[-1] if lines else type(error).__name__
        raise ValueError(
            "legend_title is drawn as matplotlib mathtext, where text between two $ "
            f"signs is a math expression, and matplotlib cannot read it: {detail}. "
            "Fix the expression, or write \\$ for a literal dollar sign; got "
            f"legend_title={title!r}."
        ) from error


def _check_contents(result: StatsResult) -> None:
    """A barstats result holds what the figure draws: its bars and their cells
    (a result fresh from the analysis and the copy a saved figure embeds
    both do)."""
    k = len(result.levels)
    if not result.bars or len(result.cells) != k * len(result.bars):
        raise ValueError(
            "This result holds no bars to draw (result.bars and result.cells are "
            "empty or do not fit its levels). Recompute it with "
            "ip.analyze.barstats(data, x, y, ...) and draw that."
        )


def _segment_artist(
    ax: Axes, x: float, bottom: float, width: float, height: float, color: theme.Color
) -> Rectangle:
    """One bar segment, before ``bar_args``."""
    segment = Rectangle(
        (x, bottom),
        width,
        height,
        facecolor=color,
        edgecolor="white",
        linewidth=SEGMENT_EDGE_PT,
        zorder=2.0,
    )
    ax.add_patch(segment)
    return segment


def _scratch_segment(ax: Axes) -> Artist:
    return _segment_artist(ax, -0.4, 0.0, 0.8, 4.0, (0.0, 0.0, 0.0, 1.0))


def _refused_style(key: str, prop: str, value: object) -> str | None:
    """Why ``bar_args`` may not set ``prop`` (given as ``key``) to ``value``, or
    None: the colour of a segment is the only key to its level of x, so
    ``bar_args`` may not colour every level alike or leave the fill out, cover
    the segments and the legend keys with a wide edge, or fade the colours
    toward the background (review of barstats, B2)."""
    allowed = (
        "bar_args may style the segments with edgecolor, hatch, linestyle, "
        f"linewidth (at most {MAX_SEGMENT_EDGE_PT:g} points) and alpha (at least "
        f"{MIN_SEGMENT_ALPHA:g})."
    )
    if prop in ("color", "facecolor", "fill"):
        what = (
            "would leave the segments unfilled"
            if prop == "fill"
            else "would paint the segments of every level of x alike"
        )
        return (
            f"bar_args sets {key!r}, which {what}, and the colour of a segment is "
            "the only key to the legend of the levels of x. Choose the colours with "
            "palette=, one per level of x in their order (for example palette=["
            "'#0072B2', '#E69F00', '#009E73'], or a colormap name such as 'Set2'). "
            f"{allowed}"
        )
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if prop == "linewidth" and number and float(value) > MAX_SEGMENT_EDGE_PT:  # pyright: ignore[reportArgumentType]
        return (
            f"bar_args sets {key}={value!r}, an edge wider than "
            f"{MAX_SEGMENT_EDGE_PT:g} points, which would cover small segments "
            "and the legend keys (a segment of 1% is about 2 points tall). Use "
            f"{key}={MAX_SEGMENT_EDGE_PT:g} or less."
        )
    if prop == "alpha" and number and float(value) < MIN_SEGMENT_ALPHA:  # pyright: ignore[reportArgumentType]
        return (
            f"bar_args sets {key}={value!r}, which would fade the colours that "
            "identify the levels of x toward the background (below "
            f"{MIN_SEGMENT_ALPHA:g}, the default palette's yellow can hardly be "
            f"told from a white background). Use {key}={MIN_SEGMENT_ALPHA:g} or "
            "more, or choose lighter colours with palette=."
        )
    return None


_BAR_ARGS: Final = LayerArgs(
    name="bar_args",
    layer="bars",
    noun="bar segments",
    make=_scratch_segment,
    advice=("Choose a visible style; the segments show each level's share of its bar."),
    refuse=_refused_style,
)


# --------------------------------------------------------------------------- #
# the drawing
# --------------------------------------------------------------------------- #

Place = Literal["inside", "outside", "omitted"]


@dataclass(slots=True, eq=False)
class _Cell:
    """A cell as drawn: bar ``j``, level ``i`` of x, its share (``percent``)
    stacked from ``bottom``; its segment, label and joining line (None where
    not drawn)."""

    j: int
    i: int
    count: int
    percent: float
    bottom: float
    words: str
    segment: Rectangle | None = None
    text: Text | None = None
    leader: Line2D | None = None
    place: Place = "inside"


class _BarDrawing(base.BaseDrawing):
    """A barstats result: stacked bars, cell labels, bar labels, legend."""

    def __init__(
        self,
        result: StatsResult,
        options: Options,
        colors: list[theme.Color],
        palette_warning: str | None,
        ax: Axes | None,
    ) -> None:
        super().__init__(result, options, colors, palette_warning, ax)
        self.options = options
        self.texts = figure_texts(result)
        self.k = len(result.levels)
        self.n_bars = len(result.bars)
        self.cells: list[_Cell] = []
        self.bar_labels: list[Text | None] = []
        self.legend: Legend | None = None
        self.legend_texts: list[Text] = []
        self.view = DrawnView(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        # The layout: pixels of a slot and of a bar, and of 0% to 100%.
        self.slot_px = MIN_SPACING_IN * self.dpi
        self.bar_px = BAR_WIDTH_IN * self.dpi
        self.bars_px = BARS_HEIGHT_IN * self.dpi
        self.column_px = 0.0
        self.extra_px = 0.0  # ax=: room right of the last slot for its labels
        self.top_px = 0.0
        self.bottom_px = 0.0
        self.tick_texts: list[str] = self.unwrapped_ticks()
        self.edge = "white"

    # -- drawing -----------------------------------------------------------------

    def run(self) -> StatsFigure:
        if not self.own:
            self.edge = _hex(axes_background(self.ax))
        self.draw_bars()
        self.draw_bar_labels()
        self.draw_axes_texts()
        self.draw_legend()
        self.draw_text_blocks()
        self.layout()
        if not self.own:
            self.finish_ink()
        self.finish_fonts()
        result = self.result
        if self.warnings:
            result = dataclasses.replace(
                result, warnings=result.warnings + tuple(self.warnings)
            )
        result = self.finish_result(result)
        inside = [c.text for c in self.cells if c.text is not None]
        inside += [t for t in self.bar_labels if t is not None]
        if not self.own:
            inside += self.legend_texts
        return StatsFigure(
            figure=self.figure,
            axes={"main": self.ax},
            result=result,
            own=self.own,
            roles=self.roles,
            inside=tuple(inside),
            subsample=None,
            view=self.view,
            colors=tuple(self.colors),
            data=self.drawn_data(),
            cell_labels=self.cell_records(),
            segments=tuple(
                (c.i, c.segment) for c in self.cells if c.segment is not None
            ),
        )

    def draw_bars(self) -> None:
        """The segments, bottom (last level) up, and their labels; a zero cell
        draws neither."""
        ax, result = self.ax, self.result
        k = self.k
        # Limits that hold everything before the layout measures it (Y1).
        ax.set_xlim(-0.5, self.n_bars - 0.5)
        ax.set_ylim(0.0, 100.0)
        for j in range(self.n_bars):
            row = result.cells[j * k : (j + 1) * k]
            bottom = 0.0
            stacked: list[_Cell] = []
            for i in reversed(range(k)):
                cell = row[i]
                stacked.append(
                    _Cell(
                        j=j,
                        i=i,
                        count=cell.count,
                        percent=cell.percent,
                        bottom=bottom,
                        words=self.texts.cell_labels[j * k + i],
                    )
                )
                bottom += cell.percent
            for cell in reversed(stacked):  # level order
                self.cells.append(cell)
                if cell.count == 0:
                    continue
                segment = _segment_artist(
                    ax,
                    j - 0.35,
                    cell.bottom,
                    0.7,
                    cell.percent,
                    self.colors[cell.i],
                )
                segment.set_edgecolor(self.edge)
                segment.set_gid(f"segment-{j}-{cell.i}")
                base.apply_args(segment, self.options.bar_args, "bar_args")
                cell.segment = segment
                if not cell.words:
                    continue
                text = ax.text(
                    float(j),
                    cell.bottom + cell.percent / 2,
                    cell.words,
                    ha="center",
                    va="center",
                    fontproperties=theme.font(theme.SIZE_LABEL),
                    parse_math=False,
                    zorder=5.0,
                    gid=f"cell-label-{j}-{cell.i}",
                )
                text.set_color(self.ink_on(segment))
                self.role(text, f"cell label of {self.cell_name(cell)}")
                cell.text = text

    def cell_name(self, cell: _Cell) -> str:
        level = self.result.levels[cell.i]
        bar = self.result.bars[cell.j].level
        return repr(level) if bar is None else f"{level!r} in {bar!r}"

    def ink_on(self, segment: Rectangle) -> str:
        """The ink with the higher contrast on the segment as drawn (over the
        Axes background)."""
        drawn = theme.over(segment.get_facecolor(), axes_background(self.ax))
        return theme.ink_for(drawn)

    def draw_bar_labels(self) -> None:
        ax = self.ax
        shown = bool(self.result.settings.proportion_test)
        for j, words in enumerate(self.texts.bar_labels):
            if not shown or not words:
                self.bar_labels.append(None)
                continue
            text = ax.text(
                float(j),
                100.0,
                words,
                ha="center",
                va="bottom",
                fontproperties=theme.font(theme.SIZE_LABEL),
                transform=offset_copy(
                    ax.transData, fig=self.figure, y=LABEL_PAD_PT, units="points"
                ),
                zorder=5.0,
                gid=f"bar-label-{j}",
            )
            self.role(text, "proportion test label", self.result.text.bar_labels[j])
            self.bar_labels.append(text)

    def draw_axes_texts(self) -> None:
        ax, options = self.ax, self.options
        ax.grid(False)
        ax.yaxis.set_major_locator(FixedLocator(list(PERCENT_TICKS)))
        ax.yaxis.set_major_formatter(FixedFormatter(list(self.texts.y_tick_labels)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="y", labelsize=theme.SIZE_TICK)
        ax.tick_params(axis="x", length=0.0, pad=6.0)
        self.set_ticks()
        title = self.texts.x_title
        name = "" if title is None else drawable(title)
        ax.set_xlabel(
            name if options.xlab is None else options.xlab,
            fontproperties=theme.font(theme.SIZE_AXIS_LABEL),
            parse_math=options.xlab is not None,
        )
        ax.set_ylabel(
            "" if options.ylab is None else options.ylab,
            fontproperties=theme.font(theme.SIZE_AXIS_LABEL),
            parse_math=options.ylab is not None,
        )

    def set_ticks(self) -> None:
        self.ax.set_xticks(
            list(range(self.n_bars)),
            labels=self.tick_texts,
            parse_math=False,
            fontproperties=theme.font(theme.SIZE_TICK),
        )

    def draw_legend(self) -> None:
        """The levels of x, first on top as in the bars; each key styled as that
        level's segments are."""
        result, options = self.result, self.options
        handles: list[Rectangle] = []
        for i in range(self.k):
            key = Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=self.colors[i])
            key.set_edgecolor(self.edge)
            key.set_linewidth(SEGMENT_EDGE_PT)
            base.apply_args(key, options.bar_args, "bar_args")
            handles.append(key)
        names = self.legend_names()
        room = LEGEND_LINE_IN * self.dpi
        title = (
            self.wrapped(
                [drawable(self.texts.legend_title or result.x)], room, bold=True
            )[0]
            if options.legend_title is None
            else options.legend_title
        )
        legend = self.ax.legend(
            handles,
            names,
            # Right of the plot in inferplot's own figure; inside a given Axes,
            # right of the bars (anchored by the layout).
            loc="upper left" if self.own else "upper right",
            frameon=False,
            prop=theme.font(theme.SIZE_LABEL),
            title=title,
            title_fontproperties=theme.font(theme.SIZE_LABEL, weight="bold"),
            alignment="left",
            handlelength=1.2,
            handleheight=1.2,
            borderaxespad=0.0,
            borderpad=0.0,
        )
        legend.set_gid("legend")
        legend.get_title().set_parse_math(options.legend_title is not None)
        self.role(legend.get_title(), "legend title")
        self.legend_texts.append(legend.get_title())
        for i, text in enumerate(legend.get_texts()):
            text.set_parse_math(False)
            self.role(text, "legend entry", plain=names[i])
            self.legend_texts.append(text)
        self.legend = legend

    def legend_names(self) -> list[str]:
        """The levels of x as the legend writes them: wrapped on spaces into lines
        LEGEND_LINE_IN wide, at most three, shortened with an ellipsis."""
        room = LEGEND_LINE_IN * self.dpi
        return self.wrapped([drawable(level) for level in self.result.levels], room)

    def wrapped(
        self, names: Sequence[str], room: float, *, bold: bool = False
    ) -> list[str]:
        weight = "bold" if bold else "normal"
        probe = self.figure.text(
            0.0,
            0.0,
            "",
            fontproperties=theme.font(theme.SIZE_LABEL, weight=weight),
            parse_math=False,
        )
        renderer = self.renderer()

        def fits(line: str) -> bool:
            probe.set_text(line)
            return float(probe.get_window_extent(renderer).width) <= room

        try:
            return wrap_names(names, fits)
        finally:
            probe.remove()

    def unwrapped_ticks(self) -> list[str]:
        """Each bar's tick label: its level of y over its n label (the n label
        alone for the single bar without y), as betweenstats writes its levels."""
        return [
            f"{name}\n{n}" if name else n
            for name, n in zip(self.texts.tick_labels, self.texts.n_labels, strict=True)
        ]

    def wrap_ticks(self, room: float) -> None:
        """The tick labels with the level names wrapped into lines ``room``
        pixels wide (the n line is kept whole)."""
        probe = self.figure.text(
            0.0, 0.0, "", fontproperties=theme.font(theme.SIZE_TICK), parse_math=False
        )
        renderer = self.renderer()

        def fits(line: str) -> bool:
            probe.set_text(line)
            return float(probe.get_window_extent(renderer).width) <= room

        try:
            named = [name for name in self.texts.tick_labels if name]
            wrapped = iter(wrap_names(named, fits))
            self.tick_texts = [
                f"{next(wrapped)}\n{n}" if name else n
                for name, n in zip(
                    self.texts.tick_labels, self.texts.n_labels, strict=True
                )
            ]
        finally:
            probe.remove()
        self.set_ticks()

    # -- layout ----------------------------------------------------------------

    def generated_texts(self) -> tuple[str | None, str | None, Sequence[str]]:
        return self.texts.subtitle, self.result.text.subtitle, self.texts.caption_lines

    def decoration_texts(self) -> list[Text]:
        texts = [c.text for c in self.cells if c.text is not None]
        return texts + [t for t in self.bar_labels if t is not None]

    def label_size(self, cell: _Cell) -> tuple[float, float]:
        assert cell.text is not None
        box = self.extent(cell.text)
        return float(box.width), float(box.height)

    def strip_px(self) -> float:
        """The room above 100% for the bars' test labels."""
        pad = px(LABEL_PAD_PT, self.dpi)
        tallest = max(
            (self.extent(t).height for t in self.bar_labels if t is not None),
            default=0.0,
        )
        return tallest + 2 * pad if tallest else pad

    def fits_inside(self, cell: _Cell, ppu: float, bar_px: float) -> bool:
        w, h = self.label_size(cell)
        pad = 2 * px(CELL_PAD_PT, self.dpi)
        return h + pad <= cell.percent * ppu and w + pad <= bar_px

    def stack_px(self, moved: Sequence[_Cell]) -> float:
        gap = px(STACK_GAP_PT, self.dpi)
        return sum(self.label_size(c)[1] + gap for c in moved) + gap

    def labelled(self) -> list[_Cell]:
        return [c for c in self.cells if c.text is not None]

    def layout(self) -> None:
        if self.own:
            self.layout_own()
        else:
            self.layout_in_axes()

    def layout_own(self) -> None:
        dpi = self.dpi
        gap = px(GAP_PT, dpi)
        labelled = self.labelled()
        pad = 2 * px(CELL_PAD_PT, dpi)
        widest = max((self.label_size(c)[0] for c in labelled), default=0.0)
        # Many bars share MAX_PLOT_IN, down to MIN_SLOT_IN each (B11).
        floor = MIN_SPACING_IN * dpi
        if self.n_bars * MIN_SPACING_IN > MAX_PLOT_IN:
            floor = max(MIN_SLOT_IN, MAX_PLOT_IN / self.n_bars) * dpi
        self.bar_px = max(min(BAR_WIDTH_IN * dpi, BAR_SHARE * floor), widest + pad)
        # 0% to 100% at least as tall as the legend, which hangs from 100%.
        bars_px = max(BARS_HEIGHT_IN * dpi, self.legend_height())
        moved: list[_Cell] = []
        for _ in range(4):
            ppu = bars_px / 100.0
            moved = [c for c in labelled if not self.fits_inside(c, ppu, self.bar_px)]
            needed = max(
                (
                    self.stack_px([c for c in moved if c.j == j])
                    for j in range(self.n_bars)
                ),
                default=0.0,
            )
            if needed <= bars_px:
                break
            bars_px = needed
        self.bars_px = bars_px
        column = max((self.label_size(c)[0] for c in moved), default=0.0)
        self.column_px = column
        slot = max(floor, self.bar_px / BAR_SHARE)
        if moved:
            slot = max(slot, self.bar_px + px(LEAD_PT, dpi) + column + gap)
        slot = max(
            [slot]
            + [self.extent(t).width + gap for t in self.bar_labels if t is not None]
        )
        self.wrap_ticks(min(TICK_LINE_IN * dpi, floor - 2 * gap))
        slot = max(slot, self.widest_tick() + 2 * gap)
        self.slot_px = slot
        strip = self.strip_px()
        left, right = self.x_limits(slot, column if moved else 0.0)
        self.check_bound(slot * (right - left), bars_px + strip)
        ppu = bars_px / 100.0
        self.set_view(left, right, 0.0, 100.0 + strip / ppu)
        self.place_cells(moved, ppu)
        self.place_legend_outside()
        self.fit_figure(slot * (right - left), bars_px + strip)

    def check_bound(self, plot_w: float, plot_h: float) -> None:
        """Refuse a plotting area beyond MAX_PLOT_IN (B11): the bars' labels need
        wider slots, or the legend is taller, than the bound allows."""
        dpi = self.dpi
        width, height = plot_w / dpi, plot_h / dpi
        if width <= MAX_PLOT_IN + 1e-9 and height <= MAX_PLOT_IN + 1e-9:
            return
        y = self.result.y
        if width > MAX_PLOT_IN:
            fit = max(1, int(MAX_PLOT_IN // (self.slot_px / dpi)))
            why = (
                f"its {self.n_bars} bars need {self.slot_px / dpi:.2f} in each for "
                f"their labels, so about {fit} fit"
            )
            fewer = f"fewer levels of {y!r} (filter the data or merge levels)"
        else:
            per = self.legend_height() / dpi / max(len(self.result.levels), 1)
            fit = max(1, int(MAX_PLOT_IN // max(per, 1e-9)))
            why = (
                f"the legend of the {len(self.result.levels)} levels of "
                f"{self.result.x!r} is {self.legend_height() / dpi:.1f} in tall, so "
                f"about {fit} levels fit"
            )
            fewer = f"fewer levels of {self.result.x!r} (merge levels)"
        raise ValueError(
            f"barstats' figure does not fit inferplot's own figure, whose plotting "
            f"area is at most {MAX_PLOT_IN:g} in wide and tall: {why}. Draw "
            f"{fewer}, or draw into an Axes of a figure of your own size with ax=, "
            "for example ax=Figure(figsize=(40, 6)).add_subplot() (from "
            "matplotlib.figure import Figure); fig.check() then reports what does "
            "not fit."
        )

    def widest_tick(self) -> float:
        probe = [
            self.figure.text(
                0.0, 0.0, s, fontproperties=theme.font(theme.SIZE_TICK),
                parse_math=False,
            )
            for s in self.tick_texts
        ]  # fmt: skip
        try:
            return max(float(self.extent(t).width) for t in probe)
        finally:
            for t in probe:
                t.remove()

    def x_limits(self, slot: float, column: float) -> tuple[float, float]:
        """Bars at 0, 1, ...; the last slot reaches past the last bar's column of
        moved labels."""
        gap = px(GAP_PT, self.dpi)
        reach = slot / 2
        if column:
            reach = max(reach, self.bar_px / 2 + px(LEAD_PT, self.dpi) + column + gap)
        return -0.5, self.n_bars - 1 + reach / slot

    def set_view(self, left: float, right: float, bottom: float, top: float) -> None:
        if not self.own:
            bottom, top = base.shared_limits(self.ax, "y", bottom, top)
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)
        self.view.xlim = (left, right)
        self.view.ylim = (bottom, top)

    def place_cells(self, moved: Sequence[_Cell], ppu: float) -> None:
        """Size the segments to the bar width, put the labels that fit in the
        middle of their segment and the others in the column right of their
        bar, stacked; with ``ax=`` a moved label the column cannot hold is left
        out."""
        width = self.bar_px / self.slot_px
        for cell in self.cells:
            if cell.segment is not None:
                cell.segment.set_x(cell.j - width / 2)
                cell.segment.set_width(width)
            if cell.text is not None:
                cell.text.set_visible(True)
                cell.text.set_position((cell.j, cell.bottom + cell.percent / 2))
                cell.text.set_horizontalalignment("center")
                cell.text.set_color(self.ink_on_cell(cell))
                cell.place = "inside"
            if cell.leader is not None:
                cell.leader.remove()
                cell.leader = None
        for j in range(self.n_bars):
            self.stack(j, [c for c in moved if c.j == j], ppu)

    def ink_on_cell(self, cell: _Cell) -> str:
        assert cell.segment is not None
        return self.ink_on(cell.segment)

    def stack(self, j: int, moved: Sequence[_Cell], ppu: float) -> None:
        """The moved labels of bar ``j`` in the column right of it, each as near
        its segment's middle as the others allow, top to bottom in stacking
        order, within 0% to 100%."""
        if not moved:
            return
        dpi = self.dpi
        gap = px(STACK_GAP_PT, dpi)
        to_data = 1.0 / ppu
        heights = [self.label_size(c)[1] * to_data for c in moved]
        wanted = [c.bottom + c.percent / 2 for c in moved]
        g = gap * to_data
        # Top to bottom (moved holds the cells in level order: top first).
        centres = list(wanted)
        top = 100.0 - g
        for n, h in enumerate(heights):
            centres[n] = min(centres[n], top - h / 2)
            top = centres[n] - h / 2 - g
        floor = g
        for n in reversed(range(len(heights))):
            h = heights[n]
            centres[n] = max(centres[n], floor + h / 2)
            floor = centres[n] + h / 2 + g
        x_edge = j + self.bar_px / 2 / self.slot_px
        x_text = x_edge + px(LEAD_PT, dpi) / self.slot_px
        fits = self.own or self.column_fits(j, moved)
        for cell, centre in zip(moved, centres, strict=True):
            text = cell.text
            assert text is not None
            if not fits or centre + 1e-9 < 0.0 or centre > 100.0 + 1e-9:
                text.set_visible(False)
                cell.place = "omitted"
                continue
            text.set_position((x_text, centre))
            text.set_horizontalalignment("left")
            text.set_color(self.outside_ink())
            cell.place = "outside"
            middle = cell.bottom + cell.percent / 2
            line = Line2D(
                [x_edge, x_text - px(1.5, dpi) / self.slot_px],
                [middle, centre],
                color=self.outside_ink(),
                linewidth=LEADER_WIDTH_PT,
                zorder=4.0,
                gid=f"leader-{cell.j}-{cell.i}",
            )
            self.ax.add_line(line)
            cell.leader = line
        if not fits:
            return
        if sum(heights) + g * (len(heights) + 1) > 100.0 + 1e-9:
            for cell in moved:
                if cell.text is not None:
                    cell.text.set_visible(False)
                    cell.place = "omitted"
                if cell.leader is not None:
                    cell.leader.remove()
                    cell.leader = None

    def column_fits(self, j: int, moved: Sequence[_Cell]) -> bool:
        """With ``ax=``: whether the moved labels of bar ``j`` fit between it and
        the next bar (or the legend, or the Axes' edge)."""
        dpi = self.dpi
        start = self.bar_px / 2 + px(LEAD_PT, dpi)
        widest = max(self.label_size(c)[0] for c in moved)
        if j < self.n_bars - 1:
            room = self.slot_px - self.bar_px / 2 - px(GAP_PT, dpi)
        else:  # the legend starts a gap right of the last slot
            room = self.slot_px / 2 + self.extra_px
        return start + widest <= room + 0.5

    def outside_ink(self) -> str:
        return theme.ink_for(axes_background(self.ax)) if not self.own else theme.INK

    def place_legend_outside(self) -> None:
        legend = self.legend
        assert legend is not None
        legend.set_bbox_to_anchor(
            (1.0, 100.0),
            transform=offset_copy(
                self.ax.get_yaxis_transform(),
                fig=self.figure,
                x=3 * GAP_PT,
                units="points",
            ),
        )

    def legend_height(self) -> float:
        legend = self.legend
        assert legend is not None
        return float(legend.get_window_extent(self.renderer()).height)

    def legend_px(self) -> float:
        """The legend's width in pixels."""
        legend = self.legend
        assert legend is not None
        return float(legend.get_window_extent(self.renderer()).width)

    def legend_room(self) -> float:
        """With ``ax=``: the pixels right of the last slot that hold the legend."""
        return self.legend_px() + 2 * px(GAP_PT, self.dpi)

    def layout_in_axes(self) -> None:
        dpi = self.dpi
        gap = px(GAP_PT, dpi)
        legend = self.legend
        assert legend is not None
        self.top_px, self.bottom_px = self.place_blocks_in_axes()
        self.wrap_ticks(TICK_LINE_IN * dpi)
        frame = self.ax.bbox
        width, height = float(frame.width), float(frame.height)
        strip = self.strip_px()
        pad = px(LABEL_PAD_PT, dpi)
        below = self.bottom_px + pad if self.bottom_px else 0.0
        bars = max(height - self.top_px - strip - below, 0.2 * height)
        ppu = bars / 100.0
        bottom = -below / ppu
        top = 100.0 + (strip + self.top_px) / ppu
        room = self.legend_room()
        labelled = self.labelled()
        extra = 0.0
        slot = bar = column = 0.0
        for _ in range(4):
            slot = max((width - room - extra) / self.n_bars, 1.0)
            bar, column, moved = self.bars_in_slot(slot, ppu, labelled)
            # The last bar's moved labels reach past its slot, toward the legend.
            last = [c for c in moved if c.j == self.n_bars - 1]
            reach = bar / 2 + px(LEAD_PT, dpi) + column + gap if last else 0.0
            need = max(0.0, reach - slot / 2)
            if abs(need - extra) <= 0.5:
                break
            extra = need
        self.slot_px = slot
        self.bar_px = bar
        self.column_px = column
        self.extra_px = extra
        left, right = -0.5, self.n_bars - 0.5 + (room + extra) / slot
        self.set_view(left, right, bottom, top)
        # The view may be wider than asked for (a shared y axis): the scale of
        # the bars is the view's.
        y0, y1 = self.ax.get_ylim()
        ppu = height / (float(y1) - float(y0))
        moved = [c for c in labelled if not self.fits_inside(c, ppu, bar)]
        self.place_cells(moved, ppu)
        legend.set_bbox_to_anchor(
            (1.0, 100.0),
            transform=offset_copy(
                self.ax.get_yaxis_transform(),
                fig=self.figure,
                x=-px(GAP_PT, dpi) * 72.0 / dpi,
                units="points",
            ),
        )

    def bars_in_slot(
        self, slot: float, ppu: float, labelled: Sequence[_Cell]
    ) -> tuple[float, float, list[_Cell]]:
        """With ``ax=``: the bar width, the width of the column of moved labels
        and the cells moved, for slots ``slot`` pixels wide.

        A bar takes BAR_SHARE of its slot, more (up to the slot less two gaps)
        where that lets its labels' width fit; where a label is moved beside its
        bar, the bar narrows (to MIN_BAR_SHARE) to make room for it.
        """
        dpi = self.dpi
        gap = px(GAP_PT, dpi)
        cell_pad = 2 * px(CELL_PAD_PT, dpi)
        widest = max((self.label_size(c)[0] for c in labelled), default=0.0)
        bar = max(BAR_SHARE * slot, min(widest + cell_pad, slot - 2 * gap))
        # The bar only narrows, so the cells moved only grow: this ends, with
        # the moved cells and the column those of the final width.
        column = 0.0
        moved: list[_Cell] = []
        for _ in range(len(labelled) + 2):
            moved = [c for c in labelled if not self.fits_inside(c, ppu, bar)]
            if not moved:
                column = 0.0
                break
            column = max(self.label_size(c)[0] for c in moved)
            beside = slot - px(LEAD_PT, dpi) - column - gap - 1.0  # 1 px slack
            narrower = min(max(MIN_BAR_SHARE * slot, beside), bar)
            if narrower >= bar:
                break
            bar = narrower
        return bar, column, moved

    # -- finishing -------------------------------------------------------------

    def ink_inside(self, ink: str) -> None:
        """Subtitle, caption, the bars' test labels, the legend and the moved
        cell labels with their lines; labels on segments keep their own ink."""
        super().ink_inside(ink)
        for text in self.bar_labels:
            if text is not None:
                text.set_color(ink)
        for text in self.legend_texts:
            text.set_color(ink)
        for cell in self.cells:
            if cell.place == "outside" and cell.text is not None:
                cell.text.set_color(ink)
            if cell.leader is not None:
                cell.leader.set_color(ink)

    def drawn_data(self) -> tuple[DrawnData, ...]:
        drawn = [c for c in self.cells if c.segment is not None]
        if not drawn:
            return ()
        xs = np.array(
            [
                [s.get_x(), s.get_x() + s.get_width()]
                for c in drawn
                if (s := c.segment) is not None
            ],
            dtype=np.float64,
        )
        ys = np.array(
            [[c.bottom, c.bottom + c.percent] for c in drawn], dtype=np.float64
        )
        return (DrawnData("main", "bar segments", xs, ys, spread="y"),)

    def cell_records(self) -> tuple[CellLabel, ...]:
        """Every cell with a label, for ``check()``."""
        return tuple(
            CellLabel(
                text=c.text,
                segment=c.segment,
                name=self.cell_name(c),
                words=c.words,
                place=c.place,
                bar=c.j,
            )
            for c in self.cells
            if c.text is not None and c.segment is not None
        )

    def finish_result(self, result: StatsResult) -> StatsResult:
        """The result with the alt text of what this figure draws: whether cell
        labels are drawn (all of them in inferplot's own figure; with ``ax=``
        some may be left out, which ``fig.check()`` reports), whether the bars'
        test labels are drawn, and the legend (always)."""
        drawn = any(c.place != "omitted" for c in self.cells if c.text is not None)
        alt = figure_alt_text(
            result,
            cell_labels=drawn,
            bar_tests=(
                None
                if result.y is None
                else any(t is not None for t in self.bar_labels)
            ),
            legend=True,
        )
        if alt == result.text.alt_text:
            return result
        return dataclasses.replace(
            result, text=dataclasses.replace(result.text, alt_text=alt)
        )


def _hex(rgb: tuple[float, float, float]) -> str:
    return to_hex(rgb)
