# matplotlib's and NumPy's stubs type many ``**kwargs`` and overloads as Unknown,
# which strict mode reports on every call; the arguments we pass are all typed.
# pyright: reportUnknownMemberType=false
"""Draw a histostats result: ggstatsplot's ``gghistostats()`` layers, our look.

* bars: the histogram of the analyzed values, with ggstatsplot's bins
  (:func:`inferplot.stats.histogram_bins`: ``binwidth``, by default (max - min) /
  sqrt(n), on the grid (k + 1/2) w), counts on the left axis and each count's
  share of n, in percent, on the right axis;
* the centrality line: dashed, at the type's centrality value (mean, median,
  trimmed mean, MAP), with ggstatsplot's label (``centrality_plotting``);
* the test value: a thin grey line at ``test_value`` with the label "test value
  18" (inferplot's addition; ggstatsplot 1.1.4 draws none, but the figure shows
  a test of the sample against that value). Where it lies more than a quarter of
  the histogram's width beyond the bars, the view is not stretched to reach it:
  an arrow at that edge names it instead ("← test value 0");
* the header (title, subtitle) and caption lines, as in the other figures.

The labels sit in a strip above the tallest bar, beside their lines, pointing
away from each other (the centrality label on the side away from the test
value); two labels that would still touch take two rows. The renderer computes
no statistics: the bins are :mod:`inferplot.stats.plot_geometry`'s, every other
number the result's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox, offset_copy
from numpy.typing import NDArray

from inferplot import _validate as v
from inferplot.render import base, theme
from inferplot.render.base import GAP_PT, LABEL_BOX_PAD, LABEL_PAD_PT, LayerArgs, px
from inferplot.render.figure import DrawnData, DrawnView, StatsFigure
from inferplot.result import StatsResult
from inferplot.stats.plot_geometry import HistogramBins, histogram_bins
from inferplot.text import (
    drawable,
    figure_alt_text,
    figure_texts,
    r_number_text,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist

__all__ = ["MAX_BARS", "check_drawing", "histostats"]

FloatArray = NDArray[np.float64]

# inferplot's own figure: the plotting area's width and the height of the region
# the bars take (the label strip comes on top), in inches.
AXES_WIDTH_IN: Final = 5.0
BARS_HEIGHT_IN: Final = 2.8
# Room above the tallest bar, as a share of its height.
HEADROOM: Final = 0.04
# Room left and right of the bars, as a share of the drawn range.
X_PAD: Final = 0.03
# The test value is drawn as a line where it lies within this share of the
# histogram's width beyond the bars; farther, an arrow at the edge names it.
TEST_VALUE_REACH: Final = 0.25
# The most bars drawn (about one per pixel of a wide figure).
MAX_BARS: Final = 10_000
BAR_ALPHA: Final = 0.8
CENTRALITY_LINE_WIDTH: Final = 1.4
TEST_VALUE_LINE_WIDTH: Final = 1.0
TEST_VALUE_COLOR: Final = "#7A7A7A"
LINE_GAP_PT: Final = 3.0  # between a line and its label


@dataclasses.dataclass(frozen=True, slots=True)
class Options:
    """The checked drawing arguments."""

    binwidth: float | None
    centrality: bool
    xlab: str | None
    ylab: str | None
    title: str | None
    subtitle: str | None
    caption: str | None
    bar_args: dict[str, object]


def check_drawing(
    *,
    binwidth: float | None,
    centrality_plotting: bool,
    xlab: str | None,
    ylab: str | None,
    title: str | None,
    subtitle: str | None,
    caption: str | None,
    bar_args: Mapping[str, object] | None,
) -> Options:
    """The checks of the drawing arguments that need no figure: ``ip.histostats``
    runs them before the analysis, so a bad one is reported first."""
    width = (
        None
        if binwidth is None
        else v.check_positive(binwidth, "binwidth", example="2")
    )
    return Options(
        binwidth=width,
        centrality=v.check_bool(centrality_plotting, "centrality_plotting"),
        xlab=v.check_optional_text(xlab, "xlab"),
        ylab=v.check_optional_text(ylab, "ylab"),
        title=v.check_optional_text(title, "title"),
        subtitle=v.check_optional_text(subtitle, "subtitle"),
        caption=v.check_optional_text(caption, "caption"),
        bar_args=v.check_artist_args(bar_args, "bar_args") or {},
    )


@v.checked
def histostats(
    result: StatsResult,
    *,
    binwidth: float | None = None,
    centrality_plotting: bool = True,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    palette: str | Sequence[object] | None = None,
    bar_args: Mapping[str, object] | None = None,
    ax: Axes | None = None,
) -> StatsFigure:
    """Draw a result of ``ip.analyze.histostats``.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        binwidth: the width of the bars; by default (max - min) / sqrt(n), as in
            ggstatsplot (which needs values that differ: for a constant column
            pass a binwidth). At most 10,000 bars are drawn.
        centrality_plotting: draw the centrality line and its label.
        xlab, ylab, title, subtitle, caption: texts, as for
            ``ip.render.betweenstats``; ``ylab`` names the count axis.
        palette: a matplotlib colormap name or a list of colors; the bars take the
            first.
        bar_args: matplotlib properties applied with ``Artist.set`` to every bar
            (a ``Rectangle``), for example ``{"alpha": 0.5}``; properties that set
            the bins (``x``, ``width``, ``height``, ...) are refused.
        ax: draw into this Axes instead of a new figure; the subtitle and caption
            then go inside it.

    The bars are ggstatsplot's histogram; the right axis gives each count as a
    percentage of n. The centrality line is dashed, at the result's centrality
    value, labelled as ggstatsplot labels it; the test value is marked by a thin
    grey line labelled "test value", or, when it lies far beyond the bars, by an
    arrow at that edge. ``fig.check()`` reports what does not fit.

    Returns:
        A :class:`~inferplot.render.figure.StatsFigure`; its ``result`` adds any
        rendering warnings to ``result.warnings``.
    """
    base.check_drawable(result, "histostats", "ip.analyze.histostats(data, x, ...)")
    options = check_drawing(
        binwidth=binwidth,
        centrality_plotting=centrality_plotting,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        bar_args=bar_args,
    )
    base.check_subtitle(result, options.subtitle)
    base.check_given_axes(ax)
    values = result.values[0]
    base.check_scale(values, result.x, "x")
    bins = _bins(values, options.binwidth)
    return base.draw(
        lambda colors, warning: _HistDrawing(
            result, options, colors, warning, ax, bins
        ),
        palette=palette,
        k=1,
        options=options,
        artist_args=lambda: base.check_artist_args(
            [(_BAR_ARGS, options.bar_args, None)]
        ),
    )


def _test_value(result: StatsResult) -> float:
    tv = result.settings.test_value
    if tv is None:
        raise ValueError(
            "result is inconsistent: a histostats result needs a test value"
        )
    return float(tv)


def _bins(values: Sequence[float], binwidth: float | None) -> HistogramBins:
    bins = histogram_bins(np.asarray(values, dtype=np.float64), binwidth=binwidth)
    count = len(bins.counts)
    if count > MAX_BARS:
        span = bins.edges[-1] - bins.edges[0]
        raise ValueError(
            f"binwidth={bins.binwidth!r} gives {count:,} bars, more than the "
            f"{MAX_BARS:,} inferplot draws (each would be narrower than a pixel). "
            f"Pass a larger binwidth, for example binwidth={span / 100:.3g} "
            "(100 bars)."
        )
    return bins


def _bar_artist(
    ax: Axes, left: float, width: float, height: float, color: theme.Color, i: int
) -> Rectangle:
    """One bar, before ``bar_args``."""
    bar = Rectangle(
        (left, 0.0),
        width,
        height,
        facecolor=(*color[:3], BAR_ALPHA),
        edgecolor="white",
        linewidth=0.6,
        zorder=2.0,
        gid=f"bar-{i}",
    )
    ax.add_patch(bar)
    return bar


def _scratch_bar(ax: Axes) -> Artist:
    return _bar_artist(ax, -0.4, 0.8, 4.0, (0.0, 0.0, 0.0, 1.0), 0)


_BAR_ARGS: Final = LayerArgs(
    name="bar_args",
    layer="bars",
    noun="bars",
    make=_scratch_bar,
    advice="Choose a visible style; the bars are the histogram of the values.",
)


class _NonNegative(MaxNLocator):
    """Integer-friendly ticks from 0 up: counts and their shares are never
    negative, also where the view reaches below 0 (the caption's room with
    ``ax=``)."""

    def tick_values(self, vmin: float, vmax: float) -> Any:
        low, high = sorted((float(vmin), float(vmax)))
        ticks = np.asarray(super().tick_values(max(low, 0.0), max(high, 0.0)))
        return ticks[(ticks >= 0.0) & (ticks <= high * (1 + 1e-9))]


def _percent(value: float, position: object) -> str:
    """A tick of the percent axis: "12.5%", without the noise of the scaling."""
    del position
    return f"{float(f'{value:.10g}'):g}%"


Side = Literal["left", "right"]


@dataclasses.dataclass(slots=True)
class _StripLabel:
    """A label in the strip above the bars: at data x ``x`` (None: at an edge of
    the Axes, ``edge``), on ``side`` of it, in ``row`` (0 the top one)."""

    text: Text
    x: float | None
    side: Side
    row: int = 0
    edge: Side = "left"


class _HistDrawing(base.BaseDrawing):
    """A histostats result: bars, centrality and test value lines and labels."""

    def __init__(
        self,
        result: StatsResult,
        options: Options,
        colors: list[theme.Color],
        palette_warning: str | None,
        ax: Axes | None,
        bins: HistogramBins,
    ) -> None:
        super().__init__(result, options, colors, palette_warning, ax)
        self.options = options
        self.bins = bins
        self.values = np.asarray(result.values[0], dtype=np.float64)
        self.texts = figure_texts(result)
        self.test_value = _test_value(result)
        d = result.descriptives[0]
        self.center = d.value if options.centrality else None
        self.bars: list[Rectangle] = []
        self.center_line: Line2D | None = None
        self.test_line: Line2D | None = None
        self.labels: list[_StripLabel] = []
        self.percent: Axes | None = None
        self.rows = 1
        self.top_px = 0.0
        self.bottom_px = 0.0
        self.view = DrawnView(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        lo, hi = bins.edges[0], bins.edges[-1]
        reach = TEST_VALUE_REACH * (hi - lo)
        self.test_mark: Literal["line", "edge"] = (
            "line" if lo - reach <= self.test_value <= hi + reach else "edge"
        )

    # -- drawing -----------------------------------------------------------------

    def run(self) -> StatsFigure:
        self.draw_bars()
        self.draw_lines()
        self.draw_axes_texts()
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
        return StatsFigure(
            figure=self.figure,
            axes={"main": self.ax},
            result=result,
            own=self.own,
            roles=self.roles,
            inside=tuple(label.text for label in self.labels),
            subsample=None,
            view=self.view,
            colors=tuple(self.colors),
            data=self.drawn_data(),
            color_names=("the bars",),
        )

    def draw_bars(self) -> None:
        edges, counts = self.bins.edges, self.bins.counts
        for i, count in enumerate(counts):
            bar = _bar_artist(
                self.ax,
                edges[i],
                edges[i + 1] - edges[i],
                float(count),
                self.colors[0],
                i,
            )
            base.apply_args(bar, self.options.bar_args, "bar_args")
            self.bars.append(bar)

    def draw_lines(self) -> None:
        ax = self.ax
        texts: list[_StripLabel] = []
        center = self.center
        if center is not None:
            self.center_line = ax.axvline(
                center,
                color=theme.CENTRALITY_COLOR,
                linewidth=CENTRALITY_LINE_WIDTH,
                linestyle=(0, (4.0, 2.5)),
                zorder=4.0,
                gid="centrality-line",
            )
            text = ax.text(
                center,
                1.0,
                self.texts.centrality_labels[0],
                fontproperties=theme.font(theme.SIZE_LABEL),
                bbox={
                    "boxstyle": f"round,pad={LABEL_BOX_PAD}",
                    "facecolor": "white",
                    "edgecolor": theme.CENTRALITY_COLOR,
                    "linewidth": 0.7,
                },
                zorder=6.0,
                gid="centrality-label",
            )
            plain = self.result.text.centrality_labels
            self.role(text, "centrality label", plain[0] if plain else None)
            texts.append(_StripLabel(text=text, x=center, side="right"))
        tv = self.test_value
        if self.test_mark == "line":
            self.test_line = ax.axvline(
                tv,
                color=TEST_VALUE_COLOR,
                linewidth=TEST_VALUE_LINE_WIDTH,
                zorder=3.5,
                gid="test-value-line",
            )
            words = f"test value {r_number_text(tv)}"
            where: float | None = tv
        else:
            below = tv < self.bins.edges[0]
            shown = r_number_text(tv)
            words = f"← test value {shown}" if below else f"test value {shown} →"
            where = None
        text = ax.text(
            tv if where is not None else 0.0,
            1.0,
            words,
            fontproperties=theme.font(theme.SIZE_LABEL),
            color=theme.SOFT_INK,
            parse_math=False,
            zorder=6.0,
            gid="test-value-label",
        )
        self.role(text, "test value label")
        label = _StripLabel(text=text, x=where, side="right")
        if where is None:
            label.edge = "left" if tv < self.bins.edges[0] else "right"
        texts.append(label)
        self.labels = texts
        # Pointing away from each other: the centrality label on the side away
        # from the test value, the test value's label on the other.
        if center is not None:
            above = tv > center
            texts[0].side = "left" if above else "right"
            texts[1].side = "right" if above else "left"

    def draw_axes_texts(self) -> None:
        ax, result, options = self.ax, self.result, self.options
        ax.set_xlabel(
            drawable(result.x) if options.xlab is None else options.xlab,
            fontproperties=theme.font(theme.SIZE_AXIS_LABEL),
            parse_math=options.xlab is not None,
        )
        ax.set_ylabel(
            "count" if options.ylab is None else options.ylab,
            fontproperties=theme.font(theme.SIZE_AXIS_LABEL),
            parse_math=options.ylab is not None,
        )
        ax.tick_params(axis="both", labelsize=theme.SIZE_TICK)
        ax.yaxis.set_major_locator(_NonNegative(integer=True))
        n = float(self.bins.n)

        def share(count: FloatArray) -> FloatArray:
            return np.asarray(count, dtype=np.float64) * 100.0 / n

        def count(share: FloatArray) -> FloatArray:
            return np.asarray(share, dtype=np.float64) * n / 100.0

        # matplotlib types the functions over ArrayLike; they get float arrays.
        percent = ax.secondary_yaxis("right", functions=cast("Any", (share, count)))
        percent.yaxis.set_major_locator(_NonNegative(steps=[1, 2, 5, 10]))
        percent.yaxis.set_major_formatter(FuncFormatter(_percent))
        percent.set_ylabel(
            "percent of values", fontproperties=theme.font(theme.SIZE_AXIS_LABEL)
        )
        percent.tick_params(axis="y", labelsize=theme.SIZE_TICK)
        # A SecondaryAxis: an Axes for every use here (labels, ticks, fonts).
        self.percent = cast("Axes", percent)

    # -- layout ----------------------------------------------------------------

    def generated_texts(self) -> tuple[str | None, str | None, Sequence[str]]:
        return self.texts.subtitle, self.result.text.subtitle, self.texts.caption_lines

    def font_axes(self) -> list[Axes]:
        return [self.ax] if self.percent is None else [self.ax, self.percent]

    def ink_inside(self, ink: str) -> None:
        super().ink_inside(ink)
        for label in self.labels:
            if label.text.get_gid() == "test-value-label":
                label.text.set_color(ink)
        if self.test_line is not None:
            self.test_line.set_color(ink)

    def decoration_texts(self) -> list[Text]:
        return [label.text for label in self.labels]

    def row_px(self) -> float:
        """The height of one row of the label strip, in pixels."""
        pad = px(LABEL_PAD_PT, self.dpi)
        tallest = max(
            (self.extent(label.text).height for label in self.labels), default=0.0
        )
        return tallest + pad

    def strip_px(self) -> float:
        return self.rows * self.row_px() + px(LABEL_PAD_PT, self.dpi)

    def layout(self) -> None:
        for _ in range(2):
            if self.own:
                self.set_limits(BARS_HEIGHT_IN * self.dpi + self.strip_px())
                self.fit_figure(
                    AXES_WIDTH_IN * self.dpi,
                    BARS_HEIGHT_IN * self.dpi + self.strip_px(),
                )
            else:
                self.top_px, self.bottom_px = self.place_blocks_in_axes()
                self.set_limits(float(self.ax.bbox.height))
            rows = self.place_labels()
            if rows == self.rows:
                return
            self.rows = rows

    def set_limits(self, height: float) -> None:
        """x: the bars (and a test value close to them), padded; y: from 0 to the
        tallest bar, with the label strip (and with ``ax=`` the subtitle and
        caption) above and below it in pixels of an Axes ``height`` high."""
        lo, hi = self.bins.edges[0], self.bins.edges[-1]
        if self.test_mark == "line":
            lo, hi = min(lo, self.test_value), max(hi, self.test_value)
        if self.center is not None:
            lo, hi = min(lo, self.center), max(hi, self.center)
        pad = X_PAD * (hi - lo)
        left, right = lo - pad, hi + pad
        top_data = max(max(self.bins.counts), 1) * (1.0 + HEADROOM)
        above = self.top_px + self.strip_px()
        below = self.bottom_px + (px(LABEL_PAD_PT, self.dpi) if self.bottom_px else 0.0)
        room = max(height - above - below, 0.2 * height)
        ppu = room / top_data
        bottom, top = -below / ppu, top_data + above / ppu
        if not self.own:
            left, right = base.shared_limits(self.ax, "x", left, right)
            bottom, top = base.shared_limits(self.ax, "y", bottom, top)
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)
        self.view.xlim = (left, right)
        self.view.ylim = (bottom, top)
        # The lines span the bars and the label strip, not the subtitle and
        # caption that ax= puts inside the Axes (in Axes fractions, so they keep
        # their span when the Axes is resized).
        span = [below / height, 1.0 - self.top_px / height]
        for line in (self.center_line, self.test_line):
            if line is not None:
                line.set_ydata(span)

    def place_labels(self) -> int:
        """Put the labels in the strip, each beside its line on its side, or on
        the other side where it would reach out of the Axes; the rows needed (2
        where the two labels would still touch)."""
        frame = self.ax.bbox
        gap = LINE_GAP_PT
        to_pt = 72.0 / self.dpi
        down = (self.top_px + px(LABEL_PAD_PT, self.dpi)) * to_pt
        row_pt = self.row_px() * to_pt
        boxes: list[Bbox] = []
        for label in self.labels:
            y = -(down + label.row * row_pt)
            if label.x is None:
                at_left = label.edge == "left"
                label.text.set_transform(
                    offset_copy(
                        self.ax.transAxes, fig=self.figure,
                        x=gap if at_left else -gap, y=y, units="points",
                    )
                )  # fmt: skip
                label.text.set_position((0.0 if at_left else 1.0, 1.0))
                label.text.set_horizontalalignment("left" if at_left else "right")
                label.text.set_verticalalignment("top")
                boxes.append(self.extent(label.text))
                continue
            tried: list[tuple[float, Side]] = []
            for side in (label.side, "left" if label.side == "right" else "right"):
                self.put(label, side, y)
                box = self.extent(label.text)
                out = max(0.0, frame.x0 - box.x0) + max(0.0, box.x1 - frame.x1)
                tried.append((out, side))
                if out <= 0.5:
                    break
            out, side = min(tried, key=lambda t: t[0])
            self.put(label, side, y)
            boxes.append(self.extent(label.text))
        if len(boxes) == 2 and self.rows == 2:
            # The test value's label, turned toward the centrality line (no room
            # on its own side), starts beyond that line, so that the line does
            # not run through it.
            label, center = self.labels[1], self.center
            assert center is not None
            if label.x is not None and (
                (label.side == "right" and center > label.x)
                or (label.side == "left" and center < label.x)
            ):
                self.put(label, label.side, -(down + row_pt), at=center)
        if len(boxes) == 2 and self.rows == 1:
            a, b = boxes
            gap_px = px(GAP_PT, self.dpi)
            if a.x0 < b.x1 + gap_px and b.x0 < a.x1 + gap_px:
                self.labels[-1].row = 1
                return 2
        return self.rows

    def put(
        self, label: _StripLabel, side: Side, y: float, *, at: float | None = None
    ) -> None:
        """Put ``label`` on ``side`` of its line (or of ``at``), ``y`` points
        below the top of the Axes."""
        assert label.x is not None
        gap = LINE_GAP_PT + (
            LABEL_BOX_PAD * theme.SIZE_LABEL if label.text.get_bbox_patch() else 0.0
        )
        label.text.set_transform(
            offset_copy(
                self.ax.get_xaxis_transform(), fig=self.figure,
                x=gap if side == "right" else -gap, y=y, units="points",
            )
        )  # fmt: skip
        label.text.set_position((label.x if at is None else at, 1.0))
        label.text.set_horizontalalignment("left" if side == "right" else "right")
        label.text.set_verticalalignment("top")
        label.side = side

    # -- for check() and the result ----------------------------------------------

    def drawn_data(self) -> tuple[DrawnData, ...]:
        edges = np.asarray(self.bins.edges, dtype=np.float64)
        counts = np.asarray(self.bins.counts, dtype=np.float64)
        bars_x = np.column_stack([edges[:-1], edges[1:], edges[:-1], edges[1:]])
        zeros = np.zeros_like(counts)
        bars_y = np.column_stack([zeros, zeros, counts, counts])
        data = [
            DrawnData("main", "values", self.values[:, None], None, spread="x"),
            DrawnData("main", "bars", bars_x, bars_y, spread="y"),
        ]
        if self.center is not None:
            data.append(
                DrawnData(
                    "main", "the centrality line", np.array([[self.center]]), None
                )
            )
        if self.test_mark == "line":
            data.append(
                DrawnData(
                    "main", "the test value line", np.array([[self.test_value]]), None
                )
            )
        return tuple(data)

    def finish_result(self, result: StatsResult) -> StatsResult:
        """The result with the alt text of what this figure draws: the
        centrality line, the test value line (where the test value lies beyond
        the bars an arrow at the edge names it instead, and no line is named) and
        the number of bars."""
        alt = figure_alt_text(
            result,
            centrality=self.center is not None,
            test_value_line=self.test_mark == "line",
            bins=len(self.bins.counts),
        )
        if alt == result.text.alt_text:
            return result
        return dataclasses.replace(
            result, text=dataclasses.replace(result.text, alt_text=alt)
        )
