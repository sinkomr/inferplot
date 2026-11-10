# matplotlib's and NumPy's stubs type many ``**kwargs`` and overloads as Unknown,
# which strict mode reports on every call; the arguments we pass are all typed.
# pyright: reportUnknownMemberType=false
"""Draw a scatterstats result: ggstatsplot's ``ggscatterstats()`` layers, our look.

* points: every complete pair; above MAX_POINTS pairs a seeded subsample of
  SUBSAMPLE_SIZE points (always with the pairs that hold the smallest and largest
  x and y, and every labelled point) over a density layer (a 2D histogram of all
  pairs), recorded in ``warnings`` and reported by ``fig.check()``;
* the least-squares line with its pointwise ``conf_level`` band (``smooth``;
  :func:`inferplot.stats.ols_band`, ggplot2's ``geom_smooth(method = "lm")``
  at 80 points); where no line exists (x constant) nothing is drawn and
  ``warnings`` says so;
* marginal histograms (``marginal``; :func:`inferplot.stats.marginal_bins`,
  ggside's 30 bins) above and right of the plot, in panels that share its axes
  exactly ("xside" and "yside" in ``fig.axes``);
* the point labels of ``label_var``, placed by
  :func:`inferplot.render.point_labels.place_labels` clear of each other and of
  the points where a free place exists, joined to their point by a thin line
  when moved away;
* the header (title, subtitle) and caption lines, as in the other figures.

The renderer computes no statistics: the line, band and bins are
:mod:`inferplot.stats.plot_geometry`'s, every other number the result's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox, offset_copy
from numpy.typing import NDArray

from inferplot import _validate as v
from inferplot.render import base, theme
from inferplot.render.base import LayerArgs, px
from inferplot.render.comparison import MAX_POINTS, SUBSAMPLE_SIZE, with_extremes
from inferplot.render.figure import (
    DrawnData,
    DrawnView,
    PointLabels,
    StatsFigure,
    Subsample,
)
from inferplot.render.histostats import checked_result
from inferplot.render.point_labels import place_labels
from inferplot.result import StatsResult
from inferplot.stats.plot_geometry import (
    HistogramBins,
    OlsBand,
    marginal_bins,
    ols_band,
)
from inferplot.text import DrawnLayers, drawable, figure_alt_text, figure_texts

if TYPE_CHECKING:
    from matplotlib.artist import Artist

__all__ = ["MAX_LABELS", "check_drawing", "checked_result", "scatterstats"]

FloatArray = NDArray[np.float64]

# inferplot's own figure: the plot's size, and the marginal panels' thickness and
# their gap to the plot, in inches.
PLOT_WIDTH_IN: Final = 4.6
PLOT_HEIGHT_IN: Final = 3.6
SIDE_IN: Final = 0.75
SIDE_GAP_IN: Final = 0.06
# Room around the data, as a share of their range.
PAD: Final = 0.05
# The most point labels placed clear of each other and of the points.
MAX_LABELS: Final = 200
POINT_SIZE: Final = 16.0  # marker area, points squared
POINT_ALPHA: Final = 0.6
LINE_WIDTH: Final = 1.4
BAND_ALPHA: Final = 0.18
MARGINAL_ALPHA: Final = 0.6
DENSITY_BINS: Final = 60
LABEL_SIZE: Final = 8.5
LABEL_GAP_PT: Final = 1.5  # between a point's marker and its label
LINK_WIDTH: Final = 0.6


@dataclasses.dataclass(frozen=True, slots=True)
class Options:
    """The checked drawing arguments."""

    marginal: bool
    smooth: bool
    xlab: str | None
    ylab: str | None
    title: str | None
    subtitle: str | None
    caption: str | None
    point_args: dict[str, object]
    smooth_args: dict[str, object]
    marginal_args: dict[str, object]


def check_drawing(
    *,
    marginal: bool,
    smooth: bool,
    xlab: str | None,
    ylab: str | None,
    title: str | None,
    subtitle: str | None,
    caption: str | None,
    point_args: Mapping[str, object] | None,
    smooth_args: Mapping[str, object] | None,
    marginal_args: Mapping[str, object] | None,
    ax: object,
) -> Options:
    """The checks of the drawing arguments that need no figure: ``ip.scatterstats``
    runs them before the analysis, so a bad one is reported first."""
    with_marginal = v.check_bool(marginal, "marginal")
    options = Options(
        marginal=with_marginal,
        smooth=v.check_bool(smooth, "smooth"),
        xlab=v.check_optional_text(xlab, "xlab"),
        ylab=v.check_optional_text(ylab, "ylab"),
        title=v.check_optional_text(title, "title"),
        subtitle=v.check_optional_text(subtitle, "subtitle"),
        caption=v.check_optional_text(caption, "caption"),
        point_args=v.check_artist_args(point_args, "point_args") or {},
        smooth_args=v.check_artist_args(smooth_args, "smooth_args") or {},
        marginal_args=v.check_artist_args(marginal_args, "marginal_args") or {},
    )
    if ax is not None and with_marginal:
        raise ValueError(
            "ax= draws into an existing Axes, which has no room for the marginal "
            "histograms (marginal=True draws them in panels of their own). Pass "
            "marginal=False to draw into ax, or leave ax out."
        )
    return options


@v.checked
def scatterstats(
    result: StatsResult,
    *,
    marginal: bool = True,
    smooth: bool = True,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    palette: str | Sequence[object] | None = None,
    point_args: Mapping[str, object] | None = None,
    smooth_args: Mapping[str, object] | None = None,
    marginal_args: Mapping[str, object] | None = None,
    ax: Axes | None = None,
) -> StatsFigure:
    """Draw a result of ``ip.analyze.scatterstats``.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        marginal: draw histograms of x and y in panels above and right of the
            plot (``fig.axes["xside"]`` and ``["yside"]``, sharing its axes).
        smooth: draw the least-squares line with its pointwise ``conf_level``
            band (the analysis' ``conf_level``).
        xlab, ylab, title, subtitle, caption: texts, as for
            ``ip.render.betweenstats``.
        palette: a matplotlib colormap name or a list of colors; the points and
            the marginal histograms take the first.
        point_args: matplotlib properties of the points (a ``PathCollection``).
        smooth_args: matplotlib properties of the line (a ``Line2D``); the band
            takes the line's colour. Needs ``smooth=True``.
        marginal_args: matplotlib properties of every bar of the marginal
            histograms (``Rectangle``). Needs ``marginal=True``.
        ax: draw into this Axes instead of a new figure; needs
            ``marginal=False``. The subtitle and caption then go inside it.

    The point labels are the result's (``label_var`` of the analysis): each
    non-empty label is drawn next to its point, where it covers no other label
    and, where a free place exists, no point; a label moved away is joined to
    its point by a thin line. Up to 200 labels are placed so; beyond that they
    are drawn right of their points and ``fig.check()`` says so. With more than
    5,000 pairs the points are a seeded subsample of 2,000 over a density layer
    of all pairs.

    Returns:
        A :class:`~inferplot.render.figure.StatsFigure`; its ``result`` adds any
        rendering warnings to ``result.warnings``.
    """
    base.check_drawable(
        result, "scatterstats", "ip.analyze.scatterstats(data, x, y, ...)"
    )
    options = check_drawing(
        marginal=marginal,
        smooth=smooth,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        point_args=point_args,
        smooth_args=smooth_args,
        marginal_args=marginal_args,
        ax=ax,
    )
    base.check_subtitle(result, options.subtitle)
    base.check_given_axes(ax)
    if len(result.values) != 2 or len(result.values[0]) != len(result.values[1]):
        raise ValueError(
            "result is inconsistent: a scatterstats result needs the x and y of "
            f"each pair; got {[len(part) for part in result.values]} values."
        )
    y_name = result.y if result.y is not None else "y"
    base.check_scale(result.values[0], result.x, "x")
    base.check_scale(result.values[1], y_name, "y")
    layers = _Layers.compute(result, options)
    return base.draw(
        lambda colors, warning: _ScatterDrawing(
            result, options, colors, warning, ax, layers
        ),
        palette=palette,
        k=1,
        options=options,
        artist_args=lambda: base.check_artist_args(
            [
                (_POINT_ARGS, options.point_args, None),
                (
                    _SMOOTH_ARGS,
                    options.smooth_args,
                    None
                    if options.smooth
                    else (
                        "smooth_args styles the least-squares line, which "
                        "smooth=False does not draw. To style it, pass "
                        "smooth=True; otherwise remove smooth_args=."
                    ),
                ),
                (
                    _MARGINAL_ARGS,
                    options.marginal_args,
                    None
                    if options.marginal
                    else (
                        "marginal_args styles the marginal histograms, which "
                        "marginal=False does not draw. To style them, pass "
                        "marginal=True; otherwise remove marginal_args=."
                    ),
                ),
            ]
        ),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _Layers:
    """The numbers the figure draws besides the pairs, computed before anything
    is drawn: the line and band (or why there is none) and the marginal bins."""

    band: OlsBand | None
    no_line: str | None
    x_bins: HistogramBins | None
    y_bins: HistogramBins | None

    @staticmethod
    def compute(result: StatsResult, options: Options) -> _Layers:
        x = np.asarray(result.values[0], dtype=np.float64)
        y = np.asarray(result.values[1], dtype=np.float64)
        band: OlsBand | None = None
        no_line: str | None = None
        if options.smooth:
            try:
                band = ols_band(x, y, conf_level=result.settings.conf_level)
            except ValueError as error:
                no_line = str(error)
        x_bins = y_bins = None
        if options.marginal:
            x_bins, y_bins = marginal_bins(x), marginal_bins(y)
        return _Layers(band=band, no_line=no_line, x_bins=x_bins, y_bins=y_bins)


def _points_artist(
    ax: Axes, x: FloatArray, y: FloatArray, color: theme.Color
) -> PathCollection:
    """The points, before ``point_args``."""
    return ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        color=color,
        alpha=POINT_ALPHA,
        linewidths=0.0,
        zorder=3.0,
        gid="points",
    )


def _line_artist(ax: Axes, x: FloatArray, y: FloatArray) -> Line2D:
    """The least-squares line, before ``smooth_args``."""
    line = Line2D(
        x,
        y,
        color=theme.INK,
        linewidth=LINE_WIDTH,
        solid_capstyle="butt",
        zorder=4.0,
        gid="smooth-line",
    )
    ax.add_line(line)
    return line


def _marginal_bar(
    ax: Axes, xy: tuple[float, float], width: float, height: float,
    color: theme.Color, gid: str,
) -> Rectangle:  # fmt: skip
    """One bar of a marginal histogram, before ``marginal_args``."""
    bar = Rectangle(
        xy,
        width,
        height,
        facecolor=(*color[:3], MARGINAL_ALPHA),
        edgecolor="white",
        linewidth=0.5,
        zorder=2.0,
        gid=gid,
    )
    ax.add_patch(bar)
    return bar


_BLACK: Final[theme.Color] = (0.0, 0.0, 0.0, 1.0)
_SCRATCH: Final = np.array([0.0, 1.0, 2.0, 3.0, 4.0])


def _scratch_points(ax: Axes) -> Artist:
    return _points_artist(ax, np.zeros_like(_SCRATCH), _SCRATCH, _BLACK)


def _scratch_line(ax: Axes) -> Artist:
    return _line_artist(ax, np.zeros_like(_SCRATCH), _SCRATCH)


def _scratch_bar(ax: Axes) -> Artist:
    return _marginal_bar(ax, (-0.4, 0.0), 0.8, 4.0, _BLACK, "xside-bar-0")


_POINT_ARGS: Final = LayerArgs(
    name="point_args",
    layer="points",
    noun="points",
    make=_scratch_points,
    advice="Choose a visible style; the points are the data.",
)
_SMOOTH_ARGS: Final = LayerArgs(
    name="smooth_args",
    layer="line",
    noun="least-squares line",
    make=_scratch_line,
    pronoun="it",
    asks=", although smooth=True asks for it",
    advice=(
        "To leave the line and its band out, pass smooth=False instead; otherwise "
        "choose a visible style."
    ),
)
_MARGINAL_ARGS: Final = LayerArgs(
    name="marginal_args",
    layer="marginal bars",
    noun="marginal histograms",
    make=_scratch_bar,
    asks=", although marginal=True asks for them",
    advice=(
        "To leave them out, pass marginal=False instead; otherwise choose a "
        "visible style."
    ),
)


class _ScatterDrawing(base.BaseDrawing):
    """A scatterstats result: points, line and band, marginals, point labels."""

    def __init__(
        self,
        result: StatsResult,
        options: Options,
        colors: list[theme.Color],
        palette_warning: str | None,
        ax: Axes | None,
        layers: _Layers,
    ) -> None:
        super().__init__(result, options, colors, palette_warning, ax)
        self.options = options
        self.layers = layers
        self.texts = figure_texts(result)
        self.x = np.asarray(result.values[0], dtype=np.float64)
        self.y = np.asarray(result.values[1], dtype=np.float64)
        labels = result.point_labels
        self.labelled = [i for i, label in enumerate(labels) if label]
        self.xside: Axes | None = None
        self.yside: Axes | None = None
        if self.own and layers.x_bins is not None:
            self.xside = self.figure.add_axes((0.15, 0.85, 0.7, 0.1), sharex=self.ax)
            self.yside = self.figure.add_axes((0.87, 0.2, 0.1, 0.6), sharey=self.ax)
        self.points: PathCollection | None = None
        self.line: Line2D | None = None
        self.band: Any = None
        self.label_texts: list[Text] = []
        self.links: list[ConnectionPatch] = []
        self.subsample: Subsample | None = None
        self.drawn_points = self.x.size
        self.view = DrawnView(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        self.top_px = 0.0
        self.bottom_px = 0.0

    # -- drawing -----------------------------------------------------------------

    def run(self) -> StatsFigure:
        self.draw_points()
        self.draw_smooth()
        self.draw_marginals()
        self.draw_axes_texts()
        self.draw_label_texts()
        self.draw_text_blocks()
        self.layout()
        self.place_point_labels()
        if not self.own:
            self.finish_ink()
        self.finish_fonts()
        result = self.result
        if self.warnings:
            result = dataclasses.replace(
                result, warnings=result.warnings + tuple(self.warnings)
            )
        result = self.finish_result(result)
        axes = {"main": self.ax}
        if self.xside is not None and self.yside is not None:
            axes |= {"xside": self.xside, "yside": self.yside}
        return StatsFigure(
            figure=self.figure,
            axes=axes,
            result=result,
            own=self.own,
            roles=self.roles,
            inside=tuple(self.label_texts),
            subsample=self.subsample,
            view=self.view,
            colors=tuple(self.colors),
            data=self.drawn_data(),
            color_names=("the points",),
            point_labels=(
                PointLabels(
                    texts=tuple(self.label_texts),
                    total=len(self.label_texts),
                    limit=MAX_LABELS,
                )
                if self.label_texts
                else None
            ),
        )

    def draw_points(self) -> None:
        ax, x, y = self.ax, self.x, self.y
        n = x.size
        shown = np.arange(n)
        if n > MAX_POINTS:
            rng = np.random.Generator(np.random.PCG64(self.result.settings.seed))
            forced = np.array(
                [
                    np.argmin(x),
                    np.argmax(x),
                    np.argmin(y),
                    np.argmax(y),
                    *self.labelled,
                ],
                dtype=np.intp,
            )
            shown = with_extremes(n, SUBSAMPLE_SIZE, forced, rng)
            self.draw_density()
            self.drawn_points = int(shown.size)
            seed = self.result.settings.seed
            shows = (
                "the density layer and the marginal histograms are drawn from all pairs"
                if self.layers.x_bins is not None
                else "the density layer is drawn from all pairs"
            )
            message = (
                f"The points show a random subsample of {shown.size:,} of the "
                f"{n:,} pairs, always with the pairs that hold the smallest and "
                f"largest x and y and every labelled point (seed {seed}); {shows}."
            )
            self.subsample = Subsample(
                shown=int(shown.size),
                total=n,
                seed=seed,
                density="density layer",
                message=message,
            )
            self.warnings.append(
                f"The plot has {n:,} pairs, more than {MAX_POINTS:,}, so the "
                f"points show a random subsample of {shown.size:,} (seed {seed}), "
                "always with the pairs that hold the smallest and largest x and y "
                f"and every labelled point; {shows}."
            )
        self.points = _points_artist(ax, x[shown], y[shown], self.colors[0])
        base.apply_args(self.points, self.options.point_args, "point_args")

    def draw_density(self) -> None:
        """A 2D histogram of every pair, under the subsampled points: white where
        no pair lies, the points' colour where most do."""
        x, y = self.x, self.y
        counts, xe, ye = np.histogram2d(x, y, bins=DENSITY_BINS)
        shown = np.ma.masked_equal(counts.T, 0.0)
        r, g, b, _ = self.colors[0]
        ramp = np.linspace(0.12, 0.7, 256)
        cmap = ListedColormap(
            np.column_stack([np.full(256, r), np.full(256, g), np.full(256, b), ramp])
        )
        self.ax.pcolormesh(
            xe, ye, shown, cmap=cmap, shading="flat", zorder=1.0, gid="density",
            rasterized=False,
        )  # fmt: skip

    def draw_smooth(self) -> None:
        layers = self.layers
        if layers.no_line is not None:
            self.warnings.append(
                f"No least-squares line is drawn: {layers.no_line}. ggstatsplot "
                "draws none there either."
            )
            return
        band = layers.band
        if band is None:
            return
        gx = np.asarray(band.x, dtype=np.float64)
        line = _line_artist(self.ax, gx, np.asarray(band.fit, dtype=np.float64))
        base.apply_args(line, self.options.smooth_args, "smooth_args")
        color = line.get_color()
        self.band = self.ax.fill_between(
            gx,
            np.asarray(band.conf_low, dtype=np.float64),
            np.asarray(band.conf_high, dtype=np.float64),
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0.0,
            zorder=2.5,
            gid="smooth-band",
        )
        self.line = line

    def draw_marginals(self) -> None:
        xside, yside = self.xside, self.yside
        layers = self.layers
        if xside is None or yside is None:
            return
        assert layers.x_bins is not None
        assert layers.y_bins is not None
        color = self.colors[0]
        args = self.options.marginal_args
        edges, counts = layers.x_bins.edges, layers.x_bins.counts
        for i, count in enumerate(counts):
            bar = _marginal_bar(
                xside, (edges[i], 0.0), edges[i + 1] - edges[i], float(count), color,
                f"xside-bar-{i}",
            )  # fmt: skip
            base.apply_args(bar, args, "marginal_args")
        edges, counts = layers.y_bins.edges, layers.y_bins.counts
        for i, count in enumerate(counts):
            bar = _marginal_bar(
                yside, (0.0, edges[i]), float(count), edges[i + 1] - edges[i], color,
                f"yside-bar-{i}",
            )  # fmt: skip
            base.apply_args(bar, args, "marginal_args")
        top = max(max(layers.x_bins.counts), 1) * 1.05
        right = max(max(layers.y_bins.counts), 1) * 1.05
        xside.set_ylim(0.0, top)
        yside.set_xlim(0.0, right)
        for side in (xside, yside):
            side.set_axis_off()

    def draw_axes_texts(self) -> None:
        ax, result, options = self.ax, self.result, self.options
        ax.set_xlabel(
            drawable(result.x) if options.xlab is None else options.xlab,
            fontproperties=theme.font(theme.SIZE_AXIS_LABEL),
            parse_math=options.xlab is not None,
        )
        measured = result.y if result.y is not None else ""
        ax.set_ylabel(
            drawable(measured) if options.ylab is None else options.ylab,
            fontproperties=theme.font(theme.SIZE_AXIS_LABEL),
            parse_math=options.ylab is not None,
        )
        ax.tick_params(axis="both", labelsize=theme.SIZE_TICK)
        if self.own:  # a given Axes keeps its own grid
            ax.grid(True, axis="both")

    def draw_label_texts(self) -> None:
        """The point labels, at their points for now (placed after the layout)."""
        labels = self.result.point_labels
        changed = 0
        for i in self.labelled:
            shown = drawable(labels[i])
            changed += shown != labels[i]
            text = self.ax.text(
                float(self.x[i]),
                float(self.y[i]),
                shown,
                fontproperties=theme.font(LABEL_SIZE),
                parse_math=False,
                zorder=6.0,
                gid=f"point-label-{len(self.label_texts)}",
            )
            self.role(text, "point label", labels[i])
            self.label_texts.append(text)
        if changed:
            self.warnings.append(
                f"{changed:,} point label{'s' if changed > 1 else ''} of "
                f"{self.result.label_var!r} hold control characters, drawn as a "
                "space (a tab) or as � (others); the result keeps the labels "
                "as they are."
            )

    # -- layout ----------------------------------------------------------------

    def place_axes(
        self, x0: float, y0: float, width: float, height: float, fig_w: float,
        fig_h: float,
    ) -> None:  # fmt: skip
        xside, yside = self.xside, self.yside
        if xside is None or yside is None:
            super().place_axes(x0, y0, width, height, fig_w, fig_h)
            return
        side, gap = SIDE_IN * self.dpi, SIDE_GAP_IN * self.dpi
        main_w, main_h = width - side - gap, height - side - gap
        self.ax.set_position((x0 / fig_w, y0 / fig_h, main_w / fig_w, main_h / fig_h))
        xside.set_position(
            (x0 / fig_w, (y0 + main_h + gap) / fig_h, main_w / fig_w, side / fig_h)
        )
        yside.set_position(
            ((x0 + main_w + gap) / fig_w, y0 / fig_h, side / fig_w, main_h / fig_h)
        )

    def panels(self) -> list[Axes]:
        return [a for a in (self.ax, self.xside, self.yside) if a is not None]

    def frame(self) -> Bbox:
        renderer = self.renderer()
        return Bbox.union([a.get_window_extent(renderer) for a in self.panels()])

    def tight_boxes(self) -> list[Bbox]:
        renderer = self.renderer()
        boxes = [a.get_tightbbox(renderer) for a in self.panels()]
        return [b for b in boxes if b is not None]

    def decoration_texts(self) -> list[Text]:
        return list(self.label_texts)

    def layout(self) -> None:
        if self.own:
            self.set_limits(PLOT_HEIGHT_IN * self.dpi)
            extra = (
                (SIDE_IN + SIDE_GAP_IN) * self.dpi if self.xside is not None else 0.0
            )
            self.fit_figure(
                PLOT_WIDTH_IN * self.dpi + extra, PLOT_HEIGHT_IN * self.dpi + extra
            )
        else:
            self.top_px, self.bottom_px = self.place_blocks_in_axes()
            self.set_limits(float(self.ax.bbox.height))

    def set_limits(self, height: float) -> None:
        """The data, padded by PAD of their range (the band too, and a constant
        variable by 0.5 either side); with ``ax=`` the subtitle and caption get
        their pixels above and below."""
        x_lo, x_hi = float(self.x.min()), float(self.x.max())
        y_lo, y_hi = float(self.y.min()), float(self.y.max())
        band = self.layers.band if self.line is not None else None
        if band is not None:
            y_lo = min(y_lo, *band.conf_low)
            y_hi = max(y_hi, *band.conf_high)
        left, right = _padded(x_lo, x_hi)
        bottom, top = _padded(y_lo, y_hi)
        if not self.own:
            above = self.top_px
            below = self.bottom_px
            room = max(height - above - below, 0.2 * height)
            ppu = room / (top - bottom)
            bottom, top = bottom - below / ppu, top + above / ppu
            left, right = base.shared_limits(self.ax, "x", left, right)
            bottom, top = base.shared_limits(self.ax, "y", bottom, top)
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)
        self.view.xlim = (left, right)
        self.view.ylim = (bottom, top)

    def place_point_labels(self) -> None:
        """Each label at the place :func:`place_labels` finds, on the drawn
        layout; a label moved away from its point gets a line to it."""
        if not self.label_texts:
            return
        self.figure.canvas.draw()
        ax = self.ax
        data = ax.transData
        anchors = np.asarray(
            data.transform(
                np.column_stack([self.x[self.labelled], self.y[self.labelled]])
            ),
            dtype=np.float64,
        )
        sizes = np.array(
            [[b.width, b.height] for b in (self.extent(t) for t in self.label_texts)],
            dtype=np.float64,
        )
        offsets = np.asarray(
            cast("Any", self.points).get_offsets() if self.points is not None else [],
            dtype=np.float64,
        ).reshape(-1, 2)
        points = np.asarray(data.transform(offsets), dtype=np.float64).reshape(-1, 2)
        marker = px(float(np.sqrt(POINT_SIZE)) / 2.0, self.dpi)
        frame = ax.get_window_extent(self.renderer())
        obstacles = [
            tuple(float(v) for v in self.extent(t).extents)
            for t in [*self.header, *self.captions]
            if not self.own
        ]
        places = place_labels(
            anchors=anchors,
            sizes=sizes,
            points=points,
            marker=marker,
            gap=px(LABEL_GAP_PT, self.dpi),
            frame=(frame.x0, frame.y0, frame.x1, frame.y1),
            obstacles=cast("list[tuple[float, float, float, float]]", obstacles),
            limit=MAX_LABELS,
        )
        to_pt = 72.0 / self.dpi
        for i, text, place in zip(self.labelled, self.label_texts, places, strict=True):
            moved = offset_copy(
                data, fig=self.figure, x=place.dx * to_pt, y=place.dy * to_pt,
                units="points",
            )  # fmt: skip
            text.set_transform(moved)
            text.set_horizontalalignment(place.ha)
            text.set_verticalalignment(place.va)
            if place.ring == 0:
                continue
            xy = (float(self.x[i]), float(self.y[i]))
            link = ConnectionPatch(
                xyA=xy,
                coordsA=data,
                xyB=xy,
                coordsB=moved,
                arrowstyle="-",
                shrinkA=float(np.sqrt(POINT_SIZE)) / 2.0 + 0.5,
                shrinkB=0.0,
                color=theme.SOFT_INK,
                linewidth=LINK_WIDTH,
                zorder=5.0,
                gid=f"point-label-link-{len(self.links)}",
            )
            ax.add_artist(link)
            self.links.append(link)

    # -- finishing -------------------------------------------------------------

    def ink_inside(self, ink: str) -> None:
        super().ink_inside(ink)
        for text in self.label_texts:
            text.set_color(ink)
        for link in self.links:
            link.set_color(ink)
        if self.line is not None and not self.options.smooth_args:
            self.line.set_color(ink)
            if self.band is not None:
                self.band.set_color(ink)
                self.band.set_alpha(BAND_ALPHA)

    def drawn_data(self) -> tuple[DrawnData, ...]:
        data = [
            DrawnData("main", "values", self.x[:, None], self.y[:, None], spread="xy")
        ]
        band = self.layers.band
        if self.line is not None and band is not None:
            gx = np.asarray(band.x, dtype=np.float64)[None, :]
            data.append(
                DrawnData(
                    "main", "the least-squares line", gx,
                    np.asarray(band.fit, dtype=np.float64)[None, :],
                )
            )  # fmt: skip
            data.append(
                DrawnData(
                    "main",
                    "the confidence band",
                    np.concatenate([gx, gx], axis=1),
                    np.asarray([*band.conf_low, *band.conf_high], dtype=np.float64)[
                        None, :
                    ],
                )
            )
        for panel, bins in (
            ("xside", self.layers.x_bins),
            ("yside", self.layers.y_bins),
        ):
            if bins is None or panel not in ("xside", "yside") or self.xside is None:
                continue
            edges = np.asarray(bins.edges, dtype=np.float64)
            counts = np.asarray(bins.counts, dtype=np.float64)
            along = np.column_stack([edges[:-1], edges[1:], edges[:-1], edges[1:]])
            zeros = np.zeros_like(counts)
            height = np.column_stack([zeros, zeros, counts, counts])
            if panel == "xside":
                data.append(DrawnData(panel, "bars", along, height, spread="y"))
            else:
                data.append(DrawnData(panel, "bars", height, along, spread="x"))
        return tuple(data)

    def finish_result(self, result: StatsResult) -> StatsResult:
        """The result told what the figure draws (its alt text is the result's
        own until the texts of scatterstats are final)."""
        drawn = DrawnLayers(
            points=self.drawn_points,
            smooth=self.line is not None,
            marginal=self.xside is not None,
            labels=len(self.label_texts),
        )
        alt = figure_alt_text(result, drawn=drawn)
        if alt == result.text.alt_text:
            return result
        return dataclasses.replace(
            result, text=dataclasses.replace(result.text, alt_text=alt)
        )


def _padded(low: float, high: float) -> tuple[float, float]:
    """``low``, ``high`` widened by PAD of their range (by 0.5 where they are
    equal, as matplotlib widens a single value)."""
    span = high - low
    if span == 0.0:
        pad = 0.5 if low == 0.0 else 0.05 * abs(low)
        return low - pad, high + pad
    return low - PAD * span, high + PAD * span
