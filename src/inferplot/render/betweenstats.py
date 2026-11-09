# matplotlib's and NumPy's stubs type many ``**kwargs`` and overloads as Unknown,
# which strict mode reports on every call; the arguments we pass are all typed.
# pyright: reportUnknownMemberType=false
"""Draw a betweenstats result: violins, boxes, points, centrality labels, brackets.

Every number and text comes from the :class:`~inferplot.result.StatsResult` and
:func:`inferplot.text.figure_texts`; the renderer computes no statistics. It does
compute drawing aids: kernel density outlines for the violins, quartiles and
whiskers for the boxes, and seeded jitter.

When inferplot creates the figure it sizes it from the rendered sizes of its texts,
so nothing is clipped and no two texts overlap (a long subtitle widens the figure).
With ``ax=`` the subtitle and caption go inside that Axes and ``fig.check()``
reports anything that does not fit.

Every argument is checked before anything is drawn, so a call that raises leaves a
given Axes as it was: the artist dicts are applied to throwaway artists of the
classes that receive them, and the user's texts are laid out once as mathtext.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox, IdentityTransform, offset_copy
from numpy.typing import NDArray

from inferplot import _validate as v
from inferplot.render import theme
from inferplot.render.brackets import Bracket, VerticalLayout, stack, vertical_layout
from inferplot.render.figure import (
    StatsFigure,
    Subsample,
    describe_text,
    root_figure,
)
from inferplot.result import StatsResult
from inferplot.text import FigureTexts, figure_texts

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.backend_bases import RendererBase

__all__ = ["MAX_POINTS", "SUBSAMPLE_SIZE", "betweenstats"]

FloatArray = NDArray[np.float64]

# A panel with more values than this draws a subsample of SUBSAMPLE_SIZE points.
MAX_POINTS: Final = 5_000
SUBSAMPLE_SIZE: Final = 2_000

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


@v.checked
def betweenstats(
    result: StatsResult,
    *,
    centrality_plotting: bool = True,
    layers: Sequence[str] = ("violin", "box", "points"),
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    palette: str | Sequence[object] | None = None,
    point_args: Mapping[str, object] | None = None,
    boxplot_args: Mapping[str, object] | None = None,
    violin_args: Mapping[str, object] | None = None,
    ax: Axes | None = None,
) -> StatsFigure:
    """Draw a result of ``ip.analyze.betweenstats``.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        centrality_plotting: draw each level's centrality point and label.
        layers: any of "violin", "box" and "points", drawn back to front.
        xlab, ylab: axis labels; default to the column names.
        title: a title above the subtitle.
        subtitle: your own subtitle; allowed only for a result computed with
            ``results_subtitle=False`` (otherwise the results subtitle is drawn).
        caption: your own caption, drawn below the generated caption lines.
        palette: a matplotlib colormap name or a list of at least one color per
            level; default Okabe-Ito (``tab20`` for more than 8 levels).
        point_args, boxplot_args, violin_args: matplotlib properties applied with
            ``Artist.set`` to each level's points, box and violin. Each needs its
            layer ("points", "box", "violin") in ``layers``; unknown properties and
            values matplotlib rejects raise before anything is drawn.
        ax: draw into this Axes instead of a new figure; the subtitle and caption
            then go inside it.

    ``title``, ``subtitle``, ``caption``, ``xlab`` and ``ylab`` are matplotlib
    text, which follows matplotlib's convention: text between two dollar signs is
    a mathtext expression (``r"Weight ($\\mu$g)"``). Write ``\\$`` for a literal
    dollar sign (``r"Revenue in \\$"``). An expression matplotlib cannot read
    raises ``ValueError`` at the call. Column names drawn as default labels are
    never read as mathtext.

    Returns:
        A :class:`~inferplot.render.figure.StatsFigure`; its ``result`` adds any
        rendering warnings to ``result.warnings``.
    """
    if not isinstance(result, StatsResult):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            "render.betweenstats() draws the result of ip.analyze.betweenstats(); "
            f"got {type(result).__name__}."
        )
    if result.analysis != "betweenstats":
        raise TypeError(
            "render.betweenstats() draws results of ip.analyze.betweenstats(); this "
            f"result comes from ip.analyze.{result.analysis}(). Draw it with "
            f"ip.render.{result.analysis}(result)."
        )
    k = len(result.levels)
    if k >= 1 and not result.values:
        raise ValueError(
            "This result holds no observations to draw: the copy of a result that "
            "save() embeds in a figure file leaves them out. Recompute it with "
            "ip.analyze.betweenstats(data, x, y, ...) and draw that."
        )
    if not (len(result.values) == len(result.descriptives) == k and k >= 1):
        raise ValueError(
            "result is inconsistent: it needs one entry of values and descriptives "
            f"per level; got {k} levels, {len(result.values)} value groups and "
            f"{len(result.descriptives)} descriptives."
        )
    options = _Options(
        centrality=v.check_bool(centrality_plotting, "centrality_plotting"),
        layers=v.check_layers(layers),
        xlab=v.check_optional_text(xlab, "xlab"),
        ylab=v.check_optional_text(ylab, "ylab"),
        title=v.check_optional_text(title, "title"),
        subtitle=v.check_optional_text(subtitle, "subtitle"),
        caption=v.check_optional_text(caption, "caption"),
        point_args=v.check_artist_args(point_args, "point_args") or {},
        boxplot_args=v.check_artist_args(boxplot_args, "boxplot_args") or {},
        violin_args=v.check_artist_args(violin_args, "violin_args") or {},
    )
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
    colors, palette_warning = theme.resolve_palette(palette, k)
    _check_artist_args(options)
    with theme.context():
        _check_mathtext(options)
        return _Drawing(result, options, colors, palette_warning, ax).run()


@dataclasses.dataclass(frozen=True, slots=True)
class _Options:
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


def _apply(artist: Artist, args: Mapping[str, object], name: str) -> None:
    """``artist.set(**args)``, with a message naming the argument on failure."""
    if not args:
        return
    try:
        artist.set(**args)
    except (AttributeError, TypeError) as error:
        raise TypeError(
            f"{name} has a property matplotlib's {type(artist).__name__} does not "
            f"accept: {error}"
        ) from error
    except ValueError as error:
        raise ValueError(f"{name} has a value matplotlib rejects: {error}") from error


# The layer each artist dict styles, and a throwaway artist of the class that
# receives it there (ax.scatter's PathCollection, the box PathPatch, the violin
# Polygon).
_ARTIST_LAYERS: Final = (
    ("point_args", "points"),
    ("boxplot_args", "box"),
    ("violin_args", "violin"),
)


def _throwaway(layer: str) -> Artist:
    if layer == "points":
        return PathCollection(
            (Path.unit_circle(),),
            sizes=[14.0],
            offsets=np.zeros((1, 2)),
            offset_transform=IdentityTransform(),
        )
    if layer == "box":
        return PathPatch(_box_path(np.array([0.0, 1.0]), 0.0))
    return Polygon(np.zeros((3, 2)), closed=True)


def _check_artist_args(options: _Options) -> None:
    """Each non-empty artist dict styles a layer that is drawn and is accepted by
    the class of artist it is applied to; checked before anything is drawn."""
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
        _apply(_throwaway(layer), args, name)


_USER_TEXTS: Final = ("title", "subtitle", "caption", "xlab", "ylab")


def _check_mathtext(options: _Options) -> None:
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
    sd = float(np.std(x, ddof=1))
    q75, q25 = (float(q) for q in np.percentile(x, [75.0, 25.0]))
    lo = min(sd, (q75 - q25) / 1.34)
    if lo <= 0.0:
        lo = sd if sd > 0.0 else 1.0
    return 0.9 * lo * x.size ** (-0.2)


def _density_outline(x: FloatArray) -> tuple[FloatArray, FloatArray] | None:
    """Gaussian kernel density on [min, max], scaled to the violin's half width.

    None when the values do not span a range (fewer than two different values).
    Large samples are binned first so drawing stays fast.
    """
    if x.size < 2 or float(x.min()) == float(x.max()):
        return None
    bw = _bandwidth(x)
    grid = np.linspace(float(x.min()), float(x.max()), KDE_GRID)
    if x.size > KDE_BINS:
        counts, edges = np.histogram(x, bins=KDE_BINS)
        centers = (edges[:-1] + edges[1:]) / 2
        z = (grid[:, None] - centers[None, :]) / bw
        density = (np.exp(-0.5 * z * z) * counts[None, :]).sum(axis=1)
    else:
        z = (grid[:, None] - x[None, :]) / bw
        density = np.exp(-0.5 * z * z).sum(axis=1)
    peak = float(density.max())
    half = np.asarray(VIOLIN_HALF_WIDTH * density / peak, dtype=np.float64)
    return np.asarray(grid, dtype=np.float64), half


def _box_path(x: FloatArray, center: float) -> Path:
    """Box (quartiles, R's type 7), median line and Tukey whiskers as one path."""
    q1, med, q3 = (float(q) for q in np.percentile(x, [25.0, 50.0, 75.0]))
    iqr = q3 - q1
    low = float(x[x >= q1 - 1.5 * iqr].min())
    high = float(x[x <= q3 + 1.5 * iqr].max())
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


def _split_subsample(
    values: Sequence[FloatArray], rng: np.random.Generator
) -> list[FloatArray]:
    """A uniform random subsample of SUBSAMPLE_SIZE values, split by level."""
    sizes = [x.size for x in values]
    total = sum(sizes)
    keep = np.sort(rng.choice(total, size=SUBSAMPLE_SIZE, replace=False))
    bounds = np.cumsum([0, *sizes])
    out: list[FloatArray] = []
    for i, x in enumerate(values):
        mask = (keep >= bounds[i]) & (keep < bounds[i + 1])
        out.append(x[keep[mask] - bounds[i]])
    return out


# --------------------------------------------------------------------------- #
# the drawing
# --------------------------------------------------------------------------- #


def _px(points: float, dpi: float) -> float:
    return points * dpi / 72.0


class _Drawing:
    """One rendering of a result; holds the artists while laying them out."""

    def __init__(
        self,
        result: StatsResult,
        options: _Options,
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
        self.bracket_lines: list[Line2D] = []
        self.bracket_labels: list[Text] = []
        self.bracket_pairs: list[tuple[int, int]] = []
        self.bracket_names: list[str] = []
        self.header: list[Text] = []  # title, subtitle (own figure)
        self.captions: list[Text] = []

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
            pad = 2 * _px(LABEL_BOX_PAD * float(text.get_fontsize()), self.dpi)
            box = Bbox.from_bounds(box.x0 - pad / 2, box.y0 - pad / 2,
                                   box.width + pad, box.height + pad)  # fmt: skip
        return box

    def role(self, text: Text, name: str, plain: str | None = None) -> Text:
        """Record what ``text`` is (and its plain wording) for check() messages."""
        self.roles[id(text)] = describe_text(
            name, text.get_text() if plain is None else plain
        )
        return text

    # -- main ------------------------------------------------------------------

    def run(self) -> StatsFigure:
        self.draw_data()
        self.draw_ticks_and_labels()
        self.draw_text_blocks()
        if self.own:
            self.layout_own()
        else:
            self.layout_in_axes()
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
        )

    def draw_data(self) -> None:
        ax, result, options = self.ax, self.result, self.options
        layers = options.layers
        seed = result.settings.seed
        rng = np.random.Generator(np.random.PCG64(seed))
        shown = self.values
        total = sum(x.size for x in self.values)
        if "points" in layers and total > MAX_POINTS:
            shown = _split_subsample(self.values, rng)
            density = "violins" if "violin" in layers else "density outlines"
            self.subsample = Subsample(
                shown=SUBSAMPLE_SIZE, total=total, seed=seed, density=density
            )
            self.warnings.append(
                f"The plot has {total:,} values, more than {MAX_POINTS:,}, so the "
                f"points show a random subsample of {SUBSAMPLE_SIZE:,} (seed {seed}); "
                f"the {density} and boxes show all values."
            )
        outline = "violin" in layers or self.subsample is not None
        for i, (level, x) in enumerate(zip(result.levels, self.values, strict=True)):
            color = self.colors[i]
            if outline:
                self.draw_density(i, level, x, color)
            if "box" in layers and x.size > 0:
                box = PathPatch(
                    _box_path(x, float(i)),
                    facecolor=(1.0, 1.0, 1.0, 0.85),
                    edgecolor=theme.SOFT_INK,
                    linewidth=0.9,
                    zorder=2.0,
                    gid=f"box-{i}",
                )
                ax.add_patch(box)
                _apply(box, options.boxplot_args, "boxplot_args")
            if "points" in layers and shown[i].size > 0:
                jitter = rng.uniform(
                    -JITTER_HALF_WIDTH, JITTER_HALF_WIDTH, size=shown[i].size
                )
                points = ax.scatter(
                    i + jitter,
                    shown[i],
                    s=14.0,
                    color=color,
                    alpha=0.6,
                    linewidths=0.0,
                    zorder=3.0,
                    gid=f"points-{i}",
                )
                _apply(points, options.point_args, "point_args")
        if options.centrality:
            self.draw_centrality()
        self.make_bracket_labels()

    def draw_density(
        self, i: int, level: str, x: FloatArray, color: theme.Color
    ) -> None:
        drawn = _density_outline(x)
        what = "violin" if "violin" in self.options.layers else "density outline"
        if drawn is None:
            self.warnings.append(
                f"The {what} of level {level!r} is not drawn: it needs at least two "
                "different values."
            )
            return
        grid, half = drawn
        xs = np.concatenate([i - half, (i + half)[::-1]])
        ys = np.concatenate([grid, grid[::-1]])
        if what == "violin":
            patch = Polygon(
                np.column_stack([xs, ys]),
                closed=True,
                facecolor=(*color[:3], 0.25),
                edgecolor=color,
                linewidth=0.8,
                zorder=1.0,
                gid=f"violin-{i}",
            )
            self.ax.add_patch(patch)
            _apply(patch, self.options.violin_args, "violin_args")
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
            ax.add_line(
                Line2D(
                    [i, i + LABEL_OFFSET],
                    [d.value, d.value],
                    color=theme.CENTRALITY_COLOR,
                    linewidth=0.7,
                    zorder=4.0,
                    gid=f"centrality-link-{i}",
                )
            )
            ax.scatter(
                [i],
                [d.value],
                s=40.0,
                color=theme.CENTRALITY_COLOR,
                zorder=5.0,
                gid=f"centrality-{i}",
            )
            text = ax.text(
                i + LABEL_OFFSET,
                d.value,
                label,
                ha="left",
                va="center",
                fontproperties=theme.font(theme.SIZE_LABEL),
                bbox={
                    "boxstyle": f"round,pad={LABEL_BOX_PAD}",
                    "facecolor": "white",
                    "edgecolor": theme.CENTRALITY_COLOR,
                    "linewidth": 0.7,
                },
                zorder=6.0,
                gid=f"centrality-label-{i}",
            )
            self.centrality_labels.append(
                self.role(
                    text,
                    f"centrality label of {d.level!r}",
                    self.result.text.centrality_labels[i],
                )
            )
            self.inside.append(text)

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
            labels=list(self.texts.tick_labels),
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

    def x_needs(self) -> tuple[float, float]:
        """Widest centrality label and widest tick label, in pixels."""
        label_w = max(
            (self.extent(t).width for t in self.centrality_labels), default=0.0
        )
        probe = [
            self.figure.text(
                0.0,
                0.0,
                s,
                fontproperties=theme.font(theme.SIZE_TICK),
                parse_math=False,
            )
            for s in self.texts.tick_labels
        ]
        tick_w = max(self.extent(t).width for t in probe)
        for t in probe:
            t.remove()
        return label_w, tick_w

    def x_layout(self, spacing: float | None) -> tuple[float, float, float]:
        """x-limits and pixels per level; ``spacing`` None fits the Axes width."""
        k = len(self.result.levels)
        gap = _px(GAP_PT, self.dpi)
        label_w, tick_w = self.x_needs()
        reach = LABEL_OFFSET if self.centrality_labels else 0.0
        if spacing is None:
            width = float(self.ax.bbox.width)
            share = (label_w + gap) / width if self.centrality_labels else 0.0
            span = float(k)
            if share < 1.0:
                span = max(span, (k - 0.5 + reach) / (1.0 - share))
            spacing = width / span
        else:
            spacing = max(spacing, label_w + 2 * gap, tick_w + 2 * gap)
        left, right = -0.5, k - 0.5
        if self.centrality_labels:
            right = max(right, (k - 1) + reach + (label_w + gap) / spacing)
        return left, right, spacing

    def place_brackets(
        self, left: float, right: float, spacing: float
    ) -> tuple[float, float, int, list[int]]:
        gap = max(0.08, _px(2 * GAP_PT, self.dpi) / spacing)
        specs = [
            Bracket(
                i=i,
                j=j,
                width=(self.extent(t).width + _px(2 * LABEL_PAD_PT, self.dpi))
                / spacing,
            )
            for (i, j), t in zip(self.bracket_pairs, self.bracket_labels, strict=True)
        ]
        levels = list(stack(specs, gap=gap))
        for spec in specs:
            lo, hi = spec.extent()
            left = min(left, lo - 0.05)
            right = max(right, hi + 0.05)
        return left, right, (max(levels) + 1 if levels else 0), levels

    def y_layout(
        self,
        n_levels: int,
        *,
        top_px: float = 0.0,
        bottom_px: float = 0.0,
        height_px: float | None = None,
    ) -> VerticalLayout:
        data = [x for x in self.values if x.size > 0]
        data_min = min(float(x.min()) for x in data)
        data_max = max(float(x.max()) for x in data)
        centers = (
            [d.value for d in self.result.descriptives]
            if self.centrality_labels
            else []
        )
        center_half = max(
            (self.extent(t).height / 2 for t in self.centrality_labels), default=0.0
        )
        label_h = max((self.extent(t).height for t in self.bracket_labels), default=0.0)
        fixed = height_px is not None
        return vertical_layout(
            data_min=data_min,
            data_max=data_max,
            centers=centers,
            n_levels=n_levels,
            center_half_px=center_half,
            label_px=label_h,
            pad_px=_px(LABEL_PAD_PT, self.dpi),
            tip_px=_px(BRACKET_TIP_PT, self.dpi),
            line_px=_px(BRACKET_LINEWIDTH_PT, self.dpi),
            top_px=top_px,
            bottom_px=bottom_px,
            min_height_px=None if fixed else MIN_AXES_HEIGHT_IN * self.dpi,
            data_height_px=MIN_DATA_HEIGHT_IN * self.dpi,
            height_px=height_px,
        )

    def apply_limits(
        self, left: float, right: float, layout: VerticalLayout, levels: list[int]
    ) -> None:
        ax = self.ax
        ax.set_xlim(left, right)
        ax.set_ylim(layout.bottom, layout.top)
        for m, ((i, j), text, level) in enumerate(
            zip(self.bracket_pairs, self.bracket_labels, levels, strict=True)
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
            text.set_position(((i + j) / 2, y + layout.label_pad))

    def layout_own(self) -> None:
        dpi = self.dpi
        left, right, spacing = self.x_layout(MIN_LEVEL_SPACING_IN * dpi)
        left, right, n_levels, levels = self.place_brackets(left, right, spacing)
        layout = self.y_layout(n_levels)
        self.apply_limits(left, right, layout, levels)
        axes_w = spacing * (right - left)
        axes_h = layout.height_px
        margin, gap = _px(MARGIN_PT, dpi), _px(GAP_PT, dpi)
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
        pad, gap = _px(LABEL_PAD_PT, dpi), _px(GAP_PT, dpi)
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
        self.apply_limits(left, right, layout, levels)

    def finish_fonts(self) -> None:
        """DejaVu Sans, also for mathtext, on every text of the Axes and figure."""
        # The own figure was drawn by the layout, so its tick labels exist.
        owner = self.figure if self.own else self.ax
        for item in owner.findobj(Text):
            item.set_fontfamily(theme.FONT_FAMILY)
            item.set_math_fontfamily(theme.MATH_FONT_FAMILY)
