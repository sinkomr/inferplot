# matplotlib's and NumPy's stubs type many ``**kwargs`` and overloads as Unknown,
# which strict mode reports on every call; the arguments we pass are all typed.
# pyright: reportUnknownMemberType=false
"""Draw a withinstats result: betweenstats' layers plus the subject and centrality
paths of ggstatsplot's ``ggwithinstats()``.

The drawing is :class:`inferplot.render.comparison.Drawing` with four changes:

* points: each subject keeps one seeded horizontal offset at every level, so its
  path is the unjittered path moved sideways and ends exactly on its drawn points;
  more than MAX_POINTS values draw every value of a seeded random subsample of
  whole subjects (SUBSAMPLE_SIZE // k of them), so every drawn path is complete,
  always with the subjects that hold each condition's smallest and largest value;
* subject paths (``point_path``): one line per drawn subject, in one collection,
  above the boxes and below the points; only with two conditions and the points
  layer, as ggwithinstats draws them only for two conditions;
* centrality path (``centrality_path``): through the centrality points in level
  order, above the points and below the centrality points; only with
  ``centrality_plotting``, as in ggwithinstats; it breaks at a level without a
  centrality value (D55);
* centrality labels stay off the paths: with two conditions the first level's
  label goes to the left of its point, and with three or more each label of a
  middle level sits above or below the point's height, on the side the path to
  the next level leaves free.

An argument that then has no effect (``point_path=False`` with three or more
conditions or without points, ``centrality_path=False`` with
``centrality_plotting=False``) adds a ``warnings`` entry (D39); the defaults add
nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from inferplot import _validate as v
from inferplot.render import comparison, theme
from inferplot.render.comparison import (
    JITTER_HALF_WIDTH,
    MAX_POINTS,
    SUBSAMPLE_SIZE,
    FloatArray,
    LabelPlace,
)
from inferplot.render.figure import DrawnPaths, Subsample, shows_all
from inferplot.result import StatsResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from inferplot.render.figure import StatsFigure

__all__ = ["withinstats"]

# Subject paths: thin, neutral and translucent, so that overlapping paths darken
# where many subjects run, and the points (drawn above them) stay visible.
SUBJECT_PATH_COLOR: Final = theme.SOFT_INK
SUBJECT_PATH_ALPHA: Final = 0.35
SUBJECT_PATH_WIDTH: Final = 0.6
# The centrality path: ggwithinstats' red path, in the centrality color.
CENTRALITY_PATH_ALPHA: Final = 0.6
CENTRALITY_PATH_WIDTH: Final = 1.5


@v.checked
def withinstats(
    result: StatsResult,
    *,
    point_path: bool = True,
    centrality_path: bool = True,
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
    """Draw a result of ``ip.analyze.withinstats``.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        point_path: join each subject's points with a line. Drawn only with two
            conditions and the "points" layer, as ggwithinstats draws it only for
            two conditions; ``fig.check()`` reports paths so many that they hide
            the boxes and violins (then pass ``point_path=False``).
        centrality_path: join the levels' centrality points with a line; drawn
            only with ``centrality_plotting=True``.
        centrality_plotting: draw each level's centrality point and label.
        layers, xlab, ylab, title, subtitle, caption, palette, point_args,
            boxplot_args, violin_args, ax: as for ``ip.render.betweenstats``.

    Each subject's points share one seeded horizontal offset, so its path ends on
    them. With more than 5,000 values the points and paths show every value of a
    seeded random subsample of whole subjects (2,000 values in all, rounded down
    to whole subjects); the violins and boxes show all values. ``point_path=False``
    where no subject paths are drawn, and ``centrality_path=False`` with
    ``centrality_plotting=False``, have no effect and add a ``warnings`` entry.

    Returns:
        A :class:`~inferplot.render.figure.StatsFigure`; its ``result`` adds any
        rendering warnings to ``result.warnings``.
    """
    comparison.checked_result(
        result,
        "withinstats",
        "ip.analyze.withinstats(data, x, y, subject_id=..., ...)",
    )
    n = len(result.subjects)
    if n < 1 or any(len(level) != n for level in result.values):
        raise ValueError(
            "result is inconsistent: a within-subjects result needs one value per "
            "subject at every level; got "
            f"{n} subjects and {[len(level) for level in result.values]} values "
            "per level."
        )
    options = comparison.Options(
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
        point_path=v.check_bool(point_path, "point_path"),
        centrality_path=v.check_bool(centrality_path, "centrality_path"),
    )
    comparison.check_options(result, options, ax)
    return comparison.draw(_WithinDrawing, result, options, palette, ax)


def _no_effect(result: StatsResult, options: comparison.Options) -> list[str]:
    """Warnings for path arguments that have no effect on this figure (D39).

    Only a value other than the default can be such an argument; the defaults
    add nothing.
    """
    out: list[str] = []
    k = len(result.levels)
    if not options.point_path:
        if k >= 3:
            out.append(
                f"point_path=False has no effect: subject paths are drawn only with "
                f"two conditions (as in ggwithinstats), and x={result.x!r} has {k}."
            )
        elif "points" not in options.layers:
            out.append(
                "point_path=False has no effect: subject paths join the points, and "
                f"layers={options.layers!r} draws no points."
            )
    if not options.centrality_path and not options.centrality:
        out.append(
            "centrality_path=False has no effect: the centrality path joins the "
            "centrality points, which centrality_plotting=False leaves out."
        )
    return out


class _WithinDrawing(comparison.Drawing):
    """A withinstats result: points per subject, subject and centrality paths."""

    def __init__(
        self,
        result: StatsResult,
        options: comparison.Options,
        colors: list[theme.Color],
        palette_warning: str | None,
        ax: Axes | None,
    ) -> None:
        super().__init__(result, options, colors, palette_warning, ax)
        self.warnings.extend(_no_effect(result, options))
        k = len(result.levels)
        self.with_subject_paths = (
            options.point_path and k == 2 and "points" in options.layers
        )
        self.with_centrality_path = options.centrality and options.centrality_path
        self.subject_paths: LineCollection | None = None
        self.centrality_line: Line2D | None = None

    def place_points(self, rng: np.random.Generator) -> list[tuple[FloatArray, ...]]:
        """Every subject's points share one offset; large panels draw whole
        subjects."""
        layers = self.options.layers
        k = len(self.values)
        if "points" not in layers:
            empty = np.empty(0, dtype=np.float64)
            return [(empty, empty)] * k
        n = self.values[0].size
        total = n * k
        keep = np.arange(n)
        if total > MAX_POINTS:
            seed = self.result.settings.seed
            # The subjects holding each condition's smallest and largest value
            # are always drawn, the others at random.
            extremes = np.array(
                [int(np.argmin(level)) for level in self.values]
                + [int(np.argmax(level)) for level in self.values],
                dtype=np.intp,
            )
            m = max(SUBSAMPLE_SIZE // k, int(np.unique(extremes).size))
            keep = comparison.with_extremes(n, m, extremes, rng)
            density = "violins" if "violin" in layers else "density outlines"
            self.subsample = Subsample(
                shown=m * k, total=total, seed=seed, density=density, subjects=(m, n)
            )
        offsets = rng.uniform(-JITTER_HALF_WIDTH, JITTER_HALF_WIDTH, size=keep.size)
        return [(j + offsets, self.values[j][keep]) for j in range(k)]

    def subsample_warning(self, sub: Subsample) -> str:
        what = "points and subject paths" if self.with_subject_paths else "points"
        m, n = sub.subjects if sub.subjects is not None else (0, 0)
        return (
            f"The plot has {sub.total:,} values, more than {MAX_POINTS:,}, so the "
            f"{what} show every value of a random subsample of {m:,} of the "
            f"{n:,} subjects ({sub.shown:,} values, seed {sub.seed}), always with "
            "the subjects that hold each condition's smallest and largest value; "
            f"{shows_all(sub, 'show')}."
        )

    def draw_paths(self, points: list[tuple[FloatArray, ...]]) -> None:
        if not self.with_subject_paths:
            return
        # (subjects, levels, 2): each subject's drawn points, in level order.
        segments = np.stack([np.column_stack(xy) for xy in points], axis=1)
        paths = LineCollection(
            list(segments),
            colors=[SUBJECT_PATH_COLOR],
            linewidths=[SUBJECT_PATH_WIDTH],
            alpha=SUBJECT_PATH_ALPHA,
            zorder=2.5,
            gid="subject-paths",
        )
        self.ax.add_collection(paths, autolim=False)
        self.subject_paths = paths

    def label_place(self, i: int) -> LabelPlace:
        k = len(self.result.levels)
        if not (self.with_subject_paths or self.with_centrality_path):
            return LabelPlace()
        if k == 2:
            # The space between the two conditions is the paths'.
            return LabelPlace(side="left" if i == 0 else "right")
        if i == k - 1:
            return LabelPlace()
        here = self.result.descriptives[i].value
        after = self.result.descriptives[i + 1].value
        if here is None or after is None:
            return LabelPlace()  # no path leaves this point to the right
        # Right of the point, on the side the path to the next level leaves free.
        return LabelPlace(rise="below" if after > here else "above")

    def draw_centrality(self) -> None:
        super().draw_centrality()
        if not self.with_centrality_path:
            return
        centers = [
            np.nan if d.value is None else d.value for d in self.result.descriptives
        ]
        line = Line2D(
            list(range(len(centers))),
            centers,
            color=theme.CENTRALITY_COLOR,
            alpha=CENTRALITY_PATH_ALPHA,
            linewidth=CENTRALITY_PATH_WIDTH,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3.5,
            gid="centrality-path",
        )
        self.ax.add_line(line)
        self.centrality_line = line

    def drawn_paths(self) -> DrawnPaths:
        return DrawnPaths(
            subjects=self.subject_paths,
            centrality=self.centrality_line,
            levels=self.result.levels,
        )
