# NumPy's stubs leave overloads of sort and searchsorted partially unknown, which
# strict mode reports on every call; the arrays passed are all typed.
# pyright: reportUnknownMemberType=false
"""What the histostats and scatterstats figures draw, as numbers: the histogram's
bins, the OLS line with its pointwise confidence band, and the marginal
histograms. The renderers draw these; nothing here plots.

The R references are the layers ggstatsplot 1.1.4 builds (ggplot2 4.0.3): the
fixtures record each built layer, and every function here reproduces them on all
76 gghistostats and 64 ggscatterstats cases (tests/stats/test_plot_geometry.py).
The binning rules are ggplot2's ``stat_bin`` as its documentation describes it
(bins of ``binwidth`` laid on a grid with a bin boundary at ``boundary``, default
half a bin width, or ``bins`` bins centred on the smallest value, intervals closed
on the right) and as the built layers show it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np

from inferplot.stats.correlation import cross_moments
from inferplot.stats.descriptive import FloatArray, SampleLike, as_sample, exact_mean
from inferplot.stats.noncentral import check_conf_level
from inferplot.stats.smm import smm_quantile

__all__ = [
    "MARGINAL_BINS",
    "SMOOTH_POINTS",
    "HistogramBins",
    "OlsBand",
    "histogram_bins",
    "histogram_binwidth",
    "marginal_bins",
    "ols_band",
]

SMOOTH_POINTS: Final = 80
"""geom_smooth's ``n``: the line and band are evaluated at 80 points."""
MARGINAL_BINS: Final = 30
"""ggside's marginal histograms (``geom_xsidehistogram``) keep stat_bin's 30 bins."""
# stat_bin refuses more bins than this (ggplot2: "The number of histogram bins must
# be less than 1,000,000").
_MAX_BINS: Final = 1_000_000
# A constant variable gets one bin this wide (ggplot2's width for zero-range data,
# the width default_expansion() gives it).
_ZERO_RANGE_WIDTH: Final = 0.1
# stat_bin's fuzz: 1e-8 of the median bin width, so a value within rounding of an
# inner break counts in the bin on its left.
_FUZZ: Final = 1e-8


@dataclass(frozen=True, slots=True)
class HistogramBins:
    """Bins laid edge to edge: bin i spans ``edges[i]`` to ``edges[i + 1]``
    (closed on the right; the first also on the left) and holds ``counts[i]``
    values, ``proportions[i]`` = count / n of them. ``binwidth`` is the width
    asked for (the edges are its multiples from the first edge, as R forms them).
    """

    edges: tuple[float, ...]
    counts: tuple[int, ...]
    proportions: tuple[float, ...]
    binwidth: float
    n: int


@dataclass(frozen=True, slots=True)
class OlsBand:
    """The least-squares line y = intercept + slope x of the pairs and its
    pointwise confidence band at ``conf_level``, evaluated at ``x``: ``fit``, its
    standard error ``se`` and ``conf_low`` / ``conf_high`` = fit -/+
    ``critical_value`` se, the critical value the t quantile at (1 + conf_level) /
    2 on ``df`` = n - 2."""

    x: tuple[float, ...]
    fit: tuple[float, ...]
    se: tuple[float, ...]
    conf_low: tuple[float, ...]
    conf_high: tuple[float, ...]
    slope: float
    intercept: float
    df: float
    critical_value: float
    conf_level: float


def histogram_binwidth(x: SampleLike) -> float:
    """gghistostats' default bin width: (max(x) - min(x)) / sqrt(n), ggstatsplot's
    ``.binwidth()`` as the fixtures record it (all 75 cases without ``binwidth``).

    Raises:
        ValueError: x is empty or not finite, or every value is the same (the
            width would be 0; pass a ``binwidth``).
    """
    arr = as_sample(x)
    spread = float(arr.max()) - float(arr.min())
    if spread == 0.0:
        raise ValueError(
            "every value of x is the same, so the default bin width "
            "(max - min) / sqrt(n) is 0; give the bin width (binwidth=)"
        )
    if math.isinf(spread):
        raise ValueError(
            "the range of x exceeds the largest double (about 1.8e308); rescale x"
        )
    return spread / math.sqrt(arr.size)


def _seq_by(start: float, stop: float, step: float) -> FloatArray:
    """R's ``seq(start, stop, by = step)`` for start <= stop: start + k step for
    k = 0 .. floor((stop - start) / step + 1e-10)."""
    count = (stop - start) / step
    if count > _MAX_BINS:
        raise ValueError(
            f"the bin width {step!r} gives about {count:,.0f} bins over the range of "
            f"the values, more than ggplot2 draws ({_MAX_BINS:,}); use a larger "
            "bin width"
        )
    k = math.floor(count + 1e-10)
    return start + np.arange(k + 1, dtype=np.float64) * step


def _bins(values: FloatArray, width: float, boundary: float) -> HistogramBins:
    """stat_bin's bins of ``width`` with a break at ``boundary``: the first break
    is boundary + floor((min - boundary) / width) width, then breaks every width
    until max + (1 - 1e-8) width; each value counted once, in (left, right]
    (the first bin also takes its left edge), every inner break moved right by
    1e-8 of the median width."""
    low, high = float(values.min()), float(values.max())
    shift = math.floor((low - boundary) / width)
    origin = boundary + shift * width
    edges = _seq_by(origin, high + (1.0 - _FUZZ) * width, width)
    if edges.size == 1:
        edges = np.array([edges[0], edges[0] + width])
    fuzz = _FUZZ * float(np.median(np.diff(edges)))
    fuzzy = edges + fuzz
    fuzzy[0] = edges[0] - fuzz
    index = np.searchsorted(fuzzy, values, side="left") - 1
    index = np.clip(index, 0, edges.size - 2)
    counts = np.bincount(index, minlength=edges.size - 1)
    n = values.size
    return HistogramBins(
        edges=tuple(edges.tolist()),
        counts=tuple(int(c) for c in counts.tolist()),
        proportions=tuple(int(c) / n for c in counts.tolist()),
        binwidth=width,
        n=n,
    )


def histogram_bins(x: SampleLike, *, binwidth: float | None = None) -> HistogramBins:
    """The histogram gghistostats draws (ggplot2's ``stat_bin(binwidth = w)``):
    bins of width w = ``binwidth``, else ``histogram_binwidth(x)``, on the grid
    (k + 1/2) w (ggplot2's default boundary, half a bin width), from the bin
    holding min(x) to the one holding max(x), every value counted once (intervals
    closed on the right). ``proportions`` is count / n, what the right axis shows
    as a percentage. Every one of the 76 gghistostats fixtures' built bins: edges
    within rounding, counts exactly.

    Raises:
        ValueError: x empty or not finite, a binwidth that is not a finite positive
            number, every value the same without a binwidth, or more than a
            million bins.
    """
    arr = as_sample(x)
    if binwidth is None:
        width = histogram_binwidth(arr)
    else:
        width = float(binwidth)
        if not (math.isfinite(width) and width > 0.0):
            raise ValueError(f"binwidth must be a finite number > 0; got {binwidth!r}")
    return _bins(arr, width, 0.5 * width)


def marginal_bins(x: SampleLike, *, bins: int = MARGINAL_BINS) -> HistogramBins:
    """A marginal histogram of the scatter plot (ggside's ``geom_xsidehistogram``
    and ``geom_ysidehistogram``, ggplot2's ``stat_bin(bins = 30)``): ``bins`` bins
    of width (max - min) / (bins - 1) centred on min(x), so the first and last
    bins are centred on the smallest and largest values; a constant variable gets
    one bin 0.1 wide centred on its value (edge_corr_constant_x). Every marginal of
    the ggscatterstats fixtures: edges within rounding, counts exactly.

    Raises:
        ValueError: x empty or not finite, or ``bins`` below 2.
    """
    if isinstance(bins, bool) or int(bins) != bins or bins < 2:
        raise ValueError(f"bins must be a whole number >= 2; got {bins!r}")
    arr = as_sample(x)
    low, high = float(arr.min()), float(arr.max())
    if low == high:
        width = _ZERO_RANGE_WIDTH
    else:
        width = (high - low) / (int(bins) - 1)
        if math.isinf(width):
            raise ValueError(
                "the range of x exceeds the largest double (about 1.8e308); rescale x"
            )
    return _bins(arr, width, low - 0.5 * width)


def _seq_length(start: float, stop: float, count: int) -> FloatArray:
    """R's ``seq(start, stop, length.out = count)``: start + i (stop - start) /
    (count - 1), the last value stop itself."""
    step = (stop - start) / (count - 1)
    out = start + np.arange(count, dtype=np.float64) * step
    out[-1] = stop
    return out


def ols_band(x: SampleLike, y: SampleLike, *, conf_level: float = 0.95) -> OlsBand:
    """The least-squares line of y on x and its pointwise confidence band, as
    ggplot2's ``geom_smooth(method = "lm", formula = y ~ x, level = conf_level)``
    draws it (R's ``predict.lm(se.fit = TRUE)`` at 80 points from min(x) to max(x)):
    slope b = S_xy / S_xx, fit(x0) = mean(y) + b (x0 - mean(x)), se(x0) = s sqrt(1 /
    n + (x0 - mean(x))^2 / S_xx) with s^2 = (S_yy - S_xy^2 / S_xx) / (n - 2), and
    the band fit -/+ t se, t the quantile of Student's t at (1 + conf_level) / 2
    on n - 2 df (``smm_quantile`` with one comparison, accurate to 1e-14 at any
    SciPy). The sums of squares and products are exact (rationals from the exact
    sums of the doubles), rounded once. Every lm fixture: fit, se and band to 1e-8.

    Raises:
        ValueError: fewer than 3 pairs (no residual df), unequal lengths, a
            non-finite value, a constant x (no line: ggplot2 draws none), or
            conf_level not strictly between 0 and 1.
    """
    level = check_conf_level(conf_level)
    a = as_sample(x, name="x", min_size=3)
    b = as_sample(y, name="y", min_size=3)
    if a.size != b.size:
        raise ValueError(
            f"x and y need the same number of values (pairs); got {a.size} and {b.size}"
        )
    low, high = float(a.min()), float(a.max())
    if low == high:
        raise ValueError(
            "every value of x is the same, so the least-squares line is undefined "
            "(no line or band is drawn)"
        )
    n = a.size
    mx, my = exact_mean(a), exact_mean(b)
    m = cross_moments(a, b)
    kx, ky = Fraction(m.x_scale), Fraction(m.y_scale)
    # Back to the units of the data (the scales are powers of two: exact).
    sxx = m.sxx / (kx * kx)
    sxy = m.sxy / (kx * ky)
    slope = sxy / sxx
    s2 = (m.syy - m.sxy * m.sxy / m.sxx) / (ky * ky) / (n - 2)
    df = float(n - 2)
    critical = smm_quantile(1, df, conf_level=level)
    grid = _seq_length(low, high, SMOOTH_POINTS)
    fits: list[float] = []
    ses: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for x0 in grid.tolist():
        offset = Fraction(x0) - mx
        fit = my + slope * offset
        var = s2 * (Fraction(1, n) + offset * offset / sxx)
        se = math.sqrt(float(var))
        fits.append(float(fit))
        ses.append(se)
        lows.append(float(fit) - critical * se)
        highs.append(float(fit) + critical * se)
    return OlsBand(
        x=tuple(grid.tolist()),
        fit=tuple(fits),
        se=tuple(ses),
        conf_low=tuple(lows),
        conf_high=tuple(highs),
        slope=float(slope),
        intercept=float(my - slope * mx),
        df=df,
        critical_value=critical,
        conf_level=level,
    )
