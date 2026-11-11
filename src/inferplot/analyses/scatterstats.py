"""``scatterstats``: test the correlation of two numeric columns.

The analysis that ``ip.analyze.scatterstats`` exposes, the counterpart of
ggstatsplot 1.1.4's ``ggscatterstats()``. It reads the data through
:func:`inferplot._data.read_pair` (the complete pairs in row order, and the
``label_var`` text of each), validates every argument with
:mod:`inferplot._validate`, and returns a :class:`~inferplot.result.StatsResult`
with ``design="correlation"``. It never imports matplotlib.

What each type reproduces (SPEC.md section 4, "scatterstats"), as
``correlation::correlation()`` computes it for ``statsExpressions::corr_test()``;
every correlation is tested two-sided (ggscatterstats has no ``alternative``):

* parametric: Pearson's r with its t test on n - 2 df and the Fisher z interval,
  and the Bayesian Pearson correlation caption;
* nonparametric: Spearman's rho with the t-approximation p and the Fieller,
  Hartley and Pearson interval;
* robust: the Pearson correlation of x and y each winsorized at ``tr``, tested as
  Pearson's r;
* bayes: the Bayesian Pearson correlation's Bayes factor (stretched beta prior of
  width ``bf_prior``).

Each type's analysis is a function ``(data, args) -> _Parts`` in ``_ANALYSES``,
run after every argument has been validated and the data read (``_prepare``);
``_assemble`` builds the result from its parts. A column without spread, too few
pairs for the Fisher z interval, and a column that winsorizes to one number raise
a plain ``ValueError`` naming the columns before the method runs. Exactly
collinear data give r = +/-1, t = +/-inf, p = 0 and the Bayes factor's limit, with
a ``warnings`` entry (D85).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from inferplot import stats
from inferplot._data import PairData, data_notes, read_pair
from inferplot._validate import (
    canonical_type,
    check_bool,
    check_conf_level,
    check_digits,
    check_optional_bool,
    check_positive,
    check_seed,
    check_tr,
    checked,
)
from inferplot.analyses._no_levels import (
    check_bayes_caption,
    check_type_arguments,
    no_effect_warnings,
)
from inferplot.result import (
    BayesFactor,
    EffectSize,
    Settings,
    StatsResult,
    TestResult,
    correlation_tied_count,
    current_provenance,
)
from inferplot.text import (
    build_texts,
    format_value,
    r_number_text,
    with_zero_p_note,
)

__all__ = ["scatterstats"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Args:
    """The validated arguments every type's analysis reads."""

    x: str
    y: str
    type: str
    conf_level: float
    prior_scale: float
    bf_message: bool | None
    results_subtitle: bool
    tr: float
    seed: int
    label_var: str | None = None
    # The arguments as the caller gave them, for the warnings that name them.
    passed: Mapping[str, object] = field(default_factory=dict[str, object])


@dataclass(frozen=True, slots=True)
class _Parts:
    """What one type's analysis computed: the test, the correlation as the effect
    size with its interval, and the Bayes factor where one is computed.
    ``warnings`` are the type's own: ``_assemble`` adds the data notes and the
    arguments without effect (D39)."""

    test: TestResult | None
    effect_size: EffectSize | None
    bayes: BayesFactor | None
    warnings: tuple[str, ...]
    citations: tuple[str, ...]


# --------------------------------------------------------------------------- #
# the types (each takes the complete pairs and the validated arguments)
# --------------------------------------------------------------------------- #

# The coefficient of each type, for the messages.
_METHODS: Final[dict[str, str]] = {
    "parametric": "Pearson's r",
    "nonparametric": "Spearman's rho",
    "robust": "the winsorized correlation",
    "bayes": "the Bayesian Pearson correlation",
}
# The Fisher-z interval has standard error sqrt(c / (n - 3)): it needs 4 pairs.
_MIN_PAIRS_INTERVAL: Final = 4
# The trimming proportions offered, largest first, when the one given winsorizes a
# column to one number (tr=0 winsorizes nothing).
_SMALLER_TR: Final = (0.2, 0.1, 0.0)

# Short references (SPEC.md section 4).
_PEARSON_1895: Final = (
    "Pearson, K. (1895). Proceedings of the Royal Society of London 58: 240-242 "
    "(the product-moment correlation)."
)
_FISHER_1915: Final = (
    "Fisher, R. A. (1915). Biometrika 10: 507-521 (the z transformation of r)."
)
_SPEARMAN_1904: Final = (
    "Spearman, C. (1904). American Journal of Psychology 15: 72-101 (the rank "
    "correlation)."
)
_FIELLER_1957: Final = (
    "Fieller, E. C., Hartley, H. O. & Pearson, E. S. (1957). Biometrika 44: "
    "470-481 (the variance of Spearman's rho on the z scale)."
)
_WILCOX_2017: Final = (
    "Wilcox, R. R. (2017). Introduction to Robust Estimation and Hypothesis "
    "Testing, 4th ed. Academic Press, section 9.3.2 (the winsorized correlation)."
)
_LY_2016: Final = (
    "Ly, A., Verhagen, J. & Wagenmakers, E.-J. (2016). Journal of Mathematical "
    "Psychology 72: 19-32 (Bayes factor of a correlation)."
)
_POSTERIOR_WARNING: Final = (
    "The posterior correlation and its credible interval, which ggstatsplot adds to "
    "the Bayes factor subtitle, arrive in inferplot 0.2."
)


def _capital(text: str) -> str:
    return text[0].upper() + text[1:]


def _same_within_rounding(values: FloatArray) -> bool:
    """Every value the same number, exactly or to within the rounding of the
    values in double precision (D74)."""
    return stats.differences_same_within_rounding(values, np.zeros_like(values))


def _no_spread(name: str, column: str, values: FloatArray) -> str | None:
    """What is wrong with a column whose values are one number, or None."""
    if not _same_within_rounding(values):
        return None
    low, high = float(values.min()), float(values.max())
    if low == high:
        return f"every value of {name}={column!r} is {r_number_text(low)}"
    return (
        f"the values of {name}={column!r} differ only by the rounding of double "
        f"precision (from {low!r} to {high!r}): as written they are one number"
    )


def _check_pairs(data: PairData, args: _Args) -> None:
    """Plain errors, naming the columns, for pairs no type can use (a column
    without spread: ggstatsplot prints NA for every number, and its Bayesian
    correlation stops) or too few for the type's interval, before the method
    runs."""
    for name, column, values in (
        ("x", data.x, data.x_values),
        ("y", data.y, data.y_values),
    ):
        problem = _no_spread(name, column, values)
        if problem is not None:
            raise ValueError(
                f"{_capital(problem)}, so its standard deviation is 0 and a "
                "correlation with it is undefined for every type (ggstatsplot prints "
                "NA for every number). Check the data for entry errors."
            )
    n = data.n_obs
    if args.type != "bayes" and n < _MIN_PAIRS_INTERVAL:
        variance = "1.06 / (n - 3)" if args.type == "nonparametric" else "1 / (n - 3)"
        raise ValueError(
            f"x={data.x!r} and y={data.y!r} have {n} complete pairs; the "
            f"Fisher z confidence interval of {_METHODS[args.type]} needs at least "
            f"{_MIN_PAIRS_INTERVAL} (its variance is {variance}). Collect more "
            "pairs, or use type='bayes', whose Bayes factor needs 3."
        )


def _winsorized(values: FloatArray, tr: float) -> FloatArray:
    """Each value clipped to the (g + 1)-th smallest and largest, g = floor(tr n)
    (``datawizard::winsorize``, as the winsorized correlation does)."""
    ordered = values.copy()
    ordered.sort()
    g = stats.trim_count(values.size, tr)
    return np.minimum(np.maximum(values, ordered[g]), ordered[values.size - g - 1])


def _winsorized_flat(data: PairData, tr: float) -> tuple[str, str] | None:
    """The first column that winsorizing at ``tr`` leaves one number, or None."""
    for name, column, values in (
        ("x", data.x, data.x_values),
        ("y", data.y, data.y_values),
    ):
        if _same_within_rounding(_winsorized(values, tr)):
            return name, column
    return None


def _check_robust(data: PairData, args: _Args) -> None:
    """Type robust: neither column winsorizes to one number (its winsorized
    standard deviation would be 0). The advice names a smaller tr that works (tr=0
    winsorizes nothing, and the columns vary) and the rank correlation."""
    flat = _winsorized_flat(data, args.tr)
    if flat is None:
        return
    name, column = flat
    smaller = next(
        t for t in _SMALLER_TR if t < args.tr and _winsorized_flat(data, t) is None
    )
    raise ValueError(
        f"After winsorizing {_percent(args.tr)}% at each end (tr={args.tr!r}), every "
        f"value of {name}={column!r} is the same number, so its winsorized standard "
        "deviation is 0 and the winsorized correlation is undefined. Pass a smaller "
        f"tr (tr={smaller!r}), or use type='nonparametric'."
    )


def _percent(tr: float) -> str:
    """``tr`` as a percentage the way R prints it: 0.2 -> "20", 0.125 -> "12.5"."""
    return r_number_text(tr * 100)


def _collinear_warning(
    data: PairData, args: _Args, estimate: float, bayes: BayesFactor | None
) -> str:
    """|r| = 1 exactly (D85): t is infinite, p is 0, the interval is [r, r] and
    the Bayes factor is its limit; ggstatsplot prints R's rounding noise for t, S
    and p there and an empty Bayes factor (edge_corr_perfect)."""
    bound = "1" if estimate > 0 else "-1"
    linear = (
        f"the columns x={data.x!r} and y={data.y!r} are exactly linearly related "
        "(every pair lies on one straight line), so Pearson's r"
    )
    what = {
        "parametric": linear,
        "bayes": linear,
        "robust": (
            f"the winsorized values of x={data.x!r} and y={data.y!r} lie on one "
            "straight line, so the winsorized correlation"
        ),
        "nonparametric": (
            f"the ranks of x={data.x!r} and y={data.y!r} agree exactly (y "
            f"{'rises' if estimate > 0 else 'falls'} whenever x rises), so "
            "Spearman's rho"
        ),
    }[args.type]
    out = [f"{_capital(what)} is exactly {bound}."]
    if args.type != "bayes":
        out.append(
            f"Its t is {'' if estimate > 0 else '-'}Inf and its p is 0 (the limit of "
            "the t approximation, not an exact probability; the subtitle prints "
            "p = 2.23e-308, as ggstatsplot prints a p-value of 0), and its confidence "
            f"interval has no width: [{bound}, {bound}]."
        )
        example = (
            "S = 3.7e-14 and p = 1.1e-62 for y = 2x + 1"
            if args.type == "nonparametric"
            else "t = 134217728 and p = 1.1e-62 for y = 2x + 1"
        )
        out.append(
            "ggstatsplot prints the rounding of R's arithmetic for these numbers "
            f"({example})."
        )
    if bayes is not None:
        size = "infinite" if math.isinf(bayes.bf10) else format_value(bayes.bf10)
        out.append(
            f"The Bayes factor is its limit as r approaches {bound} (BF10 {size}), "
            "where ggstatsplot leaves it empty."
        )
    out.append(
        "A rank correlation needs no straight line for this: y only has to rise or "
        "fall with x."
        if args.type == "nonparametric"
        else "Check that neither column is computed from the other."
    )
    return " ".join(out)


def _correlation_parts(
    data: PairData, args: _Args, result: stats.Correlation
) -> tuple[TestResult, EffectSize]:
    names = {
        "pearson": (
            "pearson_correlation",
            "Pearson correlation",
            "Pearson correlation",
            "pearson_r",
            "Pearson's r",
            "normal approximation (Fisher z)",
        ),
        "spearman": (
            "spearman_correlation",
            "Spearman correlation",
            "Spearman correlation",
            "spearman_rho",
            "Spearman's rho",
            "normal approximation (Fisher z, variance 1.06 / (n - 3))",
        ),
        "winsorized": (
            "winsorized_correlation",
            f"Winsorized Pearson correlation ({_percent(args.tr)}% winsorized)",
            "Winsorized Pearson correlation",
            "winsorized_r",
            "winsorized r",
            "normal approximation (Fisher z)",
        ),
    }
    name, label, method, es_name, es_label, ci_method = names[result.method]
    spearman = result.method == "spearman"
    test = TestResult(
        name=name,
        label=label,
        method=method,
        statistic_name="S" if spearman else "t",
        statistic=result.statistic,
        df=None,
        df_error=None if spearman else result.df,
        p_value=result.p_value,
        alternative="two-sided",
    )
    effect_size = EffectSize(
        name=es_name,
        label=es_label,
        estimate=result.estimate,
        conf_level=result.conf_level,
        conf_low=result.conf_low,
        conf_high=result.conf_high,
        ci_method=ci_method,
        sidedness="two-sided",
    )
    return test, effect_size


def _bayes_factor(data: PairData, args: _Args) -> BayesFactor:
    """The Bayesian Pearson correlation: ``prior_scale`` is the parameter of the
    stretched beta(1 / kappa, 1 / kappa) prior, 1 / bf_prior, as ggstatsplot
    prints it (``r_beta^JZS == 1.41`` for bf_prior 0.707) and correlation's row
    records it (``prior.scale``); ``settings.bf_prior`` keeps kappa."""
    bf10, log_bf01 = stats.correlation_bf10(
        data.x_values, data.y_values, prior_scale=args.prior_scale
    )
    return BayesFactor(
        bf10=bf10,
        log_bf01=log_bf01,
        prior="beta",
        prior_scale=1.0 / args.prior_scale,
        test="pearson_correlation",
    )


def _frequentist(data: PairData, args: _Args, compute: str) -> _Parts:
    """Types parametric, nonparametric and robust: the coefficient with its test
    and interval; type parametric adds the Bayes factor caption."""
    _check_pairs(data, args)
    x, y, level = data.x_values, data.y_values, args.conf_level
    if compute == "pearson":
        result = stats.pearson(x, y, conf_level=level)
        citations = [_PEARSON_1895, _FISHER_1915]
    elif compute == "spearman":
        result = stats.spearman(x, y, conf_level=level)
        citations = [_SPEARMAN_1904, _FIELLER_1957]
    else:
        _check_robust(data, args)
        result = stats.winsorized_correlation(x, y, tr=args.tr, conf_level=level)
        citations = [_WILCOX_2017, _FISHER_1915]
    test, effect_size = _correlation_parts(data, args, result)
    bayes: BayesFactor | None = None
    if compute == "pearson" and args.results_subtitle and args.bf_message is not False:
        bayes = _bayes_factor(data, args)
        citations.append(_LY_2016)
    warnings: list[str] = []
    if math.isinf(result.t):
        warnings.append(_collinear_warning(data, args, result.estimate, bayes))
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


def _parametric(data: PairData, args: _Args) -> _Parts:
    """Pearson's r with its t test and Fisher z interval, and the Bayesian Pearson
    correlation caption."""
    return _frequentist(data, args, "pearson")


def _nonparametric(data: PairData, args: _Args) -> _Parts:
    """Spearman's rho with the t-approximation p and the Fieller, Hartley and
    Pearson interval."""
    return _frequentist(data, args, "spearman")


def _robust(data: PairData, args: _Args) -> _Parts:
    """The winsorized correlation at ``tr``, tested as Pearson's r."""
    return _frequentist(data, args, "winsorized")


def _bayes(data: PairData, args: _Args) -> _Parts:
    """The Bayesian Pearson correlation's Bayes factor (with the subtitle; without
    it nothing is computed, as for the other designs' type bayes)."""
    _check_pairs(data, args)
    if not args.results_subtitle:
        return _Parts(
            test=None, effect_size=None, bayes=None, warnings=(), citations=()
        )
    bayes = _bayes_factor(data, args)
    warnings = [_POSTERIOR_WARNING]
    moments = stats.pair_moments(data.x_values, data.y_values)
    if moments.one_minus_r2 == 0:
        warnings.insert(0, _collinear_warning(data, args, moments.r, bayes))
    return _Parts(
        test=None,
        effect_size=None,
        bayes=bayes,
        warnings=tuple(warnings),
        citations=(_LY_2016,),
    )


_ANALYSES: Final[dict[str, Callable[[PairData, _Args], _Parts]]] = {
    "parametric": _parametric,
    "nonparametric": _nonparametric,
    "robust": _robust,
    "bayes": _bayes,
}


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def _assemble(
    data: PairData, args: _Args, parts: _Parts, *, digits: int
) -> StatsResult:
    """The result of a correlation from the parts its type computed.

    ``values`` holds two tuples, x and y of the complete pairs in row order;
    ``point_labels`` the ``label_var`` text of each pair ("" for no label; empty
    without ``label_var``); ``n_obs`` the number of complete pairs and
    ``n_dropped`` the rows with a missing x or y.
    """
    xs = tuple(float(v) for v in data.x_values.tolist())
    ys = tuple(float(v) for v in data.y_values.tolist())
    settings = Settings(
        conf_level=args.conf_level,
        digits=digits,
        alternative="two-sided",
        bf_prior=args.prior_scale,
        bf_message=args.bf_message,
        results_subtitle=args.results_subtitle,
        seed=args.seed,
        tr=args.tr,
        # Settings of the pairwise comparisons, which a correlation does not
        # have: the result records their defaults.
        p_adjust_method="holm",
        pairwise_display="significant",
        pairwise_alpha=0.05,
    )
    texts = build_texts(
        x=data.x,
        y=data.y,
        descriptives=(),
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=(),
        n_obs=data.n_obs,
        settings=settings,
        design="correlation",
        values=(xs, ys),
        point_labels=data.labels,
        label_var=data.label_var,
    )
    warnings = no_effect_warnings(
        type_name=args.type,
        bayes_computed=parts.bayes is not None,
        subtitle_on=args.results_subtitle,
        show_bf=args.bf_message,
        prior_scale=args.prior_scale,
        conf_level=args.conf_level,
        passed=args.passed,
    )
    return StatsResult(
        analysis="scatterstats",
        design="correlation",
        type=args.type,
        x=data.x,
        y=data.y,
        subject_id=None,
        levels=(),
        values=(xs, ys),
        subjects=(),
        descriptives=(),
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=(),
        n_obs=data.n_obs,
        n_dropped=data.n_dropped,
        dropped_subjects=0,
        dropped_subject_rows=0,
        n_tied=correlation_tied_count(xs, ys),
        warnings=with_zero_p_note(
            (*data_notes(data), *parts.warnings, *warnings),
            parts.test,
            shown=args.results_subtitle,
        ),
        provenance=current_provenance(seed=args.seed, citations=parts.citations),
        settings=settings,
        text=texts,
        point_labels=data.labels,
        label_var=data.label_var,
    )


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #


def _prepare(
    data: object,
    x: str,
    y: str,
    *,
    type: str = "parametric",
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    label_var: str | None = None,
    seed: int = 0,
) -> tuple[PairData, _Args, int]:
    """Every argument checked and the data read: the complete pairs, the
    validated arguments and ``digits``. The public function runs the type's
    analysis on them; tests use it to check the data reading alone."""
    type_name = canonical_type(type)
    level = check_conf_level(conf_level)
    n_digits = check_digits(digits)
    trim = check_tr(tr)
    prior_scale = check_positive(bf_prior, "bf_prior")
    show_bf = check_optional_bool(bf_message, "bf_message")
    subtitle_on = check_bool(results_subtitle, "results_subtitle")
    rng_seed = check_seed(seed)
    check_type_arguments(type_name, tr=tr, trim=trim, show_bf=show_bf)
    check_bayes_caption(show_bf=show_bf, subtitle_on=subtitle_on)

    pair = read_pair(data, x, y, label_var=label_var)
    args = _Args(
        x=x,
        y=y,
        type=type_name,
        conf_level=level,
        prior_scale=prior_scale,
        bf_message=show_bf,
        results_subtitle=subtitle_on,
        tr=trim,
        seed=rng_seed,
        label_var=label_var,
        passed={"bf_prior": bf_prior, "conf_level": conf_level},
    )
    return pair, args, n_digits


@checked
def scatterstats(
    data: object,
    x: str,
    y: str,
    *,
    type: str = "parametric",
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    label_var: str | None = None,
    seed: int = 0,
) -> StatsResult:
    """Test the correlation of two numeric columns; no plotting.

    The counterpart of ggstatsplot 1.1.4's ``ggscatterstats()``. A row is analyzed
    when both ``x`` and ``y`` are present (the others are counted in
    ``n_dropped``); n is the number of complete pairs. Correlations are always
    tested two-sided, as in ggscatterstats, which has no ``alternative``.

    The methods follow ``ggscatterstats()`` for each ``type`` (SPEC.md section 4):

    * "parametric" ("p"): Pearson's r with its t test and Fisher z interval, and
      the Bayesian Pearson correlation caption.
    * "nonparametric" ("np"): Spearman's rho, reported with S =
      (n^3 - n)(1 - rho) / 6 and the p of the t approximation.
    * "robust" ("r"): the correlation of x and y each winsorized at ``tr``,
      tested as Pearson's r.
    * "bayes" ("bf"): the Bayes factor of the Bayesian Pearson correlation, with
      prior width ``bf_prior``; ``test`` and ``effect_size`` are None. The
      posterior estimate arrives in 0.2.

    The coefficient with its interval is the effect size. A correlation of
    exactly +/-1 is reported as it is (t infinite, p 0) with a ``warnings`` entry.
    ``result.bayes.prior_scale`` is the beta prior's parameter 1 / bf_prior, as
    ggstatsplot prints it.

    Args:
        data: a pandas, polars or pyarrow data frame.
        x, y: the two numeric columns.
        type: "parametric" ("p"), "nonparametric" ("np"), "robust" ("r") or
            "bayes" ("bf").
        conf_level: confidence level of the correlation's interval.
        digits: decimals in the subtitle and caption.
        tr: winsorizing proportion of the robust type; another value than 0.2
            with a type that winsorizes nothing raises.
        bf_prior: width of the prior of the Bayes factor (0.707).
        bf_message: None draws the Bayes factor caption where ggstatsplot draws
            it (type "parametric"); True requires it; False never draws it.
        results_subtitle: whether the statistical results are reported as a
            subtitle (False also drops the Bayes factor caption).
        label_var: a column whose values label the points (any type, read as
            text); a missing or empty value leaves its point unlabelled. To
            label only some points, set the others' labels to missing.
        seed: recorded in the result. No correlation draws random numbers; the
            drawing uses it to choose the points it shows of more than 5,000.

    Returns:
        A :class:`~inferplot.result.StatsResult` holding every number and text;
        ``result.point_labels`` holds the label of each analyzed pair.

    Raises:
        TypeError, ValueError: a bad argument or unusable data (a column that is
            not numeric, x and y the same column, fewer than three complete
            pairs, a column whose values are all one number, fewer than four
            pairs for a type with a confidence interval, a column that
            winsorizing at ``tr`` leaves as one number); the message says how to
            fix the call.
        KeyError: a column that the data frame does not have.

    Example:
        >>> import pandas as pd
        >>> import inferplot as ip
        >>> df = pd.DataFrame({
        ...     "x": [1.0, 2, 3, 4, 5, 6, 7, 8],
        ...     "y": [2.1, 2.9, 3.2, 4.8, 5.1, 5.0, 7.2, 7.9],
        ... })
        >>> result = ip.analyze.scatterstats(df, "x", "y", type="nonparametric")
        >>> print(result.text.apa)
        Spearman correlation: S = 2.00, p < .001, Spearman's rho = .98, 95% CI [.86,
        1.00], n = 8 pairs.
    """
    pair, args, n_digits = _prepare(
        data,
        x,
        y,
        type=type,
        conf_level=conf_level,
        digits=digits,
        tr=tr,
        bf_prior=bf_prior,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        label_var=label_var,
        seed=seed,
    )
    parts = _ANALYSES[args.type](pair, args)
    return _assemble(pair, args, parts, digits=n_digits)
