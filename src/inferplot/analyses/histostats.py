"""``histostats``: test one numeric column against a value.

The analysis that ``ip.analyze.histostats`` exposes, the counterpart of ggstatsplot
1.1.4's ``gghistostats()``. It reads the data through
:func:`inferplot._data.read_one` (the non-missing values of ``x`` in row order),
validates every argument with :mod:`inferplot._validate`, and returns a
:class:`~inferplot.result.StatsResult` with ``design="one-sample"``. It never
imports matplotlib.

What each type reproduces (SPEC.md section 4, "histostats"):

* parametric: the one-sample t-test against ``test_value`` with Hedges' g and its
  noncentral-t CI (both follow ``alternative``), and the one-sample JZS Bayes
  factor caption; the mean as centrality;
* nonparametric: the one-sample Wilcoxon signed-rank test of x - test_value with
  the rank-biserial r; the median;
* robust: the bootstrap-t test of the ``tr``-trimmed mean (WRS2's trimcibt), the
  trimmed mean with its bootstrap-t CI; the trimmed mean;
* bayes: the one-sample JZS Bayes factor; the MAP estimate.

Each type's analysis is a function ``(data, args) -> _Parts`` in ``_ANALYSES``,
run after every argument has been validated and the data read (``_prepare``);
``_assemble`` builds the result from its parts. Data a type cannot use (no spread,
too few values after trimming, every value equal to ``test_value``) raise a plain
``ValueError`` naming the column before the method runs, advising only what works.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, TypeVar

import numpy as np
import numpy.typing as npt

from inferplot import stats
from inferplot._data import OneData, data_notes, read_one
from inferplot._validate import (
    canonical_type,
    check_alternative,
    check_bool,
    check_conf_level,
    check_digits,
    check_finite_number,
    check_optional_bool,
    check_positive,
    check_seed,
    check_tr,
    checked,
)
from inferplot.analyses._no_levels import (
    canonical_alternative,
    check_bayes_caption,
    check_type_arguments,
    no_effect_warnings,
)
from inferplot.result import (
    BayesFactor,
    Descriptive,
    EffectSize,
    Settings,
    StatsResult,
    TestResult,
    current_provenance,
    one_sample_tied_count,
)
from inferplot.text import build_texts, format_value, r_number_text

__all__ = ["histostats"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Args:
    """The validated arguments every type's analysis reads."""

    x: str
    type: str
    test_value: float
    conf_level: float
    alternative: stats.Alternative
    prior_scale: float
    bf_message: bool | None
    results_subtitle: bool
    tr: float
    seed: int
    # The arguments as the caller gave them, for the warnings that name them.
    passed: Mapping[str, object] = field(default_factory=dict[str, object])


@dataclass(frozen=True, slots=True)
class _Parts:
    """What one type's analysis computed.

    ``centrality`` names the centrality ("mean", "median", "trimmed_mean", "map")
    and ``center`` is its value (None for a MAP without a density, D55).
    ``warnings`` are the type's own: ``_assemble`` adds the data notes, the
    arguments without effect (D39) and, with a one-sided ``alternative``, the note
    that the Bayes factor caption stays two-sided.
    """

    test: TestResult | None
    effect_size: EffectSize | None
    bayes: BayesFactor | None
    centrality: str
    center: float | None
    warnings: tuple[str, ...]
    citations: tuple[str, ...]


# --------------------------------------------------------------------------- #
# plain errors for data a type cannot use (raised before its method runs)
# --------------------------------------------------------------------------- #

# The method each type's checks name.
_METHODS: Final[dict[str, str]] = {
    "parametric": "the one-sample t-test",
    "nonparametric": "the signed-rank test",
    "robust": "the bootstrap-t test of the trimmed mean",
    "bayes": "the one-sample t-test behind the Bayes factor",
}
# The trimming proportions offered, largest first, when the one given leaves too
# few values or a winsorized standard deviation of 0 (tr=0 trims nothing).
_SMALLER_TR: Final = (0.2, 0.1, 0.0)


def _same_within_rounding(values: FloatArray) -> bool:
    """Every value the same number, exactly or to within the rounding of the
    values in double precision (D74: 0.1 + 0.2 and 0.3 are one number as
    written)."""
    return stats.differences_same_within_rounding(values, np.zeros_like(values))


def _all_test_value(data: OneData, args: _Args) -> bool:
    """Every value equals test_value exactly: the signed-rank test drops every
    value, so no type has a difference to test."""
    return bool((data.values == args.test_value).all())


def _no_spread_text(data: OneData) -> str:
    """What is wrong with a column whose values are one number."""
    values = data.values
    low, high = float(values.min()), float(values.max())
    if low == high:
        return f"every value of x={data.x!r} is {r_number_text(low)}"
    return (
        f"the values of x={data.x!r} differ only by the rounding of double "
        f"precision (from {low!r} to {high!r}): as written they are one number"
    )


def _nonparametric_advice(data: OneData, args: _Args) -> str:
    """Advice that leads to a call that works: the signed-rank test needs only a
    value that differs from test_value."""
    if _all_test_value(data, args):
        return (
            f"Every value equals test_value={r_number_text(args.test_value)}, so "
            "there is no difference to test; check the data for entry errors."
        )
    return (
        "Use type='nonparametric': the signed-rank test needs only values that "
        "differ from test_value."
    )


def _capital(text: str) -> str:
    return text[0].upper() + text[1:]


def _check_spread(data: OneData, args: _Args) -> None:
    """Types parametric and bayes: the one-sample t needs a standard deviation
    (R's t.test stops on constant data: "data are essentially constant")."""
    if _same_within_rounding(data.values):
        raise ValueError(
            f"{_capital(_no_spread_text(data))}, so {_METHODS[args.type]} has no "
            f"standard deviation. {_nonparametric_advice(data, args)}"
        )


def _winsorized(values: FloatArray, tr: float) -> FloatArray:
    """The values winsorized at floor(tr n) at each end, in sorted order."""
    ordered = values.copy()
    ordered.sort()
    g = stats.trim_count(ordered.size, tr)
    if g:
        ordered[:g] = ordered[g]
        ordered[ordered.size - g :] = ordered[ordered.size - g - 1]
    return ordered


def _robust_problem(values: FloatArray, tr: float) -> str | None:
    """Why the bootstrap-t test cannot run on ``values`` at ``tr``, or None."""
    n = values.size
    g = stats.trim_count(n, tr)
    if n - 2 * g < 2:
        return (
            f"trimming {_percent(tr)}% at each end (tr={tr!r}) leaves {n - 2 * g} of "
            f"the {n} values; the trimmed mean's bootstrap-t test needs at least 2"
        )
    if _same_within_rounding(_winsorized(values, tr)):
        return (
            f"after winsorizing {_percent(tr)}% at each end (tr={tr!r}) every value "
            "is the same number, so the winsorized standard deviation is 0 and the "
            "bootstrap-t test of the trimmed mean is undefined"
        )
    return None


def _percent(tr: float) -> str:
    """``tr`` as a percentage the way R prints it: 0.2 -> "20", 0.125 -> "12.5"."""
    return r_number_text(tr * 100)


def _check_robust(data: OneData, args: _Args) -> None:
    """Type robust: enough values after trimming, a winsorized standard deviation
    above 0, and a conf_level the bootstrap-t rule can resolve; each message
    advises only what works (a smaller tr that passes these checks, or the
    signed-rank test)."""
    values = data.values
    if _same_within_rounding(values):
        raise ValueError(
            f"{_capital(_no_spread_text(data))}, so {_METHODS['robust']} has no "
            f"standard deviation. {_nonparametric_advice(data, args)}"
        )
    problem = _robust_problem(values, args.tr)
    if problem is not None:
        smaller = next(
            (
                t
                for t in _SMALLER_TR
                if t < args.tr and _robust_problem(values, t) is None
            ),
            None,
        )
        options = [] if smaller is None else [f"pass a smaller tr (tr={smaller!r})"]
        if not _all_test_value(data, args):
            options.append("use type='nonparametric'")
        advice = (
            _capital(", or ".join(options)) + "."
            if options
            else "Check the data for entry errors."
        )
        raise ValueError(f"For x={data.x!r}, {problem}. {advice}")
    try:
        # trimcibt's rule on placeholder values: it refuses a level it cannot
        # resolve with BOOTSTRAP_RESAMPLES resamples, before any resampling.
        stats.bootstrap_t_critical_value(
            np.zeros(stats.BOOTSTRAP_RESAMPLES), args.conf_level
        )
    except ValueError:
        end = "1" if args.conf_level > 0.5 else "0"
        raise ValueError(
            f"conf_level={args.passed.get('conf_level', args.conf_level)!r} is too "
            f"close to {end} for the bootstrap-t interval from "
            f"{stats.BOOTSTRAP_RESAMPLES:,} resamples: its bounds would come from "
            "the most extreme resample whatever the level. Pass a conf_level "
            "between 0.001 and 0.999, or use type='parametric', whose interval is "
            "not resampled."
        ) from None


# --------------------------------------------------------------------------- #
# the types (each takes the analyzed values and the validated arguments)
# --------------------------------------------------------------------------- #

# Short references (SPEC.md section 4).
_STUDENT_1908: Final = "Student (1908). Biometrika 6: 1-25 (the one-sample t-test)."
_HEDGES_1981: Final = (
    "Hedges, L. V. (1981). Journal of Educational Statistics 6: 107-128 (Hedges' g)."
)
_STEIGER_2004: Final = (
    "Steiger, J. H. (2004). Psychological Methods 9: 164-182 (noncentral confidence "
    "intervals for effect sizes)."
)
_ROUDER_2009: Final = (
    "Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D. & Iverson, G. (2009). "
    "Psychonomic Bulletin & Review 16: 225-237 (JZS Bayes factor)."
)
_WILCOXON_1945: Final = (
    "Wilcoxon, F. (1945). Biometrics Bulletin 1: 80-83 (the signed-rank test)."
)
_KERBY_2014: Final = (
    "Kerby, D. S. (2014). Comprehensive Psychology 3: 11.IT.3.1 (the rank-biserial "
    "correlation)."
)
_WILCOX_2017: Final = (
    "Wilcox, R. R. (2017). Introduction to Robust Estimation and Hypothesis "
    "Testing, 4th ed. Academic Press, section 4.4.3 (the bootstrap-t test of a "
    "trimmed mean: trimcibt)."
)
_SHEATHER_JONES_1991: Final = (
    "Sheather, S. J. & Jones, M. C. (1991). Journal of the Royal Statistical "
    "Society B 53: 683-690 (kernel density bandwidth of the MAP estimate)."
)

_HEDGES_J_ZERO_WARNING: Final = (
    "Hedges' g is 0 by construction with 2 values: its small-sample correction "
    "J(n - 1) is 0 at 1 degree of freedom, so g and its confidence interval carry no "
    "information about the size of the difference. Collect more values."
)
_POSTERIOR_WARNING: Final = (
    "The posterior difference and its credible interval, which ggstatsplot adds to "
    "the Bayes factor subtitle, arrive in inferplot 0.2."
)


_T = TypeVar("_T")


def _checked(data: OneData, compute: Callable[[], _T]) -> _T:
    """A statistic of x, or its ValueError (a spread below what double precision
    resolves, a statistic beyond the largest double) naming the column."""
    try:
        return compute()
    except ValueError as error:
        text = str(error).rstrip(".")
        raise ValueError(
            f"{_capital(text)} (x={data.x!r}). Check the data and test_value for "
            "entry or unit errors."
        ) from None


def _one_sample_bayes(data: OneData, args: _Args) -> BayesFactor:
    bf10, log_bf01 = _checked(
        data,
        lambda: stats.jzs_bf10_one_sample(
            data.values, mu=args.test_value, prior_scale=args.prior_scale
        ),
    )
    return BayesFactor(
        bf10=bf10,
        log_bf01=log_bf01,
        prior="cauchy",
        prior_scale=args.prior_scale,
        test="jzs_one_sample",
    )


def _parametric(data: OneData, args: _Args) -> _Parts:
    """One-sample t-test against ``test_value`` with Hedges' g and the one-sample
    JZS Bayes factor caption; the mean as centrality."""
    _check_spread(data, args)
    x, mu, side = data.values, args.test_value, args.alternative
    t = _checked(data, lambda: stats.one_sample_t(x, mu=mu, alternative=side))
    g = _checked(
        data,
        lambda: stats.one_sample_hedges_g(
            x, mu=mu, conf_level=args.conf_level, alternative=side
        ),
    )
    test = TestResult(
        name="one_sample_t",
        label="One-sample t-test",
        method="One Sample t-test",
        statistic_name="t",
        statistic=t.t,
        df=None,
        df_error=t.df,
        p_value=t.p_value,
        alternative=side,
    )
    effect_size = EffectSize(
        name="hedges_g",
        label="Hedges' g",
        estimate=g.estimate,
        conf_level=g.conf_level,
        conf_low=g.conf_low,
        conf_high=g.conf_high,
        ci_method="noncentral t",
        sidedness=g.sidedness,
    )
    warnings: list[str] = []
    citations = [_STUDENT_1908, _HEDGES_1981, _STEIGER_2004]
    if t.n == 2:
        warnings.append(_HEDGES_J_ZERO_WARNING)
    bayes: BayesFactor | None = None
    if args.results_subtitle and args.bf_message is not False:
        bayes = _one_sample_bayes(data, args)
        citations.append(_ROUDER_2009)
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        centrality="mean",
        center=stats.mean(x),
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


def _nonparametric(data: OneData, args: _Args) -> _Parts:
    """One-sample Wilcoxon signed-rank test of x - test_value with the
    rank-biserial r; the median as centrality."""
    if _all_test_value(data, args):
        raise ValueError(
            f"Every value of x={data.x!r} equals test_value="
            f"{r_number_text(args.test_value)}, so the signed-rank test has nothing "
            "to rank (its statistic and the rank-biserial r are 0/0). There is no "
            "difference to test; check the data for entry errors."
        )
    x, mu, side = data.values, args.test_value, args.alternative
    wilcoxon = stats.one_sample_signed_rank(x, mu=mu, alternative=side)
    r = stats.one_sample_rank_biserial(
        x, mu=mu, conf_level=args.conf_level, alternative=side
    )
    test = TestResult(
        name="wilcoxon_signed_rank",
        label="Wilcoxon signed-rank test",
        method="Wilcoxon signed rank test",
        statistic_name="V",
        statistic=wilcoxon.statistic,
        df=None,
        df_error=None,
        p_value=wilcoxon.p_value,
        alternative=side,
    )
    effect_size = EffectSize(
        name="rank_biserial",
        label="rank-biserial r",
        estimate=r.estimate,
        conf_level=r.conf_level,
        conf_low=r.conf_low,
        conf_high=r.conf_high,
        ci_method="normal approximation (Fisher z)",
        sidedness=r.sidedness,
    )
    warnings: list[str] = []
    if abs(r.estimate) == 1.0 and r.conf_low == r.conf_high:
        side_word, bound = ("above", "1") if r.estimate > 0 else ("below", "-1")
        warnings.append(
            f"Every value of x={data.x!r} that differs from test_value="
            f"{r_number_text(mu)} lies {side_word} it, so the rank-biserial r is at "
            f"its bound ({bound}) and its confidence interval has no width: "
            f"[{bound}, {bound}]. The interval is built on the Fisher z scale, where "
            "r = 1 or -1 lies infinitely far out; it does not mean the effect is "
            "known exactly. ggstatsplot prints the same interval."
        )
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=None,
        centrality="median",
        center=stats.median(x),
        warnings=tuple(warnings),
        citations=(_WILCOXON_1945, _KERBY_2014),
    )


def _robust(data: OneData, args: _Args) -> _Parts:
    """Bootstrap-t test of the trimmed mean (WRS2's trimcibt) with the trimmed
    mean and its bootstrap-t CI; the trimmed mean as centrality."""
    _check_robust(data, args)
    x, tr = data.values, args.tr
    try:
        boot = stats.trimmed_mean_bootstrap_t(
            x, mu=args.test_value, tr=tr, conf_level=args.conf_level, seed=args.seed
        )
    except ValueError as error:
        text = str(error).rstrip(".")
        advice = "" if _all_test_value(data, args) else " Use type='nonparametric'."
        raise ValueError(f"{_capital(text)} (x={data.x!r}).{advice}") from None
    test = TestResult(
        name="trimmed_mean_bootstrap_t",
        label=f"Bootstrap-t test of the {_percent(tr)}% trimmed mean",
        method="Bootstrap-t method for one-sample test",
        statistic_name="t",
        statistic=boot.statistic,
        df=None,
        df_error=None,
        p_value=boot.p_value,
        alternative="two-sided",
    )
    effect_size = EffectSize(
        name="trimmed_mean",
        label="trimmed mean",
        estimate=boot.estimate,
        conf_level=boot.conf_level,
        conf_low=boot.conf_low,
        conf_high=boot.conf_high,
        ci_method="bootstrap-t",
        sidedness="two-sided",
        bootstrap_resamples=boot.resamples,
        bootstrap_seed=boot.seed,
        undefined_resamples=boot.undefined_resamples,
        infinite_resamples=boot.infinite_resamples,
    )
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=None,
        centrality="trimmed_mean",
        center=boot.estimate,
        warnings=tuple(
            _resample_warnings(
                boot, effect_size, nonparametric=not _all_test_value(data, args)
            )
        ),
        citations=(_WILCOX_2017,),
    )


def _resample_warnings(
    boot: stats.BootstrapTrimmedMeanTest,
    effect_size: EffectSize,
    *,
    nonparametric: bool,
) -> list[str]:
    """How many bootstrap t values were 0 / 0 (left out, D65's rule) or infinite
    (kept); ``check()`` reports the infinite ones as ``infinite_effect_size``.
    A bootstrap p-value of exactly 0 is named for what it is: below the
    resolution of the resamples."""
    total = boot.resamples
    out: list[str] = []
    if boot.undefined_resamples:
        undefined = boot.undefined_resamples
        out.append(
            f"{undefined:,} of {total:,} bootstrap resamples give an undefined (0/0) "
            "bootstrap t, because every winsorized value of the resample equals the "
            "trimmed mean of the data. They are left out: the critical value, the "
            f"interval and the p-value come from the other {total - undefined:,}."
        )
    if boot.infinite_resamples:
        infinite = boot.infinite_resamples
        low, high = effect_size.conf_low, effect_size.conf_high
        interval = f"[{format_value(low)}, {format_value(high)}]"
        kept = (
            f"so the trimmed mean's interval is unbounded: {interval}"
            if math.isinf(low) or math.isinf(high)
            else f"so the trimmed mean's interval {interval} would be unbounded at a "
            "conf_level high enough to reach them"
        )
        out.append(
            f"{infinite:,} of {total:,} bootstrap resamples give an infinite bootstrap "
            "t, because every winsorized value of the resample is the same number "
            f"(a standard error of 0). They are kept, as in WRS2, {kept}"
            + (
                ". Collect more values, or use type='nonparametric'."
                if nonparametric
                else ". Collect more values."
            )
        )
    if boot.p_value == 0.0:
        defined = total - boot.undefined_resamples
        out.append(
            f"None of the {defined:,} bootstrap resamples gave a t as extreme as "
            "the observed one, so the bootstrap p-value is 0 as computed: all it "
            f"shows is that p is below about 1 in {defined:,} "
            f"({format_value(1.0 / defined, 4)}). The subtitle prints "
            "p = 2.23e-308, as ggstatsplot prints a p-value of 0; report "
            f"p < {format_value(1.0 / defined, 4)}."
        )
    return out


def _bayes(data: OneData, args: _Args) -> _Parts:
    """One-sample JZS Bayes factor; the MAP estimate as centrality."""
    _check_spread(data, args)
    warnings: list[str] = []
    bayes: BayesFactor | None = None
    if args.results_subtitle:
        bayes = _one_sample_bayes(data, args)
        warnings.append(_POSTERIOR_WARNING)
    try:
        center: float | None = stats.map_estimate(data.values).estimate
    except ValueError as error:
        center = None
        warnings.append(
            f"x={data.x!r} (n = {data.n_obs}) has no MAP estimate, so no centrality "
            f"line or label is drawn: {error}. ggstatsplot draws an empty label "
            "there. Use type='nonparametric' to show the median."
        )
    return _Parts(
        test=None,
        effect_size=None,
        bayes=bayes,
        centrality="map",
        center=center,
        warnings=tuple(warnings),
        citations=(_ROUDER_2009, _SHEATHER_JONES_1991),
    )


_ANALYSES: Final[dict[str, Callable[[OneData, _Args], _Parts]]] = {
    "parametric": _parametric,
    "nonparametric": _nonparametric,
    "robust": _robust,
    "bayes": _bayes,
}


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def _warnings(args: _Args, parts: _Parts) -> list[str]:
    """The arguments without effect (D39) and the two-sided Bayes caption note."""
    out = no_effect_warnings(
        type_name=args.type,
        bayes_computed=parts.bayes is not None,
        subtitle_on=args.results_subtitle,
        show_bf=args.bf_message,
        prior_scale=args.prior_scale,
        conf_level=args.conf_level,
        passed=args.passed,
    )
    if args.seed != _DEFAULT_SEED and args.type != "robust":
        out.append(
            f"seed={args.passed.get('seed', args.seed)!r} has no effect: "
            f"type={args.type!r} draws no random numbers (only type='robust' "
            "resamples, for its bootstrap-t test), and the histogram is drawn "
            "from every value."
        )
    if parts.bayes is not None and args.alternative != "two-sided":
        out.append(
            "The Bayes factor in the caption is two-sided: it weighs any difference "
            f"from test_value, while alternative={args.alternative!r} applies to "
            "the test and its effect size interval only (as in ggstatsplot)."
        )
    return out


_DEFAULT_SEED: Final = 0


def _assemble(data: OneData, args: _Args, parts: _Parts, *, digits: int) -> StatsResult:
    """The result of a one-sample analysis from the parts its type computed.

    ``values`` holds one tuple, the analyzed values of x in row order;
    ``descriptives`` one entry named after the column; ``n_obs`` the number of
    values and ``n_dropped`` the rows with a missing x.
    """
    observed = tuple(float(v) for v in data.values.tolist())
    descriptives = (
        Descriptive(
            level=data.x, n=data.n_obs, centrality=parts.centrality, value=parts.center
        ),
    )
    settings = Settings(
        conf_level=args.conf_level,
        digits=digits,
        alternative=args.alternative,
        bf_prior=args.prior_scale,
        bf_message=args.bf_message,
        results_subtitle=args.results_subtitle,
        seed=args.seed,
        tr=args.tr,
        test_value=args.test_value,
        # Settings of the pairwise comparisons, which a one-sample design does
        # not have: the result records their defaults.
        p_adjust_method="holm",
        pairwise_display="significant",
        pairwise_alpha=0.05,
    )
    texts = build_texts(
        x=data.x,
        y=None,
        descriptives=descriptives,
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=(),
        n_obs=data.n_obs,
        settings=settings,
        design="one-sample",
        values=(observed,),
    )
    return StatsResult(
        analysis="histostats",
        design="one-sample",
        type=args.type,
        x=data.x,
        y=None,
        subject_id=None,
        levels=(),
        values=(observed,),
        subjects=(),
        descriptives=descriptives,
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=(),
        n_obs=data.n_obs,
        n_dropped=data.n_dropped,
        dropped_subjects=0,
        dropped_subject_rows=0,
        n_tied=one_sample_tied_count(observed, args.test_value),
        warnings=(*data_notes(data), *parts.warnings, *_warnings(args, parts)),
        provenance=current_provenance(seed=args.seed, citations=parts.citations),
        settings=settings,
        text=texts,
    )


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #


def _check_alternative(type_name: str, side: str) -> None:
    """A one-sided ``alternative`` exists only for the t-test and the signed-rank
    test; ggstatsplot ignores it silently for the other types
    (ToothGrowth-greater-robust equals the two-sided case)."""
    if side == "two-sided" or type_name in ("parametric", "nonparametric"):
        return
    why = (
        "the bootstrap-t test of the trimmed mean is always two-sided"
        if type_name == "robust"
        else "the Bayes factor is always two-sided"
    )
    raise ValueError(
        f"alternative={side!r} applies to type='parametric' and "
        f"type='nonparametric' only: with type={type_name!r} {why} (ggstatsplot "
        "silently ignores alternative there). Remove alternative=, or use "
        "type='parametric' (the one-sample t-test) or type='nonparametric' (the "
        "signed-rank test) for a one-sided test."
    )


def _prepare(
    data: object,
    x: str,
    *,
    type: str = "parametric",
    test_value: float = 0.0,
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    alternative: str = "two-sided",
    seed: int = 0,
) -> tuple[OneData, _Args, int]:
    """Every argument checked and the data read: the analyzed values, the
    validated arguments and ``digits``. The public function runs the type's
    analysis on them; tests use it to check the data reading alone."""
    type_name = canonical_type(type)
    mu = check_finite_number(test_value, "test_value", example="0")
    level = check_conf_level(conf_level)
    n_digits = check_digits(digits)
    trim = check_tr(tr)
    prior_scale = check_positive(bf_prior, "bf_prior")
    show_bf = check_optional_bool(bf_message, "bf_message")
    subtitle_on = check_bool(results_subtitle, "results_subtitle")
    side = canonical_alternative(check_alternative(alternative))
    rng_seed = check_seed(seed)
    check_type_arguments(type_name, tr=tr, trim=trim, show_bf=show_bf)
    _check_alternative(type_name, side)
    check_bayes_caption(show_bf=show_bf, subtitle_on=subtitle_on)

    one = read_one(data, x)
    args = _Args(
        x=x,
        type=type_name,
        test_value=mu,
        conf_level=level,
        alternative=side,
        prior_scale=prior_scale,
        bf_message=show_bf,
        results_subtitle=subtitle_on,
        tr=trim,
        seed=rng_seed,
        passed={"bf_prior": bf_prior, "conf_level": conf_level, "seed": seed},
    )
    return one, args, n_digits


@checked
def histostats(
    data: object,
    x: str,
    *,
    type: str = "parametric",
    test_value: float = 0.0,
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    alternative: str = "two-sided",
    seed: int = 0,
) -> StatsResult:
    """Test whether one numeric column differs from ``test_value``; no plotting.

    The counterpart of ggstatsplot 1.1.4's ``gghistostats()``. Rows with a missing
    ``x`` are left out (and counted in ``n_dropped``); n is the number of values
    analyzed.

    The methods follow ``gghistostats()`` for each ``type`` (SPEC.md section 4):

    * "parametric" ("p"): the one-sample t-test with Hedges' g and the one-sample
      JZS Bayes factor caption. Centrality: the mean.
    * "nonparametric" ("np"): the one-sample Wilcoxon signed-rank test of
      x - test_value with the rank-biserial r. Centrality: the median.
    * "robust" ("r"): the bootstrap-t test of the ``tr``-trimmed mean (2,000
      resamples drawn with ``seed``), reporting the signed bootstrap t, its
      bootstrap p and the trimmed mean with its bootstrap-t interval as the
      effect size. Centrality: the trimmed mean.
    * "bayes" ("bf"): the one-sample JZS Bayes factor, reported as log_e(BF01)
      with the prior scale ``bf_prior``; ``test`` and ``effect_size`` are None.
      Centrality: the MAP estimate. Posterior estimates arrive in 0.2.

    At very large |t| R's noncentral-t interval of Hedges' g and BayesFactor's
    one-sample Bayes factor are measurably off; inferplot reports the accurate
    values (SPEC.md section 5).

    Args:
        data: a pandas, polars or pyarrow data frame.
        x: the numeric column to test.
        type: "parametric" ("p"), "nonparametric" ("np"), "robust" ("r") or
            "bayes" ("bf").
        test_value: the value x is tested against (ggstatsplot's ``test.value``);
            a finite number.
        conf_level: confidence level of the effect-size interval.
        digits: decimals in the subtitle, caption and labels.
        tr: trimming proportion of the robust type; another value than 0.2 with
            a type that trims nothing raises.
        bf_prior: scale of the Cauchy prior of the Bayes factor (0.707).
        bf_message: None draws the Bayes factor caption where ggstatsplot draws
            it (type "parametric"); True requires it; False never draws it.
        results_subtitle: whether the statistical results are reported as a
            subtitle (False also drops the Bayes factor caption).
        alternative: "two-sided", "less" or "greater"; one-sided only with types
            "parametric" and "nonparametric". "greater" tests whether x tends to
            be greater than ``test_value``, "less" whether it tends to be less.
        seed: random seed of the bootstrap (type "robust"), recorded in the
            result; another value with another type adds a ``warnings`` entry.

    Returns:
        A :class:`~inferplot.result.StatsResult` holding every number and text.

    Raises:
        TypeError, ValueError: a bad argument or unusable data (a column that is
            not numeric, fewer than two values, values that are all one number
            for the types that need a standard deviation, every value equal to
            ``test_value`` for the signed-rank test, too few values left after
            trimming); the message says how to fix the call.
        KeyError: a column that the data frame does not have.
    """
    one, args, n_digits = _prepare(
        data,
        x,
        type=type,
        test_value=test_value,
        conf_level=conf_level,
        digits=digits,
        tr=tr,
        bf_prior=bf_prior,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        alternative=alternative,
        seed=seed,
    )
    parts = _ANALYSES[args.type](one, args)
    return _assemble(one, args, parts, digits=n_digits)
