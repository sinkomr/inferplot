"""``betweenstats``: compare a numeric ``y`` across independent groups of ``x``.

The analysis that ``ip.analyze.betweenstats`` exposes. It reads the data through
:mod:`inferplot._data`, validates every argument with :mod:`inferplot._validate`,
computes every number with :mod:`inferplot.stats` and every text with
:mod:`inferplot.text`, and returns a :class:`~inferplot.result.StatsResult`. It never
imports matplotlib.

What it reproduces is ggstatsplot 1.1.4's ``ggbetweenstats()`` (SPEC.md section 4):

* two groups, type parametric: Welch's t-test, Hedges' g with a noncentral-t CI that
  follows ``alternative``, and the two-sided JZS Bayes factor caption (D28, D32);
* three or more groups, type parametric: Welch's ANOVA, partial omega squared with
  a one-sided noncentral-F CI, and Games-Howell pairwise comparisons adjusted with
  ``p_adjust_method`` over all k(k - 1)/2 pairs. The raw Games-Howell p-values come
  from an accurate studentized range tail, not from R's ``ptukey`` (D31);
* type nonparametric: the Mann-Whitney U test (normal approximation with
  continuity correction) with the rank-biserial r, or the Kruskal-Wallis test
  with rank epsilon squared (seeded percentile bootstrap CI) and Dunn's pairwise
  z; medians as centrality.
* type robust: Yuen's test with the AKP effect size, or the trimmed-means ANOVA
  with the explanatory effect size and lincon's pairwise comparisons (WRS2), all
  at trimming ``tr``, with seeded percentile bootstrap CIs; trimmed means as
  centrality. p-values are the accurate tails, not WRS2's 1 - cdf (D43).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from inferplot import stats
from inferplot._data import data_notes, read_between
from inferplot._validate import (
    canonical_pairwise_display,
    canonical_type,
    check_alternative,
    check_bool,
    check_conf_level,
    check_digits,
    check_optional_bool,
    check_p_adjust_method,
    check_positive,
    check_probability,
    check_seed,
    check_tr,
    checked,
)
from inferplot.result import (
    BayesFactor,
    Descriptive,
    EffectSize,
    PairwiseComparison,
    Settings,
    StatsResult,
    TestResult,
    current_provenance,
    tied_count,
)
from inferplot.stats.parametric import TINY
from inferplot.stats.robust import SMM_ROWS
from inferplot.text import build_texts, format_value, r_number_text

__all__ = ["betweenstats"]

FloatArray = npt.NDArray[np.float64]

# Short references for every method a result can use (SPEC.md section 4).
_WELCH_1947: Final = "Welch, B. L. (1947). Biometrika 34: 28-35 (Welch's t-test)."
_WELCH_1951: Final = "Welch, B. L. (1951). Biometrika 38: 330-336 (Welch's ANOVA)."
_HEDGES_1981: Final = (
    "Hedges, L. V. (1981). Journal of Educational Statistics 6: 107-128 (Hedges' g)."
)
_DELACRE_2021: Final = (
    "Delacre, M., Lakens, D., Ley, C., Liu, L. & Leys, C. (2021). International "
    "Review of Social Psychology 34: 1-21 (standardizer for unequal variances)."
)
_STEIGER_2004: Final = (
    "Steiger, J. H. (2004). Psychological Methods 9: 164-182 (noncentral confidence "
    "intervals for effect sizes)."
)
_OLEJNIK_ALGINA_2003: Final = (
    "Olejnik, S. & Algina, J. (2003). Psychological Methods 8: 434-447 (partial "
    "omega squared)."
)
_GAMES_HOWELL_1976: Final = (
    "Games, P. A. & Howell, J. F. (1976). Journal of Educational Statistics 1: "
    "113-125 (Games-Howell pairwise comparisons)."
)
_ROUDER_2009: Final = (
    "Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D. & Iverson, G. (2009). "
    "Psychonomic Bulletin & Review 16: 225-237 (JZS Bayes factor)."
)
_ADJUSTMENT_CITATIONS: Final[dict[str, str]] = {
    "holm": "Holm, S. (1979). Scandinavian Journal of Statistics 6: 65-70 "
    "(Holm adjustment).",
    "hochberg": "Hochberg, Y. (1988). Biometrika 75: 800-802 (Hochberg adjustment).",
    "hommel": "Hommel, G. (1988). Biometrika 75: 383-386 (Hommel adjustment).",
    "bonferroni": "Bonferroni, C. E. (1936). Pubblicazioni del R Istituto Superiore "
    "di Scienze Economiche e Commerciali di Firenze 8: 3-62 (Bonferroni adjustment).",
    "BH": "Benjamini, Y. & Hochberg, Y. (1995). Journal of the Royal Statistical "
    "Society B 57: 289-300 (false discovery rate adjustment).",
    "fdr": "Benjamini, Y. & Hochberg, Y. (1995). Journal of the Royal Statistical "
    "Society B 57: 289-300 (false discovery rate adjustment).",
    "BY": "Benjamini, Y. & Yekutieli, D. (2001). Annals of Statistics 29: 1165-1188 "
    "(Benjamini-Yekutieli adjustment).",
}

# ggstatsplot's default trimming; another value is allowed only where it is used.
_DEFAULT_TR: Final = 0.2
# Defaults of the arguments that can have no effect: a non-default value where it
# has none gets a warnings entry (not an error, so code that loops over data sets
# with two and with more groups keeps working; nothing reported would be wrong).
_DEFAULT_P_ADJUST: Final = "holm"
_DEFAULT_DISPLAY: Final = "significant"
_DEFAULT_ALPHA: Final = 0.05
_DEFAULT_BF_PRIOR: Final = 0.707
# Every pairwise Bayes factor uses this prior scale, whatever bf_prior says
# (statsExpressions' pairwise_comparisons(); SPEC.md section 4).
_PAIRWISE_BF_PRIOR: Final = 0.707

_HEDGES_J_ZERO_WARNING: Final = (
    "Hedges' g is 0 by construction at 1 degree of freedom. Welch's df is 1 here "
    "because one group has two values and the other group's variance is 0 or "
    "negligible next to it, so g and its confidence interval carry no information "
    "about the size of the difference."
)

# Below this Welch df Hedges' correction J is under 0.57 (J(2) = 0.564) and falls to
# 0 at df 1: g says little about the size of the difference.
_HEDGES_J_SMALL_DF: Final = 2.0


def _hedges_j_small_warning(df: float, j: float) -> str:
    """Welch's df between 1 and 2: g is the standardized difference times ``j``."""
    return (
        "Hedges' g is the standardized difference times a small-sample correction, "
        f"which is only {j:.2f} here: Welch's df is {df:.2f}, below 2 because one "
        "group has two values and most of the variance. g and its confidence "
        "interval are shrunk toward 0 by that factor and understate the size of "
        "the difference; collect more observations in the two-value group."
    )


# ggstatsplot 1.1.4 does not pass conf.level to WRS2's lincon, so the robust pairwise
# intervals are always 95% while the omnibus effect size follows conf_level
# (PlantGrowth-cl90-robust). Replicated, like Holm on Games-Howell (DECISIONS.md D53).
_LINCON_CONF_LEVEL: Final = 0.95


def _lincon_level_warning(conf_level: float) -> str:
    return (
        f"The pairwise confidence intervals are 95% intervals, not conf_level="
        f"{conf_level!r}, which applies to the effect size: ggstatsplot does not "
        "pass conf.level to the pairwise comparisons (WRS2's lincon), and inferplot "
        "follows it. Each pair records its conf_level."
    )


_BAYES_K3_WARNING: Final = (
    "The Bayes factor caption for three or more groups arrives in inferplot 0.2."
)


def _names(levels: Sequence[str]) -> str:
    """``'a'``, ``'a' and 'b'``, ``'a', 'b' and 'c'``."""
    quoted = [repr(level) for level in levels]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _group_word(levels: Sequence[str]) -> str:
    return "group" if len(levels) == 1 else "groups"


# What a group needs next to the others, in plain words. The statistics run on y
# times one power of two that brings the largest |y| into [1/2, 1); there a
# variance term v / n below the smallest normal double (2^-1022) cannot be computed
# accurately. In units of y that is roughly a standard deviation below 1e-154 times
# the largest |y| (between 1.5e-154 and 3e-154 times sqrt(n)).
_TOO_LITTLE: Final = (
    "vary by less than double precision can resolve next to the largest |y| (a "
    "standard deviation below roughly 1e-154 times it)"
)
_NO_RESCALING: Final = (
    "Rescaling y does not help, since it multiplies the spread and the largest |y| "
    "alike; check y for entry or unit errors"
)


def _scaled_variances(values: Sequence[FloatArray]) -> list[float]:
    """Each group's variance in the common scale of all groups, as the methods
    compute it (the expressions below repeat theirs, so the checks agree bit for
    bit)."""
    scale = stats.common_scale(*values)
    return [stats.scaled_variance(v * scale) for v in values]


def _check_groups(
    x: str, levels: Sequence[str], values: Sequence[FloatArray], *, bayes: bool
) -> None:
    """Raise ``ValueError`` naming the levels the parametric methods cannot use
    (``bayes``: the two-group Bayes factor, on Student's t, will be computed)."""
    small = [
        f"{level!r} (n = {v.size})"
        for level, v in zip(levels, values, strict=True)
        if v.size < 2
    ]
    if small:
        raise ValueError(
            f"Every group of x={x!r} needs at least 2 observations to estimate its "
            f"variance; {'this group has' if len(small) == 1 else 'these groups have'}"
            f" fewer: {', '.join(small)}. Remove {'it' if len(small) == 1 else 'them'}"
            " from the data or collect more observations."
        )
    huge = [
        level
        for level, v in zip(levels, values, strict=True)
        if math.isinf(float(np.max(v)) - float(np.min(v)))
    ]
    if huge:
        raise ValueError(
            f"The values of y in the {_group_word(huge)} {_names(huge)} of x={x!r} "
            "range more widely than a double can represent: their largest minus "
            "their smallest value exceeds the largest double (about 1.8e308). "
            "Rescale y, for example by changing its units or dividing it by 1e10."
        )
    # Constant means every value equal, not a variance that rounds to 0.
    constant = [
        level
        for level, v in zip(levels, values, strict=True)
        if float(np.ptp(v)) == 0.0
    ]
    if len(levels) == 2 and len(constant) == 2:
        raise ValueError(
            f"Both groups of x={x!r} ({_names(levels)}) have no variability (every "
            "value within each group is the same), so Welch's t-test has no standard "
            "error and cannot be computed."
        )
    if len(levels) >= 3 and constant:
        raise ValueError(
            f"Welch's ANOVA and the Games-Howell comparisons weight each group by "
            f"its variance, which is 0 for the {_group_word(constant)} "
            f"{_names(constant)} of x={x!r} (every value is the same). Remove "
            f"{'that group' if len(constant) == 1 else 'those groups'} or analyze "
            "the groups that vary."
        )
    variances = _scaled_variances(values)
    if len(levels) == 2:
        # One group is constant (both constant raised above) and the other varies
        # too little next to it: the standard error is below what doubles resolve.
        # Only possible when the group with the largest |y| is the constant one.
        (v_a, v_b), (n_a, n_b) = variances, (values[0].size, values[1].size)
        varying = [level for level in levels if level not in constant]
        welch_term = v_a / n_a + v_b / n_b  # as stats.welch_t_scaled
        pooled = ((n_a - 1) * v_a + (n_b - 1) * v_b) / (n_a + n_b - 2)
        student_term = pooled * (1.0 / n_a + 1.0 / n_b)  # as stats.student_t
        if welch_term < TINY or (bayes and student_term < TINY):
            which = (
                "Welch's t-test"
                if welch_term < TINY
                else "the Bayes factor (on Student's t)"
            )
            fix = (
                ""
                if welch_term < TINY
                else " Or leave the Bayes factor out with bf_message=False."
            )
            raise ValueError(
                f"The values of y in the {_group_word(varying)} {_names(varying)} of "
                f"x={x!r} {_TOO_LITTLE}, and the {_group_word(constant)} "
                f"{_names(constant)} has no variability, so {which} has no accurate "
                f"standard error. {_NO_RESCALING}.{fix}"
            )
    if len(levels) >= 3:
        # Welch's ANOVA weights each group by n / v: a variance term below 2^-1022
        # in the common scale would give an inaccurate or infinite weight.
        tiny = [
            level
            for level, v, sample in zip(levels, variances, values, strict=True)
            # as stats.parametric.checked_scaled_variances
            if not v / sample.size >= TINY
        ]
        if tiny:
            whose = (
                "that group by its variance"
                if len(tiny) == 1
                else "those groups by their variances"
            )
            raise ValueError(
                f"The values of y in the {_group_word(tiny)} {_names(tiny)} of "
                f"x={x!r} {_TOO_LITTLE}, so Welch's ANOVA and the Games-Howell "
                f"comparisons cannot weight {whose}. {_NO_RESCALING}, or analyze "
                "groups of similar magnitude separately."
            )


def _no_effect_warnings(
    *,
    x: str,
    k: int,
    type_name: str,
    bayes_computed: bool,
    subtitle_on: bool,
    show_bf: bool | None,
    passed: dict[str, object],
    canonical: dict[str, object],
) -> list[str]:
    """One plain-English entry per argument that was changed from its default but
    has no effect on these data (``passed``: the values as given; ``canonical``:
    validated, to compare with the defaults)."""
    out: list[str] = []
    if k == 2:
        reason = (
            f"x={x!r} has 2 groups, so there are no pairwise comparisons to {{what}}."
        )
        for name, default, what in (
            ("p_adjust_method", _DEFAULT_P_ADJUST, "adjust"),
            ("pairwise_display", _DEFAULT_DISPLAY, "display"),
            ("pairwise_alpha", _DEFAULT_ALPHA, "call significant"),
        ):
            if canonical[name] != default:
                out.append(
                    f"{name}={passed[name]!r} has no effect: "
                    + reason.format(what=what)
                )
    if not bayes_computed and canonical["bf_prior"] != _DEFAULT_BF_PRIOR:
        if type_name == "bayes" and k >= 3:
            # Bayes factors are computed, for every pair, at the fixed scale.
            scale = r_number_text(_PAIRWISE_BF_PRIOR)
            out.append(
                f"bf_prior={passed['bf_prior']!r} has no effect: the pairwise Bayes "
                f"factors always use the prior scale {scale}, as in ggstatsplot; the "
                "Bayesian ANOVA arrives in inferplot 0.2."
            )
            return out
        if type_name in ("nonparametric", "robust"):
            why = f"type={type_name!r} has no Bayes factor"
        elif k >= 3:
            why = (
                f"x={x!r} has {k} groups, and the Bayes factor for three or more "
                "groups arrives in inferplot 0.2"
            )
        elif not subtitle_on:
            why = "results_subtitle=False leaves the Bayes factor out"
        else:
            why = f"bf_message={show_bf!r} leaves the Bayes factor out"
        out.append(
            f"bf_prior={passed['bf_prior']!r} has no effect: no Bayes factor is "
            f"computed ({why})."
        )
    return out


def _alternative(value: str) -> stats.Alternative:
    for allowed in stats.ALTERNATIVES:
        if value == allowed:
            return allowed
    raise AssertionError(f"unvalidated alternative {value!r}")  # pragma: no cover


def _adjust_method(value: str) -> stats.PAdjustMethod:
    for allowed in stats.P_ADJUST_METHODS:
        if value == allowed:
            return allowed
    raise AssertionError(f"unvalidated p_adjust_method {value!r}")  # pragma: no cover


def _displayed(display: str, significant: bool) -> bool:
    if display == "all":
        return True
    if display == "significant":
        return significant
    if display == "non-significant":
        return not significant
    return False  # "none": computed and returned, never drawn


def _games_howell_pairs(
    levels: Sequence[str],
    values: Sequence[FloatArray],
    *,
    p_adjust_method: stats.PAdjustMethod,
    pairwise_alpha: float,
    pairwise_display: str,
) -> tuple[PairwiseComparison, ...]:
    """Games-Howell rows as ggstatsplot 1.1.4 reports them.

    Orientation: group1 is the earlier level in the analysis order and the statistic
    is q = (mean(group2) - mean(group1)) / SE, which ``stats.games_howell`` returns
    for each pair i < j. Row order: statsExpressions 2.2.1's
    ``pairwise_comparisons()`` (R/pairwise-comparisons.R, MIT) converts the factor
    columns to character and calls ``dplyr::arrange(group1, group2)``, so rows are
    sorted by group1 and then group2 as text; dplyr 1.1 sorts character vectors in
    the C locale, which is the code-point order of Python's ``sorted()``. Pairs are
    unique, so there are no ties. ggstatsplot 1.1.4's ``.ggsignif_adder()``
    (R/ggbetweenstats-helpers.R, MIT) keeps the pairs with ``p.value <
    pairwise.alpha`` ("significant") or ``>=`` ("non-significant"), where
    ``p.value`` is the adjusted p, and arranges them the same way before drawing
    the brackets. The adjustment runs over all pairs and does not depend on their
    order.
    """
    rows: list[PairwiseComparison] = []
    for pair in stats.games_howell(values, p_adjust_method=p_adjust_method):
        significant = pair.p_adjusted < pairwise_alpha
        rows.append(
            PairwiseComparison(
                group1=levels[pair.i],
                group2=levels[pair.j],
                statistic_name="q",
                statistic=pair.statistic,
                df=pair.df,
                p_value=pair.p_value,
                p_adjusted=pair.p_adjusted,
                adjust_method=p_adjust_method,
                test="games_howell",
                test_label="Games-Howell",
                significant=significant,
                displayed=_displayed(pairwise_display, significant),
            )
        )
    rows.sort(key=lambda row: (row.group1, row.group2))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _Args:
    """The validated arguments every type's analysis reads."""

    x: str
    type: str
    conf_level: float
    alternative: stats.Alternative
    p_adjust_method: stats.PAdjustMethod
    pairwise_alpha: float
    pairwise_display: str
    prior_scale: float
    bf_message: bool | None
    results_subtitle: bool
    tr: float
    seed: int


@dataclass(frozen=True, slots=True)
class _Parts:
    """What one type's analysis computed."""

    test: TestResult | None
    effect_size: EffectSize | None
    bayes: BayesFactor | None
    pairwise: tuple[PairwiseComparison, ...]
    centrality: str
    centers: tuple[float | None, ...]
    warnings: tuple[str, ...]
    citations: tuple[str, ...]


def _sorted_pairs(rows: list[PairwiseComparison]) -> tuple[PairwiseComparison, ...]:
    """Rows sorted by group1 and then group2 as text in the C locale (code-point
    order, Python's ``sorted()``), as statsExpressions 2.2.1's
    ``pairwise_comparisons()`` arranges them (see ``_games_howell_pairs``)."""
    return tuple(sorted(rows, key=lambda row: (row.group1, row.group2)))


def _adjustment_citation(method: str) -> list[str]:
    return [_ADJUSTMENT_CITATIONS[method]] if method in _ADJUSTMENT_CITATIONS else []


# --------------------------------------------------------------------------- #
# type parametric
# --------------------------------------------------------------------------- #


def _parametric(
    levels: Sequence[str], values: Sequence[FloatArray], args: _Args
) -> _Parts:
    """Welch's t or ANOVA, Hedges' g or partial omega squared, Games-Howell, and
    the two-group JZS Bayes factor caption (D28, D32)."""
    side, level = args.alternative, args.conf_level
    warnings: list[str] = []
    citations: list[str] = []
    bayes: BayesFactor | None = None
    pairwise: tuple[PairwiseComparison, ...] = ()
    if len(levels) == 2:
        a, b = values
        # Welch's t in the samples' common scale: t, df and p are reported, never
        # the mean difference, which can exceed the largest double in y units.
        welch = stats.welch_t_scaled(a, b)
        g = stats.hedges_g(a, b, conf_level=level, alternative=side)
        test = TestResult(
            name="welch_t",
            label="Welch's t-test",
            method="Welch Two Sample t-test",
            statistic_name="t",
            statistic=welch.t,
            df=None,
            df_error=welch.df,
            p_value=stats.t_p_value(welch.t, welch.df, side),
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
        citations += [_WELCH_1947, _HEDGES_1981, _DELACRE_2021, _STEIGER_2004]
        correction = stats.hedges_j(welch.df)
        if correction == 0.0:
            warnings.append(_HEDGES_J_ZERO_WARNING)
        elif welch.df < _HEDGES_J_SMALL_DF:
            warnings.append(_hedges_j_small_warning(welch.df, correction))
        if args.results_subtitle and args.bf_message is not False:
            bf10, log_bf01 = stats.jzs_bf10_two_sample(
                a, b, prior_scale=args.prior_scale
            )
            bayes = BayesFactor(
                bf10=bf10,
                log_bf01=log_bf01,
                prior="cauchy",
                prior_scale=args.prior_scale,
                test="jzs_two_sample",
            )
            citations.append(_ROUDER_2009)
            if side != "two-sided":
                warnings.append(
                    "The Bayes factor in the caption is two-sided: it weighs any "
                    f"difference between the groups, while alternative={side!r} "
                    "applies to the t-test and the Hedges' g interval only (as in "
                    "ggstatsplot)."
                )
    else:
        anova = stats.welch_anova(values)
        omega = stats.partial_omega_squared(
            anova.f, anova.df1, anova.df2, conf_level=level
        )
        test = TestResult(
            name="welch_anova",
            label="Welch's ANOVA",
            method="One-way analysis of means (not assuming equal variances)",
            statistic_name="F",
            statistic=anova.f,
            df=anova.df1,
            df_error=anova.df2,
            p_value=anova.p_value,
            alternative="two-sided",
        )
        effect_size = EffectSize(
            name="partial_omega_squared",
            label="partial omega squared",
            estimate=omega.estimate,
            conf_level=omega.conf_level,
            conf_low=omega.conf_low,
            conf_high=omega.conf_high,
            ci_method="noncentral F",
            sidedness=omega.sidedness,
        )
        pairwise = _games_howell_pairs(
            levels,
            values,
            p_adjust_method=args.p_adjust_method,
            pairwise_alpha=args.pairwise_alpha,
            pairwise_display=args.pairwise_display,
        )
        citations += [_WELCH_1951, _OLEJNIK_ALGINA_2003, _STEIGER_2004]
        citations.append(_GAMES_HOWELL_1976)
        citations += _adjustment_citation(args.p_adjust_method)
        if args.results_subtitle and args.bf_message is None:
            warnings.append(_BAYES_K3_WARNING)
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        centrality="mean",
        centers=tuple(stats.mean(v) for v in values),
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


# --------------------------------------------------------------------------- #
# type nonparametric
# --------------------------------------------------------------------------- #

_MANN_WHITNEY_1947: Final = (
    "Mann, H. B. & Whitney, D. R. (1947). Annals of Mathematical Statistics 18: "
    "50-60 (Mann-Whitney U test)."
)
_KERBY_2014: Final = (
    "Kerby, D. S. (2014). Comprehensive Psychology 3: 11.IT.3.1 (rank-biserial "
    "correlation)."
)
_KRUSKAL_WALLIS_1952: Final = (
    "Kruskal, W. H. & Wallis, W. A. (1952). Journal of the American Statistical "
    "Association 47: 583-621 (Kruskal-Wallis test)."
)
_TOMCZAK_2014: Final = (
    "Tomczak, M. & Tomczak, E. (2014). Trends in Sport Sciences 21: 19-25 (rank "
    "epsilon squared)."
)
_DUNN_1964: Final = (
    "Dunn, O. J. (1964). Technometrics 6: 241-252 (Dunn's pairwise comparisons)."
)
_EFRON_TIBSHIRANI_1993: Final = (
    "Efron, B. & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. "
    "Chapman & Hall (percentile bootstrap intervals)."
)


def _check_ranks(x: str, values: Sequence[FloatArray]) -> None:
    """The rank tests need at least two distinct values in all the data."""
    lows = [float(np.min(v)) for v in values]
    if min(lows) == max(float(np.max(v)) for v in values):
        raise ValueError(
            f"Every value of y is the same ({lows[0]!r}) in all groups of "
            f"x={x!r}, so the rank tests have nothing to rank: every observation "
            "ties with every other. There is no difference to test."
        )


def _nonparametric(
    levels: Sequence[str], values: Sequence[FloatArray], args: _Args
) -> _Parts:
    """Mann-Whitney U with the rank-biserial r, or Kruskal-Wallis with rank
    epsilon squared (bootstrap CI) and Dunn's pairwise z (SPEC.md section 4)."""
    _check_ranks(args.x, values)
    side, level = args.alternative, args.conf_level
    pairwise: tuple[PairwiseComparison, ...] = ()
    warnings: list[str] = []
    if len(levels) == 2:
        a, b = values
        test_result = stats.mann_whitney(a, b, alternative=side)
        r = stats.rank_biserial(a, b, conf_level=level, alternative=side)
        test = TestResult(
            name="mann_whitney",
            label="Mann-Whitney U test",
            method="Wilcoxon rank sum test",
            statistic_name="W",
            statistic=test_result.statistic,
            df=None,
            df_error=None,
            p_value=test_result.p_value,
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
        citations = [_MANN_WHITNEY_1947, _KERBY_2014]
    else:
        kw = stats.kruskal_wallis(values)
        try:
            epsilon = stats.rank_epsilon_squared(
                values, conf_level=level, seed=args.seed
            )
        except ValueError as error:
            raise ValueError(
                f"The bootstrap interval of rank epsilon squared cannot be computed "
                f"for these groups of x={args.x!r}: {error}. Use type='parametric' "
                "or type='robust', or collect more varied data."
            ) from None
        test = TestResult(
            name="kruskal_wallis",
            label="Kruskal-Wallis test",
            method="Kruskal-Wallis rank sum test",
            statistic_name="chi2",
            statistic=kw.statistic,
            df=None,
            df_error=kw.df,
            p_value=kw.p_value,
            alternative="two-sided",
        )
        effect_size = EffectSize(
            name="rank_epsilon_squared",
            label="rank epsilon squared",
            estimate=epsilon.estimate,
            conf_level=epsilon.conf_level,
            conf_low=epsilon.conf_low,
            conf_high=epsilon.conf_high,
            ci_method="percentile bootstrap",
            sidedness=epsilon.sidedness,
            bootstrap_resamples=epsilon.resamples,
            bootstrap_seed=epsilon.seed,
            undefined_resamples=epsilon.undefined_resamples,
            infinite_resamples=epsilon.infinite_resamples,
        )
        pairwise = _dunn_pairs(levels, values, args)
        if epsilon.undefined_resamples:
            warnings.append(_tied_resamples_warning(epsilon))
        citations = [
            _KRUSKAL_WALLIS_1952,
            _TOMCZAK_2014,
            _EFRON_TIBSHIRANI_1993,
            _DUNN_1964,
            *_adjustment_citation(args.p_adjust_method),
        ]
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=None,
        pairwise=pairwise,
        centrality="median",
        centers=tuple(stats.median(v) for v in values),
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


def _tied_resamples_warning(epsilon: stats.BootstrapEffectSizeCI) -> str:
    """D54: the resamples left out of rank epsilon squared's interval."""
    tied, total = epsilon.undefined_resamples, epsilon.resamples
    return (
        f"{tied:,} of {total:,} bootstrap resamples have every value tied, where "
        "rank epsilon squared is undefined; its confidence interval comes from the "
        f"other {total - tied:,}, as in ggstatsplot."
    )


def _dunn_pairs(
    levels: Sequence[str], values: Sequence[FloatArray], args: _Args
) -> tuple[PairwiseComparison, ...]:
    """Dunn's z for every pair i < j in level order (unsigned, as ggstatsplot
    reports it), raw p adjusted with ``p_adjust_method`` over all pairs, rows in
    ggstatsplot's order (``_sorted_pairs``)."""
    pairs = stats.dunn(values)
    adjusted = stats.p_adjust([pair.p_value for pair in pairs], args.p_adjust_method)
    rows: list[PairwiseComparison] = []
    for pair, p_adjusted in zip(pairs, adjusted, strict=True):
        significant = p_adjusted < args.pairwise_alpha
        rows.append(
            PairwiseComparison(
                group1=levels[pair.i],
                group2=levels[pair.j],
                statistic_name="z",
                statistic=pair.statistic,
                df=None,
                p_value=pair.p_value,
                p_adjusted=p_adjusted,
                adjust_method=args.p_adjust_method,
                test="dunn",
                test_label="Dunn",
                significant=significant,
                displayed=_displayed(args.pairwise_display, significant),
            )
        )
    return _sorted_pairs(rows)


# --------------------------------------------------------------------------- #
# type robust
# --------------------------------------------------------------------------- #

_YUEN_1974: Final = (
    "Yuen, K. K. (1974). Biometrika 61: 165-170 (Yuen's trimmed-means t-test)."
)
_WILCOX_2017: Final = (
    "Wilcox, R. R. (2017). Introduction to Robust Estimation and Hypothesis "
    "Testing, 4th ed. Academic Press (trimmed-means ANOVA, lincon)."
)
_AKP_2005: Final = (
    "Algina, J., Keselman, H. J. & Penfield, R. D. (2005). Psychological Methods "
    "10: 317-328 (AKP robust standardized difference)."
)
_WILCOX_TIAN_2011: Final = (
    "Wilcox, R. R. & Tian, T. S. (2011). Journal of Applied Statistics 38: "
    "1359-1368 (explanatory measure of effect size)."
)
_STOLINE_URY_1979: Final = (
    "Stoline, M. R. & Ury, H. K. (1979). Technometrics 21: 87-93 (Studentized "
    "maximum modulus critical values)."
)

_GROUP_NUMBER = re.compile(r"\bgroups? (\d+)(?: and (\d+))?|\bsample ([ab])\b")


def _percent(tr: float) -> str:
    """``tr`` as a percentage the way R prints it: 0.2 -> "20", 0.125 -> "12.5"."""
    return r_number_text(tr * 100)


def _named_groups(message: str, levels: Sequence[str], x: str) -> str:
    """A statistics-layer message with its group numbers ("group 2", "sample a")
    replaced by the level names of ``x``."""

    def name(match: re.Match[str]) -> str:
        if match.group(3) is not None:  # "sample a" or "sample b": two groups
            if len(levels) != 2:
                return match.group(0)
            return f"the group {levels['ab'.index(match.group(3))]!r} of x={x!r}"
        first = levels[int(match.group(1)) - 1]
        if match.group(2) is None:
            return f"the group {first!r} of x={x!r}"
        second = levels[int(match.group(2)) - 1]
        return f"the groups {first!r} and {second!r} of x={x!r}"

    # One pass, so a level name put in is never read again as a group number or
    # as "sample b"; whole words only ("resample beyond" holds "sample b").
    out = _GROUP_NUMBER.sub(name, message)
    if f"x={x!r}" not in out:
        stem = out.removesuffix(".")
        out = f"{stem} (the groups of x={x!r})" + out[len(stem) :]
    return out[0].upper() + out[1:]


def _with_advice(message: str, advice: str, *, given: str) -> str:
    """``message`` ended by one period, then ``advice`` unless the message already
    gives it (contains ``given``)."""
    text = message.rstrip().rstrip(".") + "."
    return text if given in text else f"{text} {advice}"


def _winsorized_constant(values: FloatArray, tr: float) -> bool:
    """Whether winsorizing g = floor(tr n) values at each end leaves one distinct
    value: the (g + 1)-th smallest equals the (g + 1)-th largest."""
    ordered = values.copy()
    ordered.sort()
    g = math.floor(tr * ordered.size)
    return bool(ordered[g] == ordered[ordered.size - g - 1])


def _check_robust(
    x: str, levels: Sequence[str], values: Sequence[FloatArray], args: _Args
) -> None:
    """Raise plain errors, naming the levels, for data the trimmed-means methods
    cannot use (SPEC.md section 4), before any of them runs."""
    tr, pct = args.tr, _percent(args.tr)
    # The percentile rules of the two effect sizes (akp_effect: floor; t1way's
    # explanatory measure: round) with this many resamples beyond each bound.
    resamples = stats.BOOTSTRAP_RESAMPLES
    beyond = (1.0 - args.conf_level) / 2.0 * resamples
    if (math.floor(beyond) if len(levels) == 2 else round(beyond)) < 1:
        raise ValueError(
            f"conf_level={args.conf_level!r} is too close to 1 for the bootstrap "
            f"confidence interval of the robust effect size: with {resamples:,} "
            "resamples its bounds would be the most extreme resampled values "
            "whatever the level, so the interval would stop widening. Pass "
            "conf_level=0.999 or lower, or use type='parametric', whose intervals "
            "are not resampled."
        )
    small = [
        f"{level!r} (n = {v.size})"
        for level, v in zip(levels, values, strict=True)
        if v.size < 2
    ]
    if small:
        has = "this group has" if len(small) == 1 else "these groups have"
        raise ValueError(
            f"Every group of x={x!r} needs at least 2 observations for the "
            f"trimmed-means methods; {has} fewer: {', '.join(small)}. Remove "
            f"{'it' if len(small) == 1 else 'them'} from the data or collect more "
            "observations."
        )
    # WRS2 trims g = floor(tr n) values at each end and needs 2 left.
    short = [
        f"{level!r} (n = {v.size})"
        for level, v in zip(levels, values, strict=True)
        if v.size - 2 * math.floor(tr * v.size) < 2
    ]
    if short:
        raise ValueError(
            f"Trimming {pct}% of the values from each end (tr={tr!r}) leaves fewer "
            f"than 2 values in the {_group_word(short)} {', '.join(short)} of "
            f"x={x!r}. Pass a smaller tr (for example tr=0.1), or collect more "
            "observations."
        )
    # The winsorized variance is 0 exactly when the winsorized values are all the
    # same, which is asked of the values themselves: their squares underflow or
    # overflow at extreme units (y in units of 1e-165 or 1e156).
    flat = [
        level
        for level, v in zip(levels, values, strict=True)
        if _winsorized_constant(v, tr)
    ]
    k = len(levels)
    why = (
        f"after winsorizing {pct}% of the values at each end every remaining value "
        "is the same"
    )
    if k >= 3 and flat:
        raise ValueError(
            f"The trimmed-means ANOVA weights each group by its winsorized variance, "
            f"which is 0 for the {_group_word(flat)} {_names(flat)} of x={x!r} "
            f"({why}). Remove {'that group' if len(flat) == 1 else 'those groups'}, "
            "pass a smaller tr, or use type='nonparametric'."
        )
    if k == 2 and len(flat) == 2:
        raise ValueError(
            f"Both groups of x={x!r} ({_names(levels)}) have a winsorized variance "
            f"of 0 ({why}), so Yuen's test has no standard error. Pass a smaller "
            "tr, or use type='nonparametric'."
        )
    if k == 2 and levels[0] in flat:
        raise ValueError(
            f"The AKP effect size divides by the winsorized standard deviation of "
            f"the first group, {levels[0]!r} of x={x!r}, which is 0 ({why}). Put "
            "the other group first with order=, pass a smaller tr, or use "
            "type='nonparametric'."
        )
    if k >= 3:
        # Any number of pairs (SPEC sections 2 and 4): above 28 pairs WRS2 simulates
        # its critical values and inferplot computes the exact quantile, which
        # exists from df 1, but neither is defined where lincon's table is not.
        low_df = [
            f"{levels[i]!r} and {levels[j]!r}"
            for i in range(k)
            for j in range(i + 1, k)
            if stats.yuen_t(values[i], values[j], tr=tr).df < SMM_ROWS[0]
        ]
        if low_df:
            raise NotImplementedError(
                f"The pairwise comparison of {', '.join(low_df)} (x={x!r}) has "
                f"fewer than {SMM_ROWS[0]} degrees of freedom, where the "
                "critical values of WRS2's lincon table stop (ggstatsplot stops "
                "with an error there with up to 8 groups, and inferplot with any "
                "number). Collect more observations per group, or use "
                "type='nonparametric'."
            )


def _robust(levels: Sequence[str], values: Sequence[FloatArray], args: _Args) -> _Parts:
    """Yuen's test with the AKP effect size, or the trimmed-means ANOVA with the
    explanatory effect size and lincon's pairwise comparisons, all at trimming
    ``tr``; bootstrap CIs seeded with ``seed`` (SPEC.md section 4)."""
    x, tr, level = args.x, args.tr, args.conf_level
    _check_robust(x, levels, values, args)
    trimmed = f"{_percent(tr)}% trimmed means"
    pairwise: tuple[PairwiseComparison, ...] = ()
    try:
        if len(levels) == 2:
            a, b = values
            yuen = stats.yuen_t(a, b, tr=tr)
            akp = stats.akp_effect(a, b, tr=tr, conf_level=level, seed=args.seed)
            test = TestResult(
                name="yuen_t",
                label=f"Yuen's t-test ({trimmed})",
                method="Yuen's test on trimmed means for independent samples",
                statistic_name="t",
                statistic=yuen.t,
                df=None,
                df_error=yuen.df,
                p_value=yuen.p_value,
                alternative="two-sided",
            )
            effect_size = EffectSize(
                name="akp_delta",
                label="AKP robust standardized difference",
                estimate=akp.estimate,
                conf_level=akp.conf_level,
                conf_low=akp.conf_low,
                conf_high=akp.conf_high,
                ci_method="percentile bootstrap",
                sidedness=akp.sidedness,
                bootstrap_resamples=akp.resamples,
                bootstrap_seed=akp.seed,
                undefined_resamples=akp.undefined_resamples,
                infinite_resamples=akp.infinite_resamples,
            )
            citations = [_YUEN_1974, _AKP_2005, _EFRON_TIBSHIRANI_1993]
        else:
            anova = stats.trimmed_means_anova(values, tr=tr)
            xi = stats.explanatory_effect(
                values, tr=tr, conf_level=level, seed=args.seed
            )
            test = TestResult(
                name="trimmed_means_anova",
                label=f"Trimmed-means ANOVA ({trimmed})",
                method="A heteroscedastic one-way ANOVA for trimmed means",
                statistic_name="F",
                statistic=anova.f,
                df=anova.df1,
                df_error=anova.df2,
                p_value=anova.p_value,
                alternative="two-sided",
            )
            effect_size = EffectSize(
                name="explanatory_xi",
                label="explanatory effect size xi",
                estimate=xi.estimate,
                conf_level=xi.conf_level,
                conf_low=xi.conf_low,
                conf_high=xi.conf_high,
                ci_method="percentile bootstrap",
                sidedness=xi.sidedness,
                bootstrap_resamples=xi.resamples,
                bootstrap_seed=xi.seed,
                undefined_resamples=xi.undefined_resamples,
                infinite_resamples=xi.infinite_resamples,
            )
            pairwise = _lincon_pairs(levels, values, args)
            citations = [
                _WILCOX_2017,
                _WILCOX_TIAN_2011,
                _EFRON_TIBSHIRANI_1993,
                _YUEN_1974,
                _STOLINE_URY_1979,
                *_adjustment_citation(args.p_adjust_method),
            ]
    except ValueError as error:
        raise ValueError(
            _with_advice(
                _named_groups(str(error), levels, x),
                "Pass a smaller tr, or use type='nonparametric'.",
                given="Pass a smaller tr",
            )
        ) from None
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=None,
        pairwise=pairwise,
        centrality="trimmed_mean",
        centers=tuple(stats.trimmed_mean(v, tr=tr) for v in values),
        warnings=(
            *_infinite_effect_warnings(
                effect_size,
                two_groups=len(levels) == 2,
                # t1way's estimate with unequal sizes is the bootstrap median.
                median_estimate=len(levels) > 2 and len({v.size for v in values}) > 1,
            ),
            *(
                (_lincon_level_warning(level),)
                if pairwise and level != _LINCON_CONF_LEVEL
                else ()
            ),
        ),
        citations=tuple(citations),
    )


def _infinite_effect_warnings(
    effect_size: EffectSize, *, two_groups: bool, median_estimate: bool
) -> list[str]:
    """The robust effect size or its bootstrap values are infinite: a winsorized
    variance of 0 under trimmed means that differ. WRS2 does the same arithmetic,
    so the values are kept (DECISIONS.md log); these warnings say how many and
    why, and ``check()`` reports ``infinite_effect_size``."""
    why = (
        "the first group's winsorized variance is 0 while the trimmed means differ"
        if two_groups
        else "the pooled winsorized variance is 0 while the trimmed means differ"
    )
    out: list[str] = []
    if math.isinf(effect_size.estimate):
        where = (
            "in more than half of the bootstrap resamples, whose median it is "
            "(unequal group sizes)"
            if median_estimate
            else "in the data"
        )
        out.append(f"The {effect_size.label} is infinite: {why} {where}.")
    infinite = effect_size.infinite_resamples or 0
    if infinite:
        total = effect_size.bootstrap_resamples or 0
        out.append(
            f"{infinite:,} of {total:,} bootstrap resamples give an infinite "
            f"{effect_size.label}, because in them {why}. They are kept, so "
            "the confidence interval "
            f"[{format_value(effect_size.conf_low)}, "
            f"{format_value(effect_size.conf_high)}] can be unbounded. Pass a smaller "
            "tr, or use type='nonparametric'."
        )
    return out


def _lincon_pairs(
    levels: Sequence[str], values: Sequence[FloatArray], args: _Args
) -> tuple[PairwiseComparison, ...]:
    """lincon's comparison of every pair i < j in level order: the trimmed-mean
    difference group1 - group2 with its CI, Yuen's t (signed, as the difference)
    and df, and the raw p adjusted with ``p_adjust_method`` over all pairs; rows in
    ggstatsplot's order (``_sorted_pairs``)."""
    pairs = stats.lincon(
        values,
        tr=args.tr,
        conf_level=_LINCON_CONF_LEVEL,
        p_adjust_method=args.p_adjust_method,
    )
    rows: list[PairwiseComparison] = []
    for pair in pairs:
        significant = pair.p_adjusted < args.pairwise_alpha
        rows.append(
            PairwiseComparison(
                group1=levels[pair.i],
                group2=levels[pair.j],
                statistic_name="t",
                statistic=pair.t,
                df=pair.df,
                p_value=pair.p_value,
                p_adjusted=pair.p_adjusted,
                adjust_method=args.p_adjust_method,
                test="yuen_trimmed_means",
                test_label="Yuen's trimmed means",
                significant=significant,
                displayed=_displayed(args.pairwise_display, significant),
                estimate=pair.estimate,
                conf_level=_LINCON_CONF_LEVEL,
                conf_low=pair.conf_low,
                conf_high=pair.conf_high,
            )
        )
    return _sorted_pairs(rows)


# --------------------------------------------------------------------------- #
# type bayes
# --------------------------------------------------------------------------- #

_SHEATHER_JONES_1991: Final = (
    "Sheather, S. J. & Jones, M. C. (1991). Journal of the Royal Statistical "
    "Society B 53: 683-690 (kernel density bandwidth of the MAP estimate)."
)
_CHECK_DATA: Final = "Check the data for entry or unit errors."
_POSTERIOR_WARNING: Final = (
    "The posterior difference and its credible interval, which ggstatsplot adds to "
    "the Bayes factor subtitle, arrive in inferplot 0.2."
)


def _map_estimates(
    x: str, levels: Sequence[str], values: Sequence[FloatArray]
) -> tuple[tuple[float | None, ...], tuple[str, ...]]:
    """Each group's MAP estimate (the centrality of type bayes). A group without a
    kernel density (the Sheather-Jones bandwidth fails, as on a zero interquartile
    range) has None, no label is drawn for it, and a warning says why; ggstatsplot
    prints an empty label there (probe-map, DECISIONS.md D55)."""
    out: list[float | None] = []
    warnings: list[str] = []
    for level, v in zip(levels, values, strict=True):
        try:
            out.append(stats.map_estimate(v).estimate)
        except ValueError as error:
            out.append(None)
            warnings.append(
                f"The group {level!r} of x={x!r} (n = {v.size}) has no MAP estimate, "
                f"so no centrality label is drawn for it: {error}. ggstatsplot "
                "prints an empty label there. Use type='nonparametric' to show "
                "medians."
            )
    return tuple(out), tuple(warnings)


def _check_bayes(x: str, levels: Sequence[str], values: Sequence[FloatArray]) -> None:
    small = [
        f"{level!r} (n = {v.size})"
        for level, v in zip(levels, values, strict=True)
        if v.size < 2
    ]
    if small:
        has = "this group has" if len(small) == 1 else "these groups have"
        raise ValueError(
            f"Every group of x={x!r} needs at least 2 observations for the Bayes "
            f"factor (Student's t) and the MAP estimate; {has} fewer: "
            f"{', '.join(small)}. Remove {'it' if len(small) == 1 else 'them'} from "
            "the data or collect more observations."
        )
    constant = [
        level
        for level, v in zip(levels, values, strict=True)
        if float(np.ptp(v)) == 0.0
    ]
    if len(constant) >= 2:
        raise ValueError(
            f"The groups {_names(constant)} of x={x!r} have no variability (every "
            "value within each group is the same), so Student's t of a pair of "
            "them, and its Bayes factor, has no standard error."
        )


def _bayes(levels: Sequence[str], values: Sequence[FloatArray], args: _Args) -> _Parts:
    """Two groups: the JZS Bayes factor subtitle (log_e BF01 and the prior scale);
    three or more groups, only with results_subtitle=False: a JZS Bayes factor for
    every pair, Student's t, prior 0.707, unadjusted (D41). MAP estimates as
    centrality."""
    x = args.x
    _check_bayes(x, levels, values)
    centers, map_warnings = _map_estimates(x, levels, values)
    warnings: list[str] = list(map_warnings)
    citations = [_ROUDER_2009, _SHEATHER_JONES_1991]
    bayes: BayesFactor | None = None
    pairwise: tuple[PairwiseComparison, ...] = ()
    try:
        if len(levels) == 2:
            if args.results_subtitle:
                a, b = values
                bf10, log_bf01 = stats.jzs_bf10_two_sample(
                    a, b, prior_scale=args.prior_scale
                )
                bayes = BayesFactor(
                    bf10=bf10,
                    log_bf01=log_bf01,
                    prior="cauchy",
                    prior_scale=args.prior_scale,
                    test="jzs_two_sample",
                )
                warnings.append(_POSTERIOR_WARNING)
        else:
            pairwise = _bayes_pairs(levels, values, args)
    except ValueError as error:
        # A pair's message already names its levels; renaming it again would read
        # a level called "group 7" as the seventh group.
        named = isinstance(error, _NamedPairError)
        raise ValueError(
            _with_advice(
                str(error) if named else _named_groups(str(error), levels, x),
                _CHECK_DATA,
                given="Check the data for entry or unit errors",
            )
        ) from None
    return _Parts(
        test=None,
        effect_size=None,
        bayes=bayes,
        pairwise=pairwise,
        centrality="map",
        centers=centers,
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


class _NamedPairError(ValueError):
    """A pair's error whose message already names the pair's levels."""


def _bayes_pairs(
    levels: Sequence[str], values: Sequence[FloatArray], args: _Args
) -> tuple[PairwiseComparison, ...]:
    """A JZS Bayes factor for every pair i < j in level order on Student's t
    (mean(group1) - mean(group2)), prior 0.707 whatever ``bf_prior`` is, not
    adjusted; every pair drawn unless ``pairwise_display`` is "none"; rows in
    ggstatsplot's order (``_sorted_pairs``)."""
    rows: list[PairwiseComparison] = []
    k = len(levels)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = values[i], values[j]
            try:
                student = stats.student_t(a, b)
                bf10, log_bf01 = stats.jzs_bf10_two_sample(
                    a, b, prior_scale=_PAIRWISE_BF_PRIOR
                )
            except ValueError as error:
                # "sample a" and "sample b" are this pair's levels.
                detail = _named_groups(str(error), (levels[i], levels[j]), args.x)
                raise _NamedPairError(
                    f"The Bayes factor of {levels[i]!r} against {levels[j]!r} cannot "
                    f"be computed: {detail[0].lower()}{detail[1:]}"
                ) from None
            rows.append(
                PairwiseComparison(
                    group1=levels[i],
                    group2=levels[j],
                    statistic_name="t",
                    statistic=student.t,
                    df=student.df,
                    p_value=None,
                    p_adjusted=None,
                    adjust_method="none",
                    test="jzs_two_sample",
                    test_label="Student's t",
                    significant=None,
                    displayed=args.pairwise_display != "none",
                    bayes=BayesFactor(
                        bf10=bf10,
                        log_bf01=log_bf01,
                        prior="cauchy",
                        prior_scale=_PAIRWISE_BF_PRIOR,
                        test="jzs_two_sample",
                    ),
                )
            )
    return _sorted_pairs(rows)


def _bayes_no_effect_warnings(
    *, k: int, passed: dict[str, object], canonical: dict[str, object]
) -> list[str]:
    """Arguments type bayes does not use (D39): conf_level (no interval in v0.1),
    and with three or more groups the significance and adjustment arguments
    (Bayes factors are neither tested nor adjusted; every pair is drawn)."""
    out: list[str] = []
    if canonical["conf_level"] != 0.95:
        out.append(
            f"conf_level={passed['conf_level']!r} has no effect: type='bayes' shows "
            "no interval in this version (the posterior credible interval arrives "
            "in inferplot 0.2)."
        )
    if k >= 3:
        if canonical["p_adjust_method"] != _DEFAULT_P_ADJUST:
            out.append(
                f"p_adjust_method={passed['p_adjust_method']!r} has no effect: "
                "Bayes factors are not adjusted for multiple comparisons (as in "
                "ggstatsplot)."
            )
        if canonical["pairwise_alpha"] != _DEFAULT_ALPHA:
            out.append(
                f"pairwise_alpha={passed['pairwise_alpha']!r} has no effect: Bayes "
                "factors are not compared with a significance level, and every pair "
                "is drawn."
            )
        if canonical["pairwise_display"] == "non-significant":
            out.append(
                f"pairwise_display={passed['pairwise_display']!r} has no effect: "
                "Bayes factors have no significance, so every pair is drawn (as in "
                "ggstatsplot); pass pairwise_display='none' to draw none."
            )
    return out


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #


def _check_type_arguments(
    type_name: str, *, tr: object, trim: float, side: str, show_bf: bool | None
) -> None:
    """Arguments that cannot apply to ``type_name`` whatever the data (SPEC.md
    section 2): they would change what the reported numbers mean, so they raise
    instead of being ignored as ggstatsplot ignores them."""
    if trim != _DEFAULT_TR and type_name != "robust":
        # ggstatsplot silently ignores tr outside the robust type; inferplot never
        # drops an argument (CLAUDE.md rule 3), as for alternative (SPEC section 2).
        raise ValueError(
            f"tr={tr!r} sets the trimming of the robust methods (type='robust'), but "
            f"type={type_name!r} trims nothing, so it would have no effect. Remove "
            "tr=, or pass type='robust' to compare trimmed means."
        )
    if side != "two-sided" and type_name in ("robust", "bayes"):
        why = (
            "Yuen's trimmed-means test and its AKP interval are always two-sided"
            if type_name == "robust"
            else "the Bayes factor is always two-sided"
        )
        raise ValueError(
            f"alternative={side!r} applies to type='parametric' and "
            f"type='nonparametric' only: with type={type_name!r} {why} (ggstatsplot "
            "silently ignores alternative there). Remove alternative=, or use "
            "type='nonparametric' for a one-sided rank test."
        )
    if show_bf is True and type_name != "parametric":
        raise ValueError(
            f"bf_message=True asks for a Bayes factor caption, which ggstatsplot "
            f"draws only for type='parametric'; type={type_name!r} has none. Remove "
            "bf_message=True"
            + (
                "; type='bayes' reports the Bayes factor in the subtitle."
                if type_name == "bayes"
                else "."
            )
        )


@checked
def betweenstats(
    data: object,
    x: str,
    y: str,
    *,
    type: str = "parametric",
    pairwise_display: str = "significant",
    p_adjust_method: str = "holm",
    pairwise_alpha: float = 0.05,
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    alternative: str = "two-sided",
    seed: int = 0,
    order: Iterable[str | int | float] | None = None,
) -> StatsResult:
    """Compare ``y`` across the independent groups of ``x``; no plotting.

    The methods follow ggstatsplot 1.1.4's ``ggbetweenstats()`` for each ``type``
    (SPEC.md section 4):

    * "parametric" ("p"): Welch's t-test with Hedges' g and the JZS Bayes factor
      caption (two groups); Welch's ANOVA with partial omega squared and
      Games-Howell comparisons (three or more). Centrality: means.
    * "nonparametric" ("np"): the Mann-Whitney U test with the rank-biserial r (two
      groups); the Kruskal-Wallis test with rank epsilon squared (bootstrap CI) and
      Dunn's comparisons (three or more). Centrality: medians.
    * "bayes" ("bf"): the JZS Bayes factor of Student's t, reported as log_e(BF01)
      with the prior scale ``bf_prior`` (two groups); with three or more groups
      only with ``results_subtitle=False`` (the Bayesian ANOVA arrives in 0.2): a
      Bayes factor for every pair, prior 0.707, unadjusted. Centrality: MAP
      estimates. Posterior estimates arrive in 0.2 (D41).
    * "robust" ("r"): Yuen's test on ``tr``-trimmed means with the AKP robust
      standardized difference (two groups); the heteroscedastic trimmed-means
      ANOVA with the explanatory effect size and lincon's comparisons (three or
      more). Both effect sizes have percentile bootstrap CIs. Centrality: trimmed
      means.

    Args:
        data: a pandas, polars or pyarrow data frame in long format.
        x: the grouping column.
        y: the numeric outcome column.
        type: "parametric" ("p"), "nonparametric" ("np"), "robust" ("r") or
            "bayes" ("bf").
        pairwise_display: which pairs the figure draws: "significant" ("s"),
            "non-significant" ("ns"), "all" or "none". Every pair is computed.
        p_adjust_method: R's ``p.adjust`` method for the pairwise p-values.
        pairwise_alpha: a pair is significant when its adjusted p is below this.
        conf_level: confidence level of the effect-size interval.
        digits: decimals in the subtitle, caption and labels.
        tr: trimming proportion of the robust methods; another value than 0.2
            with a type that trims nothing raises.
        bf_prior: scale of the Cauchy prior of the Bayes factor (0.707).
        bf_message: None draws the Bayes factor caption where it is available
            (type "parametric"); True requires it; False never draws it.
        results_subtitle: whether the statistical results are reported as a
            subtitle (False also drops the Bayes factor caption).
        alternative: "two-sided", "less" or "greater"; one-sided only with two
            groups and types "parametric" and "nonparametric" (Welch's t-test,
            the Mann-Whitney U test). "greater" tests whether the first level of
            x is greater than the second, and "less" whether it is less. The
            first level is the first in ``order`` when given, else the first
            declared category, else the first in sorted order: pass ``order`` to
            fix the direction. The APA sentence and the alt text name it ("Welch's
            t-test (one-sided: control greater than treatment)"); a program reads
            it from ``result.test.alternative`` and ``result.levels``.
        seed: random seed of the bootstrap intervals (type "robust"; type
            "nonparametric" with three or more groups), recorded in the result.
        order: the level order, a permutation of the levels present.

    Returns:
        A :class:`~inferplot.result.StatsResult` holding every number and text.

    Raises:
        TypeError, ValueError: a bad argument or unusable data; the message says
            how to fix the call.
        KeyError: a column that the data frame does not have.
        NotImplementedError: a design this version does not have: the Bayes factor
            caption or subtitle for three or more groups (a Bayesian ANOVA), and
            the robust pairwise comparisons when a pair has fewer than 2 degrees of
            freedom (where WRS2's lincon stops).
    """
    type_name = canonical_type(type)
    display = canonical_pairwise_display(pairwise_display)
    adjust = _adjust_method(check_p_adjust_method(p_adjust_method))
    alpha = check_probability(pairwise_alpha, "pairwise_alpha")
    level = check_conf_level(conf_level)
    n_digits = check_digits(digits)
    trim = check_tr(tr)
    prior_scale = check_positive(bf_prior, "bf_prior")
    show_bf = check_optional_bool(bf_message, "bf_message")
    subtitle_on = check_bool(results_subtitle, "results_subtitle")
    side = _alternative(check_alternative(alternative))
    rng_seed = check_seed(seed)
    if show_bf is True and not subtitle_on:
        raise ValueError(
            "bf_message=True asks for the Bayes factor caption, which is drawn only "
            "together with the results subtitle; results_subtitle=False turns both "
            "off. Pass results_subtitle=True, or drop bf_message=True."
        )
    _check_type_arguments(type_name, tr=tr, trim=trim, side=side, show_bf=show_bf)

    between = read_between(data, x, y, order=order)
    levels, values = between.levels, between.values
    k = len(levels)
    if k >= 3 and side != "two-sided":
        omnibus = {
            "parametric": "Welch's ANOVA",
            "nonparametric": "the Kruskal-Wallis test",
            "robust": "the trimmed-means ANOVA",
            "bayes": "the pairwise Bayes factors",
        }[type_name]
        raise ValueError(
            f"alternative={side!r} applies only to two groups; x={x!r} has {k} "
            f"groups, where {omnibus} and the pairwise comparisons are always "
            "two-sided. Remove alternative= or compare two groups."
        )
    if type_name == "parametric":
        if k >= 3 and show_bf is True:
            raise NotImplementedError(
                "bf_message=True: the Bayes factor caption for three or more groups "
                "(a Bayesian ANOVA) arrives in inferplot 0.2. Remove "
                "bf_message=True; the default (None) leaves the caption out and "
                "says so in result.warnings."
            )
        _check_groups(
            x, levels, values, bayes=k == 2 and subtitle_on and show_bf is not False
        )
    if type_name == "bayes" and k >= 3 and subtitle_on:
        raise NotImplementedError(
            f"type='bayes' with {k} groups: the subtitle ggstatsplot draws here is a "
            "Bayesian ANOVA, which arrives in inferplot 0.2. Pass "
            "results_subtitle=False for the rest of ggstatsplot's output (MAP "
            "estimates and a Bayes factor for every pair), or use "
            "type='parametric'."
        )

    args = _Args(
        x=x,
        type=type_name,
        conf_level=level,
        alternative=side,
        p_adjust_method=adjust,
        pairwise_alpha=alpha,
        pairwise_display=display,
        prior_scale=prior_scale,
        bf_message=show_bf,
        results_subtitle=subtitle_on,
        tr=trim,
        seed=rng_seed,
    )
    analyses = {
        "parametric": _parametric,
        "nonparametric": _nonparametric,
        "robust": _robust,
        "bayes": _bayes,
    }
    parts = analyses[type_name](levels, values, args)

    warnings: list[str] = []
    if between.empty_levels:
        empty = between.empty_levels
        warnings.append(
            f"The declared {'level' if len(empty) == 1 else 'levels'} "
            f"{_names(empty)} of {x!r} {'has' if len(empty) == 1 else 'have'} no "
            "observations with both x and y present and "
            f"{'was' if len(empty) == 1 else 'were'} left out of the analysis."
        )
    warnings += data_notes(between)
    warnings += parts.warnings
    if type_name == "bayes":
        warnings += _bayes_no_effect_warnings(
            k=k,
            passed={
                "conf_level": level,
                "p_adjust_method": p_adjust_method,
                "pairwise_alpha": alpha,
                "pairwise_display": pairwise_display,
            },
            canonical={
                "conf_level": level,
                "p_adjust_method": adjust,
                "pairwise_alpha": alpha,
                "pairwise_display": display,
            },
        )
    warnings += _no_effect_warnings(
        x=x,
        k=k,
        type_name=type_name,
        bayes_computed=parts.bayes is not None,
        subtitle_on=subtitle_on,
        show_bf=show_bf,
        passed={
            "p_adjust_method": p_adjust_method,
            "pairwise_display": pairwise_display,
            "pairwise_alpha": alpha,  # numbers as plain floats (NumPy scalars too)
            "bf_prior": prior_scale,
        },
        canonical={
            "p_adjust_method": adjust,
            "pairwise_display": display,
            "pairwise_alpha": alpha,
            "bf_prior": prior_scale,
        },
    )

    descriptives = tuple(
        Descriptive(level=name, n=int(v.size), centrality=parts.centrality, value=c)
        for name, v, c in zip(levels, values, parts.centers, strict=True)
    )
    settings = Settings(
        conf_level=level,
        digits=n_digits,
        alternative=side,
        p_adjust_method=adjust,
        pairwise_display=display,
        pairwise_alpha=alpha,
        bf_prior=prior_scale,
        bf_message=show_bf,
        results_subtitle=subtitle_on,
        seed=rng_seed,
        tr=trim,
    )
    texts = build_texts(
        x=x,
        y=y,
        descriptives=descriptives,
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=parts.pairwise,
        n_obs=between.n_obs,
        settings=settings,
    )
    observations = tuple(tuple(float(v) for v in arr.tolist()) for arr in values)
    return StatsResult(
        analysis="betweenstats",
        design="between",
        type=type_name,
        x=x,
        y=y,
        subject_id=None,
        levels=levels,
        values=observations,
        subjects=(),
        descriptives=descriptives,
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=parts.pairwise,
        n_obs=between.n_obs,
        n_dropped=between.n_dropped,
        dropped_subjects=0,
        dropped_subject_rows=0,
        n_tied=tied_count(observations),
        warnings=tuple(warnings),
        provenance=current_provenance(seed=rng_seed, citations=parts.citations),
        settings=settings,
        text=texts,
    )
