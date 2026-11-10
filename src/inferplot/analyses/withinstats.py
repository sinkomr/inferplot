"""``withinstats``: compare a numeric ``y`` across repeated measures of subjects.

The analysis that ``ip.analyze.withinstats`` exposes, the counterpart of ggstatsplot
1.1.4's ``ggwithinstats()``. It reads the data through
:func:`inferplot._data.read_within` (complete subjects only, paired by
``subject_id``, never by row order: SPEC.md section 5 item 1), validates every
argument with :mod:`inferplot._validate`, and returns a
:class:`~inferplot.result.StatsResult` with ``design="within"``. It never imports
matplotlib.

What it reproduces is ggstatsplot 1.1.4's ``ggwithinstats()`` (SPEC.md section 4,
"withinstats"), on the complete subjects only; n is their number, printed as
``n["pairs"]``:

* type parametric, two conditions: the paired t-test (first level minus second),
  Hedges' g of the differences with a noncentral-t CI that follows
  ``alternative``, and the two-sided paired JZS Bayes factor caption;
* type parametric, three or more: the repeated-measures ANOVA with the
  Greenhouse-Geisser correction (corrected df), partial omega squared with a
  one-sided noncentral-F CI at the uncorrected df, and paired t-tests of every
  pair adjusted with ``p_adjust_method``; means as centrality;
* type nonparametric: the Wilcoxon signed-rank test (normal approximation with
  tie and continuity corrections) with the matched-pairs rank-biserial r, or
  Friedman's test with Kendall's W (seeded percentile bootstrap CI over whole
  subjects) and Durbin-Conover's pairwise t; medians as centrality;
* type robust: Yuen's test for dependent trimmed means with the AKP effect size of
  the paired differences, or WRS2's repeated-measures trimmed-means ANOVA (Huynh-
  Feldt df) with the average AKP over the pairs and rmmcp's pairwise comparisons,
  all at trimming ``tr``; the bootstrap intervals are always 95%, as in
  ggstatsplot (D66); trimmed means as centrality. p-values are the accurate
  tails, not WRS2's 1 - cdf (D43);
* type bayes: the paired JZS Bayes factor (two conditions); with three or more,
  only with ``results_subtitle=False``, a paired Bayes factor for every pair at
  prior 0.707; MAP estimates as centrality.

Each type's analysis is a function ``(data, args) -> _Parts`` in ``_ANALYSES``,
run after every argument has been validated and the data read; ``_assemble``
builds the result from its parts. Only the Bayesian repeated-measures ANOVA (type
bayes with three or more conditions and the subtitle on) raises
``NotImplementedError``: it arrives in inferplot 0.2.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from inferplot import stats
from inferplot._data import WithinData, data_notes, read_within
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
    describe_value,
    subject_id_required,
)
from inferplot.analyses.betweenstats import check_one_sided_bootstrap_level
from inferplot.result import (
    BayesFactor,
    Descriptive,
    EffectSize,
    PairwiseComparison,
    Settings,
    StatsResult,
    TestResult,
    current_provenance,
    paired_tied_count,
)
from inferplot.text import build_texts, format_value, r_number_text

__all__ = ["withinstats"]

FloatArray = npt.NDArray[np.float64]

# ggstatsplot's default trimming; another value is allowed only where it is used.
_DEFAULT_TR: Final = 0.2
# From three conditions the omnibus test is an ANOVA; the trimmed-means one needs
# three subjects (with two its sphericity correction is 0 / 0 for any data).
_MIN_CONDITIONS_ANOVA: Final = 3
_MIN_SUBJECTS_TRIMMED_ANOVA: Final = 3


@dataclass(frozen=True, slots=True)
class _Args:
    """The validated arguments every type's analysis reads."""

    x: str
    subject_id: str
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
    # The arguments as the caller gave them, for the warnings that name them
    # (betweenstats' rule: "pairwise_display='ns' has no effect").
    passed: Mapping[str, object] = field(default_factory=dict[str, object])

    def shown(self, name: str, value: object) -> str:
        """``name``'s value as given (else ``value``), as a warning names it."""
        return repr(self.passed.get(name, value))


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


# --------------------------------------------------------------------------- #
# the types (each takes the complete subjects and the validated arguments)
# --------------------------------------------------------------------------- #


def _parametric(data: WithinData, args: _Args) -> _Parts:
    """Paired t-test with Hedges' g and the paired JZS Bayes factor caption (two
    conditions); repeated-measures ANOVA with the Greenhouse-Geisser correction,
    partial omega squared and paired t pairwise comparisons (three or more)."""
    x, levels, columns = args.x, data.levels, data.by_level()
    side, level = args.alternative, args.conf_level
    warnings: list[str] = []
    bayes: BayesFactor | None = None
    pairwise: tuple[PairwiseComparison, ...] = ()
    if len(levels) == 2:
        a, b = columns
        paired = _checked_paired_t(x, levels[0], levels[1], a, b, side=side)
        g = stats.paired_hedges_g(a, b, conf_level=level, alternative=side)
        test = TestResult(
            name="paired_t",
            label="Paired t-test",
            method="Paired t-test",
            statistic_name="t",
            statistic=paired.t,
            df=None,
            df_error=paired.df,
            p_value=paired.p_value,
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
        citations = [_STUDENT_1908, _HEDGES_1981, _STEIGER_2004]
        if paired.n == 2:
            warnings.append(_HEDGES_J_ZERO_WARNING)
        if args.results_subtitle and args.bf_message is not False:
            bf10, log_bf01 = stats.jzs_bf10_paired(a, b, prior_scale=args.prior_scale)
            bayes = BayesFactor(
                bf10=bf10,
                log_bf01=log_bf01,
                prior="cauchy",
                prior_scale=args.prior_scale,
                test="jzs_paired",
            )
            citations.append(_ROUDER_2009)
            if side != "two-sided":
                warnings.append(
                    "The Bayes factor in the caption is two-sided: it weighs any "
                    f"difference between the conditions, while alternative={side!r} "
                    "applies to the t-test and the Hedges' g interval only (as in "
                    "ggstatsplot)."
                )
    else:
        anova = _checked_rm_anova(x, columns)
        pairs = _paired_t_pairs(x, levels, columns, args)
        omega = stats.rm_partial_omega_squared(columns, conf_level=level)
        test = TestResult(
            name="rm_anova",
            label="Repeated-measures ANOVA (Greenhouse-Geisser)",
            method="ANOVA estimation for factorial designs using 'afex'",
            statistic_name="F",
            statistic=anova.f,
            df=anova.df1_corrected,
            df_error=anova.df2_corrected,
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
        pairwise = pairs
        citations = [
            _FISHER_1925,
            _GREENHOUSE_GEISSER_1959,
            _OLEJNIK_ALGINA_2003,
            _STEIGER_2004,
            _STUDENT_1908,
            *_within_adjustment_citation(args.p_adjust_method),
        ]
        if args.results_subtitle and args.bf_message is None:
            warnings.append(_BAYES_K3_WARNING)
    warnings += _within_no_effect_warnings(data, args, bayes_computed=bayes is not None)
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        centrality="mean",
        centers=tuple(stats.mean(column) for column in columns),
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


# Short references for the methods of types parametric and bayes (SPEC.md section 4).
_STUDENT_1908: Final = "Student (1908). Biometrika 6: 1-25 (the paired t-test)."
_FISHER_1925: Final = (
    "Fisher, R. A. (1925). Statistical Methods for Research Workers. Oliver & Boyd "
    "(analysis of variance)."
)
_GREENHOUSE_GEISSER_1959: Final = (
    "Greenhouse, S. W. & Geisser, S. (1959). Psychometrika 24: 95-112 "
    "(Greenhouse-Geisser correction)."
)
_HEDGES_1981: Final = (
    "Hedges, L. V. (1981). Journal of Educational Statistics 6: 107-128 (Hedges' g)."
)
_STEIGER_2004: Final = (
    "Steiger, J. H. (2004). Psychological Methods 9: 164-182 (noncentral confidence "
    "intervals for effect sizes)."
)
_OLEJNIK_ALGINA_2003: Final = (
    "Olejnik, S. & Algina, J. (2003). Psychological Methods 8: 434-447 (partial "
    "omega squared)."
)
_ROUDER_2009: Final = (
    "Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D. & Iverson, G. (2009). "
    "Psychonomic Bulletin & Review 16: 225-237 (JZS Bayes factor)."
)
_WITHIN_ADJUSTMENT_CITATIONS: Final[dict[str, str]] = {
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

_HEDGES_J_ZERO_WARNING: Final = (
    "Hedges' g is 0 by construction with 2 subjects: its small-sample correction "
    "J(n - 1) is 0 at 1 degree of freedom, so g and its confidence interval carry no "
    "information about the size of the difference. Collect more subjects."
)
_BAYES_K3_WARNING: Final = (
    "The Bayes factor caption for three or more conditions arrives in inferplot 0.2."
)

# Defaults of the arguments that can have no effect: a non-default value where it
# has none gets a warnings entry (D39), as in betweenstats.
_DEFAULT_P_ADJUST: Final = "holm"
_DEFAULT_DISPLAY: Final = "significant"
_DEFAULT_ALPHA: Final = 0.05
_DEFAULT_BF_PRIOR: Final = 0.707
# Every pairwise Bayes factor uses this prior scale, whatever bf_prior says
# (statsExpressions 2.2.1's pairwise_comparisons(); bugs_long-prior08-bayes).
_PAIRWISE_BF_PRIOR: Final = 0.707


def _within_adjustment_citation(method: str) -> list[str]:
    citation = _WITHIN_ADJUSTMENT_CITATIONS.get(method)
    return [] if citation is None else [citation]


def _pair_names(first: str, second: str, x: str) -> str:
    return f"{first!r} and {second!r} of x={x!r}"


def _checked_paired_t(
    x: str,
    first: str,
    second: str,
    a: FloatArray,
    b: FloatArray,
    *,
    side: stats.Alternative = "two-sided",
    method: str = "the paired t-test",
) -> stats.PairedT:
    """The paired t-test of ``first`` minus ``second``, or a plain ValueError naming
    the two levels and ``method`` when the differences have no usable spread."""
    if stats.paired_differences_constant(a, b):
        what = (
            f"Every subject has the same value at {first!r} as at {second!r}"
            if bool((a == b).all())
            else f"Every subject's value at {first!r} differs from its value at "
            f"{second!r} by the same amount (exactly, or to within the rounding of "
            "y in double precision)"
        )
        raise ValueError(
            f"{what} (x={x!r}), so the paired differences have no variability and "
            f"{method} has no standard error. Check the data for entry errors, or "
            "use type='nonparametric'."
        )
    try:
        return stats.paired_t(a, b, alternative=side)
    except ValueError:
        raise ValueError(
            f"The differences between the levels {_pair_names(first, second, x)} "
            "vary by less than double precision can resolve next to the largest "
            "|y| (a standard deviation of the differences below roughly 1e-154 "
            f"times it), so {method} has no accurate standard error. "
            "Rescaling y does not help, since it multiplies the spread and the "
            "largest |y| alike; check y for entry or unit errors."
        ) from None


def _checked_rm_anova(
    x: str, columns: Sequence[FloatArray]
) -> stats.RepeatedMeasuresAnova:
    """The repeated-measures ANOVA, or a plain ValueError naming x when it has no
    error variance."""
    try:
        return stats.rm_anova(columns)
    except ValueError as error:
        if "differ by the same amounts" in str(error):
            raise ValueError(
                f"Every subject's values at the levels of x={x!r} are the condition "
                "means plus a constant of the subject's own (the conditions differ "
                "by the same amounts for every subject), so the repeated-measures "
                "ANOVA has no error variance and its F cannot be computed. Check "
                "the data for entry errors, or use type='nonparametric'."
            ) from None
        raise ValueError(
            f"The repeated-measures ANOVA of the levels of x={x!r} cannot be "
            f"computed: {error}."
        ) from None


def _displayed(display: str, significant: bool) -> bool:
    if display == "all":
        return True
    if display == "significant":
        return significant
    if display == "non-significant":
        return not significant
    return False  # "none": computed and returned, never drawn


def _sorted_pairs(rows: list[PairwiseComparison]) -> tuple[PairwiseComparison, ...]:
    """Rows sorted by group1 and then group2 as text in the C locale (code-point
    order), as statsExpressions 2.2.1's ``pairwise_comparisons()`` arranges them
    (``dplyr::arrange(group1, group2)``; the fixtures' rows and brackets)."""
    return tuple(sorted(rows, key=lambda row: (row.group1, row.group2)))


def _paired_t_pairs(
    x: str, levels: Sequence[str], columns: Sequence[FloatArray], args: _Args
) -> tuple[PairwiseComparison, ...]:
    """A paired t-test for every pair i < j in level order (t of group1 minus
    group2, df n - 1), raw p adjusted with ``p_adjust_method`` over all pairs,
    rows in ggstatsplot's order; R's test label is "Student's t"."""
    k = len(levels)
    for i in range(k):
        for j in range(i + 1, k):
            _checked_paired_t(x, levels[i], levels[j], columns[i], columns[j])
    rows: list[PairwiseComparison] = []
    for pair in stats.paired_t_pairs(columns, p_adjust_method=args.p_adjust_method):
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
                test="paired_t",
                test_label="Student's t",
                significant=significant,
                displayed=_displayed(args.pairwise_display, significant),
            )
        )
    return _sorted_pairs(rows)


def _within_no_effect_warnings(
    data: WithinData, args: _Args, *, bayes_computed: bool
) -> list[str]:
    """One plain-English entry per argument changed from its default that has no
    effect on these data (D39; betweenstats' rules with "conditions")."""
    x, k = args.x, len(data.levels)
    out: list[str] = []
    if k == 2:
        reason = (
            f"x={x!r} has 2 conditions, so there are no pairwise comparisons to "
            "{what}."
        )
        for name, value, default, what in (
            ("p_adjust_method", args.p_adjust_method, _DEFAULT_P_ADJUST, "adjust"),
            ("pairwise_display", args.pairwise_display, _DEFAULT_DISPLAY, "display"),
            ("pairwise_alpha", args.pairwise_alpha, _DEFAULT_ALPHA, "call significant"),
        ):
            if value != default:
                out.append(
                    f"{name}={args.shown(name, value)} has no effect: "
                    + reason.format(what=what)
                )
    out += _bf_message_no_effect(args)
    if bayes_computed or args.prior_scale == _DEFAULT_BF_PRIOR:
        return out
    prior = args.shown("bf_prior", args.prior_scale)
    if args.type == "bayes" and k >= 3:
        # Bayes factors are computed, for every pair, at the fixed scale.
        out.append(
            f"bf_prior={prior} has no effect: the pairwise Bayes "
            f"factors always use the prior scale {r_number_text(_PAIRWISE_BF_PRIOR)}"
            ", as in ggstatsplot; the Bayesian repeated-measures ANOVA arrives in "
            "inferplot 0.2."
        )
        return out
    if args.type in ("nonparametric", "robust"):
        why = f"type={args.type!r} has no Bayes factor"
    elif k >= 3:
        why = (
            f"x={x!r} has {k} conditions, and the Bayes factor for three or more "
            "conditions arrives in inferplot 0.2"
        )
    elif not args.results_subtitle:
        why = "results_subtitle=False leaves the Bayes factor out"
    else:
        why = f"bf_message={args.bf_message!r} leaves the Bayes factor out"
    out.append(f"bf_prior={prior} has no effect: no Bayes factor is computed ({why}).")
    return out


def _bf_message_no_effect(args: _Args) -> list[str]:
    """``bf_message=False`` where no Bayes factor caption could be drawn: types
    other than parametric (ggstatsplot draws the caption for type parametric
    only) and ``results_subtitle=False``. With type parametric and the subtitle it
    acts: two conditions lose the caption, three or more the note that their
    caption arrives in 0.2. ``True`` there raises instead (``withinstats``)."""
    if args.bf_message is not False:
        return []
    if args.type != "parametric":
        where = (
            "; type='bayes' reports its Bayes factor in the subtitle"
            if args.type == "bayes"
            else ""
        )
        return [
            "bf_message=False has no effect: the Bayes factor caption is drawn only "
            f"for type='parametric' (as in ggstatsplot), and type={args.type!r} "
            f"draws none{where}."
        ]
    if not args.results_subtitle:
        return [
            "bf_message=False has no effect: results_subtitle=False leaves the Bayes "
            "factor caption out."
        ]
    return []


def _nonparametric(data: WithinData, args: _Args) -> _Parts:
    """Wilcoxon signed-rank test with the rank-biserial r (two conditions);
    Friedman test with Kendall's W and Durbin-Conover comparisons (three or more).
    Medians of the complete subjects' values as centrality."""
    columns = data.by_level()
    if len(data.levels) == 2:
        test, effect_size, pairwise, warnings, citations = _signed_rank_parts(
            data, args
        )
    else:
        test, effect_size, pairwise, warnings, citations = _friedman_parts(data, args)
    warnings += _within_no_effect_warnings(data, args, bayes_computed=False)
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=None,
        pairwise=pairwise,
        centrality="median",
        centers=tuple(stats.median(column) for column in columns),
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


_NonparametricParts = tuple[
    TestResult, EffectSize, tuple[PairwiseComparison, ...], list[str], list[str]
]


def _signed_rank_parts(data: WithinData, args: _Args) -> _NonparametricParts:
    """Two conditions: the signed-rank test of the first level minus the second
    and the matched-pairs rank-biserial r."""
    x, levels = args.x, data.levels
    side, level = args.alternative, args.conf_level
    warnings: list[str] = []
    a, b = data.by_level()
    if bool((a == b).all()):
        raise ValueError(
            f"Every subject has the same value at {levels[0]!r} as at {levels[1]!r} "
            f"(x={x!r}), so every paired difference is 0 and the signed-rank test "
            "has nothing to rank: its statistic and the rank-biserial r are 0/0. "
            "There is no difference to test; check the data for entry errors."
        )
    wilcoxon = stats.signed_rank(a, b, alternative=side)
    r = stats.paired_rank_biserial(a, b, conf_level=level, alternative=side)
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
    if abs(r.estimate) == 1.0 and r.conf_low == r.conf_high:
        warnings.append(_rank_biserial_bound_warning(x, levels[0], levels[1], r))
    return test, effect_size, (), warnings, [_WILCOXON_1945, _KERBY_2014]


def _friedman_parts(data: WithinData, args: _Args) -> _NonparametricParts:
    """Three or more conditions: Friedman's test, Kendall's W with its seeded
    bootstrap interval and Durbin-Conover's comparisons of every pair."""
    x, levels, columns = args.x, data.levels, data.by_level()
    k = len(levels)
    if all(bool((column == columns[0]).all()) for column in columns[1:]):
        raise ValueError(
            f"Every subject has the same value at all {k} levels of x={x!r}, so "
            "Friedman's test has nothing to rank within the subjects: its statistic "
            "and Kendall's W are 0/0. There is no difference to test; check the "
            "data for entry errors."
        )
    friedman = stats.friedman(columns)
    check_one_sided_bootstrap_level(args.conf_level, "Kendall's W")
    try:
        w = stats.kendalls_w(columns, conf_level=args.conf_level, seed=args.seed)
    except ValueError as error:
        raise ValueError(
            f"The bootstrap interval of Kendall's W cannot be computed for these "
            f"subjects (x={x!r}): {error}. Use type='parametric', or collect more "
            "subjects whose values differ between the conditions."
        ) from None
    test = TestResult(
        name="friedman",
        label="Friedman test",
        method="Friedman rank sum test",
        statistic_name="chi2",
        statistic=friedman.statistic,
        df=None,
        df_error=friedman.df,
        p_value=friedman.p_value,
        alternative="two-sided",
    )
    effect_size = EffectSize(
        name="kendalls_w",
        label="Kendall's W",
        estimate=w.estimate,
        conf_level=w.conf_level,
        conf_low=w.conf_low,
        conf_high=w.conf_high,
        ci_method="percentile bootstrap",
        sidedness=w.sidedness,
        bootstrap_resamples=w.resamples,
        bootstrap_seed=w.seed,
        undefined_resamples=w.undefined_resamples,
        infinite_resamples=w.infinite_resamples,
    )
    pairwise = _durbin_conover_pairs(x, levels, columns, args)
    warnings: list[str] = []
    if w.undefined_resamples:
        warnings.append(_tied_subject_resamples_warning(w))
    if w.estimate == 1.0:
        warnings.append(_identical_rankings_warning(x, data.n_subjects))
    citations = [
        _FRIEDMAN_1937,
        _KENDALL_1939,
        _EFRON_TIBSHIRANI_1993,
        _CONOVER_1999,
        *_within_adjustment_citation(args.p_adjust_method),
    ]
    return test, effect_size, pairwise, warnings, citations


def _durbin_conover_pairs(
    x: str, levels: Sequence[str], columns: Sequence[FloatArray], args: _Args
) -> tuple[PairwiseComparison, ...]:
    """Durbin-Conover's |t| for every pair i < j in level order (unsigned, as
    ggstatsplot reports it), on (n - 1)(k - 1) df, raw p adjusted with
    ``p_adjust_method`` over all pairs, rows in ggstatsplot's order."""
    try:
        pairs = stats.durbin_conover(columns, p_adjust_method=args.p_adjust_method)
    except ValueError as error:
        if "share their rank in every subject" not in str(error):
            raise
        tied = _rank_tied_pair(levels, columns)
        raise ValueError(
            f"Every subject ranks the levels of x={x!r} in the same order, and "
            f"{tied} share their rank in every subject, so the Durbin-Conover "
            "comparison of that pair has no difference and no error variance (its t "
            "is 0/0). Check the data for entry errors, or compare the other levels "
            "without one of these two."
        ) from None
    rows: list[PairwiseComparison] = []
    for pair in pairs:
        significant = pair.p_adjusted < args.pairwise_alpha
        rows.append(
            PairwiseComparison(
                group1=levels[pair.i],
                group2=levels[pair.j],
                statistic_name="t",
                statistic=pair.statistic,
                df=pair.df,
                p_value=pair.p_value,
                p_adjusted=pair.p_adjusted,
                adjust_method=args.p_adjust_method,
                test="durbin_conover",
                test_label="Durbin-Conover",
                significant=significant,
                displayed=_displayed(args.pairwise_display, significant),
            )
        )
    return _sorted_pairs(rows)


def _rank_tied_pair(levels: Sequence[str], columns: Sequence[FloatArray]) -> str:
    """The first two levels whose values tie within every subject."""
    k = len(levels)
    for i in range(k):
        for j in range(i + 1, k):
            if bool((columns[i] == columns[j]).all()):
                return f"{levels[i]!r} and {levels[j]!r}"
    raise AssertionError("no pair of levels ties in every subject")  # pragma: no cover


# Short references for the methods of type nonparametric (SPEC.md section 4).
_WILCOXON_1945: Final = (
    "Wilcoxon, F. (1945). Biometrics Bulletin 1: 80-83 (the signed-rank test)."
)
_KERBY_2014: Final = (
    "Kerby, D. S. (2014). Comprehensive Psychology 3: 11.IT.3.1 (the matched-pairs "
    "rank-biserial correlation)."
)
_FRIEDMAN_1937: Final = (
    "Friedman, M. (1937). Journal of the American Statistical Association 32: "
    "675-701 (Friedman's rank test)."
)
_KENDALL_1939: Final = (
    "Kendall, M. G. & Babington Smith, B. (1939). Annals of Mathematical "
    "Statistics 10: 275-287 (Kendall's W)."
)
_EFRON_TIBSHIRANI_1993: Final = (
    "Efron, B. & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. "
    "Chapman & Hall (percentile bootstrap intervals)."
)
_CONOVER_1999: Final = (
    "Conover, W. J. (1999). Practical Nonparametric Statistics, 3rd ed. Wiley, "
    "section 5.8 (Durbin-Conover pairwise comparisons)."
)


def _tied_subject_resamples_warning(w: stats.BootstrapEffectSizeCI) -> str:
    """D54: the resamples left out of Kendall's W's interval."""
    tied, total = w.undefined_resamples, w.resamples
    return (
        f"{tied:,} of {total:,} bootstrap resamples draw only subjects with the "
        "same value in every condition, where Kendall's W is undefined; its "
        f"confidence interval comes from the other {total - tied:,}."
    )


def _identical_rankings_warning(x: str, n: int) -> str:
    """W = 1: a degenerate bootstrap and infinite Durbin-Conover t."""
    return (
        f"All {n} subjects rank the levels of x={x!r} in the same order, so "
        "Kendall's W is 1 and so is every bootstrap resample: its confidence "
        "interval is [1, 1], which shows the identical rankings, not an effect "
        "known exactly (ggstatsplot's bootstrap fails here and it leaves out the "
        "results subtitle). The Durbin-Conover comparisons then have no error "
        "variance: each t is infinite and each p is 0, as ggstatsplot reports "
        "them; that p is the limit of the t approximation, not an exact "
        "probability, so report Friedman's test for these data."
    )


def _rank_biserial_bound_warning(
    x: str, first: str, second: str, r: stats.EffectSizeCI
) -> str:
    """|r| = 1 collapses the Fisher-z interval to the point r."""
    bound = "1" if r.estimate > 0 else "-1"
    return (
        f"Every nonzero difference between {first!r} and {second!r} of x={x!r} has "
        f"the same sign, so the rank-biserial r is at its bound ({bound}) and its "
        f"confidence interval has no width: [{bound}, {bound}]. The interval is "
        "built on the Fisher z scale, where r = 1 or -1 lies infinitely far out; "
        "it does not mean the effect is known exactly. ggstatsplot prints the "
        "same interval."
    )


def _robust(data: WithinData, args: _Args) -> _Parts:
    """Yuen's test for dependent trimmed means with WRS2's dep.effect AKP (two
    conditions); WRS2's rmanova with the average AKP (wmcpAKP) and rmmcp's
    comparisons (three or more). Every interval of this type is a 95% interval
    whatever conf_level says, as in ggstatsplot (SPEC.md section 4)."""
    x, levels, columns, tr = args.x, data.levels, data.by_level(), args.tr
    _check_within_robust(data, args)
    trimmed = f"{_trim_percent(tr)}% trimmed means"
    pairwise: tuple[PairwiseComparison, ...] = ()
    try:
        if len(levels) == 2:
            a, b = columns
            yuen = stats.yuen_dependent_t(a, b, tr=tr)
            akp = stats.dependent_akp(
                a, b, tr=tr, conf_level=_ROBUST_CONF_LEVEL, seed=args.seed
            )
            test = TestResult(
                name="yuen_dependent_t",
                label=f"Yuen's t-test for dependent samples ({trimmed})",
                method="Yuen's test on trimmed means for dependent samples",
                statistic_name="t",
                statistic=yuen.t,
                df=None,
                df_error=yuen.df,
                p_value=yuen.p_value,
                alternative="two-sided",
            )
            name, label = "akp_delta", "AKP robust standardized difference"
            citations = [_WILCOX_2017, _AKP_2005, _EFRON_TIBSHIRANI_1993]
        else:
            anova = stats.trimmed_rm_anova(columns, tr=tr)
            akp = stats.average_dependent_akp(
                columns, tr=tr, conf_level=_ROBUST_CONF_LEVEL, seed=args.seed
            )
            pairwise = _rmmcp_pairs(levels, columns, args)
            test = TestResult(
                name="trimmed_rm_anova",
                label=f"Repeated-measures trimmed-means ANOVA ({trimmed})",
                method=(
                    "A heteroscedastic one-way repeated measures ANOVA for trimmed "
                    "means"
                ),
                statistic_name="F",
                statistic=anova.f,
                df=anova.df1,
                df_error=anova.df2,
                p_value=anova.p_value,
                alternative="two-sided",
            )
            name, label = "akp_delta_avg", "average AKP robust standardized difference"
            citations = [
                _WILCOX_2017,
                _HUYNH_FELDT_1976,
                _AKP_2005,
                _EFRON_TIBSHIRANI_1993,
                *_within_adjustment_citation(args.p_adjust_method),
            ]
    except ValueError as error:
        raise ValueError(
            _robust_message(str(error), levels, x)
            + " Pass a smaller tr, or use type='nonparametric'."
        ) from None
    effect_size = EffectSize(
        name=name,
        label=label,
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
    warnings = _robust_resample_warnings(akp, effect_size, k=len(levels))
    if args.conf_level != _ROBUST_CONF_LEVEL:
        warnings.append(
            _robust_level_warning(
                args.shown("conf_level", args.conf_level), k=len(levels)
            )
        )
    warnings += _within_no_effect_warnings(data, args, bayes_computed=False)
    return _Parts(
        test=test,
        effect_size=effect_size,
        bayes=None,
        pairwise=pairwise,
        centrality="trimmed_mean",
        centers=tuple(stats.trimmed_mean(column, tr=tr) for column in columns),
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


# ggstatsplot calls dep.effect, wmcpAKP and rmmcp at their default alpha 0.05:
# conf.level reaches none of them (sleep-cl90-robust, bugs_long-cl90-robust).
_ROBUST_CONF_LEVEL: Final = 0.95
_WILCOX_2017: Final = (
    "Wilcox, R. R. (2017). Introduction to Robust Estimation and Hypothesis "
    "Testing, 4th ed. Academic Press (trimmed means of dependent groups: yuend, "
    "rmanova, rmmcp)."
)
_HUYNH_FELDT_1976: Final = (
    "Huynh, H. & Feldt, L. S. (1976). Journal of Educational Statistics 1: 69-82 "
    "(Huynh-Feldt correction of the degrees of freedom)."
)
_AKP_2005: Final = (
    "Algina, J., Keselman, H. J. & Penfield, R. D. (2005). Psychological Methods "
    "10: 317-328 (AKP robust standardized difference)."
)
_CONDITION_NUMBERS: Final = re.compile(r"\bconditions (\d+) and (\d+)")


def _trim_percent(tr: float) -> str:
    """``tr`` as a percentage the way R prints it: 0.2 -> "20", 0.125 -> "12.5"."""
    return r_number_text(tr * 100)


def _robust_message(message: str, levels: Sequence[str], x: str) -> str:
    """A statistics-layer message with its condition numbers replaced by the level
    names (two conditions: the levels added), first letter capitalized and ended
    with one period."""

    def name(match: re.Match[str]) -> str:
        first, second = levels[int(match.group(1)) - 1], levels[int(match.group(2)) - 1]
        return f"the levels {first!r} and {second!r} of x={x!r}"

    out = _CONDITION_NUMBERS.sub(name, message).rstrip().rstrip(".")
    if f"x={x!r}" not in out:
        out = f"{out} (the levels {_level_list(levels)} of x={x!r})"
    return out[0].upper() + out[1:] + "."


def _level_list(levels: Sequence[str]) -> str:
    names = [repr(level) for level in levels]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _check_within_robust(data: WithinData, args: _Args) -> None:
    """Plain errors, naming x and the levels, for data the trimmed-means methods
    cannot use, before any of them runs (SPEC.md section 4)."""
    n, tr = data.n_subjects, args.tr
    g = math.floor(tr * n)
    if n - 2 * g < 2:
        raise ValueError(
            f"Trimming {_trim_percent(tr)}% of the {n} complete subjects from each "
            f"end (tr={tr!r}) leaves {n - 2 * g} at the levels of x={args.x!r}; the "
            "trimmed-means methods need at least 2. Pass a smaller tr (for example "
            "tr=0.1), or collect more subjects."
        )
    columns = data.by_level()
    levels = data.levels
    if len(levels) >= _MIN_CONDITIONS_ANOVA and n < _MIN_SUBJECTS_TRIMMED_ANOVA:
        raise ValueError(
            f"The trimmed-means repeated-measures ANOVA corrects its degrees of "
            f"freedom by the Huynh-Feldt epsilon, which is undefined (0 / 0) with "
            f"{n} complete subjects at the levels of x={args.x!r}; it needs at "
            f"least {_MIN_SUBJECTS_TRIMMED_ANOVA}. Collect more subjects, or use "
            "type='parametric' or type='nonparametric'."
        )
    for i, j in itertools.combinations(range(len(levels)), 2):
        if stats.paired_differences_winsorize_to_constant(
            columns[i], columns[j], tr=tr
        ):
            raise ValueError(
                f"After winsorizing {_trim_percent(tr)}% at each end, every "
                f"subject's difference between the levels {levels[i]!r} and "
                f"{levels[j]!r} of x={args.x!r} is the same (exactly, or to within "
                "the rounding of y in double precision), so the AKP effect size "
                "of the paired differences (their trimmed mean over their winsorized "
                "standard deviation) is undefined. Pass a smaller tr, or use "
                "type='nonparametric'."
            )


def _rmmcp_pairs(
    levels: Sequence[str], columns: Sequence[FloatArray], args: _Args
) -> tuple[PairwiseComparison, ...]:
    """rmmcp's comparison of every pair i < j in level order: the trimmed mean of
    the differences group1 - group2 with its 95% Bonferroni interval, its t and
    df, the raw p adjusted with ``p_adjust_method`` over all pairs; rows in
    ggstatsplot's order (``_sorted_pairs``)."""
    pairs = stats.rmmcp(
        columns,
        tr=args.tr,
        conf_level=_ROBUST_CONF_LEVEL,
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
                test="rmmcp",
                test_label="Yuen's trimmed means",
                significant=significant,
                displayed=_displayed(args.pairwise_display, significant),
                estimate=pair.estimate,
                conf_level=_ROBUST_CONF_LEVEL,
                conf_low=pair.conf_low,
                conf_high=pair.conf_high,
            )
        )
    return _sorted_pairs(rows)


def _robust_level_warning(conf_level: str, *, k: int) -> str:
    """D53's rule for the within methods: conf_level reaches no interval."""
    what = (
        "the AKP effect size's bootstrap interval is always a 95% interval"
        if k == 2
        else "the average AKP effect size's bootstrap interval and the pairwise "
        "intervals are always 95% intervals"
    )
    return (
        f"conf_level={conf_level} has no effect: {what}, as in "
        "ggstatsplot, which does not pass conf.level to WRS2 here; the effect size "
        "records conf_level 0.95."
    )


def _robust_resample_warnings(
    akp: stats.BootstrapEffectSizeCI, effect_size: EffectSize, *, k: int
) -> list[str]:
    """How many bootstrap values were 0 / 0 (left out, D54's rule) or infinite
    (kept); ``check()`` reports the infinite ones as ``infinite_effect_size``.

    With three or more conditions the counts are totals over the pairs, in
    values (resamples times pairs); the result's ``undefined_resamples`` and
    ``infinite_resamples`` are the largest count of one pair, in resamples."""
    pairs = k * (k - 1) // 2
    total = akp.resamples if k == 2 else akp.resamples * pairs
    where = (
        f"{total:,} bootstrap resamples"
        if k == 2
        else f"{total:,} bootstrap values ({akp.resamples:,} resamples of the "
        f"subjects for each of the {pairs} pairs)"
    )
    undefined = akp.undefined_resamples if k == 2 else sum(akp.pair_undefined)
    infinite = akp.infinite_resamples if k == 2 else sum(akp.pair_infinite)
    out: list[str] = []
    if undefined:
        most = (
            ""
            if k == 2
            else f" (at most {akp.undefined_resamples:,} of one pair's "
            f"{akp.resamples:,})"
        )
        out.append(
            f"{undefined:,} of the {where} give an undefined (0/0) AKP{most}, "
            "because in them every winsorized difference is 0. They are left "
            "out: each pair's confidence interval comes from the others."
        )
    if infinite:
        most = (
            ""
            if k == 2
            else f" (at most {akp.infinite_resamples:,} of one pair's "
            f"{akp.resamples:,})"
        )
        out.append(
            f"{infinite:,} of the {where} give an infinite AKP{most}, because "
            "in them every winsorized difference is the same nonzero value. They "
            "are kept, so the confidence interval "
            f"[{format_value(effect_size.conf_low)}, "
            f"{format_value(effect_size.conf_high)}] can be unbounded. Pass a "
            "smaller tr, or use type='nonparametric'."
        )
    return out


def _bayes(data: WithinData, args: _Args) -> _Parts:
    """Two conditions: the paired JZS Bayes factor subtitle (log_e BF01 and the
    prior scale). Three or more conditions, only with results_subtitle=False (the
    Bayesian repeated-measures ANOVA arrives in 0.2): a paired JZS Bayes factor for
    every pair, prior 0.707, unadjusted (D41). MAP estimates of the complete
    subjects' values as centrality."""
    x, levels, columns = args.x, data.levels, data.by_level()
    k = len(levels)
    if k >= 3 and args.results_subtitle:
        raise NotImplementedError(
            f"type='bayes' with {k} conditions: the subtitle ggstatsplot draws here "
            "is a Bayesian repeated-measures ANOVA, which arrives in inferplot 0.2. "
            "Pass results_subtitle=False for the rest of ggstatsplot's output (MAP "
            "estimates and a Bayes factor for every pair), or use type='parametric'."
        )
    bayes: BayesFactor | None = None
    pairwise: tuple[PairwiseComparison, ...] = ()
    warnings: list[str] = []
    if k == 2:
        if args.results_subtitle:
            a, b = columns
            _checked_paired_t(x, levels[0], levels[1], a, b, method=_BF_METHOD)
            bf10, log_bf01 = stats.jzs_bf10_paired(a, b, prior_scale=args.prior_scale)
            bayes = BayesFactor(
                bf10=bf10,
                log_bf01=log_bf01,
                prior="cauchy",
                prior_scale=args.prior_scale,
                test="jzs_paired",
            )
            warnings.append(_POSTERIOR_WARNING)
    else:
        pairwise = _paired_bayes_pairs(x, levels, columns, args)
    centers, map_warnings = _within_map_estimates(x, levels, columns)
    warnings += map_warnings
    warnings += _within_bayes_no_effect_warnings(k, args)
    warnings += _within_no_effect_warnings(data, args, bayes_computed=bayes is not None)
    return _Parts(
        test=None,
        effect_size=None,
        bayes=bayes,
        pairwise=pairwise,
        centrality="map",
        centers=centers,
        warnings=tuple(warnings),
        citations=(_ROUDER_2009, _SHEATHER_JONES_1991),
    )


_SHEATHER_JONES_1991: Final = (
    "Sheather, S. J. & Jones, M. C. (1991). Journal of the Royal Statistical "
    "Society B 53: 683-690 (kernel density bandwidth of the MAP estimate)."
)
_BF_METHOD: Final = "the paired t-test behind the Bayes factor"
_POSTERIOR_WARNING: Final = (
    "The posterior difference and its credible interval, which ggstatsplot adds to "
    "the Bayes factor subtitle, arrive in inferplot 0.2."
)


def _within_map_estimates(
    x: str, levels: Sequence[str], columns: Sequence[FloatArray]
) -> tuple[tuple[float | None, ...], tuple[str, ...]]:
    """Each level's MAP estimate of the complete subjects' values (betweenstats'
    rule, D55): a level without a kernel density has None, no label is drawn for
    it, and a warning says why."""
    out: list[float | None] = []
    warnings: list[str] = []
    for level, column in zip(levels, columns, strict=True):
        try:
            out.append(stats.map_estimate(column).estimate)
        except ValueError as error:
            out.append(None)
            warnings.append(
                f"The condition {level!r} of x={x!r} (n = {column.size}) has no MAP "
                f"estimate, so no centrality label is drawn for it: {error}. "
                "ggstatsplot prints an empty label there. Use type='nonparametric' "
                "to show medians."
            )
    return tuple(out), tuple(warnings)


def _paired_bayes_pairs(
    x: str, levels: Sequence[str], columns: Sequence[FloatArray], args: _Args
) -> tuple[PairwiseComparison, ...]:
    """A paired JZS Bayes factor for every pair i < j in level order (the paired t
    of group1 minus group2), prior 0.707 whatever ``bf_prior`` is, not adjusted;
    every pair drawn unless ``pairwise_display`` is "none"; rows in ggstatsplot's
    order (``_sorted_pairs``)."""
    k = len(levels)
    rows: list[PairwiseComparison] = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = columns[i], columns[j]
            paired = _checked_paired_t(x, levels[i], levels[j], a, b, method=_BF_METHOD)
            bf10, log_bf01 = stats.jzs_bf10_paired(a, b, prior_scale=_PAIRWISE_BF_PRIOR)
            rows.append(
                PairwiseComparison(
                    group1=levels[i],
                    group2=levels[j],
                    statistic_name="t",
                    statistic=paired.t,
                    df=paired.df,
                    p_value=None,
                    p_adjusted=None,
                    adjust_method="none",
                    test="jzs_paired",
                    test_label="Student's t",
                    significant=None,
                    displayed=args.pairwise_display != "none",
                    bayes=BayesFactor(
                        bf10=bf10,
                        log_bf01=log_bf01,
                        prior="cauchy",
                        prior_scale=_PAIRWISE_BF_PRIOR,
                        test="jzs_paired",
                    ),
                )
            )
    return _sorted_pairs(rows)


def _within_bayes_no_effect_warnings(k: int, args: _Args) -> list[str]:
    """Arguments type bayes does not use (D39): conf_level (no interval in v0.1),
    and with three or more conditions the significance and adjustment arguments
    (Bayes factors are neither tested nor adjusted; every pair is drawn)."""
    out: list[str] = []
    if args.conf_level != 0.95:
        out.append(
            f"conf_level={args.shown('conf_level', args.conf_level)} has no "
            "effect: type='bayes' shows no "
            "interval in this version (the posterior credible interval arrives in "
            "inferplot 0.2)."
        )
    if k >= 3:
        if args.p_adjust_method != _DEFAULT_P_ADJUST:
            out.append(
                f"p_adjust_method={args.shown('p_adjust_method', args.p_adjust_method)}"
                " has no effect: Bayes "
                "factors are not adjusted for multiple comparisons (as in "
                "ggstatsplot)."
            )
        if args.pairwise_alpha != _DEFAULT_ALPHA:
            out.append(
                f"pairwise_alpha={args.shown('pairwise_alpha', args.pairwise_alpha)}"
                " has no effect: Bayes factors "
                "are not compared with a significance level, and every pair is "
                "drawn."
            )
        if args.pairwise_display == "non-significant":
            display = args.shown("pairwise_display", args.pairwise_display)
            out.append(
                f"pairwise_display={display} has no effect: Bayes "
                "factors have no significance, so every pair is drawn (as in "
                "ggstatsplot); pass pairwise_display='none' to draw none."
            )
    return out


_ANALYSES: Final[dict[str, Callable[[WithinData, _Args], _Parts]]] = {
    "parametric": _parametric,
    "nonparametric": _nonparametric,
    "robust": _robust,
    "bayes": _bayes,
}


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def _plural(items: tuple[str, ...], one: str, many: str) -> str:
    return one if len(items) == 1 else many


def _data_warnings(data: WithinData) -> list[str]:
    """Plain-English notes on what the data reading left out. Subject labels are
    not named: warnings are part of the copy ``save()`` embeds in figure files,
    which leaves the user's data out (D36)."""
    out: list[str] = []
    if data.empty_levels:
        empty = data.empty_levels
        names = ", ".join(repr(level) for level in empty)
        out.append(
            f"The declared {_plural(empty, 'level', 'levels')} {names} of {data.x!r} "
            f"{_plural(empty, 'has', 'have')} no rows with x, y and subject_id all "
            f"present and {_plural(empty, 'was', 'were')} left out of the analysis."
        )
    out += data_notes(data)
    dropped = data.dropped_subjects
    if dropped:
        out.append(
            f"{len(dropped)} {_plural(dropped, 'subject was', 'subjects were')} left "
            f"out of the analysis and the plot because "
            f"{_plural(dropped, 'it lacks', 'they lack')} a value of {data.y!r} at "
            f"some level of {data.x!r}; {data.n_subjects} complete subjects remain."
        )
    return out


def _assemble(
    data: WithinData, args: _Args, parts: _Parts, *, digits: int
) -> StatsResult:
    """The result of a within-subjects analysis from the parts its type computed.

    ``values`` holds, per level, the complete subjects' values in ``subjects``
    order (the columns of the subjects-by-levels matrix), and ``n_obs`` the number
    of complete subjects.
    """
    columns = data.by_level()
    n_subjects = data.n_subjects
    descriptives = tuple(
        Descriptive(level=name, n=n_subjects, centrality=parts.centrality, value=c)
        for name, c in zip(data.levels, parts.centers, strict=True)
    )
    settings = Settings(
        conf_level=args.conf_level,
        digits=digits,
        alternative=args.alternative,
        p_adjust_method=args.p_adjust_method,
        pairwise_display=args.pairwise_display,
        pairwise_alpha=args.pairwise_alpha,
        bf_prior=args.prior_scale,
        bf_message=args.bf_message,
        results_subtitle=args.results_subtitle,
        seed=args.seed,
        tr=args.tr,
    )
    texts = build_texts(
        x=data.x,
        y=data.y,
        descriptives=descriptives,
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=parts.pairwise,
        n_obs=n_subjects,
        settings=settings,
        design="within",
    )
    observations = tuple(tuple(float(v) for v in col.tolist()) for col in columns)
    return StatsResult(
        analysis="withinstats",
        design="within",
        type=args.type,
        x=data.x,
        y=data.y,
        subject_id=data.subject_id,
        levels=data.levels,
        values=observations,
        subjects=data.subjects,
        descriptives=descriptives,
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=parts.pairwise,
        n_obs=n_subjects,
        n_dropped=data.n_dropped,
        dropped_subjects=len(data.dropped_subjects),
        dropped_subject_rows=data.n_rows_of_dropped_subjects,
        n_tied=paired_tied_count(observations),
        warnings=(*_data_warnings(data), *parts.warnings),
        provenance=current_provenance(seed=args.seed, citations=parts.citations),
        settings=settings,
        text=texts,
    )


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
        raise ValueError(
            f"tr={tr!r} sets the trimming of the robust methods (type='robust'), but "
            f"type={type_name!r} trims nothing, so it would have no effect. Remove "
            "tr=, or pass type='robust' to compare trimmed means."
        )
    if side != "two-sided" and type_name in ("robust", "bayes"):
        why = (
            "the trimmed-means tests are always two-sided"
            if type_name == "robust"
            else "the Bayes factor is always two-sided"
        )
        raise ValueError(
            f"alternative={side!r} applies to type='parametric' and "
            f"type='nonparametric' only: with type={type_name!r} {why} (ggstatsplot "
            "silently ignores alternative there). Remove alternative=, or use "
            "type='nonparametric' for a one-sided signed-rank test."
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


def _check_design(data: WithinData, args: _Args) -> None:
    """Arguments that cannot apply to the conditions found in the data."""
    k = len(data.levels)
    if k >= 3 and args.alternative != "two-sided":
        raise ValueError(
            f"alternative={args.alternative!r} applies only to two conditions; "
            f"x={args.x!r} has {k}, where the omnibus test and the pairwise "
            "comparisons are always two-sided. Remove alternative= or compare two "
            "conditions."
        )
    if k >= 3 and args.type == "parametric" and args.bf_message is True:
        raise NotImplementedError(
            "bf_message=True: the Bayes factor caption for three or more conditions "
            "(a Bayesian repeated-measures ANOVA) arrives in inferplot 0.2. Remove "
            "bf_message=True."
        )


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


@checked
def withinstats(
    data: object,
    x: str,
    y: str,
    *,
    subject_id: str,
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
    """Compare ``y`` across the levels of ``x`` measured on the same subjects; no
    plotting.

    The counterpart of ggstatsplot 1.1.4's ``ggwithinstats()``. ``data`` is in long
    format: one row per subject and level of ``x``, with ``subject_id`` naming the
    column that identifies the subject. Measurements are paired by that column,
    never by row order. Subjects without a value at every level are left out (and
    counted in ``dropped_subjects``), as ggstatsplot does; n is the number of
    complete subjects.

    The methods follow ggstatsplot 1.1.4's ``ggwithinstats()`` for each ``type``
    (SPEC.md section 4):

    * "parametric" ("p"): the paired t-test with Hedges' g of the differences and
      the paired JZS Bayes factor caption (two conditions); the repeated-measures
      ANOVA with the Greenhouse-Geisser correction, partial omega squared and
      paired t-tests of every pair (three or more). Centrality: means.
    * "nonparametric" ("np"): the Wilcoxon signed-rank test with the matched-pairs
      rank-biserial r (two conditions); Friedman's test with Kendall's W
      (bootstrap CI) and Durbin-Conover comparisons (three or more). Centrality:
      medians.
    * "robust" ("r"): Yuen's test for dependent ``tr``-trimmed means with the AKP
      robust standardized difference of the paired differences (two
      conditions); the repeated-measures trimmed-means ANOVA with the average AKP
      over the pairs and rmmcp's comparisons (three or more). The bootstrap and
      pairwise intervals are always 95%, as in ggstatsplot. Centrality: trimmed
      means.
    * "bayes" ("bf"): the paired JZS Bayes factor, reported as log_e(BF01) with
      the prior scale ``bf_prior`` (two conditions); with three or more
      conditions only with ``results_subtitle=False`` (the Bayesian
      repeated-measures ANOVA arrives in 0.2): a paired Bayes factor for every
      pair, prior 0.707, unadjusted. Centrality: MAP estimates. Posterior
      estimates arrive in 0.2.

    Args:
        data: a pandas, polars or pyarrow data frame in long format.
        x: the column of conditions (the repeated factor).
        y: the numeric outcome column.
        subject_id: the column that identifies each subject. Required.
        type: "parametric" ("p"), "nonparametric" ("np"), "robust" ("r") or
            "bayes" ("bf").
        pairwise_display: which pairs the figure draws: "significant" ("s"),
            "non-significant" ("ns"), "all" or "none".
        p_adjust_method: R's ``p.adjust`` method for the pairwise p-values.
        pairwise_alpha: a pair is significant when its adjusted p is below this.
        conf_level: confidence level of the effect-size interval.
        digits: decimals in the subtitle, caption and labels.
        tr: trimming proportion of the robust methods; another value than 0.2
            with a type that trims nothing raises.
        bf_prior: scale of the Cauchy prior of the Bayes factor (0.707).
        bf_message: None draws the Bayes factor caption where it is available;
            True requires it; False never draws it.
        results_subtitle: whether the statistical results are reported as a
            subtitle.
        alternative: "two-sided", "less" or "greater"; one-sided only with two
            conditions and type "parametric" or "nonparametric" (the paired
            t-test, the Wilcoxon signed-rank test). "greater" tests whether the
            first level of x is greater than the second (the differences first
            minus second are positive), and "less" whether it is less. The first
            level is the first in ``order`` when given, else the first declared
            category, else the first in sorted order: pass ``order`` to fix the
            direction. The APA sentence and the alt text name it ("Paired t-test
            (one-sided: pre less than post)"); a program reads it from
            ``result.test.alternative`` and ``result.levels``.
        seed: random seed of the bootstrap intervals, recorded in the result.
        order: the level order, a permutation of the levels present.

    Returns:
        A :class:`~inferplot.result.StatsResult` holding every number and text.

    Raises:
        TypeError, ValueError: a bad argument or unusable data (a subject repeated
            within a level, fewer than two complete subjects or two levels); the
            message says how to fix the call. A call without ``subject_id`` raises
            ``TypeError``.
        KeyError: a column that the data frame does not have.
        NotImplementedError: what arrives in inferplot 0.2 (the Bayesian
            repeated-measures ANOVA: type "bayes" with three or more conditions
            unless ``results_subtitle=False``, and ``bf_message=True`` there).
    """
    if subject_id is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise subject_id_required("withinstats", given=describe_value(subject_id))
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

    within = read_within(data, x, y, subject_id, order=order)
    args = _Args(
        x=x,
        subject_id=subject_id,
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
        passed={
            "p_adjust_method": p_adjust_method,
            "pairwise_display": pairwise_display,
            "pairwise_alpha": pairwise_alpha,
            "conf_level": conf_level,
            "bf_prior": bf_prior,
        },
    )
    _check_design(within, args)
    parts = _ANALYSES[type_name](within, args)
    return _assemble(within, args, parts, digits=n_digits)
