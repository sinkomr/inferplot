"""``betweenstats``: compare a numeric ``y`` across independent groups of ``x``.

The analysis that ``ip.analyze.betweenstats`` exposes. It reads the data through
:mod:`inferplot._data`, validates every argument with :mod:`inferplot._validate`,
computes every number with :mod:`inferplot.stats` and every text with
:mod:`inferplot.text`, and returns a :class:`~inferplot.result.StatsResult`. It never
imports matplotlib.

What it reproduces is ggstatsplot 1.1.4's ``ggbetweenstats()`` (SPEC.md section 4):

* two groups, type parametric: Welch's t-test, Hedges' g with a noncentral-t CI that
  follows ``alternative``, and the two-sided JZS Bayes factor caption (D28, D32);
* three or more groups: Welch's ANOVA, partial omega squared with a one-sided
  noncentral-F CI, and Games-Howell pairwise comparisons adjusted with
  ``p_adjust_method`` over all k(k - 1)/2 pairs. The raw Games-Howell p-values come
  from an accurate studentized range tail, not from R's ``ptukey`` (D31).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from inferplot import stats
from inferplot._data import read_between
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
)
from inferplot.stats.parametric import TINY
from inferplot.text import build_texts

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

_HEDGES_J_ZERO_WARNING: Final = (
    "Hedges' g is 0 by construction at 1 degree of freedom (Welch's df is 1 here: "
    "one group has no variability and the other has two values), so g and its "
    "confidence interval carry no information about the size of the difference."
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
            "span more than the largest number a double can hold (about 1.8e308), "
            "so their standard deviation cannot be represented. Rescale y, for "
            "example by changing its units or dividing it by 1e10."
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
        if k >= 3:
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


def _pairwise(
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

    Two groups: Welch's t-test, Hedges' g (unpooled standardizer, exact small-sample
    correction at the Welch df) with a noncentral-t CI, and the JZS Bayes factor.
    Three or more groups: Welch's ANOVA, partial omega squared with a one-sided
    noncentral-F CI, and Games-Howell comparisons of every pair adjusted with
    ``p_adjust_method``. Defaults are ggstatsplot 1.1.4's ``ggbetweenstats()``.

    Args:
        data: a pandas, polars or pyarrow data frame in long format.
        x: the grouping column.
        y: the numeric outcome column.
        type: "parametric" (alias "p"); "nonparametric", "robust" and "bayes" are
            not available yet in this development version.
        pairwise_display: which pairs the figure draws: "significant" ("s"),
            "non-significant" ("ns"), "all" or "none". Every pair is computed.
        p_adjust_method: R's ``p.adjust`` method for the pairwise p-values.
        pairwise_alpha: a pair is significant when its adjusted p is below this.
        conf_level: confidence level of the effect-size interval.
        digits: decimals in the subtitle, caption and labels.
        tr: trimming proportion of the robust methods; another value than 0.2
            with type="parametric" raises, since nothing would be trimmed.
        bf_prior: scale of the Cauchy prior of the Bayes factor (0.707).
        bf_message: None draws the Bayes factor caption where it is available;
            True requires it; False never draws it.
        results_subtitle: whether the statistical results are reported as a
            subtitle (False also drops the Bayes factor caption).
        alternative: "two-sided", "less" or "greater"; one-sided only with two
            groups.
        seed: random seed, recorded in the result (no random step here).
        order: the level order, a permutation of the levels present.

    Returns:
        A :class:`~inferplot.result.StatsResult` holding every number and text.

    Raises:
        TypeError, ValueError: a bad argument or unusable data; the message says
            how to fix the call.
        KeyError: a column that the data frame does not have.
        NotImplementedError: a type, or a Bayes factor caption, that this version
            does not have.
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
    if type_name != "parametric":
        shown = repr(type) if type == type_name else f"{type!r} ({type_name})"
        raise NotImplementedError(
            f"type={shown} is not available yet in this development version of "
            "inferplot; only type='parametric' exists for now."
        )
    if trim != _DEFAULT_TR:
        # ggstatsplot silently ignores tr outside the robust type; inferplot never
        # drops an argument (CLAUDE.md rule 3), as for alternative (SPEC section 2).
        raise ValueError(
            f"tr={tr!r} sets the trimming of the robust methods (type='robust'), but "
            f"type={type_name!r} trims nothing, so it would have no effect. Remove "
            "tr=; the robust methods arrive later in this development version."
        )
    if show_bf is True and not subtitle_on:
        raise ValueError(
            "bf_message=True asks for the Bayes factor caption, which is drawn only "
            "together with the results subtitle; results_subtitle=False turns both "
            "off. Pass results_subtitle=True, or drop bf_message=True."
        )

    between = read_between(data, x, y, order=order)
    levels, values = between.levels, between.values
    k = len(levels)
    if k >= 3 and side != "two-sided":
        raise ValueError(
            f"alternative={side!r} applies only to two groups; x={x!r} has {k} "
            "groups, where Welch's ANOVA and the pairwise comparisons are always "
            "two-sided. Remove alternative= or compare two groups."
        )
    if k >= 3 and show_bf is True:
        raise NotImplementedError(
            "bf_message=True: the Bayes factor caption for three or more groups (a "
            "Bayesian ANOVA) arrives in inferplot 0.2. Remove bf_message=True; the "
            "default (None) leaves the caption out and says so in result.warnings."
        )
    _check_groups(
        x, levels, values, bayes=k == 2 and subtitle_on and show_bf is not False
    )

    warnings: list[str] = []
    if between.empty_levels:
        empty = between.empty_levels
        warnings.append(
            f"The declared {'level' if len(empty) == 1 else 'levels'} "
            f"{_names(empty)} of {x!r} {'has' if len(empty) == 1 else 'have'} no "
            "observations with both x and y present and "
            f"{'was' if len(empty) == 1 else 'were'} left out of the analysis."
        )

    citations: list[str] = []
    bayes: BayesFactor | None = None
    pairwise: tuple[PairwiseComparison, ...] = ()
    if k == 2:
        a, b = values
        welch = stats.welch_t(a, b, alternative=side)
        g = stats.hedges_g(a, b, conf_level=level, alternative=side)
        test = TestResult(
            name="welch_t",
            label="Welch's t-test",
            method="Welch Two Sample t-test",
            statistic_name="t",
            statistic=welch.t,
            df=None,
            df_error=welch.df,
            p_value=welch.p_value,
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
        if stats.hedges_j(welch.df) == 0.0:
            warnings.append(_HEDGES_J_ZERO_WARNING)
        if subtitle_on and show_bf is not False:
            bf10, log_bf01 = stats.jzs_bf10_two_sample(a, b, prior_scale=prior_scale)
            bayes = BayesFactor(
                bf10=bf10,
                log_bf01=log_bf01,
                prior="cauchy",
                prior_scale=prior_scale,
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
        pairwise = _pairwise(
            levels,
            values,
            p_adjust_method=adjust,
            pairwise_alpha=alpha,
            pairwise_display=display,
        )
        citations += [_WELCH_1951, _OLEJNIK_ALGINA_2003, _STEIGER_2004]
        citations.append(_GAMES_HOWELL_1976)
        if adjust in _ADJUSTMENT_CITATIONS:
            citations.append(_ADJUSTMENT_CITATIONS[adjust])
        if subtitle_on and show_bf is None:
            warnings.append(_BAYES_K3_WARNING)

    warnings += _no_effect_warnings(
        x=x,
        k=k,
        bayes_computed=bayes is not None,
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
        Descriptive(level=name, n=int(v.size), centrality="mean", value=stats.mean(v))
        for name, v in zip(levels, values, strict=True)
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
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        n_obs=between.n_obs,
        settings=settings,
    )
    return StatsResult(
        analysis="betweenstats",
        design="between",
        type=type_name,
        x=x,
        y=y,
        subject_id=None,
        levels=levels,
        values=tuple(tuple(float(v) for v in arr.tolist()) for arr in values),
        descriptives=descriptives,
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        n_obs=between.n_obs,
        n_dropped=between.n_dropped,
        dropped_subjects=0,
        warnings=tuple(warnings),
        provenance=current_provenance(seed=rng_seed, citations=tuple(citations)),
        settings=settings,
        text=texts,
    )
