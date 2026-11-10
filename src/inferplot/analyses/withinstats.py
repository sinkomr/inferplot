"""``withinstats``: compare a numeric ``y`` across repeated measures of subjects.

The analysis that ``ip.analyze.withinstats`` exposes, the counterpart of ggstatsplot
1.1.4's ``ggwithinstats()``. It reads the data through
:func:`inferplot._data.read_within` (complete subjects only, paired by
``subject_id``, never by row order: SPEC.md section 5 item 1), validates every
argument with :mod:`inferplot._validate`, and returns a
:class:`~inferplot.result.StatsResult` with ``design="within"``. It never imports
matplotlib.

The statistics of each type (SPEC.md section 4, "withinstats") are not implemented
yet: every type raises ``NotImplementedError`` naming the type and the number of
conditions, after every argument has been validated and the data read. Each type's
analysis is a function ``(data, args) -> _Parts`` in ``_ANALYSES``; ``_assemble``
builds the result from its parts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

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
from inferplot.text import build_texts

__all__ = ["withinstats"]

# ggstatsplot's default trimming; another value is allowed only where it is used.
_DEFAULT_TR: Final = 0.2

# The method of each type, for the messages of the types not implemented yet
# (SPEC.md section 4, "withinstats").
_METHODS: Final[dict[tuple[str, bool], str]] = {
    ("parametric", False): "the paired t-test",
    ("parametric", True): "the repeated-measures ANOVA",
    ("nonparametric", False): "the Wilcoxon signed-rank test",
    ("nonparametric", True): "the Friedman test",
    ("robust", False): "Yuen's test for dependent trimmed means",
    ("robust", True): "the repeated-measures trimmed-means ANOVA",
    ("bayes", False): "the paired Bayes factor",
    ("bayes", True): "the Bayesian repeated-measures ANOVA",
}


def _not_implemented(type_name: str, data: WithinData) -> NotImplementedError:
    """The error of a type whose analysis is not implemented yet (CLAUDE.md rule 3:
    no fallback to another method)."""
    k = len(data.levels)
    method = _METHODS[type_name, k >= 3]
    later = (
        "arrives in inferplot 0.2"
        if (type_name, k >= 3) == ("bayes", True)
        else "is not implemented yet in this version of inferplot"
    )
    return NotImplementedError(
        f"withinstats(type={type_name!r}) with {k} conditions ({method}) {later}. "
        f"The data were read: {data.n_subjects} subjects with a value at each of "
        f"the {k} levels of x={data.x!r}."
    )


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
    """Paired t-test with Hedges' g (two conditions); repeated-measures ANOVA with
    partial omega squared and paired t pairwise comparisons (three or more)."""
    raise _not_implemented(args.type, data)


def _nonparametric(data: WithinData, args: _Args) -> _Parts:
    """Wilcoxon signed-rank test with the rank-biserial r (two conditions);
    Friedman test with Kendall's W and Durbin-Conover comparisons (three or more)."""
    raise _not_implemented(args.type, data)


def _robust(data: WithinData, args: _Args) -> _Parts:
    """Yuen's test for dependent trimmed means with WRS2's dep.effect (two
    conditions); WRS2's rmanova with rmmcp comparisons (three or more)."""
    raise _not_implemented(args.type, data)


def _bayes(data: WithinData, args: _Args) -> _Parts:
    """The paired JZS Bayes factor (two conditions); three or more: inferplot 0.2."""
    raise _not_implemented(args.type, data)


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
    )
    return StatsResult(
        analysis="withinstats",
        design="within",
        type=args.type,
        x=data.x,
        y=data.y,
        subject_id=data.subject_id,
        levels=data.levels,
        values=tuple(tuple(float(v) for v in col.tolist()) for col in columns),
        subjects=data.subjects,
        descriptives=descriptives,
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=parts.bayes,
        pairwise=parts.pairwise,
        n_obs=n_subjects,
        n_dropped=data.n_dropped,
        dropped_subjects=len(data.dropped_subjects),
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
    counted in ``dropped_subjects``), as ggstatsplot does.

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
            conditions and type "parametric" or "nonparametric".
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
        NotImplementedError: this version computes no within-subjects statistics
            yet; the message names the type and the number of conditions.
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
    )
    _check_design(within, args)
    parts = _ANALYSES[type_name](within, args)
    return _assemble(within, args, parts, digits=n_digits)
