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
``_assemble`` builds the result from its parts. The statistics are not in this
version yet: every type raises ``NotImplementedError`` naming itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

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
from inferplot.text import build_texts, r_number_text

__all__ = ["histostats"]


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
# the types (each takes the analyzed values and the validated arguments)
# --------------------------------------------------------------------------- #

# The method of each type, for the messages of the types not implemented yet.
_METHODS: Final[dict[str, str]] = {
    "parametric": "the one-sample t-test with Hedges' g",
    "nonparametric": "the one-sample Wilcoxon signed-rank test",
    "robust": "the bootstrap-t test of the trimmed mean",
    "bayes": "the one-sample Bayes factor",
}


def _not_implemented(data: OneData, args: _Args) -> NotImplementedError:
    """The error of a type whose statistics are not in this version yet
    (CLAUDE.md rule 3: no fallback to another method)."""
    return NotImplementedError(
        f"histostats(type={args.type!r}) ({_METHODS[args.type]}) is not implemented "
        "yet in this version of inferplot; it arrives with the Phase 4 histostats "
        f"statistics. The data were read: {data.n_obs} values of x={data.x!r}, to "
        f"be tested against {r_number_text(args.test_value)}."
    )


def _parametric(data: OneData, args: _Args) -> _Parts:
    """One-sample t-test against ``test_value`` with Hedges' g and the one-sample
    JZS Bayes factor caption; the mean as centrality."""
    raise _not_implemented(data, args)


def _nonparametric(data: OneData, args: _Args) -> _Parts:
    """One-sample Wilcoxon signed-rank test of x - test_value with the
    rank-biserial r; the median as centrality."""
    raise _not_implemented(data, args)


def _robust(data: OneData, args: _Args) -> _Parts:
    """Bootstrap-t test of the trimmed mean (WRS2's trimcibt) with the trimmed
    mean and its bootstrap-t CI; the trimmed mean as centrality."""
    raise _not_implemented(data, args)


def _bayes(data: OneData, args: _Args) -> _Parts:
    """One-sample JZS Bayes factor; the MAP estimate as centrality."""
    raise _not_implemented(data, args)


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
    if parts.bayes is not None and args.alternative != "two-sided":
        out.append(
            "The Bayes factor in the caption is two-sided: it weighs any difference "
            f"from test_value, while alternative={args.alternative!r} applies to "
            "the test and its effect size interval only (as in ggstatsplot)."
        )
    return out


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
    analysis on them; tests use it to build results before the statistics
    exist."""
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
        passed={"bf_prior": bf_prior, "conf_level": conf_level},
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
    * "robust" ("r"): the bootstrap-t test of the ``tr``-trimmed mean, reporting
      the trimmed mean with its bootstrap-t interval. Centrality: the trimmed
      mean.
    * "bayes" ("bf"): the one-sample JZS Bayes factor, reported as log_e(BF01)
      with the prior scale ``bf_prior``. Centrality: the MAP estimate. Posterior
      estimates arrive in 0.2.

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
            result.

    Returns:
        A :class:`~inferplot.result.StatsResult` holding every number and text.

    Raises:
        TypeError, ValueError: a bad argument or unusable data (a column that is
            not numeric, fewer than two values); the message says how to fix the
            call.
        KeyError: a column that the data frame does not have.
        NotImplementedError: the statistics of each type are not in this version
            yet.
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
