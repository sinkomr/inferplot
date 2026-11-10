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
``_assemble`` builds the result from its parts. The statistics are not in this
version yet: every type raises ``NotImplementedError`` naming itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

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
from inferplot.text import build_texts

__all__ = ["scatterstats"]


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

# The method of each type, for the messages of the types not implemented yet.
_METHODS: Final[dict[str, str]] = {
    "parametric": "Pearson's correlation",
    "nonparametric": "Spearman's rank correlation",
    "robust": "the winsorized correlation",
    "bayes": "the Bayesian Pearson correlation",
}


def _not_implemented(data: PairData, args: _Args) -> NotImplementedError:
    """The error of a type whose statistics are not in this version yet
    (CLAUDE.md rule 3: no fallback to another method)."""
    return NotImplementedError(
        f"scatterstats(type={args.type!r}) ({_METHODS[args.type]}) is not "
        "implemented yet in this version of inferplot; it arrives with the Phase 4 "
        f"scatterstats statistics. The data were read: {data.n_obs} complete pairs "
        f"of x={data.x!r} and y={data.y!r}."
    )


def _parametric(data: PairData, args: _Args) -> _Parts:
    """Pearson's r with its t test and Fisher z interval, and the Bayesian Pearson
    correlation caption."""
    raise _not_implemented(data, args)


def _nonparametric(data: PairData, args: _Args) -> _Parts:
    """Spearman's rho with the t-approximation p and the Fieller, Hartley and
    Pearson interval."""
    raise _not_implemented(data, args)


def _robust(data: PairData, args: _Args) -> _Parts:
    """The winsorized correlation at ``tr``, tested as Pearson's r."""
    raise _not_implemented(data, args)


def _bayes(data: PairData, args: _Args) -> _Parts:
    """The Bayesian Pearson correlation's Bayes factor."""
    raise _not_implemented(data, args)


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
        warnings=(*data_notes(data), *parts.warnings, *warnings),
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
    analysis on them; tests use it to build results before the statistics
    exist."""
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
    * "nonparametric" ("np"): Spearman's rho.
    * "robust" ("r"): the correlation of x and y each winsorized at ``tr``,
      tested as Pearson's r.
    * "bayes" ("bf"): the Bayes factor of the Bayesian Pearson correlation, with
      prior width ``bf_prior``. The posterior estimate arrives in 0.2.

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
        seed: random seed, recorded in the result.

    Returns:
        A :class:`~inferplot.result.StatsResult` holding every number and text;
        ``result.point_labels`` holds the label of each analyzed pair.

    Raises:
        TypeError, ValueError: a bad argument or unusable data (a column that is
            not numeric, x and y the same column, fewer than three complete
            pairs); the message says how to fix the call.
        KeyError: a column that the data frame does not have.
        NotImplementedError: the statistics of each type are not in this version
            yet.
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
