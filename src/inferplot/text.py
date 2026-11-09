"""Every text inferplot puts on a figure or into a result.

ggstatsplot 1.1.4 draws its subtitle, Bayes caption, bracket labels and centrality
labels as R plotmath expressions built by statsExpressions 2.2.1 (MIT) from glue
templates. This module builds the *same* expression as a small AST (:class:`Str`,
:class:`Num`, :class:`Sym`, :class:`Call`, :class:`BinOp`, :class:`Index`,
:class:`Paren`, mirroring R's parse tree) and renders it two ways:

* :func:`to_mathtext`: matplotlib mathtext, drawn with the bundled DejaVu Sans;
* :func:`to_plain`: plain Unicode, stored in :class:`~inferplot.result.Texts`.

Parity with ggstatsplot is AST equality with the fixture's deparsed expression
(tests parse it with ``tests/plotmath.py``), so every symbol, italic, subscript and
number string is checked, not only the numbers.

Number formatting reproduces statsExpressions' four routes (read from its MIT
sources ``R/add-expression-col.R``, ``R/pairwise-comparisons.R`` and
``R/centrality-description.R``):

* quoted statistics, p-values, estimates, CI bounds, prior scales and log Bayes
  factors: ``insight::format_value(x, digits)`` -> :func:`format_value`;
* degrees of freedom: ``format_value(df, digits.df)`` pasted *unquoted*, so R parses
  the text back into a number and prints it as R prints a double ->
  :func:`format_df`;
* sample sizes: ``format_value(n, digits = 0, big_mark = ",")`` -> :func:`format_n`;
* the CI level: ``paste0(conf.level * 100L, "%")`` -> :func:`format_conf_level`.

insight (GPL-3) source was not read: :func:`format_value`'s rules come from insight's
reference manual, the oracle fixtures (every number string of every fixture) and the
oracle probe ``probe-format`` (``tests/fixtures/oracle/probe-format.json``), which
records ``insight::format_value``, statsExpressions' ``.to_char`` and ``.prettyNum``,
the df route and the CI-level label in R for a grid of 1,002 doubles (decimal ties
and their neighbours one ulp away, subnormals, the largest double, Inf, NA, NaN) at
digits 0 to 4. The functions below reproduce every one of those 5,010 rows. The one
rule the probe could not pin is kept in :data:`UNPINNED_KEEP_NEGATIVE_ZERO`.

This module imports nothing outside the standard library and the result types; in
particular it never imports matplotlib.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from inferplot.result import (
    BayesFactor,
    Descriptive,
    EffectSize,
    PairwiseComparison,
    Settings,
    StatsResult,
    TestResult,
    Texts,
)

__all__ = [
    "ALT_TEXT_MAX_CHARS",
    "DOUBLE_XMIN",
    "EFFECT_SIZE_SYMBOLS",
    "N_BIG_MARK",
    "PRIOR_SYMBOLS",
    "SCI_LARGE_THRESHOLD",
    "STATISTIC_SYMBOLS",
    "UNPINNED_KEEP_NEGATIVE_ZERO",
    "BinOp",
    "Call",
    "FigureTexts",
    "Index",
    "Node",
    "Num",
    "Paren",
    "Str",
    "Sym",
    "alt_text",
    "apa_sentence",
    "bayes_expression",
    "bayes_factor_expression",
    "bayes_sentence",
    "build_texts",
    "caption_lines",
    "centrality_expression",
    "figure_texts",
    "format_bf01_log",
    "format_bf10",
    "format_conf_level",
    "format_df",
    "format_n",
    "format_p_subtitle",
    "format_value",
    "frequentist_expression",
    "p_adjust_label",
    "pairwise_bayes_sentence",
    "pairwise_bf_expression",
    "pairwise_caption_line",
    "pairwise_p_expression",
    "r_number_text",
    "results_sentence",
    "subtitle_expression",
    "tick_label",
    "to_mathtext",
    "to_plain",
]

# =========================================================================== #
# Number formatting
# =========================================================================== #

# Pinned by probe-format ("values"): a non-zero |x| at or above 1e5 prints in
# scientific notation, judged on the unrounded value: 99999.996 prints "100000.00" at
# digits 2 and 1e5 prints "1.00e+05"; 99999.99999999997 prints "100000" at digits 0.
# The largest double and 1e300 follow the same rule.
SCI_LARGE_THRESHOLD: Final[float] = 1e5

# Pinned by probe-format ("values"): small numbers switch to scientific notation when
# log10(|x|) < -digits, so the threshold moves with digits: 0.000999 prints "1e-03"
# at digits 0, "9.990e-04" at digits 3 and "0.0010" at digits 4; 0.0095 prints
# "9.50e-03" at digits 2 and "0.009" at digits 3. Like the large switch it is judged
# on the unrounded value (0.0099999 prints "1.00e-02" at digits 2, "0.010" at 3).
#
# Also pinned there: NA and NaN print as the ``missing`` text, "" from format_value
# itself and "NA" through statsExpressions' .to_char; Inf and -Inf print as "Inf" and
# "-Inf" whatever digits is.

# UNPINNED: an exact negative zero prints as C sprintf prints it ("-0.00"). The probe
# could not settle this: R handed its -0 grid point and the end-to-end estimate -0 to
# format_value as +0 (every "values" row reports negative_zero false, and the
# end-to-end subtitle prints "0.00"), so no row shows what format_value does with -0.
UNPINNED_KEEP_NEGATIVE_ZERO: Final[bool] = True

# Pinned by probe-format ("counts"): statsExpressions' .prettyNum puts a comma every
# three digits from 1,000 on and never switches to scientific notation, for doubles
# and integers alike ("100,000", "1,000,000,000").
N_BIG_MARK: Final[str] = ","

# R's .Machine$double.xmin: add_expression_col() replaces an exact 0 p-value by it.
DOUBLE_XMIN: Final[float] = sys.float_info.min


def format_value(x: float | None, digits: int = 2, *, missing: str = "NA") -> str:
    """Format ``x`` as ``insight::format_value(x, digits)`` does in ggstatsplot.

    Fixed notation with ``digits`` decimals, unless ``x`` is non-zero and
    ``log10(|x|) < -digits`` or ``|x| >= SCI_LARGE_THRESHOLD`` (1e5), both judged on
    the unrounded value: then scientific, ``digits`` mantissa decimals and a signed
    exponent of at least two digits, as "7.09e-03". Rounding is C ``sprintf``
    rounding of the binary double (correctly rounded), which Python's formatting
    reproduces exactly. Infinite values print as "Inf" and "-Inf". All of it is
    pinned by the oracle probe ``probe-format`` (every row, digits 0 to 4).

    ``missing`` is returned for None and NaN (R's NA and NaN): ``"NA"`` on
    statsExpressions' ``.to_char`` route (subtitle and caption tables), ``""`` for
    its direct ``format_value`` calls (pairwise labels, the log Bayes factor,
    centrality labels).
    """
    if digits < 0:
        raise ValueError(f"digits must be >= 0, got {digits}")
    if x is None or math.isnan(x):
        return missing
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x != 0.0:
        magnitude = abs(x)
        if math.log10(magnitude) < -digits or magnitude >= SCI_LARGE_THRESHOLD:
            return f"{x:.{digits}e}"
    elif not UNPINNED_KEEP_NEGATIVE_ZERO:
        x = 0.0
    return f"{x:.{digits}f}"


def format_p_subtitle(p: float, digits: int = 2) -> str:
    """Format a subtitle p-value: an exact 0 becomes ``.Machine$double.xmin``.

    ``add_expression_col()`` does this before formatting, so ``p == 0`` prints as
    "2.23e-308". Pairwise labels do not; use :func:`format_value` there.
    """
    return format_value(DOUBLE_XMIN if p == 0.0 else p, digits)


def format_bf01_log(bf10: float, digits: int = 2) -> str:
    """Format ``log_e(BF01) = -log(bf10)`` as the caption and pairwise labels do.

    As in R, a Bayes factor of 0 gives "Inf" and an infinite one "-Inf" (probe-format,
    "end_to_end": ``bf10 = exp(-123456.7)``, which is 0, prints "Inf"); a missing one
    prints as "".
    """
    if math.isnan(bf10):
        return format_value(None, digits, missing="")
    if bf10 < 0:
        raise ValueError(f"a Bayes factor cannot be negative, got {bf10}")
    log_bf01 = math.inf if bf10 == 0.0 else -math.log(bf10)
    return format_value(log_bf01, digits, missing="")


def format_bf10(bf10: float, log_bf01: float, digits: int = 2) -> str:
    """Format a Bayes factor BF10 as :func:`format_value` does, also beyond doubles.

    A BF10 that overflows a double (stored as inf) or underflows it (0 or
    subnormal) while ``log_bf01`` is finite is printed from ``log_bf01``: its
    mantissa and exponent come from -log_bf01 / ln(10), as "4.17e+330". Elsewhere
    the result is ``format_value(bf10, digits)`` ("NA" when missing). The stored
    ``bf10`` keeps the double (inf or 0); ``log_bf01`` carries the value.
    """
    if math.isnan(bf10):
        return "NA"
    beyond = math.isinf(bf10) or bf10 < DOUBLE_XMIN
    if not beyond or not math.isfinite(log_bf01):
        return format_value(bf10, digits, missing="NA")
    log10 = -log_bf01 / math.log(10.0)
    exponent = math.floor(log10)
    mantissa = 10.0 ** (log10 - exponent)
    text = f"{mantissa:.{digits}f}"
    if float(text) >= 10.0:  # 9.996 rounds up to 10.00: carry into the exponent
        exponent += 1
        text = f"{mantissa / 10.0:.{digits}f}"
    sign = "-" if exponent < 0 else "+"
    return f"{text}e{sign}{abs(exponent):02d}"


def r_number_text(x: float) -> str:
    """Print a double the way R's ``deparse()`` and ``as.character()`` do.

    R's ``formatReal`` with 15 significant digits and ``scipen = 0``: the fewest
    significant digits (at most 15) that represent ``x`` to 15 significant digits,
    in fixed notation when its width does not exceed the scientific width ("1e+05"
    for 100000, "123456" for 123456, "1e-04" for 0.0001, "0.001" for 0.001).
    Non-finite values print as "NaN", "Inf" and "-Inf".
    """
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == 0.0:
        return "0"
    neg = x < 0
    mantissa, exp_text = f"{abs(x):.14e}".split("e")
    kpower = int(exp_text)
    digits_only = mantissa.replace(".", "").rstrip("0") or "0"
    nsig = len(digits_only)
    rgt_sci = nsig - 1
    exp_width = 3 if abs(kpower) >= 100 else 2
    width_sci = int(neg) + 1 + (rgt_sci + 1 if rgt_sci > 0 else 0) + 2 + exp_width
    left = kpower + 1 if kpower >= 0 else 1
    rgt_fix = max(0, nsig - kpower - 1)
    width_fix = int(neg) + left + (rgt_fix + 1 if rgt_fix > 0 else 0)
    if width_fix <= width_sci:
        return f"{x:.{rgt_fix}f}"
    sci_mantissa = digits_only[0] + ("." + digits_only[1:] if nsig > 1 else "")
    sign = "-" if neg else ""
    exp_sign = "-" if kpower < 0 else "+"
    return f"{sign}{sci_mantissa}e{exp_sign}{abs(kpower):0{exp_width}d}"


def format_df(df: float, digits: int) -> str:
    """Format degrees of freedom as they appear in a ggstatsplot subtitle.

    statsExpressions formats ``df`` with ``format_value(df, digits.df)`` and pastes
    the text unquoted, so plotmath parses it back to a number and R prints that
    number (:func:`r_number_text`): "97.4" for 97.40, "2" for 2.00. ``digits`` is the
    analysis' ``digits`` for fractional df (Welch t and F, Yuen, t1way;
    within-subjects parametric and robust) and 0 otherwise.
    """
    text = format_value(df, digits)
    if text in ("NA", "Inf", "-Inf", "NaN"):
        return text
    return r_number_text(float(text))


def format_n(n: int) -> str:
    """Format a sample size as statsExpressions' ``.prettyNum(n)``: "1,234".

    A comma every three digits from 1,000 on, never scientific notation (pinned by
    probe-format's "counts" for doubles and integers up to 1e9).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return f"{n:,d}".replace(",", N_BIG_MARK)


def format_conf_level(level: float) -> str:
    """Format the CI level as ``paste0(conf.level * 100L, "%")``: 0.95 -> "95%".

    ``as.character`` prints the product with 15 significant digits, so 0.89 * 100
    (89.00000000000001) is "89%" and 0.9999 * 100 is "99.99%" (probe-format,
    "conf_levels").
    """
    return r_number_text(level * 100) + "%"


# =========================================================================== #
# plotmath AST
# =========================================================================== #


@dataclass(frozen=True, slots=True)
class Str:
    """A quoted R string, drawn upright: ``"Welch"``, ``"0.06"``."""

    value: str


@dataclass(frozen=True, slots=True)
class Num:
    """An unquoted R number as R prints it, such as a df: ``97.4``, ``2``."""

    token: str


@dataclass(frozen=True, slots=True)
class Sym:
    """An R symbol: ``p``, ``mu``, ``log``, ``adj.``, ``ETI``, ``NA``, ``Inf``."""

    name: str


@dataclass(frozen=True, slots=True)
class Call:
    """A call ``fn(args)``: ``list(...)``, ``italic("t")``, ``F["Welch"](2, 17.13)``."""

    fn: Node
    args: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class BinOp:
    """A binary operator: ``*``, ``~``, ``==``, ``-``, ``+`` or ``^``."""

    op: str
    lhs: Node
    rhs: Node


@dataclass(frozen=True, slots=True)
class Index:
    """A subscript ``base[index]``."""

    base: Node
    index: Node


@dataclass(frozen=True, slots=True)
class Paren:
    """Parentheses ``(inner)``, drawn by plotmath."""

    inner: Node


Node: TypeAlias = Str | Num | Sym | Call | BinOp | Index | Paren

# small constructors -------------------------------------------------------- #


def _call(name: str, *args: Node) -> Call:
    return Call(Sym(name), tuple(args))


def _italic(x: Node) -> Call:
    return _call("italic", x)


def _hat(x: Node) -> Call:
    return _call("widehat", x)


def _rel(lhs: Node, rhs: Node) -> BinOp:
    return BinOp("==", lhs, rhs)


def _times(*parts: Node) -> Node:
    """``a * b * c``: left-associative, as R parses it."""
    node = parts[0]
    for part in parts[1:]:
        node = BinOp("*", node, part)
    return node


def _pow(base: Node, sup: Node) -> BinOp:
    return BinOp("^", base, sup)


# =========================================================================== #
# Symbol tables (statsExpressions 2.2.1, R/switch-functions.R)
# =========================================================================== #


def _istr(letter: str) -> Call:
    return _italic(Str(letter))


# extract_statistic_text(): statistic symbol by key.
STATISTIC_SYMBOLS: Final[dict[str, Node]] = {
    "welch_t": Index(_istr("t"), Str("Welch")),
    "student_t": Index(_istr("t"), Str("Student")),
    "bootstrap_t": Index(_istr("t"), Str("bootstrapped")),
    "yuen_t": Index(_istr("t"), Str("Yuen")),
    "fisher_f": Index(_istr("F"), Str("Fisher")),
    "welch_f": Index(_istr("F"), Str("Welch")),
    "trimmed_means_f": Index(_istr("F"), Str("trimmed-means")),
    "mann_whitney_w": Index(_istr("W"), Str("Mann-Whitney")),
    "wilcoxon_v": Index(_istr("V"), Str("Wilcoxon")),
    "spearman_s": _istr("S"),
    "meta_z": _istr("z"),
    "friedman_chi2": _pow(Index(Sym("chi"), Str("Friedman")), Num("2")),
    "kruskal_wallis_chi2": _pow(Index(Sym("chi"), Str("Kruskal-Wallis")), Num("2")),
    "gof_chi2": _pow(Index(Sym("chi"), Str("gof")), Num("2")),
    "pearson_chi2": _pow(Index(Sym("chi"), Str("Pearson")), Num("2")),
    "mcnemar_chi2": _pow(Index(Sym("chi"), Str("McNemar")), Num("2")),
}

# extract_estimate_type(): effect-size symbol by inferplot's effect-size name.
EFFECT_SIZE_SYMBOLS: Final[dict[str, Node]] = {
    "pearsons_c": Index(_hat(_istr("C")), Str("Pearson")),
    "cohens_d": Index(_hat(_istr("d")), Str("Cohen")),
    "hedges_g": Index(_hat(_istr("g")), Str("Hedges")),
    "rank_biserial": _pow(Index(_hat(_istr("r")), Str("biserial")), Str("rank")),
    "cramers_v": Index(_hat(_istr("V")), Str("Cramer")),
    "cohens_g": Index(_hat(_istr("g")), Str("Cohen")),
    "bayes_r2": Index(_hat(_italic(_pow(Sym("R"), Str("2")))), Str("Bayesian")),
    "spearman_rho": Index(_hat(Sym("rho")), Str("Spearman")),
    "bayes_pearson_rho": Index(_hat(Sym("rho")), Str("Pearson")),
    "posterior_difference": Index(_hat(Sym("delta")), Str("difference")),
    "akp_delta": _pow(Index(_hat(Sym("delta")), Str("R")), Str("AKP")),
    "akp_delta_avg": _pow(Index(_hat(Sym("delta")), Str("R-avg")), Str("AKP")),
    "partial_eta_squared": _hat(_pow(Index(Sym("eta"), Str("p")), Num("2"))),
    "partial_omega_squared": _hat(_pow(Index(Sym("omega"), Str("p")), Num("2"))),
    "trimmed_mean": Index(_hat(Sym("mu")), Str("trimmed")),
    "rank_epsilon_squared": _pow(Index(_hat(Sym("epsilon")), Str("ordinal")), Num("2")),
    "explanatory_xi": _hat(Sym("xi")),
    "meta_beta": _pow(Index(_hat(Sym("beta")), Str("summary")), Str("meta")),
    "pearson_r": Index(_hat(_istr("r")), Str("Pearson")),
    "winsorized_r": Index(_hat(_istr("r")), Str("Winsorized")),
    "kendalls_w": Index(_hat(_istr("W")), Str("Kendall")),
}

# prior_switch(): prior symbol by prior name.
PRIOR_SYMBOLS: Final[dict[str, Node]] = {
    "cauchy": _pow(Index(_istr("r"), Str("Cauchy")), Str("JZS")),
    "beta": _pow(Index(_istr("r"), Str("beta")), Str("JZS")),
    "gunel_dickey": Index(_istr("a"), Str("Gunel-Dickey")),
}

# The unquoted conf.method symbol in CI["95%"]^ETI.
_CONF_METHODS: Final = frozenset({"ETI", "HDI", "BCI", "SI", "SPI"})

# centrality_description(): Descriptive.centrality -> subscript of widehat(mu)[...].
_CENTRALITY_SUBSCRIPTS: Final[dict[str, str]] = {
    "mean": "mean",
    "median": "median",
    "trimmed_mean": "trimmed",
    "map": "MAP",
}

# TestResult.name -> (statistic symbol key, df fractional, df_error fractional).
# statsExpressions rounds fractional df to `digits` and the others to 0 decimals
# (two_sample_test(), oneway_anova()).
_TEST_TEMPLATES: Final[dict[str, tuple[str, bool, bool]]] = {
    "welch_t": ("welch_t", False, True),
    "welch_anova": ("welch_f", False, True),
    "mann_whitney": ("mann_whitney_w", False, False),
    "kruskal_wallis": ("kruskal_wallis_chi2", False, False),
    "yuen_t": ("yuen_t", False, True),
    "trimmed_means_anova": ("trimmed_means_f", False, True),
}


def _lookup(table: dict[str, Node], key: str, what: str) -> Node:
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"unknown {what} {key!r}; known: {sorted(table)}") from None


# =========================================================================== #
# Expression builders (statsExpressions 2.2.1 glue templates)
# =========================================================================== #


def _df_node(text: str) -> Node:
    """An unquoted df as R parses it: a number, or the constants NA and Inf."""
    if text in ("NA", "Inf", "NaN"):
        return Sym(text)
    if text.startswith("-"):
        raise ValueError(f"degrees of freedom cannot be negative: {text}")
    return Num(text)


def _ci_items(
    conf_level: float,
    conf_low: float,
    conf_high: float,
    digits: int,
    conf_method: str | None,
) -> tuple[Node, Node]:
    """``CI['95%'](^method) ~ '[' * 'lo'`` and ``'hi' * ']'`` (two list items)."""
    head: Node = Index(Sym("CI"), Str(format_conf_level(conf_level)))
    if conf_method is not None:
        if conf_method not in _CONF_METHODS:
            raise ValueError(f"unknown CI method symbol {conf_method!r}")
        head = _pow(head, Sym(conf_method))
    low = BinOp("~", head, _times(Str("["), Str(format_value(conf_low, digits))))
    high = _times(Str(format_value(conf_high, digits)), Str("]"))
    return low, high


def _log_bf01_item(log_bf01: float, digits: int) -> BinOp:
    """``log[e]*(BF['01'])=='{format_value(-log(bf10), digits)}'``."""
    lhs = _times(Index(Sym("log"), Sym("e")), Paren(Index(Sym("BF"), Str("01"))))
    return _rel(lhs, Str(format_value(log_bf01, digits, missing="")))


def frequentist_expression(
    *,
    statistic: str,
    statistic_value: float,
    p_value: float,
    effect_size: str,
    estimate: float,
    conf_level: float,
    conf_low: float,
    conf_high: float,
    n_obs: int,
    df: float | None = None,
    df_error: float | None = None,
    digits: int = 2,
    digits_df: int = 0,
    digits_df_error: int | None = None,
    paired: bool = False,
) -> Call:
    """statsExpressions' frequentist template (``add_expression_col()``).

    ``statistic`` keys :data:`STATISTIC_SYMBOLS`, ``effect_size`` keys
    :data:`EFFECT_SIZE_SYMBOLS`. With no df the statistic term is
    ``stat=='v'``; with one df column ``stat*'('*df.error*')'=='v'`` (a lone ``df``
    is moved to ``df.error``, as for chi-square); with both ``stat(df, df.error)=='v'``.
    ``digits_df_error`` defaults to ``digits_df``. The n term is
    ``italic("n")["obs"]``, or ``["pairs"]`` when ``paired``.
    """
    stat = _lookup(STATISTIC_SYMBOLS, statistic, "statistic")
    es = _lookup(EFFECT_SIZE_SYMBOLS, effect_size, "effect size")
    if digits_df_error is None:
        digits_df_error = digits_df
    value = Str(format_value(statistic_value, digits))
    if df is None and df_error is None:
        lhs: Node = stat
    elif df is None or df_error is None:
        only = df_error if df_error is not None else df
        only_digits = digits_df_error if df_error is not None else digits_df
        assert only is not None
        lhs = _times(stat, Str("("), _df_node(format_df(only, only_digits)), Str(")"))
    else:
        lhs = Call(
            stat,
            (
                _df_node(format_df(df, digits_df)),
                _df_node(format_df(df_error, digits_df_error)),
            ),
        )
    n_label = "pairs" if paired else "obs"
    low, high = _ci_items(conf_level, conf_low, conf_high, digits, None)
    return _call(
        "list",
        _rel(lhs, value),
        _rel(_italic(Sym("p")), Str(format_p_subtitle(p_value, digits))),
        _rel(es, Str(format_value(estimate, digits))),
        low,
        high,
        _rel(Index(_istr("n"), Str(n_label)), Str(format_n(n_obs))),
    )


def _with_posterior(es: Node) -> Node:
    """``{es.text}^'posterior'`` as R parses the pasted text (``^`` is right-assoc)."""
    if isinstance(es, BinOp) and es.op == "^":
        return BinOp("^", es.lhs, _with_posterior(es.rhs))
    return _pow(es, Str("posterior"))


def bayes_expression(
    *,
    log_bf01: float,
    effect_size: str,
    estimate: float,
    conf_level: float,
    conf_method: str,
    conf_low: float,
    conf_high: float,
    prior: str,
    prior_scale: float,
    digits: int = 2,
) -> Call:
    """statsExpressions' Bayesian template: ggstatsplot's full Bayes expression.

    ``list(log[e]*(BF['01'])=='b', {es}^'posterior'=='e',
    CI['95%']^{method}~'['*'lo', 'hi'*']', {prior}=='s')``. inferplot v0.1 draws
    :func:`bayes_factor_expression` instead (posterior estimates arrive in v0.2).
    """
    es = _with_posterior(_lookup(EFFECT_SIZE_SYMBOLS, effect_size, "effect size"))
    low, high = _ci_items(conf_level, conf_low, conf_high, digits, conf_method)
    return _call(
        "list",
        _log_bf01_item(log_bf01, digits),
        _rel(es, Str(format_value(estimate, digits))),
        low,
        high,
        _rel(
            _lookup(PRIOR_SYMBOLS, prior, "prior"),
            Str(format_value(prior_scale, digits)),
        ),
    )


def bayes_factor_expression(
    *, log_bf01: float, prior: str, prior_scale: float, digits: int = 2
) -> Call:
    """inferplot v0.1 Bayes caption: ``list(log[e]*(BF['01'])=='b', {prior}=='s')``.

    The first and last items of ggstatsplot's caption, unchanged (D32); the
    posterior estimate and its interval arrive in v0.2 (SPEC section 5 item 5). It is
    also statsExpressions' template for one-way Bayesian contingency tables.
    """
    return _call(
        "list",
        _log_bf01_item(log_bf01, digits),
        _rel(
            _lookup(PRIOR_SYMBOLS, prior, "prior"),
            Str(format_value(prior_scale, digits)),
        ),
    )


_P_ADJUST_LABELS: Final[dict[str, str]] = {
    "holm": "Holm",
    "hochberg": "Hochberg",
    "hommel": "Hommel",
    "bonferroni": "Bonferroni",
    "BH": "FDR",
    "fdr": "FDR",
    "BY": "BY",
    "none": "None",
}


def p_adjust_label(method: str) -> str:
    """statsExpressions' label for ``p.adjust.method``.

    ``.pairwise_p_adjust_expr()`` (statsExpressions 2.2.1, MIT) applies
    ``insight::format_capitalize()`` (first letter upper case), then turns "BH" and
    "Fdr" into "FDR"; "None" makes the subscript ``unadj.``. Every label is pinned by
    the fixtures: "Holm" by the default cases, the others by the
    PlantGrowth-padj-{hochberg, hommel, bonferroni, BH, fdr, BY, none} cases, whose
    ``p.adjust.method`` column and bracket labels read "Hochberg", "Hommel",
    "Bonferroni", "FDR", "FDR", "BY" and "None" (``italic(p)[unadj.]``).
    """
    try:
        return _P_ADJUST_LABELS[method]
    except KeyError:
        raise ValueError(
            f"unknown p_adjust_method {method!r}; known: {sorted(_P_ADJUST_LABELS)}"
        ) from None


def pairwise_p_expression(
    *, p_adjusted: float, adjust_method: str, digits: int = 2
) -> Call:
    """Bracket label ``list(italic(p)['Holm'-adj.]=='0.47')``.

    The subscript is :func:`p_adjust_label`'s label; with no adjustment
    statsExpressions writes ``italic(p)[unadj.]``. The p-value goes through
    ``format_value`` directly, so an exact 0 prints "0.00" (not the subtitle's
    2.23e-308) and a missing one "" (probe-format, "end_to_end").
    """
    label = p_adjust_label(adjust_method)
    sub: Node = (
        Sym("unadj.") if label == "None" else BinOp("-", Str(label), Sym("adj."))
    )
    return _call(
        "list",
        _rel(
            Index(_italic(Sym("p")), sub),
            Str(format_value(p_adjusted, digits, missing="")),
        ),
    )


def pairwise_bf_expression(*, log_bf01: float, digits: int = 2) -> Call:
    """Bayes bracket label ``list(log[e]*(BF['01'])=='0.43')``."""
    return _call("list", _log_bf01_item(log_bf01, digits))


def centrality_expression(
    *, centrality: str, value: float | None, digits: int = 2
) -> Call:
    """Centrality label ``list(widehat(mu)[mean]=='5.03')``.

    ``centrality`` is :attr:`~inferplot.result.Descriptive.centrality`: "mean",
    "median", "trimmed_mean" or "map". A missing ``value`` (a MAP without a density)
    gives ggstatsplot's empty label, ``list(widehat(mu)[MAP]=='')``.
    """
    try:
        sub = _CENTRALITY_SUBSCRIPTS[centrality]
    except KeyError:
        raise ValueError(
            f"unknown centrality {centrality!r}; "
            f"known: {sorted(_CENTRALITY_SUBSCRIPTS)}"
        ) from None
    return _call(
        "list",
        _rel(
            Index(_hat(Sym("mu")), Sym(sub)),
            Str(format_value(value, digits, missing="")),
        ),
    )


def tick_label(level: str, n: int) -> str:
    """x tick label ``"<level>\\n(n = <n>)"``: plain text, never mathtext.

    ggstatsplot passes these as character labels (no parse); draw them with
    ``parse_math=False`` so a level such as "$5" or "a_b" shows literally.
    """
    return f"{level}\n(n = {format_n(n)})"


def subtitle_expression(
    test: TestResult, effect_size: EffectSize, *, n_obs: int, digits: int
) -> Call:
    """The frequentist subtitle of a result's main test, as ggstatsplot builds it."""
    try:
        statistic, frac_df, frac_df_error = _TEST_TEMPLATES[test.name]
    except KeyError:
        raise NotImplementedError(
            f"no subtitle template for the test {test.name!r} yet"
        ) from None
    return frequentist_expression(
        statistic=statistic,
        statistic_value=test.statistic,
        p_value=test.p_value,
        effect_size=effect_size.name,
        estimate=effect_size.estimate,
        conf_level=effect_size.conf_level,
        conf_low=effect_size.conf_low,
        conf_high=effect_size.conf_high,
        n_obs=n_obs,
        df=test.df,
        df_error=test.df_error,
        digits=digits,
        digits_df=digits if frac_df else 0,
        digits_df_error=digits if frac_df_error else 0,
    )


# =========================================================================== #
# Renderers
# =========================================================================== #

_GREEK: Final[dict[str, str]] = {
    "alpha": "\u03b1",  # alpha
    "beta": "\u03b2",  # beta
    "gamma": "\u03b3",  # gamma
    "delta": "\u03b4",  # delta
    "epsilon": "\u03b5",  # epsilon
    "zeta": "\u03b6",  # zeta
    "eta": "\u03b7",  # eta
    "theta": "\u03b8",  # theta
    "iota": "\u03b9",  # iota
    "kappa": "\u03ba",  # kappa
    "lambda": "\u03bb",  # lambda
    "mu": "\u03bc",  # mu
    "nu": "\u03bd",  # nu
    "xi": "\u03be",  # xi
    "pi": "\u03c0",  # pi
    "rho": "\u03c1",  # rho
    "sigma": "\u03c3",  # sigma
    "tau": "\u03c4",  # tau
    "upsilon": "\u03c5",  # upsilon
    "phi": "\u03d5",  # phi
    "chi": "\u03c7",  # chi
    "psi": "\u03c8",  # psi
    "omega": "\u03c9",  # omega
    "Gamma": "\u0393",  # Gamma
    "Delta": "\u0394",  # Delta
    "Theta": "\u0398",  # Theta
    "Lambda": "\u039b",  # Lambda
    "Xi": "\u039e",  # Xi
    "Pi": "\u03a0",  # Pi
    "Sigma": "\u03a3",  # Sigma
    "Upsilon": "\u03a5",  # Upsilon
    "Phi": "\u03a6",  # Phi
    "Psi": "\u03a8",  # Psi
    "Omega": "\u03a9",  # Omega
}

_STYLES: Final = frozenset({"italic", "plain", "bold"})
_TEXT_ESCAPES: Final = {"\\": "\\\\", "{": "\\{", "}": "\\}", "$": "\\$"}
_LETTERS_RE: Final = re.compile(r"^[A-Za-z0-9]+$")
_NAME_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9.]*$")


def _mt_text(value: str) -> str:
    """Upright literal text. ``\\text`` keeps an ASCII hyphen as U+002D, unspaced."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"mathtext cannot hold a line break: {value!r}")
    return "\\text{" + "".join(_TEXT_ESCAPES.get(ch, ch) for ch in value) + "}"


def _mt_italic(value: str) -> str:
    if not _LETTERS_RE.match(value):
        raise ValueError(f"italic() supports letters and digits only: {value!r}")
    return "\\mathit{" + value + "}"


def _split_script(node: Node) -> tuple[Node, Node | None, Node | None]:
    """``base[sub]^sup`` -> (base, sub, sup)."""
    if isinstance(node, BinOp) and node.op == "^":
        if isinstance(node.lhs, Index):
            return node.lhs.base, node.lhs.index, node.rhs
        return node.lhs, None, node.rhs
    if isinstance(node, Index):
        return node.base, node.index, None
    return node, None, None


def _mathtext(node: Node, style: str) -> str:
    if isinstance(node, Str | Num):
        value = node.value if isinstance(node, Str) else node.token
        if style == "italic":
            return _mt_italic(value)
        if style == "plain":
            return _mt_text(value)
        raise NotImplementedError(
            f"{style}() text cannot be drawn faithfully: {value!r}"
        )
    if isinstance(node, Sym):
        if node.name in _GREEK:
            if style != "plain":
                raise NotImplementedError(f"{style}() Greek letters")
            return "\\mathrm{\\" + node.name + "}"
        if style == "italic":
            return _mt_italic(node.name)
        if not _NAME_RE.match(node.name):
            raise ValueError(f"not a plotmath symbol name: {node.name!r}")
        return "\\mathrm{" + node.name + "}"
    if isinstance(node, Index) or (isinstance(node, BinOp) and node.op == "^"):
        base, sub, sup = _split_script(node)
        out = "{" + _mathtext(base, style) + "}"
        if sub is not None:
            out += "_{" + _mathtext(sub, style) + "}"
        if sup is not None:
            out += "^{" + _mathtext(sup, style) + "}"
        return out
    if isinstance(node, BinOp):
        lhs, rhs = _mathtext(node.lhs, style), _mathtext(node.rhs, style)
        if node.op == "*":
            return lhs + rhs
        if node.op == "~":
            return lhs + "\\ " + rhs
        if node.op == "==":
            return lhs + " = " + rhs
        if node.op in ("-", "+"):
            return lhs + node.op + rhs
        raise NotImplementedError(f"operator {node.op!r}")
    if isinstance(node, Paren):
        return "(" + _mathtext(node.inner, style) + ")"
    fn = node.fn
    if isinstance(fn, Sym):
        if fn.name == "list":
            return ",\\ ".join(_mathtext(a, style) for a in node.args)
        if fn.name in _STYLES and len(node.args) == 1:
            return _mathtext(node.args[0], fn.name)
        if fn.name == "widehat" and len(node.args) == 1:
            return "\\widehat{" + _mathtext(node.args[0], style) + "}"
    args = ",\\ ".join(_mathtext(a, style) for a in node.args)
    return _mathtext(fn, style) + "(" + args + ")"


def to_mathtext(node: Node) -> str:
    """Render a plotmath AST as a matplotlib mathtext string ``$...$``.

    Mechanical rules, applied uniformly: strings and numbers are ``\\text{...}``
    (only ``\\ { } $`` escaped, so hyphens in "-0.02" and "Kruskal-Wallis" stay
    ASCII hyphen-minus as ggstatsplot draws them); ``italic(x)`` is ``\\mathit{x}``;
    other symbols are upright ``\\mathrm{}``, Greek too (``\\mathrm{\\mu}``);
    ``x[a]^b`` is ``{x}_{a}^{b}``; ``widehat`` is ``\\widehat``; ``==`` is " = ",
    ``~`` a space, ``*`` juxtaposition; list and call arguments are joined by
    ``,\\ ``; a binary ``-`` (``"Holm" - adj.``) is a minus sign, as in plotmath.
    """
    return "$" + _mathtext(node, "plain") + "$"


def _plain(node: Node) -> str:
    if isinstance(node, Str):
        return node.value
    if isinstance(node, Num):
        return node.token
    if isinstance(node, Sym):
        return _GREEK.get(node.name, node.name)
    if isinstance(node, Index):
        return _plain(node.base) + "_" + _plain(node.index)
    if isinstance(node, BinOp):
        lhs, rhs = _plain(node.lhs), _plain(node.rhs)
        if node.op == "*":
            return lhs + rhs
        if node.op == "~":
            return lhs + " " + rhs
        if node.op == "==":
            return lhs + " = " + rhs
        if node.op == "^":
            return lhs + "^" + rhs
        if node.op in ("-", "+"):
            return lhs + node.op + rhs
        raise NotImplementedError(f"operator {node.op!r}")
    if isinstance(node, Paren):
        return "(" + _plain(node.inner) + ")"
    fn = node.fn
    args = [_plain(a) for a in node.args]
    if isinstance(fn, Sym):
        if fn.name == "list":
            return ", ".join(args)
        if fn.name in (*_STYLES, "widehat") and len(args) == 1:
            return args[0]
    return _plain(fn) + "(" + ", ".join(args) + ")"


def to_plain(node: Node) -> str:
    """Render a plotmath AST as plain Unicode text.

    ``x[a]`` is ``x_a``, ``^b`` is ``^b``, ``*`` juxtaposition, ``~`` a space,
    ``==`` " = ", list items are joined by ", "; ``italic``, ``plain``, ``bold`` and
    ``widehat`` keep only their content; Greek symbols become Greek letters. Example:
    ``t_Welch(55.31) = 1.92, p = 0.06, g_Hedges = 0.49, CI_95% [-0.02, 0.99],
    n_obs = 60``.
    """
    return _plain(node)


# =========================================================================== #
# Plain-English texts: caption line, APA sentence, alt text
# =========================================================================== #

_ADJUST_NAMES: Final[dict[str, str]] = {
    "holm": "Holm",
    "hochberg": "Hochberg",
    "hommel": "Hommel",
    "bonferroni": "Bonferroni",
    "BH": "Benjamini-Hochberg",
    "fdr": "Benjamini-Hochberg",
    "BY": "Benjamini-Yekutieli",
}

_DISPLAY_NOTES: Final[dict[str, str]] = {
    "significant": "significant only",
    "non-significant": "non-significant only",
}

# Effect sizes bounded by 1 in absolute value: APA style drops the leading zero.
_BOUNDED_EFFECT_SIZES: Final = frozenset(
    {
        "partial_omega_squared",
        "partial_eta_squared",
        "rank_epsilon_squared",
        "rank_biserial",
        "cramers_v",
        "pearsons_c",
        "kendalls_w",
        "pearson_r",
        "spearman_rho",
        "winsorized_r",
        "bayes_r2",
    }
)

_CENTRALITY_WORDS: Final[dict[str, str]] = {
    "mean": "mean",
    "median": "median",
    "trimmed_mean": "trimmed mean",
    "map": "MAP estimate",
}

ALT_TEXT_MAX_CHARS: Final = 400


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def pairwise_caption_line(
    pairwise: Sequence[PairwiseComparison], *, display: str, adjust_method: str
) -> str | None:
    """The caption line naming the pairwise test, adjustment and drawn pairs.

    "Pairwise: Games-Howell, Holm adjusted; showing 1 of 3 pairs (significant
    only)." None when there are no pairs or ``display`` is "none". It also states how
    many pairs were tested when only some are drawn (SPEC section 5 item 3). Bayes
    factor pairs (type "bayes", every pair drawn): "Pairwise: Bayes factors
    (Student's t, JZS prior 0.707), unadjusted; showing all 3 pairs." (D41).
    """
    if not pairwise or display == "none":
        return None
    test_label = pairwise[0].test_label
    bayes = pairwise[0].bayes
    if bayes is not None:
        return (
            f"Pairwise: Bayes factors ({test_label}, JZS prior "
            f"{r_number_text(bayes.prior_scale)}), unadjusted; showing all "
            f"{_plural(len(pairwise), 'pair')}."
        )
    if adjust_method == "none":
        method = f"{test_label}, unadjusted"
    else:
        try:
            method = f"{test_label}, {_ADJUST_NAMES[adjust_method]} adjusted"
        except KeyError:
            raise ValueError(f"unknown p_adjust_method {adjust_method!r}") from None
    total = len(pairwise)
    if display == "all":
        return f"Pairwise: {method}; showing all {_plural(total, 'pair')}."
    try:
        note = _DISPLAY_NOTES[display]
    except KeyError:
        raise ValueError(f"unknown pairwise_display {display!r}") from None
    drawn = sum(1 for pair in pairwise if pair.displayed)
    return f"Pairwise: {method}; showing {drawn} of {_plural(total, 'pair')} ({note})."


def _no_neg_zero(text: str) -> str:
    if text.startswith("-") and set(text[1:]) <= {"0", "."}:
        return text[1:]
    return text


def _apa_number(x: float, *, bounded: bool = False) -> str:
    """Two decimals, ASCII minus; no leading zero for quantities bounded by 1."""
    if math.isnan(x):
        return "NA"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    text = _no_neg_zero(f"{x:.2f}")
    if bounded and abs(x) < 1:
        text = text.replace("0.", ".", 1)
    return text


def _apa_p(p: float) -> str:
    """ "p = .061"; "p < .001" below .001; three decimals without a leading zero."""
    if math.isnan(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    text = f"{p:.3f}"
    return "p = " + (text[1:] if text.startswith("0.") else text)


def _apa_df(df: float) -> str:
    if math.isfinite(df) and df == int(df):
        return str(int(df))
    return f"{df:.2f}"


def _apa_test(test: TestResult) -> str:
    if test.df is not None and test.df_error is not None:
        dfs = f"({_apa_df(test.df)}, {_apa_df(test.df_error)})"
    elif test.df_error is not None or test.df is not None:
        only = test.df_error if test.df_error is not None else test.df
        assert only is not None
        dfs = f"({_apa_df(only)})"
    else:
        dfs = ""
    statistic = _apa_number(test.statistic)
    return f"{test.statistic_name}{dfs} = {statistic}, {_apa_p(test.p_value)}"


def _apa_effect_size(effect_size: EffectSize) -> str:
    bounded = effect_size.name in _BOUNDED_EFFECT_SIZES
    estimate = _apa_number(effect_size.estimate, bounded=bounded)
    low = _apa_number(effect_size.conf_low, bounded=bounded)
    high = _apa_number(effect_size.conf_high, bounded=bounded)
    level = format_conf_level(effect_size.conf_level)
    return f"{effect_size.label} = {estimate}, {level} CI [{low}, {high}]"


def _test_heading(test: TestResult) -> str:
    if test.alternative == "two-sided":
        return test.label
    return f"{test.label} (one-sided, {test.alternative})"


def apa_sentence(test: TestResult, effect_size: EffectSize, *, n_obs: int) -> str:
    """The APA 7 results sentence (SPEC section 7), ASCII only.

    "Welch's t-test: t(55.31) = 1.92, p = .061, Hedges' g = 0.49, 95% CI [-0.02,
    0.99], n = 60." Statistics and estimates to two decimals; p to three decimals
    without a leading zero ("p < .001" below .001); no leading zero for quantities
    bounded by 1; non-integer df to two decimals, integer df bare; one-sided tests say
    so after the test name.
    """
    return (
        f"{_test_heading(test)}: {_apa_test(test)}, {_apa_effect_size(effect_size)}, "
        f"n = {format_n(n_obs)}."
    )


def _apa_bf(bayes: BayesFactor) -> str:
    """A Bayes factor: two decimals, scientific from 1e5 and below 0.01, and from
    log(BF01) where BF10 is beyond a double (:func:`format_bf10`)."""
    return format_bf10(bayes.bf10, bayes.log_bf01, 2)


def _prior_text(bayes: BayesFactor) -> str:
    return f"JZS prior, r = {r_number_text(bayes.prior_scale)}"


def bayes_sentence(bayes: BayesFactor, *, n_obs: int) -> str:
    """The results sentence of a two-group Bayes factor (type "bayes"), ASCII.

    "Bayesian t-test (JZS prior, r = 0.707): BF10 = 1.20, log(BF01) = -0.18,
    n = 60." BF10 to two decimals (scientific from 1e5 and below 0.01, as
    ``format_value``), log(BF01) to two decimals.
    """
    return (
        f"Bayesian t-test ({_prior_text(bayes)}): BF10 = {_apa_bf(bayes)}, "
        f"log(BF01) = {_apa_number(bayes.log_bf01)}, n = {format_n(n_obs)}."
    )


def pairwise_bayes_sentence(
    pairwise: Sequence[PairwiseComparison], *, n_obs: int
) -> str:
    """The results sentence of type "bayes" with three or more groups, which has no
    omnibus test in v0.1 (D41): each pair's Bayes factor, ASCII.

    "Bayesian t-tests of each pair (JZS prior, r = 0.707): ctrl vs trt1, BF10 =
    0.65; ctrl vs trt2, BF10 = 1.77; trt1 vs trt2, BF10 = 6.36; n = 30."
    """
    first = pairwise[0].bayes
    assert first is not None
    parts: list[str] = []
    for pair in pairwise:
        assert pair.bayes is not None
        parts.append(f"{pair.group1} vs {pair.group2}, BF10 = {_apa_bf(pair.bayes)}")
    return (
        f"Bayesian t-tests of each pair ({_prior_text(first)}): "
        + "; ".join(parts)
        + f"; n = {format_n(n_obs)}."
    )


def results_sentence(
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    pairwise: Sequence[PairwiseComparison],
    *,
    n_obs: int,
) -> str:
    """The result's APA-style sentence: :func:`apa_sentence` for a frequentist
    test, :func:`bayes_sentence` or :func:`pairwise_bayes_sentence` for type
    "bayes", or a note that nothing was tested."""
    if test is not None and effect_size is not None:
        return apa_sentence(test, effect_size, n_obs=n_obs)
    if bayes is not None:
        return bayes_sentence(bayes, n_obs=n_obs)
    if pairwise and pairwise[0].bayes is not None:
        return pairwise_bayes_sentence(pairwise, n_obs=n_obs)
    return f"No test was computed (results_subtitle=False), n = {format_n(n_obs)}."


def _short(name: str, limit: int) -> str:
    # U+2026 HORIZONTAL ELLIPSIS: three periods would read as a sentence end.
    return name if len(name) <= limit else name[: limit - 1] + "\u2026"


def _centrality_word(centrality: str, tr: float) -> str:
    """The centrality in words: "mean", "median", "20% trimmed mean", "MAP
    estimate"."""
    word = _CENTRALITY_WORDS.get(centrality, centrality)
    if centrality == "trimmed_mean":
        return f"{r_number_text(tr * 100)}% {word}"
    return word


def _alt_groups_detailed(
    descriptives: Sequence[Descriptive], digits: int, limit: int, tr: float
) -> str:
    parts: list[str] = []
    for d in descriptives:
        word = _centrality_word(d.centrality, tr)
        # A missing centrality (a MAP without a density, D55): "no MAP estimate".
        center = (
            f"no {word}"
            if d.value is None or math.isnan(d.value)
            else f"{word} {format_value(d.value, digits)}"
        )
        parts.append(f"{_short(d.level, limit)} (n = {format_n(d.n)}, {center})")
    return ", ".join(parts)


def _alt_groups_summary(
    descriptives: Sequence[Descriptive], digits: int, limit: int, tr: float
) -> str:
    ns = [d.n for d in descriptives]
    n_text = (
        f"n = {format_n(ns[0])} each"
        if min(ns) == max(ns)
        else f"n = {format_n(min(ns))} to {format_n(max(ns))} per group"
    )
    finite = [
        (d, d.value)
        for d in descriptives
        if d.value is not None and math.isfinite(d.value)
    ]
    if not finite:
        return n_text
    low, low_value = min(finite, key=lambda item: item[1])
    high, high_value = max(finite, key=lambda item: item[1])
    word = _centrality_word(low.centrality, tr)
    return (
        f"{n_text}; {word}s from {format_value(low_value, digits)} "
        f"({_short(low.level, limit)}) to {format_value(high_value, digits)} "
        f"({_short(high.level, limit)})"
    )


def _alt_bayes_pairs(
    pairwise: Sequence[PairwiseComparison], *, limit: int, display: str
) -> str:
    """Type "bayes" with three or more groups: the range of the pairs' BF10, the
    level names shortened to ``limit`` characters, and what the brackets show.

    Pairs are ranked by -log(BF01), which stays finite where BF10 overflows a
    double, so the strongest pair is found even when several BF10 are inf."""
    first = pairwise[0].bayes
    assert first is not None

    def strength(pair: PairwiseComparison) -> float:
        assert pair.bayes is not None
        return -pair.bayes.log_bf01

    def described(pair: PairwiseComparison) -> str:
        assert pair.bayes is not None
        return (
            f"{_apa_bf(pair.bayes)} ({_short(pair.group1, limit)} vs "
            f"{_short(pair.group2, limit)})"
        )

    low, high = min(pairwise, key=strength), max(pairwise, key=strength)
    brackets = (
        "no brackets are drawn (pairwise_display='none')."
        if display == "none"
        else "brackets show log(BF01) for every pair."
    )
    return (
        f"Bayesian t-tests of each of the {len(pairwise)} pairs ({_prior_text(first)}, "
        f"Student's t): BF10 from {described(low)} to {described(high)}; {brackets}"
    )


def alt_text(
    *,
    x: str,
    y: str,
    descriptives: Sequence[Descriptive],
    test: TestResult | None,
    effect_size: EffectSize | None,
    pairwise: Sequence[PairwiseComparison],
    n_obs: int,
    settings: Settings,
    bayes: BayesFactor | None = None,
) -> str:
    """Alt text for the figure: 2 to 4 sentences, at most 400 characters.

    Chart type and variables, groups with n and centrality values (summarized as
    ranges when listing every group would not fit), the omnibus result, and the
    count of significant pairs. Type "bayes" with three or more groups gives the
    range of the pairs' Bayes factors instead.

    Level names, x and y are shortened (40, 20, then 12 characters) until the text
    fits. If it still does not, the group description is left out; the result
    sentence is never cut, so a result sentence too long by itself (a statistic
    printed with hundreds of digits) gives a text above the limit.
    """
    k = len(descriptives)
    pairs = ""
    bayes_pairs = bool(pairwise) and pairwise[0].bayes is not None

    def result_at(limit: int) -> str:
        if bayes_pairs:
            return _alt_bayes_pairs(
                pairwise, limit=limit, display=settings.pairwise_display
            )
        return results_sentence(test, effect_size, bayes, pairwise, n_obs=n_obs)

    if pairwise and not bayes_pairs:
        significant = sum(1 for pair in pairwise if pair.significant)
        verb = "is" if significant == 1 else "are"
        method = settings.p_adjust_method
        adjust = (
            "unadjusted"
            if method == "none"
            else f"{_ADJUST_NAMES.get(method, method)} adjusted"
        )
        pairs = (
            f" {significant} of {_plural(len(pairwise), 'pairwise comparison')} {verb} "
            f"significant ({pairwise[0].test_label}, {adjust}, alpha = "
            f"{r_number_text(settings.pairwise_alpha)})."
        )
    head = ""
    result = ""
    for limit in (40, 20, 12):
        head = (
            f"Plot comparing {_short(y, limit)} across the {k} groups of "
            f"{_short(x, limit)}"
        )
        result = result_at(limit)
        detailed = _alt_groups_detailed(
            descriptives, settings.digits, limit, settings.tr
        )
        summary = _alt_groups_summary(descriptives, settings.digits, limit, settings.tr)
        for text in (
            f"{head}: {detailed}. {result}{pairs}",
            f"{head} ({summary}). {result}{pairs}",
        ):
            if len(text) <= ALT_TEXT_MAX_CHARS:
                return text
    # Extremely long labels: keep the start and the whole result, drop the groups.
    return f"{head}. {result}{pairs}"


# =========================================================================== #
# Texts for the result and for the renderer
# =========================================================================== #


@dataclass(frozen=True, slots=True)
class _Expressions:
    subtitle: Node | None
    bayes_caption: Node | None
    centrality: tuple[Node, ...]
    pairwise: tuple[Node, ...]
    pairwise_line: str | None


def _expressions(
    *,
    descriptives: Sequence[Descriptive],
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    pairwise: Sequence[PairwiseComparison],
    n_obs: int,
    settings: Settings,
) -> _Expressions:
    digits = settings.digits
    subtitle = None
    bayes_caption = None
    if settings.results_subtitle and (test is None or effect_size is None):
        # Type bayes (D41): the two-group subtitle is the first and last items of
        # ggstatsplot's; there is no caption.
        if bayes is not None:
            subtitle = bayes_factor_expression(
                log_bf01=bayes.log_bf01,
                prior=bayes.prior,
                prior_scale=bayes.prior_scale,
                digits=digits,
            )
    elif settings.results_subtitle:
        assert test is not None
        assert effect_size is not None
        subtitle = subtitle_expression(test, effect_size, n_obs=n_obs, digits=digits)
        # bf_message=False never draws the caption, even when a Bayes factor exists.
        if bayes is not None and settings.bf_message is not False:
            bayes_caption = bayes_factor_expression(
                log_bf01=bayes.log_bf01,
                prior=bayes.prior,
                prior_scale=bayes.prior_scale,
                digits=digits,
            )
    centrality = tuple(
        centrality_expression(centrality=d.centrality, value=d.value, digits=digits)
        for d in descriptives
    )
    labels = tuple(_pair_label(pair, digits) for pair in pairwise)
    line = pairwise_caption_line(
        pairwise,
        display=settings.pairwise_display,
        adjust_method=settings.p_adjust_method,
    )
    return _Expressions(subtitle, bayes_caption, centrality, labels, line)


def _pair_label(pair: PairwiseComparison, digits: int) -> Call:
    """A bracket label: the log Bayes factor of a Bayesian pair, else its
    adjusted p."""
    if pair.bayes is not None:
        return pairwise_bf_expression(log_bf01=pair.bayes.log_bf01, digits=digits)
    if pair.p_adjusted is None:
        raise ValueError(
            f"the pair {pair.group1!r}-{pair.group2!r} has neither an adjusted p nor "
            "a Bayes factor"
        )
    return pairwise_p_expression(
        p_adjusted=pair.p_adjusted, adjust_method=pair.adjust_method, digits=digits
    )


def caption_lines(
    bayes_caption: Node | None, pairwise_line: str | None, *, mathtext: bool
) -> tuple[str, ...]:
    """The generated caption lines: the Bayes line, then the pairwise line."""
    lines: list[str] = []
    if bayes_caption is not None:
        lines.append(
            to_mathtext(bayes_caption) if mathtext else to_plain(bayes_caption)
        )
    if pairwise_line is not None:
        lines.append(pairwise_line)
    return tuple(lines)


def build_texts(
    *,
    x: str,
    y: str,
    descriptives: Sequence[Descriptive],
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    pairwise: Sequence[PairwiseComparison],
    n_obs: int,
    settings: Settings,
) -> Texts:
    """Every plain text of a result, from the parts the analysis computed.

    The subtitle is None when ``settings.results_subtitle`` is false. The caption
    holds the Bayes line (when ``bayes`` is given, the subtitle is drawn and
    ``settings.bf_message`` is not False, as ggstatsplot's ``bf.message``) and the
    pairwise line (three or more groups, unless ``pairwise_display`` is "none"),
    joined by a newline, or None when there is neither. The user's own caption is
    not part of it.
    """
    ex = _expressions(
        descriptives=descriptives,
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        n_obs=n_obs,
        settings=settings,
    )
    lines = caption_lines(ex.bayes_caption, ex.pairwise_line, mathtext=False)
    return Texts(
        subtitle=None if ex.subtitle is None else to_plain(ex.subtitle),
        caption="\n".join(lines) if lines else None,
        apa=results_sentence(test, effect_size, bayes, pairwise, n_obs=n_obs),
        alt_text=alt_text(
            x=x,
            y=y,
            descriptives=descriptives,
            test=test,
            effect_size=effect_size,
            pairwise=pairwise,
            n_obs=n_obs,
            settings=settings,
            bayes=bayes,
        ),
        centrality_labels=tuple(to_plain(node) for node in ex.centrality),
        tick_labels=tuple(tick_label(d.level, d.n) for d in descriptives),
        pairwise_labels=tuple(to_plain(node) for node in ex.pairwise),
    )


@dataclass(frozen=True, slots=True)
class FigureTexts:
    """What the renderer draws, ready for matplotlib.

    ``subtitle`` is mathtext (``$...$``) or None; ``caption_lines`` holds the Bayes
    line as mathtext and the pairwise line as plain text (draw it with
    ``parse_math=False``); ``centrality_labels`` is mathtext per level;
    ``tick_labels`` is plain text per level (``parse_math=False``);
    ``pairwise_labels`` is mathtext per pair, in the order of ``result.pairwise``.
    """

    subtitle: str | None
    caption_lines: tuple[str, ...]
    centrality_labels: tuple[str, ...]
    tick_labels: tuple[str, ...]
    pairwise_labels: tuple[str, ...]


def figure_texts(result: StatsResult) -> FigureTexts:
    """The renderer's texts for ``result``, rebuilt from its numbers and settings."""
    ex = _expressions(
        descriptives=result.descriptives,
        test=result.test,
        effect_size=result.effect_size,
        bayes=result.bayes,
        pairwise=result.pairwise,
        n_obs=result.n_obs,
        settings=result.settings,
    )
    return FigureTexts(
        subtitle=None if ex.subtitle is None else to_mathtext(ex.subtitle),
        caption_lines=caption_lines(ex.bayes_caption, ex.pairwise_line, mathtext=True),
        centrality_labels=tuple(to_mathtext(node) for node in ex.centrality),
        tick_labels=tuple(tick_label(d.level, d.n) for d in result.descriptives),
        pairwise_labels=tuple(to_mathtext(node) for node in ex.pairwise),
    )
