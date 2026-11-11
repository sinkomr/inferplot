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

import itertools
import math
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from inferplot.result import (
    Bar,
    BayesFactor,
    Cell,
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
    "APA_NAME_LIMIT",
    "DEFAULT_PAIRWISE_ALPHA",
    "DOUBLE_XMIN",
    "EFFECT_SIZE_SYMBOLS",
    "N_BIG_MARK",
    "PERCENT_TICKS",
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
    "bar_p_expression",
    "bayes_expression",
    "bayes_factor_expression",
    "bayes_sentence",
    "build_texts",
    "caption_lines",
    "cell_label",
    "centrality_expression",
    "contingency_alt_text",
    "contingency_apa",
    "contingency_figure_texts",
    "contingency_sentence",
    "contingency_subtitle_expression",
    "contingency_tests_sentence",
    "contingency_texts",
    "correlation_alt_text",
    "correlation_sentence",
    "figure_alt_text",
    "figure_texts",
    "format_bf01_log",
    "format_bf10",
    "format_conf_level",
    "format_df",
    "format_n",
    "format_p_subtitle",
    "format_value",
    "frequentist_expression",
    "n_label",
    "one_sample_alt_text",
    "one_sample_sentence",
    "p_adjust_label",
    "pairwise_bayes_sentence",
    "pairwise_bf_expression",
    "pairwise_caption_line",
    "pairwise_p_expression",
    "percent_text",
    "r_number_text",
    "results_sentence",
    "shown_test_value",
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


def zero_p_note(what: str, where: str = "The subtitle prints") -> str:
    """The ``warnings`` entry for a p-value of exactly 0 that a text prints as
    2.23e-308 (``format_p_subtitle``; D108, review B15): ``what`` names the
    test ("the chi-square test"), ``where`` the text that prints it."""
    return (
        f"The p-value of {what} is 0 as computed: it is smaller than the smallest "
        "number a double can hold (about 2.2e-308), not exactly zero. "
        f"{where} p = 2.23e-308, as ggstatsplot prints a p-value of 0; report "
        "p < 1e-300 (in APA style p < .001)."
    )


def with_zero_p_note(
    warnings: Sequence[str], test: TestResult | None, *, shown: bool
) -> tuple[str, ...]:
    """``warnings`` with :func:`zero_p_note` for the main test, where its p is
    exactly 0 and the subtitle is ``shown``, unless a warning already explains
    the 2.23e-308 (histostats' bootstrap, scatterstats' exact correlation)."""
    out = tuple(warnings)
    if (
        not shown
        or test is None
        or test.p_value != 0.0
        or any("2.23e-308" in w for w in out)
    ):
        return out
    return (*out, zero_p_note(test.label))


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
    "paired_t": ("student_t", False, False),
    "rm_anova": ("fisher_f", True, True),
    "wilcoxon_signed_rank": ("wilcoxon_v", False, False),
    "friedman": ("friedman_chi2", False, False),
    "yuen_dependent_t": ("yuen_t", False, False),
    "trimmed_rm_anova": ("trimmed_means_f", True, True),
    # One sample (gghistostats): ``stats::t.test``'s method ends in "t-test", so
    # statsExpressions prints t_Student with its integer df; WRS2's trimcibt
    # ("Bootstrap-t method ...") prints t_bootstrapped without a df; the
    # signed-rank test is the paired one's V_Wilcoxon (key above).
    "one_sample_t": ("student_t", False, False),
    "trimmed_mean_bootstrap_t": ("bootstrap_t", False, False),
    # Correlation (ggscatterstats): every method ends in "correlation", so the
    # Pearson and winsorized t is t_Student on n - 2 df; Spearman prints S alone.
    "pearson_correlation": ("student_t", False, False),
    "winsorized_correlation": ("student_t", False, False),
    "spearman_correlation": ("spearman_s", False, False),
    # Counts (ggbarstats): chisq.test's methods "Pearson's Chi-squared test" and
    # "Chi-squared test for given probabilities" and mcnemar.test's "McNemar's
    # Chi-squared test" print chi^2 with its integer df (Bowker's test of
    # symmetry beyond 2 x 2 is the same method string).
    "pearson_chi_square": ("pearson_chi2", False, False),
    "mcnemar": ("mcnemar_chi2", False, False),
    "chi_square_gof": ("gof_chi2", False, False),
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
    *, p_adjusted: float | None, adjust_method: str, digits: int = 2
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


def drawable(name: str) -> str:
    """A level name as the figure draws it: a tab as a space, and any other
    control character but the line break (which wraps the tick label) as U+FFFD,
    the replacement character, so that every file format can hold it (XML, and
    so SVG, cannot hold most control characters)."""
    return "".join(
        " "
        if char == "\t"
        else "\ufffd"
        if char != "\n" and unicodedata.category(char) == "Cc"
        else char
        for char in name
    )


def tick_label(level: str, n: int) -> str:
    """x tick label ``"<level>\\n(n = <n>)"``: plain text, never mathtext.

    ggstatsplot passes these as character labels (no parse); draw them with
    ``parse_math=False`` so a level such as "$5" or "a_b" shows literally. A
    control character in the name is drawn as :func:`drawable` says.
    """
    return f"{drawable(level)}\n(n = {format_n(n)})"


def subtitle_expression(
    test: TestResult,
    effect_size: EffectSize,
    *,
    n_obs: int,
    digits: int,
    paired: bool = False,
) -> Call:
    """The frequentist subtitle of a result's main test, as ggstatsplot builds it.

    ``paired`` prints n as ``italic("n")["pairs"]``: the number of complete
    subjects of a within-subjects result (ggwithinstats) or of complete pairs of a
    correlation (ggscatterstats); otherwise ``["obs"]`` (ggbetweenstats,
    gghistostats).
    """
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
        paired=paired,
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
# R parses a leading ``~x`` as the call `~`(x): plotmath draws a space before x.
_UNARY_TILDE: Final = Sym("~")
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
    if fn == _UNARY_TILDE and len(node.args) == 1:
        # ``~x``: plotmath draws a space, then x (ggbarstats' bar labels).
        return "\\ " + _mathtext(node.args[0], style)
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
    if fn == _UNARY_TILDE and len(args) == 1:
        return args[0]  # the leading space plotmath draws is left out
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

# ggstatsplot's pairwise.alpha; the caption states any other threshold.
DEFAULT_PAIRWISE_ALPHA: Final = 0.05


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def pairwise_caption_line(
    pairwise: Sequence[PairwiseComparison],
    *,
    display: str,
    adjust_method: str,
    alpha: float,
) -> str | None:
    """The caption line naming the pairwise test, adjustment and drawn pairs.

    "Pairwise: Games-Howell, Holm adjusted; showing 1 of 3 pairs (significant
    only)." None when there are no pairs or ``display`` is "none". It also states how
    many pairs were tested when only some are drawn (SPEC section 5 item 3). When
    ``alpha`` (``pairwise_alpha``) is not 0.05 and the pairs drawn depend on it, the
    note states the threshold: "(significant only, adjusted p < 0.1)", "(non-
    significant only, adjusted p >= 0.1)", "p" alone when the p-values are not
    adjusted. Bayes factor pairs (type "bayes", every pair drawn): "Pairwise: Bayes
    factors (Student's t, JZS prior 0.707), unadjusted; showing all 3 pairs."
    (D41).
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
    if alpha != DEFAULT_PAIRWISE_ALPHA:
        p = "p" if adjust_method == "none" else "adjusted p"
        relation = "<" if display == "significant" else ">="
        note = f"{note}, {p} {relation} {r_number_text(alpha)}"
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


# Effect sizes in the data's units (the trimmed mean of a one-sample robust
# test): printed with the subtitle's number rule, not with two fixed decimals.
DATA_UNIT_EFFECT_SIZES: Final = frozenset({"trimmed_mean"})


def data_number(x: float, digits: int) -> str:
    """A quantity in the data's units as the texts print it: the subtitle's
    number rule (:func:`format_value` with ``digits``: fixed with ``digits``
    decimals, scientific below 10^-digits and from 1e5), so that "19.24" stays
    "19.24" and 3.1e-9 prints "3.10e-09", not "0.00"; no negative zero."""
    if math.isnan(x):
        return "NA"
    return _no_neg_zero(format_value(x, digits))


def _apa_effect_size(effect_size: EffectSize, digits: int = 2) -> str:
    """ "<label> = <estimate>, 95% CI [<low>, <high>]": two decimals, without the
    leading zero where bounded by 1; in the data's units
    (DATA_UNIT_EFFECT_SIZES) with :func:`data_number` and ``digits``."""
    if effect_size.name in DATA_UNIT_EFFECT_SIZES:
        estimate, low, high = (
            data_number(x, digits)
            for x in (effect_size.estimate, effect_size.conf_low, effect_size.conf_high)
        )
        level = format_conf_level(effect_size.conf_level)
        return f"{effect_size.label} = {estimate}, {level} CI [{low}, {high}]"
    bounded = effect_size.name in _BOUNDED_EFFECT_SIZES
    estimate = _apa_number(effect_size.estimate, bounded=bounded)
    low = _apa_number(effect_size.conf_low, bounded=bounded)
    high = _apa_number(effect_size.conf_high, bounded=bounded)
    level = format_conf_level(effect_size.conf_level)
    return f"{effect_size.label} = {estimate}, {level} CI [{low}, {high}]"


# The longest level name the APA sentence prints whole; alt_text() shortens further.
APA_NAME_LIMIT: Final = 40


def _direction(alternative: str, levels: Sequence[str], limit: int) -> str:
    """A one-sided hypothesis in words: "OJ greater than VC". ``alternative``
    "greater" tests whether the first level is greater than the second, "less"
    whether it is less (the definition of the argument messages, D59)."""
    if len(levels) != 2:
        raise ValueError(
            f"a one-sided test ({alternative!r}) compares two levels; got "
            f"{len(levels)} level names, so its direction cannot be named"
        )
    short = _shortened(levels, limit)
    first, second = (short[level] for level in levels)
    if alternative == "greater":
        return f"{first} greater than {second}"
    if alternative == "less":
        return f"{first} less than {second}"
    raise ValueError(f"unknown alternative {alternative!r}")


def _test_heading(test: TestResult, levels: Sequence[str], limit: int) -> str:
    """The test's name, followed for a one-sided test by its direction with the
    level names: "Welch's t-test (one-sided: OJ greater than VC)"."""
    if test.alternative == "two-sided":
        return test.label
    return f"{test.label} (one-sided: {_direction(test.alternative, levels, limit)})"


def _n_phrase(n_obs: int, design: str, k: int) -> str:
    """What n counts: "n = 60" (observations), or in a within-subjects design
    "n = 10 pairs" (two conditions) and "n = 88 subjects" (three or more)."""
    n = f"n = {format_n(n_obs)}"
    if not _paired(design):
        return n
    if k < 2:
        raise ValueError(
            f"a within-subjects sentence needs the level names to say what n counts; "
            f"got {k}"
        )
    return f"{n} pairs" if k == 2 else f"{n} subjects"


def apa_sentence(
    test: TestResult,
    effect_size: EffectSize,
    *,
    n_obs: int,
    levels: Sequence[str] = (),
    name_limit: int = APA_NAME_LIMIT,
    design: str = "between",
) -> str:
    """The APA 7 results sentence (SPEC section 7).

    "Welch's t-test: t(55.31) = 1.92, p = .061, Hedges' g = 0.49, 95% CI [-0.02,
    0.99], n = 60." Statistics and estimates to two decimals; p to three decimals
    without a leading zero ("p < .001" below .001); no leading zero for quantities
    bounded by 1; non-integer df to two decimals, integer df bare. A one-sided test
    names its direction after the test name with the two ``levels`` (the analyzed
    order: "greater" tests whether the first is greater than the second), each
    shortened to ``name_limit`` characters with an ellipsis: "Welch's t-test
    (one-sided: OJ greater than VC): ..."; ``levels`` is required then. The
    sentence is ASCII except for the level names of a one-sided test, which are
    the user's data and printed as they are, and the ellipsis (U+2026) that
    shortens a long one. A within-subjects ``design`` says what n counts: "n = 10
    pairs" with two ``levels``, "n = 88 subjects" with more ("Paired t-test: t(9) =
    -4.06, p = .003, Hedges' g = -1.17, 95% CI [-1.94, -0.38], n = 10 pairs.").
    """
    heading = _test_heading(test, levels, name_limit)
    return (
        f"{heading}: {_apa_test(test)}, {_apa_effect_size(effect_size)}, "
        f"{_n_phrase(n_obs, design, len(levels))}."
    )


def _apa_bf(bayes: BayesFactor) -> str:
    """A Bayes factor: two decimals, scientific from 1e5 and below 0.01, and from
    log(BF01) where BF10 is beyond a double (:func:`format_bf10`)."""
    return format_bf10(bayes.bf10, bayes.log_bf01, 2)


def _prior_text(bayes: BayesFactor) -> str:
    return f"JZS prior, r = {r_number_text(bayes.prior_scale)}"


def bayes_sentence(bayes: BayesFactor, *, n_obs: int, design: str = "between") -> str:
    """The results sentence of a two-group Bayes factor (type "bayes"), ASCII
    (it names no level).

    "Bayesian t-test (JZS prior, r = 0.707): BF10 = 1.20, log(BF01) = -0.18,
    n = 60." BF10 to two decimals (scientific from 1e5 and below 0.01, as
    ``format_value``), log(BF01) to two decimals. Two conditions of a
    within-subjects ``design``: "Bayesian paired t-test (...): ..., n = 10 pairs."
    """
    name = "Bayesian paired t-test" if _paired(design) else "Bayesian t-test"
    return (
        f"{name} ({_prior_text(bayes)}): BF10 = {_apa_bf(bayes)}, "
        f"log(BF01) = {_apa_number(bayes.log_bf01)}, {_n_phrase(n_obs, design, 2)}."
    )


def pairwise_bayes_sentence(
    pairwise: Sequence[PairwiseComparison], *, n_obs: int, design: str = "between"
) -> str:
    """The results sentence of type "bayes" with three or more groups, which has no
    omnibus test in v0.1 (D41): each pair's Bayes factor. The level names are
    printed as they are, so the sentence is ASCII only when they are.

    "Bayesian t-tests of each pair (JZS prior, r = 0.707): ctrl vs trt1, BF10 =
    0.65; ctrl vs trt2, BF10 = 1.77; trt1 vs trt2, BF10 = 6.36; n = 30." A
    within-subjects ``design``: "Bayesian paired t-tests of each pair (...): ...;
    n = 88 subjects."
    """
    first = pairwise[0].bayes
    assert first is not None
    parts: list[str] = []
    for pair in pairwise:
        assert pair.bayes is not None
        parts.append(f"{pair.group1} vs {pair.group2}, BF10 = {_apa_bf(pair.bayes)}")
    name = "Bayesian paired t-tests" if _paired(design) else "Bayesian t-tests"
    return (
        f"{name} of each pair ({_prior_text(first)}): "
        + "; ".join(parts)
        + f"; {_n_phrase(n_obs, design, 3)}."
    )


def results_sentence(
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    pairwise: Sequence[PairwiseComparison],
    *,
    n_obs: int,
    levels: Sequence[str] = (),
    name_limit: int = APA_NAME_LIMIT,
    design: str = "between",
) -> str:
    """The result's APA-style sentence: :func:`apa_sentence` for a frequentist
    test (``levels`` and ``name_limit`` name a one-sided test's direction),
    :func:`bayes_sentence` or :func:`pairwise_bayes_sentence` for type "bayes", or
    a note that nothing was tested. A within-subjects ``design`` says what n counts
    (``levels``: pairs with two, subjects with more)."""
    if test is not None and effect_size is not None:
        return apa_sentence(
            test,
            effect_size,
            n_obs=n_obs,
            levels=levels,
            name_limit=name_limit,
            design=design,
        )
    if bayes is not None:
        return bayes_sentence(bayes, n_obs=n_obs, design=design)
    if pairwise and pairwise[0].bayes is not None:
        return pairwise_bayes_sentence(pairwise, n_obs=n_obs, design=design)
    n = _n_phrase(n_obs, design, len(levels))
    return f"No test was computed (results_subtitle=False), {n}."


def _short(name: str, limit: int) -> str:
    # U+2026 HORIZONTAL ELLIPSIS: three periods would read as a sentence end.
    return name if len(name) <= limit else name[: limit - 1] + "\u2026"


def _elided(name: str, start: int, head: int, limit: int) -> str:
    """``name`` kept from ``start`` (a word start) to its end, after its first
    ``head`` characters and an ellipsis; the end cut with an ellipsis when it is
    longer than ``limit`` allows."""
    tail = name[start:]
    lead = name[:head]
    if head < len(name) and not name[head].isspace() and " " in lead:
        lead = lead[: lead.rfind(" ")]  # end the start at a word's end
    lead = lead.rstrip()
    joined = f"{lead} \u2026 {tail}" if lead else f"\u2026{tail}"
    if len(joined) <= limit:
        return joined
    room = limit - 2  # the leading and the trailing ellipsis
    return f"\u2026{tail[:room].rstrip()}\u2026"


def _shortened(names: Sequence[str], limit: int) -> dict[str, str]:
    """Each of ``names`` shortened to ``limit`` characters, every one still told
    apart from the others.

    A name is cut at the end with an ellipsis (:func:`_short`), unless that gives
    the same text as another name's: then the names that would look alike keep
    what they do not share (from the word where they first differ to the end)
    after the start they share, with an ellipsis between ("Mean reaction …
    congruent trials", "Mean reaction … incongruent trials"). A name that would
    still look like another's is not shortened.
    """
    unique = list(dict.fromkeys(names))
    out = {name: _short(name, limit) for name in unique}
    groups: dict[str, list[str]] = {}
    for name in unique:
        groups.setdefault(out[name], []).append(name)
    for alike in groups.values():
        if len(alike) < 2:
            continue
        shared = 0
        while all(
            len(name) > shared and name[shared] == alike[0][shared] for name in alike
        ):
            shared += 1
        tails: dict[str, int] = {}
        for name in alike:
            # Back to the start of the word where the names part.
            tails[name] = name.rfind(" ", 0, shared) + 1
        longest = max(len(name) - tails[name] for name in alike)
        head = max(0, limit - 3 - longest)
        for name in alike:
            out[name] = _elided(name, tails[name], head, limit)
    seen: dict[str, int] = {}
    for text in out.values():
        seen[text] = seen.get(text, 0) + 1
    return {name: (name if seen[text] > 1 else text) for name, text in out.items()}


def _centrality_word(centrality: str, tr: float) -> str:
    """The centrality in words: "mean", "median", "20% trimmed mean", "MAP
    estimate"."""
    word = _CENTRALITY_WORDS.get(centrality, centrality)
    if centrality == "trimmed_mean":
        return f"{r_number_text(tr * 100)}% {word}"
    return word


def _alt_groups_detailed(
    descriptives: Sequence[Descriptive],
    digits: int,
    limit: int,
    tr: float,
    centrality: bool = True,
) -> str:
    short = _shortened([d.level for d in descriptives], limit)
    if not centrality:
        return ", ".join(
            f"{short[d.level]} (n = {format_n(d.n)})" for d in descriptives
        )
    parts: list[str] = []
    for d in descriptives:
        word = _centrality_word(d.centrality, tr)
        # A missing centrality (a MAP without a density, D55): "no MAP estimate".
        center = (
            f"no {word}"
            if d.value is None or math.isnan(d.value)
            else f"{word} {format_value(d.value, digits)}"
        )
        parts.append(f"{short[d.level]} (n = {format_n(d.n)}, {center})")
    return ", ".join(parts)


def _alt_groups_summary(
    descriptives: Sequence[Descriptive],
    digits: int,
    limit: int,
    tr: float,
    centrality: bool = True,
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
    if not finite or not centrality:
        return n_text
    short = _shortened([d.level for d in descriptives], limit)
    low, low_value = min(finite, key=lambda item: item[1])
    high, high_value = max(finite, key=lambda item: item[1])
    word = _centrality_word(low.centrality, tr)
    return (
        f"{n_text}; {word}s from {format_value(low_value, digits)} "
        f"({short[low.level]}) to {format_value(high_value, digits)} "
        f"({short[high.level]})"
    )


def _alt_bayes_pairs(
    pairwise: Sequence[PairwiseComparison],
    *,
    limit: int,
    display: str,
    paired: bool = False,
) -> str:
    """Type "bayes" with three or more groups: the range of the pairs' BF10, the
    level names shortened to ``limit`` characters, and what the brackets show.
    ``paired`` (conditions of a within-subjects design) names paired t-tests.

    Pairs are ranked by -log(BF01), which stays finite where BF10 overflows a
    double, so the strongest pair is found even when several BF10 are inf."""
    first = pairwise[0].bayes
    assert first is not None
    short = _shortened(
        [name for pair in pairwise for name in (pair.group1, pair.group2)], limit
    )

    def strength(pair: PairwiseComparison) -> float:
        assert pair.bayes is not None
        return -pair.bayes.log_bf01

    def described(pair: PairwiseComparison) -> str:
        assert pair.bayes is not None
        return f"{_apa_bf(pair.bayes)} ({short[pair.group1]} vs {short[pair.group2]})"

    low, high = min(pairwise, key=strength), max(pairwise, key=strength)
    brackets = (
        "no brackets are drawn (pairwise_display='none')."
        if display == "none"
        else "brackets show log(BF01) for every pair."
    )
    tests = (
        f"Bayesian paired t-tests of each of the {len(pairwise)} pairs "
        f"({_prior_text(first)})"
        if paired
        else f"Bayesian t-tests of each of the {len(pairwise)} pairs "
        f"({_prior_text(first)}, Student's t)"
    )
    return f"{tests}: BF10 from {described(low)} to {described(high)}; {brackets}"


def _alt_conditions_detailed(
    descriptives: Sequence[Descriptive],
    digits: int,
    limit: int,
    tr: float,
    centrality: bool = True,
) -> str:
    """Each condition with its centrality value (only the names without
    centrality points); n is the same for all and is named once, in the head."""
    short = _shortened([d.level for d in descriptives], limit)
    if not centrality:
        return ", ".join(short[d.level] for d in descriptives)
    parts: list[str] = []
    for d in descriptives:
        word = _centrality_word(d.centrality, tr)
        center = (
            f"no {word}"
            if d.value is None or math.isnan(d.value)
            else f"{word} {format_value(d.value, digits)}"
        )
        parts.append(f"{short[d.level]} ({center})")
    return ", ".join(parts)


def _alt_conditions_summary(
    descriptives: Sequence[Descriptive],
    digits: int,
    limit: int,
    tr: float,
    centrality: bool = True,
) -> str:
    """The range of the conditions' centrality values, or "" without any."""
    summary = _alt_groups_summary(descriptives, digits, limit, tr, centrality)
    _, _, values = summary.partition("; ")
    return values


def _alt_paths(
    descriptives: Sequence[Descriptive],
    *,
    n_subjects: int,
    subject_paths: int,
    centrality_path: bool,
    tr: float,
) -> str:
    """The paths drawn, as a clause: "; lines join each subject's values and the
    means", "; a line joins the medians", "" without paths. ``subject_paths`` is
    the number of subjects whose paths are drawn (fewer than all when a large
    figure draws a random sample of whole subjects)."""
    word = _centrality_word(descriptives[0].centrality, tr) + "s"
    # A centrality path is a line only where two neighbouring levels have a
    # value (a level without a MAP estimate breaks it, D55).
    centrality_path = centrality_path and any(
        a.value is not None and b.value is not None
        for a, b in itertools.pairwise(descriptives)
    )
    if subject_paths >= n_subjects:
        whose = "each subject's values"
    else:
        whose = (
            f"the values of a random sample of {subject_paths:,} of the "
            f"{n_subjects:,} subjects"
        )
    if subject_paths and centrality_path:
        return f"; lines join {whose} and the {word}"
    if subject_paths:
        return f"; lines join {whose}"
    if centrality_path:
        return f"; a line joins the {word}"
    return ""


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
    design: str = "between",
    subject_paths: int | None = None,
    centrality_path: bool | None = None,
    centrality: bool = True,
) -> str:
    """Alt text for the figure: 2 to 4 sentences, at most 400 characters.

    Chart type and variables, groups with n and centrality values (summarized as
    ranges when listing every group would not fit), the omnibus result, and the
    count of significant pairs. Type "bayes" with three or more groups gives the
    range of the pairs' Bayes factors instead.

    A within-subjects ``design`` says "conditions" and that they were "measured on
    the same <n> subjects", lists each condition's centrality value without
    repeating n, and names the paths drawn: ``subject_paths`` subjects' paths (all
    of them, a random sample, or 0) and the ``centrality_path``. Left as None, they
    are what ``ip.withinstats`` draws by default: every subject's path with two
    conditions, none with more, and the centrality path; a figure drawn otherwise
    carries its own alt text (:func:`figure_alt_text`). Its result sentence says
    what n counts ("n = 10 pairs", "n = 88 subjects").

    It describes only what the figure shows. Without centrality points
    (``centrality``, a figure drawn with ``centrality_plotting=False``) it names
    the levels without their centrality values, and a centrality path is named
    only where one is drawn. With ``results_subtitle=False`` the
    figure shows no test, so the result sentence reads "No test result is shown
    (results_subtitle=False), n = <n>." (type "bayes" with two groups computes
    none: "No test was computed ..."), while the APA sentence still reports a
    computed test. The pair sentence follows the brackets: it is there whenever
    brackets are drawn (also with ``results_subtitle=False``) and left out with
    ``pairwise_display="none"``. A one-sided test's direction names the levels
    as the APA sentence does, shortened with them.

    Level names, x and y are shortened (40, 20, then 12 characters) until the text
    fits. If it still does not, the group description is left out, then the paths;
    the result sentence is never cut, so a result sentence too long by itself (a
    statistic printed with hundreds of digits) gives a text above the limit.
    """
    k = len(descriptives)
    paired = _paired(design)
    levels = [d.level for d in descriptives]
    pairs = ""
    bayes_pairs = bool(pairwise) and pairwise[0].bayes is not None
    paths = ""
    if paired:
        drawn_subjects = (
            (n_obs if k == 2 else 0) if subject_paths is None else subject_paths
        )
        paths = _alt_paths(
            descriptives,
            n_subjects=n_obs,
            subject_paths=drawn_subjects,
            centrality_path=True if centrality_path is None else centrality_path,
            tr=settings.tr,
        )
    elif subject_paths is not None or centrality_path is not None:
        raise ValueError("independent groups have no subject or centrality paths")

    def result_at(limit: int) -> str:
        if bayes_pairs:
            return _alt_bayes_pairs(
                pairwise,
                limit=limit,
                display=settings.pairwise_display,
                paired=paired,
            )
        if not settings.results_subtitle and (test is not None or bayes is not None):
            # Computed, but the figure does not show it (U12).
            return (
                "No test result is shown (results_subtitle=False), "
                f"{_n_phrase(n_obs, design, k)}."
            )
        return results_sentence(
            test,
            effect_size,
            bayes,
            pairwise,
            n_obs=n_obs,
            levels=levels,
            name_limit=limit,
            design=design,
        )

    if pairwise and not bayes_pairs and settings.pairwise_display != "none":
        # Brackets and the caption's pairwise line are drawn (also with
        # results_subtitle=False); with pairwise_display="none" neither is.
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
    digits, tr = settings.digits, settings.tr
    for limit in (40, 20, 12):
        result = result_at(limit)
        if paired:
            head = (
                f"Plot comparing {_short(y, limit)} across the {k} conditions of "
                f"{_short(x, limit)}, measured on the same {format_n(n_obs)} subjects"
            )
            detailed = _alt_conditions_detailed(
                descriptives, digits, limit, tr, centrality
            )
            summary = _alt_conditions_summary(
                descriptives, digits, limit, tr, centrality
            )
            candidates = [f"{head}: {detailed}{paths}. {result}{pairs}"]
            if summary:
                candidates.append(f"{head} ({summary}){paths}. {result}{pairs}")
        else:
            head = (
                f"Plot comparing {_short(y, limit)} across the {k} groups of "
                f"{_short(x, limit)}"
            )
            detailed = _alt_groups_detailed(descriptives, digits, limit, tr, centrality)
            summary = _alt_groups_summary(descriptives, digits, limit, tr, centrality)
            candidates = [
                f"{head}: {detailed}. {result}{pairs}",
                f"{head} ({summary}). {result}{pairs}",
            ]
        for text in candidates:
            if len(text) <= ALT_TEXT_MAX_CHARS:
                return text
    # Extremely long labels or results: keep the start and the whole result, drop
    # the groups, then the paths.
    with_paths = f"{head}{paths}. {result}{pairs}"
    if len(with_paths) <= ALT_TEXT_MAX_CHARS:
        return with_paths
    return f"{head}. {result}{pairs}"


def figure_alt_text(
    result: StatsResult,
    *,
    subject_paths: int | None = None,
    centrality_path: bool | None = None,
    centrality: bool = True,
    test_value_line: bool | None = None,
    bins: int | None = None,
    smooth: bool | None = None,
    marginal: bool | None = None,
    labelled_points: int | None = None,
    points_shown: int | None = None,
    cell_labels: bool | None = None,
    bar_tests: bool | None = None,
    legend: bool | None = None,
) -> str:
    """The alt text of the figure the renderer drew for ``result``; it gives the
    text to the result it returns. Every keyword describes what the figure
    draws, and its default is what the default figure of the result's function
    draws (so ``figure_alt_text(result)`` equals ``result.text.alt_text`` for a
    result fresh from its analysis). A keyword of another design raises
    ``ValueError``.

    betweenstats and withinstats (:func:`alt_text`): ``centrality``, the
    centrality points and labels; withinstats only: ``subject_paths``, the number
    of subjects whose paths are drawn (0 for none), and ``centrality_path``.

    histostats (:func:`one_sample_alt_text`): ``centrality``, the centrality line
    and its label (``centrality_plotting``); ``test_value_line``, whether a line
    marks the test value (default False: ggstatsplot draws none); ``bins``, the
    number of bars drawn (default None: not named).

    scatterstats (:func:`correlation_alt_text`): ``smooth``, the least-squares
    line with its band (default True; pass False where none is drawn);
    ``marginal``, the marginal histograms (default True); ``labelled_points``,
    the number of points that carry a label (default: every non-empty entry of
    ``result.point_labels``); ``points_shown``, the number of points drawn when
    the figure draws a random sample of them (default None: every point).

    barstats (:func:`contingency_alt_text`): ``cell_labels``, the labels inside
    the bars (default True); ``bar_tests``, the p-value above each bar (default
    ``settings.proportion_test``; refused without y, where the single bar's test
    is the main test); ``legend``, the legend of the levels of x (default True).
    """
    design = result.design
    if design != "contingency":
        _check_design(design)
    passed = {
        "subject_paths": subject_paths,
        "centrality_path": centrality_path,
        "test_value_line": test_value_line,
        "bins": bins,
        "smooth": smooth,
        "marginal": marginal,
        "labelled_points": labelled_points,
        "points_shown": points_shown,
        "cell_labels": cell_labels,
        "bar_tests": bar_tests,
        "legend": legend,
    }
    own = {
        "between": (),
        "within": ("subject_paths", "centrality_path"),
        "one-sample": ("test_value_line", "bins"),
        "correlation": ("smooth", "marginal", "labelled_points", "points_shown"),
        "contingency": ("cell_labels", "bar_tests", "legend"),
    }[design]
    foreign = [k for k, v in passed.items() if v is not None and k not in own]
    if design in ("correlation", "contingency") and not centrality:
        foreign.insert(0, "centrality")
    if foreign:
        raise ValueError(
            f"{', '.join(foreign)} {'does' if len(foreign) == 1 else 'do'} not "
            f"describe a {result.analysis} figure (design {design!r})"
            + (f"; its keywords: {', '.join(own)}" if own else "")
        )
    for name in ("bins", "labelled_points", "points_shown"):
        count = passed[name]
        if count is not None and (isinstance(count, bool) or count < 0):
            raise ValueError(f"{name} must be a count of 0 or more, got {count!r}")
    if design == "contingency":
        return contingency_alt_text(
            x=result.x,
            y=result.y,
            levels=result.levels,
            test=result.test,
            effect_size=result.effect_size,
            bars=result.bars,
            cells=result.cells,
            n_obs=result.n_obs,
            settings=result.settings,
            cell_labels=True if cell_labels is None else cell_labels,
            bar_tests=bar_tests,
            legend=True if legend is None else legend,
        )
    if design == "one-sample":
        return one_sample_alt_text(
            x=result.x,
            descriptive=result.descriptives[0],
            values=result.values[0] if result.values else (),
            test=result.test,
            effect_size=result.effect_size,
            bayes=result.bayes,
            n_obs=result.n_obs,
            settings=result.settings,
            centrality=centrality,
            test_value_line=bool(test_value_line),
            bins=bins,
        )
    if design == "correlation":
        assert result.y is not None
        return correlation_alt_text(
            x=result.x,
            y=result.y,
            test=result.test,
            effect_size=result.effect_size,
            bayes=result.bayes,
            n_obs=result.n_obs,
            settings=result.settings,
            label_var=result.label_var,
            labelled_points=(
                sum(1 for label in result.point_labels if label)
                if labelled_points is None
                else labelled_points
            ),
            smooth=True if smooth is None else smooth,
            marginal=True if marginal is None else marginal,
            points_shown=points_shown,
        )
    assert result.y is not None
    return alt_text(
        x=result.x,
        y=result.y,
        descriptives=result.descriptives,
        test=result.test,
        effect_size=result.effect_size,
        pairwise=result.pairwise,
        n_obs=result.n_obs,
        settings=result.settings,
        bayes=result.bayes,
        design=result.design,
        subject_paths=subject_paths,
        centrality_path=centrality_path,
        centrality=centrality,
    )


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
    paired: bool,
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
        subtitle = subtitle_expression(
            test, effect_size, n_obs=n_obs, digits=digits, paired=paired
        )
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
        alpha=settings.pairwise_alpha,
    )
    return _Expressions(subtitle, bayes_caption, centrality, labels, line)


_DESIGNS: Final = ("between", "within", "one-sample", "correlation")


def _check_design(design: str) -> None:
    if design not in _DESIGNS:
        known = ", ".join(repr(d) for d in _DESIGNS)
        raise ValueError(f"unknown design {design!r}; known: {known}")


def _paired(design: str) -> bool:
    """Whether a result's design measures the same subjects in every level:
    "within" (withinstats) does; "between", "one-sample" and "correlation" do
    not."""
    _check_design(design)
    return design == "within"


def _n_pairs(design: str) -> bool:
    """Whether the subtitle prints n as ``italic("n")["pairs"]``: the complete
    subjects of "within" (ggwithinstats) and the complete pairs of "correlation"
    (ggscatterstats); ``["obs"]`` for "between" and "one-sample"."""
    _check_design(design)
    return design in ("within", "correlation")


def _grouped(design: str) -> bool:
    """Whether the design compares levels (groups or conditions): "between" and
    "within". "one-sample" (histostats) and "correlation" (scatterstats) have no
    levels, tick labels or pairs."""
    _check_design(design)
    return design in ("between", "within")


def _pair_label(pair: PairwiseComparison, digits: int) -> Call:
    """A bracket label: the log Bayes factor of a Bayesian pair, else its
    adjusted p."""
    if pair.bayes is not None:
        return pairwise_bf_expression(log_bf01=pair.bayes.log_bf01, digits=digits)
    if pair.p_adjusted is None and pair.no_test_reason is not None:
        # barstats: a Fisher pair without a test (its table has one column, or
        # its exact test is beyond the cost limit) gets an empty p, as a bar
        # without a test does: list(italic(p)['Holm'-adj.]=='').
        return pairwise_p_expression(
            p_adjusted=None, adjust_method=pair.adjust_method, digits=digits
        )
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


# --------------------------------------------------------------------------- #
# Designs without levels: one sample (histostats) and correlation (scatterstats)
# --------------------------------------------------------------------------- #
#
# The subtitle, Bayes caption and centrality label are ggstatsplot's expressions
# (the shared builders above, with the templates of _TEST_TEMPLATES); the APA
# sentence and the alt text are inferplot's own, in the style of the other
# designs (SPEC section 7).


# The longest test value printed as written (R's 15-significant-digit print).
TEST_VALUE_CHARS: Final = 8


def shown_test_value(test_value: float, digits: int) -> tuple[str, bool]:
    """The test value as the texts print it, and whether that is rounded.

    As written (R's print of the double, :func:`r_number_text`: "18", "5.8",
    "-10", "1.7e+09", "0.001") when that takes at most TEST_VALUE_CHARS
    characters; otherwise rounded with the subtitle's number rule
    (:func:`format_value` with ``digits``: "0.33" for 1/3, "36.12" for
    36.123456789, "1.23e+05" for 123456.789) and marked as rounded by the
    caller ("test value ≈ 0.33", "about 0.33"), so that a printed value never
    stands for a different one. The result keeps the exact value
    (``settings.test_value``).
    """
    written = r_number_text(test_value)
    if len(written) <= TEST_VALUE_CHARS or not math.isfinite(test_value):
        return written, False
    return format_value(test_value, digits), True


def _test_value_text(settings: Settings) -> str:
    """The value a one-sample test is tested against, in the APA sentence and
    the alt text: "18", or "about 0.33" where it is rounded
    (:func:`shown_test_value`)."""
    if settings.test_value is None:
        raise ValueError(
            "a one-sample result needs settings.test_value, the value its test is "
            "tested against"
        )
    shown, rounded = shown_test_value(settings.test_value, settings.digits)
    return f"about {shown}" if rounded else shown


def _no_level_n(n_obs: int, design: str) -> str:
    """What n counts without levels: "n = 60" (the values of one sample) or
    "n = 32 pairs" (the complete pairs of a correlation)."""
    n = f"n = {format_n(n_obs)}"
    return n if design == "one-sample" else f"{n} pairs"


def one_sample_sentence(
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    *,
    x: str,
    n_obs: int,
    settings: Settings,
    name_limit: int = APA_NAME_LIMIT,
) -> str:
    """The APA 7 results sentence of a one-sample result (histostats).

    "One-sample t-test against 18: t(59) = 0.82, p = .413, Hedges' g = 0.10, 95%
    CI [-0.15, 0.36], n = 60." The other types: "Wilcoxon signed-rank test against
    18: V = 1030.50, p = .397, rank-biserial r = .13, 95% CI [-.16, .40], n = 60.";
    "Bootstrap-t test of the 20% trimmed mean against 18: t = 1.00, p = .331,
    trimmed mean = 19.24, 95% CI [16.61, 21.88], n = 60."; "Bayesian one-sample
    t-test against 18 (JZS prior, r = 0.707): BF10 = 0.20, log(BF01) = 1.63, n =
    60." A one-sided test names its direction with the column ``x`` (shortened to
    ``name_limit`` characters with an ellipsis, as D68 names levels): "One-sample
    t-test (one-sided: len greater than 18): ...". Without a computed test (type
    bayes, ``results_subtitle=False``): "No test was computed
    (results_subtitle=False), n = 60."

    Numbers follow :func:`apa_sentence`: an infinite statistic or bound prints
    "Inf" or "-Inf" (a one-sided Hedges' g interval [-0.11, Inf], an unbounded
    bootstrap-t interval [-Inf, Inf]), and every p below .001 prints "p < .001",
    also a bootstrap p of 0: no resample of the 2,000 was as extreme as the data,
    so the estimate is 0 and the true p is below about 1/2,000 (R's 200 resamples
    give the same 0, which ggstatsplot prints as 2.23e-308). The sentence is ASCII
    except the column name of a one-sided test and the ellipsis that shortens it.
    """
    against = _test_value_text(settings)
    n = _no_level_n(n_obs, "one-sample")
    if test is not None and effect_size is not None:
        if test.alternative == "two-sided":
            heading = f"{test.label} against {against}"
        elif test.alternative in ("greater", "less"):
            heading = (
                f"{test.label} (one-sided: {_short(x, name_limit)} "
                f"{test.alternative} than {against})"
            )
        else:
            raise ValueError(f"unknown alternative {test.alternative!r}")
        shown = _apa_effect_size(effect_size, settings.digits)
        return f"{heading}: {_apa_test(test)}, {shown}, {n}."
    if bayes is not None:
        return (
            f"Bayesian one-sample t-test against {against} ({_prior_text(bayes)}): "
            f"BF10 = {_apa_bf(bayes)}, log(BF01) = {_apa_number(bayes.log_bf01)}, "
            f"{n}."
        )
    return f"No test was computed (results_subtitle=False), {n}."


def _beta_prior_text(bayes: BayesFactor) -> str:
    """The correlation prior as the figure prints it: the stretched beta
    distribution's parameter 1 / bf_prior to two decimals ("r_beta = 1.41" for
    bf_prior 0.707), which ggstatsplot's caption shows as r_beta^JZS (D86)."""
    return f"stretched beta prior, r_beta = {format_value(bayes.prior_scale, 2)}"


def correlation_sentence(
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    *,
    n_obs: int,
) -> str:
    """The APA 7 results sentence of a correlation (scatterstats), ASCII.

    "Pearson correlation: t(30) = -9.56, p < .001, Pearson's r = -.87, 95% CI
    [-.93, -.74], n = 32 pairs."; "Spearman correlation: S = 10292.32, p < .001,
    Spearman's rho = -.89, 95% CI [-.94, -.77], n = 32 pairs."; "Winsorized
    Pearson correlation (20% winsorized): t(30) = -9.41, ..., winsorized r =
    -.86, ..."; "Bayesian Pearson correlation (stretched beta prior, r_beta =
    1.41): BF10 = 5.62e+07, log(BF01) = -17.84, n = 32 pairs." Correlations drop
    the leading zero (bounded by 1). Exactly collinear data print their limits
    (D85, D86): "t(8) = Inf, p < .001, Pearson's r = 1.00, 95% CI [1.00, 1.00]";
    "S = 0.00"; "BF10 = Inf, log(BF01) = -Inf". Without a computed test (type
    bayes, ``results_subtitle=False``): "No test was computed
    (results_subtitle=False), n = 32 pairs."
    """
    n = _no_level_n(n_obs, "correlation")
    if test is not None and effect_size is not None:
        if test.alternative != "two-sided":
            raise ValueError(
                f"a correlation test is two-sided; got alternative={test.alternative!r}"
            )
        return f"{test.label}: {_apa_test(test)}, {_apa_effect_size(effect_size)}, {n}."
    if bayes is not None:
        return (
            f"Bayesian Pearson correlation ({_beta_prior_text(bayes)}): BF10 = "
            f"{_apa_bf(bayes)}, log(BF01) = {_apa_number(bayes.log_bf01)}, {n}."
        )
    return f"No test was computed (results_subtitle=False), {n}."


def _no_level_sentence(
    *,
    x: str,
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    n_obs: int,
    settings: Settings,
    design: str,
    name_limit: int = APA_NAME_LIMIT,
) -> str:
    if design == "one-sample":
        return one_sample_sentence(
            test,
            effect_size,
            bayes,
            x=x,
            n_obs=n_obs,
            settings=settings,
            name_limit=name_limit,
        )
    return correlation_sentence(test, effect_size, bayes, n_obs=n_obs)


def _shown_result(
    *,
    x: str,
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    n_obs: int,
    settings: Settings,
    design: str,
    name_limit: int,
) -> str:
    """The alt text's result sentence: the APA sentence where the figure shows the
    result, "No test result is shown (results_subtitle=False), n = ..." where a
    computed test is not drawn (D70)."""
    if not settings.results_subtitle and (test is not None or bayes is not None):
        return (
            "No test result is shown (results_subtitle=False), "
            f"{_no_level_n(n_obs, design)}."
        )
    return _no_level_sentence(
        x=x,
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        n_obs=n_obs,
        settings=settings,
        design=design,
        name_limit=name_limit,
    )


def _fits(text: str) -> bool:
    return len(text) <= ALT_TEXT_MAX_CHARS


def _marks_sentence(marks: Sequence[str]) -> str:
    """ "A vertical line marks the mean, 18.81." or "Vertical lines mark the
    mean, 18.81, and the test value, 18." ("" without marks)."""
    if not marks:
        return ""
    if len(marks) == 1:
        return f"A vertical line marks {marks[0]}."
    return f"Vertical lines mark {', '.join(marks[:-1])}, and {marks[-1]}."


def _value_range(low: float, high: float, digits: int) -> str:
    """ "from 4.20 to 33.90" with the subtitle's number rule; "all equal to 5.00"
    for one value; where the two ends would print alike but differ (1.7e9 plus a
    few seconds), with as many more digits as tell them apart."""
    if low == high:
        return f"all equal to {data_number(low, digits)}"
    shown = digits
    while data_number(low, shown) == data_number(high, shown) and shown < 17:
        shown += 1
    return f"from {data_number(low, shown)} to {data_number(high, shown)}"


def one_sample_alt_text(
    *,
    x: str,
    descriptive: Descriptive,
    values: Sequence[float],
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    n_obs: int,
    settings: Settings,
    centrality: bool = True,
    test_value_line: bool = False,
    bins: int | None = None,
) -> str:
    """Alt text of a histostats figure: 2 or 3 sentences, at most 400 characters.

    "Histogram of len (60 values, from 4.20 to 33.90, in 8 bins). A vertical line
    marks the mean, 18.81. One-sample t-test against 18: t(59) = 0.82, ..., n =
    60." The head names the column, the number of values, their range (from
    ``values``; left out when they are not at hand, as in a result read back from
    a saved file) and the number of ``bins`` drawn (when the figure gives it).
    The marks sentence names what the vertical lines show: the centrality line
    with its value (``centrality``: drawn unless ``centrality_plotting=False``;
    "the 20% trimmed mean, 19.24", "the MAP estimate, 24.12"; a MAP that could
    not be estimated draws no line and is named as missing) and the test value
    (``test_value_line``: when the figure marks it). The result sentence is the
    APA sentence (:func:`one_sample_sentence`), or with ``results_subtitle=False``
    "No test result is shown (results_subtitle=False), n = 60." (D70).

    The column name is shortened (40, 20, then 12 characters) until the text
    fits; then the range and bins are left out, then the marks sentence; the
    result sentence is never cut.
    """
    digits = settings.digits
    against = _test_value_text(settings)
    marks: list[str] = []
    missing = ""
    if centrality:
        word = _centrality_word(descriptive.centrality, settings.tr)
        value = descriptive.value
        if value is None or math.isnan(value):
            missing = f"No {word} could be computed, so no line marks it."
        else:
            marks.append(f"the {word}, {format_value(value, digits)}")
    if test_value_line:
        marks.append(f"the test value, {against}")
    lines = " ".join(part for part in (_marks_sentence(marks), missing) if part)
    details: list[str] = []
    if values:
        details.append(_value_range(min(values), max(values), digits))
    if bins is not None:
        details.append(f"in {format_n(bins)} {'bin' if bins == 1 else 'bins'}")
    count = f"{format_n(n_obs)} {'value' if n_obs == 1 else 'values'}"
    head = ""
    result = ""
    for limit in (40, 20, 12):
        result = _shown_result(
            x=x,
            test=test,
            effect_size=effect_size,
            bayes=bayes,
            n_obs=n_obs,
            settings=settings,
            design="one-sample",
            name_limit=limit,
        )
        name = _short(x, limit)
        head = f"Histogram of {name} ({count})."
        detailed = f"Histogram of {name} ({', '.join([count, *details])})."
        for start in (detailed, head):
            text = " ".join(part for part in (start, lines, result) if part)
            if _fits(text):
                return text
    # Still too long: the marks go, the result sentence stays whole.
    return f"{head} {result}"


def _correlation_direction(
    effect_size: EffectSize | None, *, x: str, y: str, limit: int
) -> str:
    """The direction of a correlation in words, only as far as its numbers say
    it: "mpg tends to fall as wt rises" where the interval lies below 0, "... to
    rise ..." where it lies above 0, "every point lies on one rising straight
    line" at a Pearson's r of exactly +/-1, and "" where the interval includes 0
    (the data do not settle the direction) or there is no interval (type bayes).
    Strength is left to the number in the result sentence."""
    if effect_size is None:
        return ""
    estimate, low, high = (
        effect_size.estimate,
        effect_size.conf_low,
        effect_size.conf_high,
    )
    if effect_size.name == "pearson_r" and abs(estimate) == 1.0:
        way = "rising" if estimate > 0 else "falling"
        return f"every point lies on one {way} straight line"
    if low > 0:
        return f"{_short(y, limit)} tends to rise as {_short(x, limit)} rises"
    if high < 0:
        return f"{_short(y, limit)} tends to fall as {_short(x, limit)} rises"
    return ""


def _join_words(parts: Sequence[str]) -> str:
    """ "a", "a, and b", "a, b, and c": the parts are phrases that hold an "and"
    of their own ("histograms of wt and mpg"), so each is set off by a comma."""
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def correlation_alt_text(
    *,
    x: str,
    y: str,
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    n_obs: int,
    settings: Settings,
    label_var: str | None = None,
    labelled_points: int = 0,
    smooth: bool = True,
    marginal: bool = True,
    points_shown: int | None = None,
) -> str:
    """Alt text of a scatterstats figure: 2 or 3 sentences, at most 400 characters.

    "Scatter plot of mpg against wt (32 pairs): mpg tends to fall as wt rises. It
    also shows a least-squares line with its 95% confidence band, and marginal
    histograms of wt and mpg. Pearson correlation: t(30) = -9.56, ..., n = 32
    pairs." The head names the variables, the number of pairs and, when the
    figure draws ``points_shown`` of them as a random sample, says so; the
    direction is said in words only where the figure shows the result and its
    interval excludes 0 (:func:`_correlation_direction`). The layers sentence
    names what the figure draws besides the points: the least-squares line with its
    ``conf_level`` band (``smooth``), the ``marginal`` histograms, and the
    ``labelled_points`` points labelled from ``label_var``. The result sentence is
    the APA sentence (:func:`correlation_sentence`), or with
    ``results_subtitle=False`` "No test result is shown (results_subtitle=False),
    n = 32 pairs." (D70).

    Column names are shortened (40, 20, then 12 characters) until the text fits;
    then the layers sentence is left out, then the direction; the result sentence
    is never cut.
    """
    level = format_conf_level(settings.conf_level)
    head = ""
    result = ""
    for limit in (40, 20, 12):
        sx, sy = _short(x, limit), _short(y, limit)
        result = _shown_result(
            x=x,
            test=test,
            effect_size=effect_size,
            bayes=bayes,
            n_obs=n_obs,
            settings=settings,
            design="correlation",
            name_limit=limit,
        )
        sample = (
            ""
            if points_shown is None or points_shown >= n_obs
            else f"; a random sample of {format_n(points_shown)} is drawn as points"
        )
        head = f"Scatter plot of {sy} against {sx} ({format_n(n_obs)} pairs{sample})"
        shown = settings.results_subtitle and test is not None
        direction = _correlation_direction(
            effect_size if shown else None, x=x, y=y, limit=limit
        )
        layers: list[str] = []
        if smooth:
            layers.append(f"a least-squares line with its {level} confidence band")
        if marginal:
            layers.append(f"marginal histograms of {sx} and {sy}")
        if labelled_points:
            where = (
                "every point"
                if labelled_points >= n_obs
                else f"{format_n(labelled_points)} of the {format_n(n_obs)} points"
            )
            source = "" if label_var is None else f" from {_short(label_var, limit)}"
            layers.append(f"labels{source} on {where}")
        also = f"It also shows {_join_words(layers)}." if layers else ""
        with_direction = f"{head}: {direction}." if direction else f"{head}."
        for start, middle in (
            (with_direction, also),
            (with_direction, ""),
            (f"{head}.", ""),
        ):
            text = " ".join(part for part in (start, middle, result) if part)
            if _fits(text):
                return text
    return f"{head}. {result}"


def _no_level_texts(
    *,
    x: str,
    y: str | None,
    descriptives: Sequence[Descriptive],
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    pairwise: Sequence[PairwiseComparison],
    n_obs: int,
    settings: Settings,
    design: str,
    values: Sequence[Sequence[float]],
    point_labels: Sequence[str],
    label_var: str | None,
) -> Texts:
    """``build_texts`` for designs "one-sample" (histostats) and "correlation"
    (scatterstats): ggstatsplot's subtitle, Bayes caption and (one sample) its
    centrality label; the APA sentence; the alt text of the default figure; no
    tick or pairwise labels."""
    if pairwise:
        raise ValueError(f"design {design!r} has no pairwise comparisons")
    one_sample = design == "one-sample"
    if one_sample and len(descriptives) != 1:
        raise ValueError(
            "a one-sample result has one descriptive (its column), got "
            f"{len(descriptives)}"
        )
    if not one_sample and (descriptives or y is None):
        raise ValueError("a correlation has no descriptives and needs y")
    ex = _expressions(
        descriptives=descriptives,
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=(),
        n_obs=n_obs,
        settings=settings,
        paired=_n_pairs(design),
    )
    lines = caption_lines(ex.bayes_caption, None, mathtext=False)
    if one_sample:
        alt = one_sample_alt_text(
            x=x,
            descriptive=descriptives[0],
            values=values[0] if values else (),
            test=test,
            effect_size=effect_size,
            bayes=bayes,
            n_obs=n_obs,
            settings=settings,
        )
    else:
        assert y is not None
        alt = correlation_alt_text(
            x=x,
            y=y,
            test=test,
            effect_size=effect_size,
            bayes=bayes,
            n_obs=n_obs,
            settings=settings,
            label_var=label_var,
            labelled_points=sum(1 for label in point_labels if label),
        )
    return Texts(
        subtitle=None if ex.subtitle is None else to_plain(ex.subtitle),
        caption="\n".join(lines) if lines else None,
        apa=_no_level_sentence(
            x=x,
            test=test,
            effect_size=effect_size,
            bayes=bayes,
            n_obs=n_obs,
            settings=settings,
            design=design,
        ),
        alt_text=alt,
        centrality_labels=tuple(
            "" if d.value is None else to_plain(node)
            for d, node in zip(descriptives, ex.centrality, strict=True)
        ),
        tick_labels=(),
        pairwise_labels=(),
    )


def build_texts(
    *,
    x: str,
    y: str | None,
    descriptives: Sequence[Descriptive],
    test: TestResult | None,
    effect_size: EffectSize | None,
    bayes: BayesFactor | None,
    pairwise: Sequence[PairwiseComparison],
    n_obs: int,
    settings: Settings,
    design: str = "between",
    values: Sequence[Sequence[float]] = (),
    point_labels: Sequence[str] = (),
    label_var: str | None = None,
) -> Texts:
    """Every plain text of a result, from the parts the analysis computed.

    ``design`` is the result's: "within" (withinstats) prints n as the number of
    complete subjects, ``n["pairs"]`` in the subtitle, as ggwithinstats does.
    The subtitle is None when ``settings.results_subtitle`` is false. The caption
    holds the Bayes line (when ``bayes`` is given, the subtitle is drawn and
    ``settings.bf_message`` is not False, as ggstatsplot's ``bf.message``) and the
    pairwise line (three or more groups, unless ``pairwise_display`` is "none"),
    joined by a newline, or None when there is neither. The user's own caption is
    not part of it.

    Designs "one-sample" (histostats) and "correlation" (scatterstats): the
    subtitle and Bayes caption as ggstatsplot prints them (n as ``n["obs"]`` and
    ``n["pairs"]``), one centrality label for histostats and none for
    scatterstats, no tick or pairwise labels, :func:`one_sample_sentence` or
    :func:`correlation_sentence`, and the alt text of the default figure
    (:func:`one_sample_alt_text`, :func:`correlation_alt_text`), which reads the
    analyzed ``values`` (the range of a histogram) and, for a correlation, the
    ``point_labels`` and their column ``label_var`` (the labelled points). ``y``
    is None only for "one-sample".
    """
    if not _grouped(design):
        return _no_level_texts(
            x=x,
            y=y,
            descriptives=descriptives,
            test=test,
            effect_size=effect_size,
            bayes=bayes,
            pairwise=pairwise,
            n_obs=n_obs,
            settings=settings,
            design=design,
            values=values,
            point_labels=point_labels,
            label_var=label_var,
        )
    if values or point_labels or label_var is not None:
        raise ValueError(
            f"values, point_labels and label_var describe the designs without "
            f"levels; design {design!r} reads its texts from its descriptives"
        )
    if y is None:
        raise ValueError(f"design {design!r} needs y, the measured column")
    ex = _expressions(
        descriptives=descriptives,
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        n_obs=n_obs,
        settings=settings,
        paired=_n_pairs(design),
    )
    lines = caption_lines(ex.bayes_caption, ex.pairwise_line, mathtext=False)
    return Texts(
        subtitle=None if ex.subtitle is None else to_plain(ex.subtitle),
        caption="\n".join(lines) if lines else None,
        apa=results_sentence(
            test,
            effect_size,
            bayes,
            pairwise,
            n_obs=n_obs,
            levels=[d.level for d in descriptives],
            design=design,
        ),
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
            design=design,
        ),
        # A level without a centrality value (a MAP without a density, D55)
        # draws no label: its text is empty.
        centrality_labels=tuple(
            "" if d.value is None else to_plain(node)
            for d, node in zip(descriptives, ex.centrality, strict=True)
        ),
        tick_labels=tuple(tick_label(d.level, d.n) for d in descriptives),
        pairwise_labels=tuple(to_plain(node) for node in ex.pairwise),
    )


@dataclass(frozen=True, slots=True)
class FigureTexts:
    """What the renderer draws, ready for matplotlib.

    ``subtitle`` is mathtext (``$...$``) or None; ``caption_lines`` holds the Bayes
    line as mathtext and the pairwise line as plain text (draw it with
    ``parse_math=False``); ``centrality_labels`` is mathtext per level (one for
    histostats: the centrality line's label; none for scatterstats);
    ``tick_labels`` is plain text per level (``parse_math=False``; none for
    histostats and scatterstats); ``pairwise_labels`` is mathtext per pair, in
    the order of ``result.pairwise``.

    barstats (design "contingency", :func:`contingency_figure_texts`) also fills
    ``bar_labels`` (mathtext per bar: its proportion test's p, drawn above the
    bar when ``settings.proportion_test``; "" for a bar without a test),
    ``n_labels`` (plain, under each bar), ``cell_labels`` (plain, one per entry
    of ``result.cells``, "" for a zero cell, which has no label; "both" holds a
    line break), ``legend_title`` (the name of x), ``x_title`` (the name of y;
    None without y) and ``y_tick_labels`` ("0%" to "100%" by 10); its
    ``tick_labels`` are the bars' levels of y ("" for the single bar without y).
    The other designs leave them empty.
    """

    subtitle: str | None
    caption_lines: tuple[str, ...]
    centrality_labels: tuple[str, ...]
    tick_labels: tuple[str, ...]
    pairwise_labels: tuple[str, ...]
    bar_labels: tuple[str, ...] = ()
    n_labels: tuple[str, ...] = ()
    cell_labels: tuple[str, ...] = ()
    legend_title: str | None = None
    x_title: str | None = None
    y_tick_labels: tuple[str, ...] = ()


def figure_texts(result: StatsResult) -> FigureTexts:
    """The renderer's texts for ``result``, rebuilt from its numbers and settings.

    histostats ("one-sample"): the subtitle, the Bayes caption (type parametric),
    one centrality label (the line's: ``widehat(mu)[mean] == "18.81"``, drawn
    at the centrality value unless ``centrality_plotting=False``) and no tick or
    pairwise labels. scatterstats ("correlation"): the subtitle, the Bayes
    caption, and no centrality, tick or pairwise labels. barstats
    ("contingency"): :func:`contingency_figure_texts`.
    """
    if result.design == "contingency":
        return contingency_figure_texts(result)
    grouped = _grouped(result.design)
    ex = _expressions(
        descriptives=result.descriptives,
        test=result.test,
        effect_size=result.effect_size,
        bayes=result.bayes,
        pairwise=result.pairwise,
        n_obs=result.n_obs,
        settings=result.settings,
        paired=_n_pairs(result.design),
    )
    return FigureTexts(
        subtitle=None if ex.subtitle is None else to_mathtext(ex.subtitle),
        caption_lines=caption_lines(ex.bayes_caption, ex.pairwise_line, mathtext=True),
        centrality_labels=tuple(to_mathtext(node) for node in ex.centrality),
        tick_labels=(
            tuple(tick_label(d.level, d.n) for d in result.descriptives)
            if grouped
            else ()
        ),
        pairwise_labels=tuple(to_mathtext(node) for node in ex.pairwise),
    )


# --------------------------------------------------------------------------- #
# barstats (design "contingency")
# --------------------------------------------------------------------------- #
#
# The subtitle, the p label above each bar and the pairs' labels are
# ggstatsplot's expressions (statsExpressions' template and ggbarstats'
# ``onesample_data()``); the labels inside and under the bars are ggbarstats'
# ``descriptive_data()`` and ``onesample_data()`` texts (ggstatsplot 1.1.4, MIT:
# ``R/ggpiestats-ggbarstats-helpers.R``, ``R/ggbarstats.R``); the APA sentence and
# the alt text are inferplot's own, in the style of the other designs (SPEC
# section 7). ggstatsplot's Bayes caption arrives in inferplot 0.2: there is no
# caption in 0.1 (the analysis says so in ``warnings``).

# The chi-square symbol and the effect size of each barstats design.
# statsExpressions picks the symbol from the test's method string, which the
# design fixes: "Pearson's Chi-squared test", "McNemar's Chi-squared test",
# "Chi-squared test for given probabilities".
_CONTINGENCY_STATISTICS: Final[dict[str, str]] = {
    "two-way": "pearson_chi2",
    "paired": "mcnemar_chi2",
    "one-way": "gof_chi2",
}
_CONTINGENCY_EFFECT_SIZES: Final[dict[str, str]] = {
    "two-way": "cramers_v",
    "paired": "cohens_g",
    "one-way": "pearsons_c",
}
# Cramér's V and Pearson's C lie in [0, 1] and Cohen's g in [-0.5, 0.5]: APA
# style drops the leading zero of all three.
_CONTINGENCY_BOUNDED: Final = frozenset({"cramers_v", "pearsons_c", "cohens_g"})
# What the main sentence says when results_subtitle=False computed no test.
_CONTINGENCY_NO_TEST: Final[dict[str, str]] = {
    "two-way": "No test of independence was computed",
    "paired": "No McNemar's test was computed",
    "one-way": "No goodness-of-fit test was computed",
}
# The y axis of a barstats figure: insight::format_percent(digits = 0) of the
# breaks 0, 0.1, ..., 1.
PERCENT_TICKS: Final[tuple[str, ...]] = tuple(f"{10 * i}%" for i in range(11))
# What the labels inside the bars show, by settings.label.
_CELL_LABEL_WORDS: Final[dict[str, str]] = {
    "percentage": "percentages",
    "counts": "counts",
    "both": "counts and percentages",
}


def _contingency_design(y: str | None, settings: Settings) -> str:
    """ "two-way", "paired" or "one-way" (``y`` None)."""
    if y is None:
        if settings.paired:
            raise ValueError("a paired barstats result needs y, the second measurement")
        return "one-way"
    return "paired" if settings.paired else "two-way"


def contingency_subtitle_expression(
    test: TestResult,
    effect_size: EffectSize,
    *,
    design: str,
    n_obs: int,
    digits: int,
) -> Call:
    """ggbarstats' subtitle (statsExpressions' ``contingency_table()``).

    ``list(chi["Pearson"]^2 * "(" * 2 * ")" == "8.74", italic(p) == "0.01",
    widehat(italic("V"))["Cramer"] == "0.46", CI["95%"] ~ "[" * "0.00", "0.82" *
    "]", italic("n")["obs"] == "32")``. ``design`` "two-way" prints Pearson's
    chi^2 and Cramér's V, "paired" McNemar's chi^2, Cohen's g and n as
    ``italic("n")["pairs"]``, "one-way" the goodness-of-fit chi^2
    (``chi["gof"]``) and Pearson's C. The df is an integer (``digits.df`` 0);
    an exact p of 0 prints 2.23e-308.
    """
    try:
        statistic = _CONTINGENCY_STATISTICS[design]
    except KeyError:
        known = ", ".join(repr(d) for d in _CONTINGENCY_STATISTICS)
        raise ValueError(
            f"unknown barstats design {design!r}; known: {known}"
        ) from None
    expected = _CONTINGENCY_EFFECT_SIZES[design]
    if effect_size.name != expected:
        raise ValueError(
            f"the {design} design reports {expected!r}, got the effect size "
            f"{effect_size.name!r}"
        )
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
        digits_df=0,
        paired=design == "paired",
    )


def bar_p_expression(p_value: float | None, digits: int = 2) -> Call:
    """The label above a bar: ``list(~italic(p)=='0.09')`` (ggbarstats'
    ``.p.label``, ``format_value(p.value, digits)``), its proportion test's p
    with no adjustment and no stars. The p comes from statsExpressions'
    ``contingency_table()``, which stores an exact 0 as ``.Machine$double.xmin``
    (2.23e-308); a missing p prints as "" (R's ``format_value(NA)``)."""
    if p_value is None or math.isnan(p_value):
        text = format_value(None, digits, missing="")
    else:
        text = format_p_subtitle(p_value, digits)
    return _call("list", Call(_UNARY_TILDE, (_rel(_italic(Sym("p")), Str(text)),)))


def n_label(n: int) -> str:
    """The label under a bar, ``"(n = 1,490)"`` (``.prettyNum``: a comma from
    1,000 on)."""
    return f"(n = {format_n(n)})"


def percent_text(percent: float, digits_perc: int) -> str:
    """A cell's percentage as ggbarstats prints it, without the "%":
    ``round(perc, digits.perc)`` pasted as R prints a double ("62" for 62.5 at 0
    digits, "6.2" for 6.25 at 1, "33.33" for 100 / 3 at 2, "50" for 50).

    R's ``round()`` agrees with Python's on the same double (the nearest
    representable result; an exact binary tie goes to even), on all 927 percent
    labels of the oracle probe ``probe-contingency``.
    """
    if digits_perc < 0:
        raise ValueError(f"digits_perc must be >= 0, got {digits_perc}")
    return r_number_text(round(percent, digits_perc))


def cell_label(count: int, percent: float, *, label: str, digits_perc: int) -> str:
    """The label inside a cell (ggbarstats' ``descriptive_data()``): "16%"
    (``label`` "percentage"), "1,490" ("counts"), "3\\n(16%)" ("both"); "" for
    a zero cell, which ggbarstats leaves without a label. ``percent`` is the
    cell's share of its bar, (count / bar n) * 100 (:class:`~inferplot.result.Cell`)."""
    if count == 0:
        return ""
    if label == "percentage":
        return f"{percent_text(percent, digits_perc)}%"
    if label == "counts":
        return format_n(count)
    if label == "both":
        return f"{format_n(count)}\n({percent_text(percent, digits_perc)}%)"
    raise ValueError(f"unknown cell label {label!r}; known: percentage, counts, both")


def _cell_labels(cells: Sequence[Cell], settings: Settings) -> tuple[str, ...]:
    if settings.label is None or settings.digits_perc is None:
        raise ValueError("a barstats result sets settings.label and digits_perc")
    return tuple(
        cell_label(
            c.count, c.percent, label=settings.label, digits_perc=settings.digits_perc
        )
        for c in cells
    )


def _bar_expressions(bars: Sequence[Bar], digits: int) -> tuple[Call | None, ...]:
    return tuple(
        None if bar.test is None else bar_p_expression(bar.test.p_value, digits)
        for bar in bars
    )


def _and_join(items: Sequence[str]) -> str:
    """ "a", "a and b", "a, b and c"."""
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _share_text(share: float, digits: int) -> tuple[str, bool]:
    """One expected share rounded to ``digits`` decimals (more where that would
    print 0), and whether rounding changed it."""
    places = digits
    while round(share, places) == 0.0 and places < 15:
        places += 1
    rounded = round(share, places)
    return r_number_text(rounded), abs(rounded - share) > 1e-12


def _bars_shares_text(settings: Settings, empty: Sequence[str], k: int) -> str:
    """What the bars' tests compare against (:func:`_shares_text`), naming the
    levels of x without observations that they count as categories (review
    B18): "equal shares of 3 categories, 'waitlist' among them with no
    observations"."""
    shares = _shares_text(settings)
    if not empty:
        return shares
    names = _and_join([repr(level) for level in empty])
    return f"{shares} of {k} categories ({names} with no observations)"


def _shares_text(settings: Settings) -> str:
    """What the goodness-of-fit tests compare against: "equal shares" or
    "expected shares 0.5, 0.25 and 0.25" (``ratio``, in the order of the
    levels of x).

    Equal shares read "equal shares" however they were written (``ratio=[1/3,
    1/3, 1/3]``); other shares are rounded to ``settings.digits`` decimals (more
    where that would print 0), and "of about" marks a rounding that changed one
    ("expected shares of about 0.17, 0.17, 0.17 and 0.5")."""
    ratio = settings.ratio
    if ratio is None or all(abs(r * len(ratio) - 1.0) <= 1e-9 for r in ratio):
        return "equal shares"
    shown = [_share_text(r, settings.digits) for r in ratio]
    about = "of about " if any(changed for _, changed in shown) else ""
    return f"expected shares {about}" + _and_join([text for text, _ in shown])


def _contingency_effect_text(effect_size: EffectSize) -> str:
    """ "Cramer's V = .46, 95% CI [.00, .82]"; a one-sided interval says so:
    "95% one-sided CI [.00, 1.00]"."""
    bounded = effect_size.name in _CONTINGENCY_BOUNDED
    estimate = _apa_number(effect_size.estimate, bounded=bounded)
    low = _apa_number(effect_size.conf_low, bounded=bounded)
    high = _apa_number(effect_size.conf_high, bounded=bounded)
    level = format_conf_level(effect_size.conf_level)
    sided = "" if effect_size.sidedness == "two-sided" else "one-sided "
    return f"{effect_size.label} = {estimate}, {level} {sided}CI [{low}, {high}]"


def contingency_sentence(
    test: TestResult | None,
    effect_size: EffectSize | None,
    *,
    x: str,
    y: str | None,
    n_obs: int,
    settings: Settings,
    name_limit: int = APA_NAME_LIMIT,
    compact: bool = False,
) -> str:
    """The APA 7 sentence of a barstats result's main test, ASCII except the
    column names.

    Two-way: "Pearson's chi-square test (shares of cyl across the levels of am):
    chi2(2, N = 32) = 8.74, p = .013, Cramer's V = .46, 95% CI [.00, .82]."
    Paired: "McNemar's test (before and after, the same units measured twice):
    chi2(1, N = 40) = 4.00, p = .046, Cohen's g = .25, 95% CI [.01, .40]." (N
    counts the pairs). One-way: "Chi-square goodness-of-fit test (shares of cyl
    against equal shares): chi2(2, N = 32) = 2.31, p = .315, Pearson's C = .26,
    95% CI [.00, .50]." ("against expected shares 0.5, 0.25 and 0.25" with
    ``ratio``). The chi-square has no direction, so a one-sided ``alternative``
    shows only in the interval, named "one-sided CI". Cramér's V, Pearson's C
    and Cohen's g drop the leading zero. Column names are shortened to
    ``name_limit`` characters with an ellipsis. Without a test
    (``results_subtitle=False``): "No test of independence was computed
    (results_subtitle=False), n = 32." ("No McNemar's test ..., n = 40 pairs.",
    "No goodness-of-fit test ...").

    ``compact`` (the alt text, whose first sentence already says what the bars
    compare) leaves out the parenthesis of the two-way and paired designs and
    keeps "(against equal shares)" of the one-way design.
    """
    design = _contingency_design(y, settings)
    if test is None or effect_size is None:
        pairs = " pairs" if design == "paired" else ""
        return (
            f"{_CONTINGENCY_NO_TEST[design]} (results_subtitle=False), "
            f"n = {format_n(n_obs)}{pairs}."
        )
    sx = _short(x, name_limit)
    if compact:
        what = f"against {_shares_text(settings)}" if design == "one-way" else ""
    elif design == "two-way":
        assert y is not None
        what = f"shares of {sx} across the levels of {_short(y, name_limit)}"
    elif design == "paired":
        assert y is not None
        what = f"{sx} and {_short(y, name_limit)}, the same units measured twice"
    else:
        what = f"shares of {sx} against {_shares_text(settings)}"
    df = test.df_error if test.df_error is not None else test.df
    dfs = "" if df is None else f"{_apa_df(df)}, "
    statistic = (
        f"{test.statistic_name}({dfs}N = {format_n(n_obs)}) = "
        f"{_apa_number(test.statistic)}, {_apa_p(test.p_value)}"
    )
    return (
        f"{test.label}{f' ({what})' if what else ''}: {statistic}, "
        f"{_contingency_effect_text(effect_size)}."
    )


def contingency_tests_sentence(
    bars: Sequence[Bar],
    pairwise: Sequence[PairwiseComparison],
    *,
    x: str,
    y: str | None,
    settings: Settings,
    name_limit: int = APA_NAME_LIMIT,
    empty: Sequence[str] = (),
    k: int = 0,
) -> str | None:
    """One sentence on the bars' proportion tests and the pairwise tests, or
    None where there are neither (no y): how many are below .05, the method
    named, as the other designs' alt text counts its pairs.

    "Goodness-of-fit tests within each level of am (shares of cyl against equal
    shares, unadjusted): p < .05 in 1 of 2 bars; Fisher's exact tests between
    the levels of cyl (Holm adjusted): adjusted p < .05 in 1 of 3 pairs." The
    bars' tests are never adjusted (ggstatsplot's ``onesample_data()``); the
    pairs follow ``p_adjust_method`` ("unadjusted: p < .05 in ..." for "none").
    Only the pairs with a test are counted; the others are named ("in 2 of the
    2 tested pairs; 1 pair has no test", or "none of the 3 pairs has a test").
    """
    parts: list[str] = []
    tested = [bar for bar in bars if bar.test is not None]
    sx = _short(x, name_limit)
    if tested and y is not None:
        below = sum(
            1 for bar in tested if bar.test is not None and bar.test.p_value < 0.05
        )
        parts.append(
            f"Goodness-of-fit tests within each level of {_short(y, name_limit)} "
            f"(shares of {sx} against {_bars_shares_text(settings, empty, k)}, "
            "unadjusted): "
            f"p < .05 in {below} of {_plural(len(tested), 'bar')}"
        )
    if pairwise:
        method = settings.p_adjust_method
        if method == "none":
            how, p = "unadjusted", "p"
        else:
            how, p = f"{_ADJUST_NAMES.get(method, method)} adjusted", "adjusted p"
        tested = [pair for pair in pairwise if pair.p_value is not None]
        untested = len(pairwise) - len(tested)
        below = sum(1 for pair in tested if pair.significant)
        head = f"Fisher's exact tests between the levels of {sx}"
        if not tested:
            parts.append(f"{head}: none of the {_plural(untested, 'pair')} has a test")
        elif not untested:
            parts.append(
                f"{head} ({how}): {p} < .05 in {below} of "
                f"{_plural(len(tested), 'pair')}"
            )
        else:
            verb = "has" if untested == 1 else "have"
            parts.append(
                f"{head} ({how}): {p} < .05 in {below} of the "
                f"{_plural(len(tested), 'tested pair')}; "
                f"{_plural(untested, 'pair')} {verb} no test"
            )
    if not parts:
        return None
    return "; ".join(parts) + "."


def contingency_apa(
    test: TestResult | None,
    effect_size: EffectSize | None,
    bars: Sequence[Bar],
    pairwise: Sequence[PairwiseComparison],
    *,
    x: str,
    y: str | None,
    n_obs: int,
    settings: Settings,
    empty: Sequence[str] = (),
    k: int = 0,
) -> str:
    """``result.text.apa`` of barstats: :func:`contingency_sentence`, then
    :func:`contingency_tests_sentence` where there are bar or pair tests
    (``empty``: the levels of x without observations, of ``k``)."""
    main = contingency_sentence(
        test, effect_size, x=x, y=y, n_obs=n_obs, settings=settings
    )
    tests = contingency_tests_sentence(
        bars, pairwise, x=x, y=y, settings=settings, empty=empty, k=k
    )
    return main if tests is None else f"{main} {tests}"


def _bar_compositions(
    bars: Sequence[Bar],
    cells: Sequence[Cell],
    *,
    digits_perc: int,
    short: dict[str, str],
    short_bars: dict[str, str],
    largest: bool,
) -> list[str]:
    """Each bar's shares, in level order, zero cells left out: "0 (n = 19): 4
    16%, 6 21%, 8 63%", or with ``largest`` its largest share only: "0 (n =
    19): largest 8 63%" ("largest a and b 50% each" for a tie)."""
    out: list[str] = []
    for bar in bars:
        own = [c for c in cells if c.bar == bar.level and c.count > 0]
        name = "" if bar.level is None else f"{short_bars[bar.level]} "
        head = f"{name}(n = {format_n(bar.n)})"
        if largest:
            top = max(c.count for c in own)
            tops = [c for c in own if c.count == top]
            share = f"{percent_text(tops[0].percent, digits_perc)}%"
            names = _and_join([short[c.level] for c in tops])
            each = " each" if len(tops) > 1 else ""
            out.append(f"{head}: largest {names} {share}{each}")
        else:
            shares = ", ".join(
                f"{short[c.level]} {percent_text(c.percent, digits_perc)}%" for c in own
            )
            out.append(f"{head}: {shares}")
    return out


def _second_sentence(*parts: str) -> str:
    """The parts that are there, joined by semicolons, as one sentence."""
    kept = [part for part in parts if part]
    if not kept:
        return ""
    text = "; ".join(kept)
    return text[0].upper() + text[1:] + "."


def contingency_alt_text(
    *,
    x: str,
    y: str | None,
    levels: Sequence[str],
    test: TestResult | None,
    effect_size: EffectSize | None,
    bars: Sequence[Bar],
    cells: Sequence[Cell],
    n_obs: int,
    settings: Settings,
    cell_labels: bool = True,
    bar_tests: bool | None = None,
    legend: bool = True,
) -> str:
    """Alt text of a barstats figure: 2 or 3 sentences, at most 400 characters.

    "Stacked bar chart of cyl within each level of am, each bar summing to 100%:
    0 (n = 19): 4 16%, 6 21%, 8 63%; 1 (n = 13): 4 62%, 6 23%, 8 15%. Labels
    show percentages; the p-value above each bar tests its shares against equal
    shares (0: p = 0.02, 1: p = 0.09); a legend names the 3 levels of cyl.
    Pearson's chi-square test: chi2(2, N = 32) = 8.74, p = .013, Cramer's V =
    .46, 95% CI [.00, .82]."

    The head names x and y (paired: "... within each level of after (40 units
    measured twice), ..."; one-way: "Stacked bar chart of the shares of cyl in
    one bar (n = 32): ...") and each bar's shares as its height shows them
    (rounded to ``digits_perc``, as the labels; zero cells left out). The second
    sentence names what else is drawn: the labels inside the bars
    (``cell_labels``: percentages, counts or both, as ``settings.label``), the
    p-value above each bar (``bar_tests``, default ``settings.proportion_test``;
    as the label prints it) and the legend (``legend``). The result sentence is
    :func:`contingency_sentence` with ``compact=True`` (the head says what is
    compared; the figure draws no pairs, and the bars' tests are named with
    their labels). Every keyword describes what the figure draws; the defaults
    are ``ip.barstats``' default figure.

    Shortening, until the text fits, at each length of the names of the columns
    and levels (40, 20, then 12 characters): the legend clause goes, then the
    list of the bars' p-values; then each bar keeps only its largest share; then
    the second sentence goes (all shares back); then only the largest shares;
    then each bar keeps only its n (with the short second sentence, then
    without). Last, only the head and the result. The result sentence is never
    cut.
    """
    design = _contingency_design(y, settings)
    if bar_tests is None:
        bar_tests = bool(settings.proportion_test) and design != "one-way"
    elif bar_tests and design == "one-way":
        raise ValueError(
            "bar_tests=True: without y the single bar has no test of its own (its "
            "goodness-of-fit test is the main test)"
        )
    if settings.label not in _CELL_LABEL_WORDS or settings.digits_perc is None:
        raise ValueError("a barstats result sets settings.label and digits_perc")
    digits_perc = settings.digits_perc
    bar_names = [bar.level for bar in bars if bar.level is not None]
    plain_p = [
        "" if node is None else to_plain(node)
        for node in _bar_expressions(bars, settings.digits)
    ]
    result = ""
    head = ""
    for limit in (40, 20, 12):
        sx = _short(x, limit)
        short = _shortened(list(levels), limit)
        short_bars = _shortened(bar_names, limit)
        result = contingency_sentence(
            test,
            effect_size,
            x=x,
            y=y,
            n_obs=n_obs,
            settings=settings,
            name_limit=limit,
            compact=True,
        )
        if design == "one-way":
            head = f"Stacked bar chart of the shares of {sx} in one bar"
        else:
            assert y is not None
            sy = _short(y, limit)
            same = (
                f" ({format_n(n_obs)} units measured twice)"
                if design == "paired"
                else ""
            )
            head = (
                f"Stacked bar chart of {sx} within each level of {sy}{same}, each bar "
                "summing to 100%"
            )
        joiner = " " if design == "one-way" else ": "
        heads = [
            head + joiner + "; ".join(compositions)
            for compositions in (
                _bar_compositions(
                    bars,
                    cells,
                    digits_perc=digits_perc,
                    short=short,
                    short_bars=short_bars,
                    largest=largest,
                )
                for largest in (False, True)
            )
        ]
        sizes = ", ".join(
            n_label(bar.n)
            if bar.level is None
            else f"{short_bars[bar.level]} {n_label(bar.n)}"
            for bar in bars
        )
        heads.append(head + joiner + sizes)
        cell_part = (
            f"labels show {_CELL_LABEL_WORDS[settings.label]}" if cell_labels else ""
        )
        p_items = ", ".join(
            f"{short_bars[bar.level]}: {text}"
            for bar, text in zip(bars, plain_p, strict=True)
            if text and bar.level is not None
        )
        tests_part = (
            "the p-value above each bar tests its shares against "
            f"{_bars_shares_text(settings, _empty_levels(levels, cells), len(levels))}"
            if bar_tests
            else ""
        )
        legend_part = (
            f"a legend names the {len(levels)} levels of {sx}" if legend else ""
        )
        listed = f"{tests_part} ({p_items})" if tests_part and p_items else tests_part
        full, largest, sizes_only = heads
        every = _second_sentence(cell_part, listed, legend_part)
        no_legend = _second_sentence(cell_part, listed)
        brief = _second_sentence(cell_part, tests_part)
        candidates = [
            (full, every),
            (full, no_legend),
            (full, brief),
            (largest, brief),
            (full, ""),
            (largest, ""),
            (sizes_only, brief),
            (sizes_only, ""),
        ]
        for start, middle in candidates:
            text = " ".join(part for part in (f"{start}.", middle, result) if part)
            if _fits(text):
                return text
    # Still too long: the bars and the second sentence go, the result stays whole.
    return f"{head}. {result}"


def _empty_levels(levels: Sequence[str], cells: Sequence[Cell]) -> list[str]:
    """The levels of x without observations (every cell of theirs 0)."""
    held = {cell.level for cell in cells if cell.count > 0}
    return [level for level in levels if level not in held]


def contingency_texts(
    *,
    x: str,
    y: str | None,
    levels: Sequence[str],
    y_levels: Sequence[str],
    test: TestResult | None,
    effect_size: EffectSize | None,
    bars: Sequence[Bar],
    cells: Sequence[Cell],
    pairwise: Sequence[PairwiseComparison],
    n_obs: int,
    settings: Settings,
) -> Texts:
    """Every plain text of a barstats result (design "contingency").

    ``subtitle``: ggbarstats' (:func:`contingency_subtitle_expression`), None
    with ``results_subtitle=False``; ``caption`` None (ggstatsplot's Bayes
    caption arrives in inferplot 0.2); ``apa``: :func:`contingency_apa`;
    ``alt_text``: :func:`contingency_alt_text` of the default figure;
    ``tick_labels``: the bars' levels of y ("" for the single bar without y);
    ``pairwise_labels``: each Fisher pair's adjusted p as ggstatsplot labels it
    (``p_Holm-adj. = 0.56``), never drawn; ``bar_labels``: each bar's proportion
    test, "p = 0.09" ("" for a bar without a test: the single bar without y),
    drawn above the bar when ``settings.proportion_test``; ``n_labels``: "(n =
    13)" under each bar; ``cell_labels``: one per entry of ``cells``,
    :func:`cell_label` ("" for a zero cell). No centrality labels.
    """
    del y_levels  # the bars name the levels of y that are drawn
    design = _contingency_design(y, settings)
    digits = settings.digits
    subtitle = None
    if settings.results_subtitle and test is not None and effect_size is not None:
        subtitle = to_plain(
            contingency_subtitle_expression(
                test, effect_size, design=design, n_obs=n_obs, digits=digits
            )
        )
    return Texts(
        subtitle=subtitle,
        caption=None,
        apa=contingency_apa(
            test,
            effect_size,
            bars,
            pairwise,
            x=x,
            y=y,
            n_obs=n_obs,
            settings=settings,
            empty=_empty_levels(levels, cells),
            k=len(levels),
        ),
        alt_text=contingency_alt_text(
            x=x,
            y=y,
            levels=levels,
            test=test,
            effect_size=effect_size,
            bars=bars,
            cells=cells,
            n_obs=n_obs,
            settings=settings,
        ),
        centrality_labels=(),
        tick_labels=tuple("" if b.level is None else drawable(b.level) for b in bars),
        pairwise_labels=tuple(to_plain(_pair_label(p, digits)) for p in pairwise),
        bar_labels=tuple(
            "" if node is None else to_plain(node)
            for node in _bar_expressions(bars, digits)
        ),
        n_labels=tuple(n_label(bar.n) for bar in bars),
        cell_labels=_cell_labels(cells, settings),
    )


def contingency_figure_texts(result: StatsResult) -> FigureTexts:
    """The renderer's texts for a barstats result (see :class:`FigureTexts`):
    the subtitle as mathtext, no caption lines, the bars' levels as tick labels,
    each bar's p label as mathtext (drawn when ``settings.proportion_test``),
    the n and cell labels as plain text, the legend title (the name of x), the
    x axis title (the name of y; None without y) and the percent ticks."""
    settings = result.settings
    design = _contingency_design(result.y, settings)
    subtitle = None
    if (
        settings.results_subtitle
        and result.test is not None
        and result.effect_size is not None
    ):
        subtitle = to_mathtext(
            contingency_subtitle_expression(
                result.test,
                result.effect_size,
                design=design,
                n_obs=result.n_obs,
                digits=settings.digits,
            )
        )
    return FigureTexts(
        subtitle=subtitle,
        caption_lines=(),
        centrality_labels=(),
        tick_labels=tuple(
            "" if b.level is None else drawable(b.level) for b in result.bars
        ),
        pairwise_labels=tuple(
            to_mathtext(_pair_label(p, settings.digits)) for p in result.pairwise
        ),
        bar_labels=tuple(
            "" if node is None else to_mathtext(node)
            for node in _bar_expressions(result.bars, settings.digits)
        ),
        n_labels=tuple(n_label(bar.n) for bar in result.bars),
        cell_labels=_cell_labels(result.cells, settings),
        legend_title=result.x,
        x_title=result.y,
        y_tick_labels=PERCENT_TICKS,
    )
