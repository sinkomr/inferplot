"""inferplot agent evaluation harness (SPEC section 13; read evals/README.md first).

    uv run python evals/run.py                  # dry run (the default)
    uv run python evals/run.py --tasks sleep-two-drugs,diets-three-groups

The dry run makes no model call and needs no network. It validates every task in
evals/tasks/ (schema; dataset present, readable and equal to its stated source;
every expected number equal to the oracle fixture it cites; every number
reproduced by ``ip.analyze.<name>`` or by its stated SciPy recipe within the
task's tolerance; the expected ``check()`` codes), builds the two arms' temporary
project directories, and prints the exact commands a real run would execute with
an estimate of the number of model calls.

A real run spends the owner's Claude usage. It needs the flag
``--i-understand-this-spends-usage``, is refused in CI, and runs ``claude -p``
once per task, arm and repeat in a fresh temporary directory with a time limit,
saves each transcript and the files the agent produced, and scores them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
TASKS_DIR = EVALS_DIR / "tasks"
REPO_ROOT = EVALS_DIR.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "oracle"
SKILL_DIR = REPO_ROOT / "skills" / "inferplot"

SPEND_FLAG = "--i-understand-this-spends-usage"
FUNCTIONS = ("betweenstats", "withinstats", "histostats", "scatterstats", "barstats")
ARMS = ("baseline", "inferplot")
# Both arms get the same scientific Python stack; the inferplot arm adds inferplot
# and its skill, so the package and the skill are the only difference.
BASE_PACKAGES = ("pandas", "numpy", "scipy", "matplotlib", "seaborn", "statsmodels")
DEFAULT_MAX_TURNS = 40
DEFAULT_TIME_LIMIT = 900  # seconds per run
ALLOWED_TOOLS = "Bash,Read,Write,Edit,Glob,Grep"

# --- numbers and tolerances --------------------------------------------------------

CORE = ("statistic", "df", "df_error", "p_value", "n")
EFFECT = ("estimate", "conf_low", "conf_high")
NUMBERS = (*CORE, *EFFECT, "bf10")
# The default tolerance per kind of number, for a number an agent reports (usually
# rounded to 2 to 4 digits). A number counts as right when
# |reported - expected| <= max(abs, rel * |expected|). Tasks may tighten or widen one
# number with `tol_abs` / `tol_rel`; the dry run applies the same tolerance to
# inferplot's own result. README.md explains each choice.
TOLERANCES: Mapping[str, tuple[float, float]] = {
    "statistic": (0.01, 0.005),
    "df": (0.02, 0.0),
    "df_error": (0.02, 0.0),
    "p_value": (0.001, 0.05),
    "n": (0.0, 0.0),
    "estimate": (0.01, 0.0),
    "conf_low": (0.02, 0.0),
    "conf_high": (0.02, 0.0),
    "bf10": (0.0, 0.05),
}
# A p-value below this may be reported as an inequality ("< .001").
P_FLOOR = 0.001
# Fixture values are copied into the tasks; they must equal the fixture to this.
FIXTURE_REL = 1e-12
# Recipe values are recomputed by the dry run; they must equal the task to this.
RECIPE_REL = 1e-9

# --- test families: how a reported test name is recognized -------------------------


@dataclass(frozen=True)
class Family:
    """A kind of test, recognized in a free-text test name.

    ``groups``: every group needs one of its phrases; ``none``: no phrase may occur.
    A phrase matches at the start of a word of the normalized name
    (``normalize_name``); a phrase ending in ``$`` must also end at a word end
    ("t$" is the word t, not the start of "test").
    """

    label: str
    groups: tuple[tuple[str, ...], ...]
    none: tuple[str, ...] = ()


_WELCH = (
    "welch",
    "unequal var",
    "not assuming equal var",
    "equal var false",
    "var equal false",
    "heteroscedastic",
    "separate var",
    "satterthwaite",
)
_ANOVA = ("anova", "analysis of variance", "analysis of means", "f test", "f$")
_WITHIN = (
    "repeated",
    "rm$",
    "rmanova",
    "rm anova",
    "within subject",
    "within$",
    "anovarm",
    "anova rm",
    "greenhouse",
    "huynh",
    "sphericity",
    "mauchly",
)

FAMILIES: Mapping[str, Family] = {
    "welch_t": Family(
        "Welch's t-test",
        (_WELCH,),
        (*_ANOVA, "analysis of", "one way", "oneway", "paired", "groups"),
    ),
    "student_t": Family(
        "Student's t-test (equal variances)",
        (
            (
                "student",
                "pooled",
                "equal var true",
                "var equal true",
                "equal variances assumed",
                "assuming equal var",
                "equal variance t",
                "classic t",
                "classical t",
            ),
        ),
        (
            "welch",
            "paired",
            "one sample",
            "anova",
            "unequal",
            "not assuming",
            "not assumed",
            "equal var false",
            "var equal false",
            "related",
            "dependent",
            "repeated",
        ),
    ),
    "welch_anova": Family(
        "Welch's ANOVA",
        (_WELCH, (*_ANOVA, "one way", "oneway")),
        ("trimmed", "repeated", "two way", "factorial"),
    ),
    "oneway_anova": Family(
        "one-way ANOVA (equal variances)",
        (("anova", "f oneway", "analysis of variance", "analysis of means", "f test"),),
        (
            *_WELCH,
            *_WITHIN,
            "not assuming",
            "friedman",
            "kruskal",
            "two way",
            "twoway",
            "2 way",
            "factorial",
            "interaction",
            "trimmed",
            "t1way",
            "robust",
            "brown",
            "mixed",
            "ancova",
            "rank",
            "bayes",
        ),
    ),
    "kruskal_wallis": Family("Kruskal-Wallis test", (("kruskal",),)),
    "mann_whitney": Family(
        "Mann-Whitney U / Wilcoxon rank-sum test",
        (
            (
                "mann",
                "whitney",
                "rank sum",
                "ranksum",
                "u test",
                "wilcoxon two sample",
                "two sample wilcoxon",
            ),
        ),
        ("signed", "friedman", "kruskal"),
    ),
    "paired_t": Family(
        "paired t-test",
        (
            (
                "paired",
                "dependent",
                "related",
                "ttest rel",
                "repeated",
                "matched",
                "within",
            ),
            ("t test", "ttest", "t statistic", "t$", "student"),
        ),
        (
            "wilcoxon",
            "yuen",
            "trimmed",
            "welch",
            "independent",
            "unpaired",
            "rank",
            "sign$",
            "bootstrap",
        ),
    ),
    "wilcoxon_signed_rank": Family(
        "Wilcoxon signed-rank test",
        (("signed rank", "signedrank", "wilcoxon"),),
        ("rank sum", "ranksum", "mann", "two sample"),
    ),
    "sign_test": Family(
        "sign test",
        (("sign test", "sign$", "signs$", "binomial"),),
        ("signed", "rank", "mcnemar"),
    ),
    "rm_anova": Family(
        "repeated-measures ANOVA",
        (
            _WITHIN,
            (
                "anova",
                "anovarm",
                "rmanova",
                "f test",
                "analysis of variance",
                "greenhouse",
                "huynh",
            ),
        ),
        ("friedman", "trimmed", "robust", "kruskal", "rank", "mixed", "bayes"),
    ),
    "friedman": Family("Friedman test", (("friedman",),)),
    "one_sample_t": Family(
        "one-sample t-test",
        (
            (
                "one sample",
                "single sample",
                "1samp",
                "1 sample",
                "onesample",
                "one group",
            ),
            ("t test", "ttest", "t$", "1samp", "student"),
        ),
        ("wilcoxon", "signed", "bootstrap", "trimmed", "bayes", "sign$"),
    ),
    "pearson": Family(
        "Pearson correlation",
        (("pearson", "product moment"),),
        ("chi", "spearman", "winsor", "kendall"),
    ),
    "spearman": Family("Spearman correlation", (("spearman",),)),
    "kendall": Family(
        "Kendall's tau", (("kendall",),), ("kendall w$", "concordance", "spearman")
    ),
    "winsorized": Family("winsorized correlation", (("winsor",),)),
    "yuen": Family("Yuen's trimmed-means test", (("yuen",),)),
    "trimmed_anova": Family(
        "trimmed-means ANOVA",
        (("trimmed", "t1way"), ("anova", "t1way", "f test", "analysis of")),
        ("bootstrap", "repeated"),
    ),
    "trimmed_bootstrap": Family(
        "bootstrap-t test of the trimmed mean", (("trimmed",), ("bootstrap",))
    ),
    "mcnemar": Family("McNemar's test", (("mcnemar",),)),
    "chi_square_independence": Family(
        "chi-square test of independence",
        (("chi", "contingency", "independence"),),
        (
            "goodness",
            "given prob",
            "gof",
            "mcnemar",
            "one way",
            "one sample",
            "stats chisquare$",
            "fisher",
            "friedman",
            "kruskal",
        ),
    ),
    "chi_square_gof": Family(
        "chi-square goodness-of-fit test",
        (("chi", "goodness"),),
        (
            "contingency",
            "independence",
            "mcnemar",
            "association",
            "fisher",
            "friedman",
            "kruskal",
        ),
    ),
    "fisher": Family(
        "Fisher's exact test",
        (("fisher",),),
        ("fisher z$", "pearson", "correlation", "chi"),
    ),
    "regression": Family(
        "linear regression with covariates",
        (
            (
                "regression",
                "ols",
                "linear model",
                "lm$",
                "multiple",
                "least squares",
                "ancova",
                "glm$",
            ),
        ),
        ("comparison", "logistic"),
    ),
    "two_way_anova": Family(
        "two-way (factorial) ANOVA",
        (
            (
                "two way",
                "twoway",
                "2 way",
                "factorial",
                "interaction",
                "two factor",
                "2x3",
                "2 x 3",
            ),
        ),
        ("repeated",),
    ),
    "bayes_factor": Family("a Bayes factor", (("bayes", "bf10", "bf 10", "jzs"),)),
}
# Names that only one family can mean, matched as whole normalized names: the
# `result.test.method` strings inferplot copies from ggstatsplot, which the word
# rules would read otherwise ("ANOVA estimation for factorial designs" is
# ggstatsplot's name for its repeated-measures ANOVA, not a factorial ANOVA).
EXACT_NAMES: Mapping[str, str] = {
    "anova estimation for factorial designs using afex": "rm_anova",
    "bootstrap t method for one sample test": "trimmed_bootstrap",
    "a heteroscedastic one way anova for trimmed means": "trimmed_anova",
    "one way analysis of means not assuming equal variances": "welch_anova",
}
# A reported test that says no test was run.
NO_TEST = "no_test"
NO_TEST_WORDS = frozenset(
    {
        "",
        "none",
        "null",
        "n a",
        "na",
        "not run",
        "no test",
        "refused",
        "not applicable",
        "not supported",
        "unsupported",
        "not available",
    }
)
_GREEK = {
    "\u03c7": " chi",  # chi
    "\u03c1": " rho",
    "\u03c4": " tau",
    "\u03b7": " eta",
    "\u03c9": " omega",
    "\u03b5": " epsilon",
    "\u03bc": " mu",
    "\u03b1": " alpha",
    "\u03b2": " beta",
    "\u03c3": " sigma",
}


def normalize_name(text: str) -> str:
    """Lower case ASCII words: Unicode dashes, Greek letters, superscripts,
    possessives and punctuation folded ("Welch's t\u2013test" -> "welch t test",
    "\u03c7\u00b2 test" -> "chi2 test")."""
    text = unicodedata.normalize("NFKC", text).casefold()
    for letter, latin in _GREEK.items():
        text = text.replace(letter, latin)
    text = re.sub(r"[\u2018\u2019\u02bc\u00b4`]", "'", text)
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _has(phrase: str, text: str) -> bool:
    whole = phrase.endswith("$")
    core = phrase[:-1] if whole else phrase
    pattern = r"(?:^|\s)" + re.escape(core) + (r"(?:\s|$)" if whole else "")
    return re.search(pattern, text) is not None


def families_of(name: str | None) -> frozenset[str]:
    """Every family a reported test name can mean (the task decides which count)."""
    if name is None:
        return frozenset({NO_TEST})
    text = normalize_name(str(name))
    if text in NO_TEST_WORDS:
        return frozenset({NO_TEST})
    if text in EXACT_NAMES:
        return frozenset({EXACT_NAMES[text]})
    found = set()
    for family_id, family in FAMILIES.items():
        if any(_has(p, text) for p in family.none):
            continue
        if all(any(_has(p, text) for p in group) for group in family.groups):
            found.add(family_id)
    return frozenset(found)


def family_matches(family_id: str, name: str | None) -> bool:
    return family_id in families_of(name)


# --- recipes: alternative analyses computed with SciPy ------------------------------
# Some accepted answers (a variant R does not run by default, such as Yates'
# correction, the exact McNemar test or SciPy's exact Wilcoxon p) and some trap
# answers (the analysis a trap leads to) have no ggstatsplot fixture. Their numbers
# are computed here from the task's dataset with SciPy and NumPy; the dry run
# recomputes them and compares with the task file. A recipe returns its numbers by
# name; "statistic_alt" lists the other forms of the same statistic an analyst may
# report (U or n1 n2 - U, V or its complement, z), "estimate_alt" likewise.


def _frame(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path)


def _two(df: Any, x: str, y: str, levels: Sequence[str]) -> tuple[Any, Any]:
    clean = df[[x, y]].dropna()
    keys = clean[x].astype(str)
    a = clean.loc[keys == levels[0], y].to_numpy(dtype=float)
    b = clean.loc[keys == levels[1], y].to_numpy(dtype=float)
    return a, b


def _groups(df: Any, x: str, y: str, log: bool = False) -> list[Any]:
    import numpy as np

    clean = df[[x, y]].dropna()
    groups = [g[y].to_numpy(dtype=float) for _, g in clean.groupby(x)]
    return [np.log(g) for g in groups] if log else groups


def _wide(df: Any, x: str, y: str, subject: str) -> Any:
    """Subjects by conditions, complete subjects only (pairs matched by subject)."""
    clean = df[[subject, x, y]].dropna()
    return clean.pivot_table(
        index=subject, columns=x, values=y, aggfunc="first"
    ).dropna()


def _differences(
    df: Any,
    *,
    x: str | None = None,
    y: str | None = None,
    subject: str | None = None,
    levels: Sequence[str] = (),
    column: str | None = None,
    mu: float = 0.0,
) -> Any:
    """Paired differences (levels[0] - levels[1], by subject) or values - mu."""
    if column is not None:
        return df[column].dropna().to_numpy(dtype=float) - mu
    if x is None or y is None or subject is None:
        raise ValueError("a paired design needs x, y and subject")
    wide = _wide(df, x, y, subject)
    return wide[levels[0]].to_numpy(dtype=float) - wide[levels[1]].to_numpy(dtype=float)


def _welch_df(a: Any, b: Any) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)))


def recipe_student_t(
    df: Any,
    *,
    x: str,
    y: str,
    levels: Sequence[str],
    alternative: str = "two-sided",
) -> dict[str, Any]:
    from scipy import stats

    a, b = _two(df, x, y, levels)
    side: dict[str, Any] = {"alternative": alternative}  # a SciPy literal
    res = stats.ttest_ind(a, b, equal_var=True, **side)
    return {
        "statistic": float(res.statistic),
        "df": float(len(a) + len(b) - 2),
        "p_value": float(res.pvalue),
        "n": float(len(a) + len(b)),
    }


def recipe_welch_t(
    df: Any,
    *,
    x: str,
    y: str,
    levels: Sequence[str],
    alternative: str = "two-sided",
) -> dict[str, Any]:
    from scipy import stats

    a, b = _two(df, x, y, levels)
    side: dict[str, Any] = {"alternative": alternative}  # a SciPy literal
    res = stats.ttest_ind(a, b, equal_var=False, **side)
    return {
        "statistic": float(res.statistic),
        "df": _welch_df(a, b),
        "p_value": float(res.pvalue),
        "n": float(len(a) + len(b)),
    }


def recipe_paired_by_row_order(
    df: Any, *, x: str, y: str, levels: Sequence[str]
) -> dict[str, Any]:
    """The paired t-test with values paired by their order in the file."""
    from scipy import stats

    a, b = _two(df, x, y, levels)
    res = stats.ttest_rel(a, b)
    return {
        "statistic": float(res.statistic),
        "df": float(len(a) - 1),
        "p_value": float(res.pvalue),
        "n": float(len(a)),
    }


def recipe_oneway_anova(
    df: Any, *, x: str, y: str, log: bool = False
) -> dict[str, Any]:
    from scipy import stats

    groups = _groups(df, x, y, log)
    res = stats.f_oneway(*groups)
    n = sum(len(g) for g in groups)
    return {
        "statistic": float(res.statistic),
        "df": float(len(groups) - 1),
        "df_error": float(n - len(groups)),
        "p_value": float(res.pvalue),
        "n": float(n),
    }


def recipe_welch_anova(df: Any, *, x: str, y: str, log: bool = False) -> dict[str, Any]:
    """Welch's (1951) heteroscedastic one-way ANOVA (statsmodels' anova_oneway with
    use_var="unequal", R's oneway.test); ``log`` analyses the logarithms."""
    import numpy as np
    from scipy import stats

    groups = _groups(df, x, y, log)
    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    w = n / np.array([g.var(ddof=1) for g in groups])
    centre = float((w * means).sum() / w.sum())
    a = float((w * (means - centre) ** 2).sum()) / (k - 1)
    tmp = float(((1 - w / w.sum()) ** 2 / (n - 1)).sum())
    f = a / (1 + 2 * (k - 2) / (k**2 - 1) * tmp)
    df2 = (k**2 - 1) / (3 * tmp)
    return {
        "statistic": f,
        "df": float(k - 1),
        "df_error": df2,
        "p_value": float(stats.f.sf(f, k - 1, df2)),
        "n": float(n.sum()),
    }


def _rm_parts(df: Any, x: str, y: str, subject: str) -> dict[str, Any]:
    import numpy as np

    data = _wide(df, x, y, subject).to_numpy(dtype=float)
    n, k = data.shape
    grand = data.mean()
    ss_cond = n * float(((data.mean(axis=0) - grand) ** 2).sum())
    ss_subj = k * float(((data.mean(axis=1) - grand) ** 2).sum())
    ss_total = float(((data - grand) ** 2).sum())
    ss_err = ss_total - ss_cond - ss_subj
    # Greenhouse-Geisser (1959) epsilon from the double-centred covariance matrix;
    # Huynh-Feldt (1976) epsilon from it (one group of subjects), at most 1.
    centre = np.eye(k) - 1.0 / k
    s = centre @ np.cov(data, rowvar=False) @ centre
    gg = float(np.trace(s) ** 2 / ((k - 1) * np.sum(s * s)))
    hf = min(1.0, (n * (k - 1) * gg - 2) / ((k - 1) * (n - 1 - (k - 1) * gg)))
    return {
        "n": n,
        "k": k,
        "ss_cond": ss_cond,
        "ss_subj": ss_subj,
        "ss_err": ss_err,
        "gg": gg,
        "hf": hf,
    }


def recipe_rm_anova(
    df: Any, *, x: str, y: str, subject: str, correction: str = "none"
) -> dict[str, Any]:
    """Repeated-measures ANOVA on the complete subjects; ``correction`` "none",
    "gg" (Greenhouse-Geisser) or "hf" (Huynh-Feldt) scales both df."""
    import numpy as np
    from scipy import stats

    parts = _rm_parts(df, x, y, subject)
    n, k = parts["n"], parts["k"]
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f = (parts["ss_cond"] / df1) / (parts["ss_err"] / df2)
    if not np.isfinite(f):
        raise ValueError("rm_anova: no error variance")
    eps = {"none": 1.0, "gg": parts["gg"], "hf": parts["hf"]}[correction]
    return {
        "statistic": f,
        "df": float(df1 * eps),
        "df_error": float(df2 * eps),
        "p_value": float(stats.f.sf(f, df1 * eps, df2 * eps)),
        "n": float(n),
    }


def recipe_rm_anova_uncorrected(
    df: Any, *, x: str, y: str, subject: str
) -> dict[str, Any]:
    """Repeated-measures ANOVA without a sphericity correction, complete subjects."""
    return recipe_rm_anova(df, x=x, y=y, subject=subject, correction="none")


def recipe_kruskal_untied(df: Any, *, x: str, y: str) -> dict[str, Any]:
    """Kruskal-Wallis H without the correction for ties."""
    from scipy import stats

    clean = df[[x, y]].dropna()
    ranks = stats.rankdata(clean[y].to_numpy(dtype=float))
    n_total = len(clean)
    h = 0.0
    keys = clean[x].astype(str).to_numpy()
    levels = sorted(set(keys))
    for level in levels:
        r = ranks[keys == level]
        h += float(r.sum()) ** 2 / len(r)
    h = 12.0 / (n_total * (n_total + 1)) * h - 3.0 * (n_total + 1)
    k = len(levels)
    return {
        "statistic": h,
        "df": float(k - 1),
        "p_value": float(stats.chi2.sf(h, k - 1)),
        "n": float(n_total),
    }


def recipe_wilcoxon_signed_rank(
    df: Any, *, method: str, correction: bool = False, **design: Any
) -> dict[str, Any]:
    """scipy.stats.wilcoxon on the differences (zeros dropped, as R and SciPy do
    by default): ``method`` "exact" or "approx", with or without the continuity
    correction. The statistic is SciPy's (the smaller rank sum); the larger one and,
    for the normal approximation, z are its other forms."""
    from scipy import stats

    d = _differences(df, **design)
    options: dict[str, Any] = {"method": method, "correction": correction}
    res = stats.wilcoxon(d, **options)
    nonzero = int((d != 0).sum())
    total = nonzero * (nonzero + 1) / 2.0
    stat = float(res.statistic)
    alt = [total - stat]
    z = getattr(res, "zstatistic", None)
    if method == "approx" and z is not None:
        alt.append(float(z))
    return {
        "statistic": stat,
        "statistic_alt": alt,
        "p_value": float(res.pvalue),
        "n": float(len(d)),
    }


def recipe_sign_test(df: Any, **design: Any) -> dict[str, Any]:
    """The exact sign test: the positive differences among the nonzero ones against
    a binomial(n, 1/2), two-sided (scipy.stats.binomtest)."""
    from scipy import stats

    d = _differences(df, **design)
    nonzero = d[d != 0]
    positive = int((nonzero > 0).sum())
    res = stats.binomtest(positive, len(nonzero), 0.5)
    return {
        "statistic": float(positive),
        "statistic_alt": [float(len(nonzero) - positive)],
        "p_value": float(res.pvalue),
        "n": float(len(d)),
    }


def recipe_mann_whitney(
    df: Any,
    *,
    x: str,
    y: str,
    levels: Sequence[str],
    method: str,
    use_continuity: bool = True,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """scipy.stats.mannwhitneyu: ``method`` "exact" or "asymptotic" (with or
    without the continuity correction). U of the first level, or n1 n2 - U."""
    from scipy import stats

    a, b = _two(df, x, y, levels)
    options: dict[str, Any] = {
        "method": method,
        "use_continuity": use_continuity,
        "alternative": alternative,
    }
    res = stats.mannwhitneyu(a, b, **options)
    u = float(res.statistic)
    return {
        "statistic": u,
        "statistic_alt": [len(a) * len(b) - u],
        "p_value": float(res.pvalue),
        "n": float(len(a) + len(b)),
    }


def _xy(df: Any, x: str, y: str) -> tuple[Any, Any]:
    clean = df[[x, y]].dropna()
    return clean[x].to_numpy(dtype=float), clean[y].to_numpy(dtype=float)


def recipe_spearman_t(df: Any, *, x: str, y: str) -> dict[str, Any]:
    """scipy.stats.spearmanr: rho with the t-distribution p (R uses an exact or
    Edgeworth p below 1,290 pairs without ties). rho or its t is the statistic."""
    from scipy import stats

    a, b = _xy(df, x, y)
    res = stats.spearmanr(a, b)
    rho = float(res.statistic)
    n = len(a)
    return {
        "statistic": rho,
        "statistic_alt": [rho * math.sqrt((n - 2) / (1 - rho**2))],
        "p_value": float(res.pvalue),
        "estimate": rho,
        "n": float(n),
    }


def recipe_kendall(df: Any, *, x: str, y: str, method: str) -> dict[str, Any]:
    """scipy.stats.kendalltau (tau-b): ``method`` "exact" (no ties; R's T, the
    concordant pairs, is the other form) or "asymptotic" (z the other form)."""
    from scipy import stats

    a, b = _xy(df, x, y)
    options: dict[str, Any] = {"method": method}
    res = stats.kendalltau(a, b, **options)
    tau = float(res.statistic)
    n = len(a)
    alt: list[float] = []
    if method == "exact":
        alt.append((tau + 1) / 2 * n * (n - 1) / 2)
    else:
        alt.append(math.copysign(float(stats.norm.isf(res.pvalue / 2)), tau))
    return {
        "statistic": tau,
        "statistic_alt": alt,
        "p_value": float(res.pvalue),
        "estimate": tau,
        "n": float(n),
    }


def _table(df: Any, x: str, y: str, counts: str | None = None) -> Any:
    import pandas as pd

    clean = df.dropna(subset=[x, y] + ([counts] if counts else []))
    weights = clean[counts] if counts else None
    return (
        pd.crosstab(
            clean[x].astype(str),
            clean[y].astype(str),
            values=weights,
            aggfunc="sum" if counts else None,
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )


def recipe_mcnemar_corrected(df: Any, *, x: str, y: str) -> dict[str, Any]:
    from scipy import stats

    t = _table(df, x, y)
    b, c = t[0, 1], t[1, 0]
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return {
        "statistic": float(stat),
        "df": 1.0,
        "p_value": float(stats.chi2.sf(stat, 1)),
        "n": float(t.sum()),
    }


def recipe_mcnemar_exact(df: Any, *, x: str, y: str) -> dict[str, Any]:
    """The exact (binomial) McNemar test on the discordant pairs; the statistic is
    the smaller discordant count (statsmodels' mcnemar default), or the larger."""
    from scipy import stats

    t = _table(df, x, y)
    b, c = int(t[0, 1]), int(t[1, 0])
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return {
        "statistic": float(min(b, c)),
        "statistic_alt": [float(max(b, c))],
        "p_value": float(p),
        "n": float(t.sum()),
    }


def recipe_chi2_yates(df: Any, *, x: str, y: str) -> dict[str, Any]:
    from scipy import stats

    t = _table(df, x, y)
    res = stats.chi2_contingency(t, correction=True)
    return {
        "statistic": float(res.statistic),
        "df": float(res.dof),
        "p_value": float(res.pvalue),
        "n": float(t.sum()),
    }


def recipe_fisher_2x2(df: Any, *, x: str, y: str) -> dict[str, Any]:
    from scipy import stats

    t = _table(df, x, y)
    res = stats.fisher_exact(t)
    return {"p_value": float(res.pvalue), "n": float(t.sum())}


def fisher_exact_p(table: Any) -> float:
    """Fisher's exact test of an r x 2 (or 2 x c) table (Freeman & Halton, 1951):
    the probability, given the margins, of the tables no more probable than the
    observed one (relative tolerance 1e-7, as R's fisher.test)."""
    import numpy as np
    from scipy import special

    t = np.asarray(table, dtype=int)
    if t.shape[1] != 2:
        t = t.T
    if t.shape[1] != 2:
        raise ValueError("fisher_exact_p: the table needs two rows or two columns")
    rows = t.sum(axis=1)
    first = int(t[:, 0].sum())
    total = int(rows.sum())

    def log_prob(cells: Sequence[int]) -> float:
        logp = -float(
            special.gammaln(total + 1)
            - special.gammaln(first + 1)
            - special.gammaln(total - first + 1)
        )
        for r, a in zip(rows, cells, strict=True):
            logp += float(
                special.gammaln(r + 1)
                - special.gammaln(a + 1)
                - special.gammaln(r - a + 1)
            )
        return logp

    observed = log_prob(t[:, 0].tolist())
    p = 0.0

    def walk(i: int, left: int, cells: list[int]) -> None:
        nonlocal p
        if i == len(rows) - 1:
            if 0 <= left <= rows[i]:
                logp = log_prob([*cells, left])
                if logp <= observed + math.log1p(1e-7):
                    p += math.exp(logp)
            return
        for a in range(max(0, left - int(rows[i + 1 :].sum())), min(left, rows[i]) + 1):
            walk(i + 1, left - a, [*cells, a])

    walk(0, first, [])
    return min(1.0, p)


def recipe_fisher_exact(df: Any, *, x: str, y: str) -> dict[str, Any]:
    t = _table(df, x, y)
    return {"p_value": fisher_exact_p(t), "n": float(t.sum())}


def recipe_chi2_rows_unweighted(df: Any, *, x: str, y: str) -> dict[str, Any]:
    """Chi-square on the rows of a table of counts, ignoring the counts."""
    from scipy import stats

    t = _table(df, x, y)
    res = stats.chi2_contingency(t, correction=False)
    return {
        "statistic": float(res.statistic),
        "df": float(res.dof),
        "p_value": float(res.pvalue),
        "n": float(t.sum()),
    }


def _design(clean: Any, columns: Sequence[str], factors: Sequence[str]) -> list[Any]:
    """Columns of a design matrix: numbers as they are, factors as treatment
    dummies (first level dropped)."""
    out = []
    for column in columns:
        if column in factors:
            codes = clean[column].astype(str)
            for level in sorted(set(codes))[1:]:
                out.append((codes == level).to_numpy(dtype=float))
        else:
            out.append(clean[column].to_numpy(dtype=float))
    return out


def recipe_ols_term(
    df: Any,
    *,
    y: str,
    predictors: Sequence[str],
    term: str,
    factors: Sequence[str] = (),
) -> dict[str, Any]:
    """Ordinary least squares y ~ predictors (``factors`` as categories); the
    t-test of the numeric predictor ``term``."""
    import numpy as np
    from scipy import stats

    clean = df[[y, *predictors]].dropna()
    others = [p for p in predictors if p != term]
    design = np.column_stack(
        [
            np.ones(len(clean)),
            clean[term].to_numpy(dtype=float),
            *_design(clean, others, factors),
        ]
    )
    target = clean[y].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    dof = len(clean) - design.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    t = float(coef[1] / math.sqrt(cov[1, 1]))
    return {
        "statistic": t,
        "df": float(dof),
        "p_value": float(2 * stats.t.sf(abs(t), dof)),
        "estimate": float(coef[1]),
        "n": float(len(clean)),
    }


def _nested_f(target: Any, small: Any, large: Any) -> tuple[float, float, float, float]:
    import numpy as np
    from scipy import stats

    def rss(design: Any) -> float:
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        r = target - design @ coef
        return float(r @ r)

    df1 = large.shape[1] - small.shape[1]
    df2 = len(target) - large.shape[1]
    f = ((rss(small) - rss(large)) / df1) / (rss(large) / df2)
    return f, float(df1), float(df2), float(stats.f.sf(f, df1, df2))


def recipe_two_way_interaction(
    df: Any, *, y: str, a: str, b: str, numeric: Sequence[str] = ()
) -> dict[str, Any]:
    """F-test of the a x b interaction (balanced design: any SS type); a column in
    ``numeric`` enters as a number (one slope) instead of as a factor."""
    import numpy as np

    clean = df[[y, a, b]].dropna()
    target = clean[y].to_numpy(dtype=float)
    factors = [c for c in (a, b) if c not in numeric]
    da = np.column_stack(_design(clean, [a], factors))
    db = np.column_stack(_design(clean, [b], factors))
    inter = np.column_stack(
        [da[:, i] * db[:, j] for i in range(da.shape[1]) for j in range(db.shape[1])]
    )
    additive = np.hstack([np.ones((len(clean), 1)), da, db])
    f, df1, df2, p = _nested_f(target, additive, np.hstack([additive, inter]))
    return {
        "statistic": f,
        "df": df1,
        "df_error": df2,
        "p_value": p,
        "n": float(len(clean)),
    }


RECIPES: Mapping[str, Callable[..., dict[str, Any]]] = {
    "student_t": recipe_student_t,
    "welch_t": recipe_welch_t,
    "paired_by_row_order": recipe_paired_by_row_order,
    "oneway_anova": recipe_oneway_anova,
    "welch_anova": recipe_welch_anova,
    "rm_anova": recipe_rm_anova,
    "rm_anova_uncorrected": recipe_rm_anova_uncorrected,
    "kruskal_untied": recipe_kruskal_untied,
    "wilcoxon_signed_rank": recipe_wilcoxon_signed_rank,
    "sign_test": recipe_sign_test,
    "mann_whitney": recipe_mann_whitney,
    "spearman_t": recipe_spearman_t,
    "kendall": recipe_kendall,
    "mcnemar_corrected": recipe_mcnemar_corrected,
    "mcnemar_exact": recipe_mcnemar_exact,
    "chi2_yates": recipe_chi2_yates,
    "fisher_2x2": recipe_fisher_2x2,
    "fisher_exact": recipe_fisher_exact,
    "chi2_rows_unweighted": recipe_chi2_rows_unweighted,
    "ols_term": recipe_ols_term,
    "two_way_interaction": recipe_two_way_interaction,
}

# --- effect sizes: the standard alternatives, one formula away ---------------------
# ggstatsplot reports one effect size per test (Hedges' g, omega squared, rank
# epsilon squared, ...). An answer with another standard effect size of the same
# analysis is right when it matches one of these, each computed from the data.


def _j(df: float) -> float:
    """Hedges' small-sample factor J(df) (Hedges, 1981), with the gamma function."""
    return math.exp(
        math.lgamma(df / 2) - math.log(math.sqrt(df / 2)) - math.lgamma((df - 1) / 2)
    )


def effect_cohen_d(
    df: Any, *, x: str, y: str, levels: Sequence[str], sd: str = "pooled"
) -> float:
    """(mean_1 - mean_2) / s: s pooled (Cohen's d) or "average" (the root mean of
    the two variances, d_av)."""
    a, b = _two(df, x, y, levels)
    if sd == "pooled":
        n1, n2 = len(a), len(b)
        s = math.sqrt(
            ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
        )
    else:
        s = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s)


def effect_hedges_g(df: Any, *, x: str, y: str, levels: Sequence[str]) -> float:
    """Cohen's d (pooled) times J(n1 + n2 - 2)."""
    a, b = _two(df, x, y, levels)
    return effect_cohen_d(df, x=x, y=y, levels=levels) * _j(len(a) + len(b) - 2)


def effect_cohen_dz(df: Any, **design: Any) -> float:
    """Mean difference over the SD of the differences (paired d_z, or the
    one-sample d when the design is a column against mu)."""
    d = _differences(df, **design)
    return float(d.mean() / d.std(ddof=1))


def _oneway_ss(df: Any, x: str, y: str) -> tuple[float, float, int, int]:
    import numpy as np

    groups = _groups(df, x, y)
    values = np.concatenate(groups)
    grand = values.mean()
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    ss_total = float(((values - grand) ** 2).sum())
    return ss_between, ss_total, len(groups), len(values)


def effect_eta_squared(df: Any, *, x: str, y: str) -> float:
    ss_b, ss_t, _, _ = _oneway_ss(df, x, y)
    return ss_b / ss_t


def effect_omega_squared(df: Any, *, x: str, y: str) -> float:
    ss_b, ss_t, k, n = _oneway_ss(df, x, y)
    ms_w = (ss_t - ss_b) / (n - k)
    return (ss_b - (k - 1) * ms_w) / (ss_t + ms_w)


def effect_eta_squared_h(df: Any, *, x: str, y: str) -> float:
    """(H - k + 1) / (n - k) (Tomczak & Tomczak, 2014), H with the tie correction."""
    from scipy import stats

    groups = _groups(df, x, y)
    h = float(stats.kruskal(*groups).statistic)
    n = sum(len(g) for g in groups)
    return (h - len(groups) + 1) / (n - len(groups))


def effect_probability_of_superiority(
    df: Any, *, x: str, y: str, levels: Sequence[str]
) -> float:
    """U / (n1 n2): the common-language effect size (McGraw & Wong, 1992)."""
    from scipy import stats

    a, b = _two(df, x, y, levels)
    return float(stats.mannwhitneyu(a, b).statistic) / (len(a) * len(b))


def effect_rm_eta_squared(df: Any, *, x: str, y: str, subject: str, kind: str) -> float:
    """Partial (SS_cond / (SS_cond + SS_err)) or generalized (Olejnik & Algina,
    2003: SS_cond / (SS_cond + SS_subj + SS_err)) eta squared."""
    p = _rm_parts(df, x, y, subject)
    if kind == "partial":
        return p["ss_cond"] / (p["ss_cond"] + p["ss_err"])
    return p["ss_cond"] / (p["ss_cond"] + p["ss_subj"] + p["ss_err"])


def effect_cramers_v(df: Any, *, x: str, y: str, counts: str | None = None) -> float:
    """sqrt(chi2 / (n (min(r, c) - 1))) without the bias correction and without
    Yates' correction; phi for a 2 x 2 table."""
    from scipy import stats

    t = _table(df, x, y, counts)
    chi2 = float(stats.chi2_contingency(t, correction=False).statistic)
    return math.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))


def effect_cohens_w(df: Any, *, x: str, ratio: Sequence[float]) -> float:
    """sqrt(chi2 / n) of the goodness-of-fit test (Cohen, 1988)."""
    import numpy as np

    observed = df[x].dropna().astype(str).value_counts().sort_index().to_numpy()
    observed = observed.astype(float)
    expected = np.asarray(ratio, dtype=float) * observed.sum()
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return math.sqrt(chi2 / observed.sum())


def effect_odds_ratio(df: Any, *, x: str, y: str, paired: bool = False) -> float:
    """The sample odds ratio of a 2 x 2 table (ad / bc), or for paired answers the
    ratio of the discordant counts (b / c)."""
    t = _table(df, x, y)
    if paired:
        return float(t[0, 1] / t[1, 0])
    return float(t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0]))


EFFECTS: Mapping[str, Callable[..., float]] = {
    "cohen_d": effect_cohen_d,
    "hedges_g": effect_hedges_g,
    "cohen_dz": effect_cohen_dz,
    "eta_squared": effect_eta_squared,
    "omega_squared": effect_omega_squared,
    "eta_squared_h": effect_eta_squared_h,
    "probability_of_superiority": effect_probability_of_superiority,
    "rm_eta_squared": effect_rm_eta_squared,
    "cramers_v": effect_cramers_v,
    "cohens_w": effect_cohens_w,
    "odds_ratio": effect_odds_ratio,
}
# Effect sizes whose reciprocal or complement is the same size read the other way.
EFFECT_ALT: Mapping[str, Callable[[float], float]] = {
    "probability_of_superiority": lambda v: 1.0 - v,
    "odds_ratio": lambda v: 1.0 / v,
}

# --- tasks -------------------------------------------------------------------------


class TaskError(ValueError):
    """A task file is malformed; the message names the task and the field."""


@dataclass(frozen=True)
class Expected:
    value: float
    alt: tuple[float, ...] = ()  # other values that are equally right (U vs n1 n2 - U)
    key: str | None = None  # fixture path ("stats/subtitle_data/0/p.value")
    tol_abs: float | None = None
    tol_rel: float | None = None

    def tolerance(self, name: str) -> tuple[float, float]:
        base_abs, base_rel = TOLERANCES[name]
        return (
            base_abs if self.tol_abs is None else self.tol_abs,
            base_rel if self.tol_rel is None else self.tol_rel,
        )


@dataclass(frozen=True)
class EffectAlternative:
    """Another standard effect size of the same analysis, computed from the data."""

    label: str
    recipe: str
    recipe_args: Mapping[str, Any]
    estimate: Expected


@dataclass(frozen=True)
class Block:
    """One accepted (or trap) analysis with the numbers it gives."""

    family: str
    label: str
    source: str  # "fixture" or "recipe" or "none"
    numbers: Mapping[str, Expected]
    fixture: str | None = None
    kwargs: Mapping[str, Any] | None = None  # inferplot call that reproduces it
    recipe: str | None = None
    recipe_args: Mapping[str, Any] = field(default_factory=dict)
    sign_free: bool = True
    # Words one of which the reported name must contain, unless the numbers are
    # this variant's as a whole ("log": Welch's ANOVA of the log reaction times).
    qualifier: tuple[str, ...] = ()
    # The task's [counts] that are also right for n (besides the value).
    n_also: tuple[str, ...] = ()
    effects: tuple[EffectAlternative, ...] = ()


# How the task's trap is scored (Score.trap_avoided).
TRAP_SCORING = {
    "traps": "the numbers of a [[traps]] analysis, or a rejected test",
    "test": "the choice of test (a rejected test is the trap)",
    "n": "n (the rows with missing values must be left out)",
    "none": "no trap (a baseline task)",
    "manual": "not scored automatically; read the answer for it (read_for)",
}


@dataclass(frozen=True)
class Task:
    id: str
    path: Path
    function: str
    trap: str
    trap_scoring: str
    read_for: str | None
    request: str
    data_file: Path
    data_source: str | None
    data_columns: Mapping[str, str]
    data_levels: Mapping[str, Mapping[str, str]]
    data_note: str
    design: str
    args: tuple[str, ...]
    kwargs: Mapping[str, Any]
    raises: str | None
    check_codes: tuple[str, ...]
    counts: Mapping[str, float]
    accepted: tuple[Block, ...]
    traps: tuple[Block, ...]
    rejected: tuple[str, ...]
    correct_rule: str


def _need(
    table: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str
) -> Any:
    if key not in table:
        raise TaskError(f"{where}: missing '{key}'")
    value = table[key]
    if not isinstance(value, kind):
        raise TaskError(f"{where}: '{key}' must be {kind}, not {type(value).__name__}")
    return value


def _expected(raw: Any, where: str) -> Expected:
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return Expected(float(raw))
    if not isinstance(raw, dict):
        raise TaskError(f"{where}: a number or a table with 'value'")
    unknown = set(raw) - {"value", "alt", "key", "tol_abs", "tol_rel"}
    if unknown:
        raise TaskError(f"{where}: unknown keys {sorted(unknown)}")
    value = _need(raw, "value", (int, float), where)
    alt = tuple(float(v) for v in raw.get("alt", ()))
    return Expected(
        float(value), alt, raw.get("key"), raw.get("tol_abs"), raw.get("tol_rel")
    )


BLOCK_KEYS = {
    "family",
    "label",
    "source",
    "numbers",
    "fixture",
    "kwargs",
    "recipe",
    "recipe_args",
    "sign_free",
    "qualifier",
    "n_also",
    "effects",
}


def _effect(raw: Any, where: str) -> EffectAlternative:
    if not isinstance(raw, dict):
        raise TaskError(f"{where}: must be a table")
    unknown = set(raw) - {"label", "recipe", "recipe_args", "estimate"}
    if unknown:
        raise TaskError(f"{where}: unknown keys {sorted(unknown)}")
    recipe = _need(raw, "recipe", str, where)
    if recipe not in EFFECTS:
        raise TaskError(f"{where}: unknown effect recipe '{recipe}'")
    return EffectAlternative(
        label=_need(raw, "label", str, where),
        recipe=recipe,
        recipe_args=raw.get("recipe_args", {}),
        estimate=_expected(_need(raw, "estimate", (int, float, dict), where), where),
    )


def _block(raw: Any, where: str, counts: Mapping[str, float]) -> Block:
    if not isinstance(raw, dict):
        raise TaskError(f"{where}: must be a table")
    unknown_keys = set(raw) - BLOCK_KEYS
    if unknown_keys:
        raise TaskError(f"{where}: unknown keys {sorted(unknown_keys)}")
    family = _need(raw, "family", str, where)
    if family != NO_TEST and family not in FAMILIES:
        raise TaskError(f"{where}: unknown family '{family}'")
    source = _need(raw, "source", str, where)
    if source not in ("fixture", "recipe", "none"):
        raise TaskError(f"{where}: source must be fixture, recipe or none")
    numbers_raw = raw.get("numbers", {})
    unknown = set(numbers_raw) - set(NUMBERS)
    if unknown:
        raise TaskError(f"{where}: unknown numbers {sorted(unknown)}")
    numbers = {k: _expected(v, f"{where}.numbers.{k}") for k, v in numbers_raw.items()}
    if source != "none" and not numbers:
        raise TaskError(f"{where}: a {source} block needs numbers")
    if source == "fixture":
        _need(raw, "fixture", str, where)
        missing_keys = [k for k, v in numbers.items() if v.key is None]
        if missing_keys:
            raise TaskError(f"{where}: numbers {missing_keys} need the fixture 'key'")
    if source == "recipe":
        recipe = _need(raw, "recipe", str, where)
        if recipe not in RECIPES:
            raise TaskError(f"{where}: unknown recipe '{recipe}'")
    n_also = tuple(raw.get("n_also", ()))
    for name in n_also:
        if name not in counts:
            raise TaskError(f"{where}: n_also '{name}' is not in the task's [counts]")
    if n_also:
        if "n" not in numbers:
            raise TaskError(f"{where}: n_also without an n")
        n = numbers["n"]
        numbers["n"] = dataclasses.replace(
            n, alt=(*n.alt, *(counts[name] for name in n_also))
        )
    effects = tuple(
        _effect(e, f"{where}.effects[{i}]")
        for i, e in enumerate(raw.get("effects", []))
    )
    return Block(
        family=family,
        label=str(
            raw.get(
                "label", FAMILIES[family].label if family in FAMILIES else "no test"
            )
        ),
        source=source,
        numbers=numbers,
        fixture=raw.get("fixture"),
        kwargs=raw.get("kwargs"),
        recipe=raw.get("recipe"),
        recipe_args=raw.get("recipe_args", {}),
        sign_free=bool(raw.get("sign_free", True)),
        qualifier=tuple(raw.get("qualifier", ())),
        n_also=n_also,
        effects=effects,
    )


TASK_KEYS = {
    "id",
    "function",
    "trap",
    "trap_scoring",
    "read_for",
    "request",
    "design",
    "correct_rule",
    "data",
    "reference",
    "counts",
    "accepted",
    "traps",
    "rejected",
}


def load_task(path: Path) -> Task:
    where = path.parent.name
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise TaskError(f"{where}: {error}") from error
    unknown = set(raw) - TASK_KEYS
    if unknown:
        raise TaskError(f"{where}: unknown keys {sorted(unknown)}")
    task_id = _need(raw, "id", str, where)
    if task_id != where:
        raise TaskError(f"{where}: id '{task_id}' differs from the directory name")
    function = _need(raw, "function", str, where)
    if function not in FUNCTIONS:
        raise TaskError(f"{where}: unknown function '{function}'")
    trap_scoring = _need(raw, "trap_scoring", str, where)
    if trap_scoring not in TRAP_SCORING:
        raise TaskError(f"{where}: trap_scoring must be one of {sorted(TRAP_SCORING)}")
    read_for = raw.get("read_for")
    if trap_scoring == "manual" and not read_for:
        raise TaskError(f"{where}: a manual trap needs 'read_for'")
    data = _need(raw, "data", dict, where)
    reference = _need(raw, "reference", dict, where)
    counts_raw = _need(raw, "counts", dict, where)
    unknown_counts = set(counts_raw) - set(COUNTS)
    if unknown_counts:
        raise TaskError(f"{where}: unknown counts {sorted(unknown_counts)}")
    counts = {
        k: float(_need(counts_raw, k, int, f"{where}.counts")) for k in counts_raw
    }
    accepted = tuple(
        _block(b, f"{where}.accepted[{i}]", counts)
        for i, b in enumerate(_need(raw, "accepted", list, where))
    )
    if not accepted:
        raise TaskError(f"{where}: no accepted analysis")
    traps = tuple(
        _block(b, f"{where}.traps[{i}]", counts)
        for i, b in enumerate(raw.get("traps", []))
    )
    rejected = tuple(raw.get("rejected", []))
    for family in rejected:
        if family not in FAMILIES:
            raise TaskError(f"{where}: unknown rejected family '{family}'")
    request = _need(raw, "request", str, where).strip()
    if len(request) < 40:
        raise TaskError(
            f"{where}: the request is too short to be a researcher's question"
        )
    return Task(
        id=task_id,
        path=path.parent,
        function=function,
        trap=_need(raw, "trap", str, where),
        trap_scoring=trap_scoring,
        read_for=read_for,
        request=request,
        data_file=path.parent / _need(data, "file", str, f"{where}.data"),
        data_source=data.get("source"),
        data_columns=data.get("columns", {}),
        data_levels=data.get("levels", {}),
        data_note=_need(data, "note", str, f"{where}.data"),
        design=_need(raw, "design", str, where),
        args=tuple(_need(reference, "args", list, f"{where}.reference")),
        kwargs=reference.get("kwargs", {}),
        raises=reference.get("raises"),
        check_codes=tuple(reference.get("check_codes", [])),
        counts=counts,
        accepted=accepted,
        traps=traps,
        rejected=rejected,
        correct_rule=_need(raw, "correct_rule", str, where),
    )


def load_tasks(only: Sequence[str] | None = None) -> list[Task]:
    paths = sorted(TASKS_DIR.glob("*/task.toml"))
    tasks = [load_task(p) for p in paths]
    if only:
        known = {t.id for t in tasks}
        missing = [o for o in only if o not in known]
        if missing:
            raise TaskError(f"no such task: {', '.join(missing)}")
        tasks = [t for t in tasks if t.id in only]
    return tasks


# --- counts: every n an answer may report ------------------------------------------
# The prompt asks for "the number of observations (or subjects, or pairs) the test
# used", so each defensible count of the design is right for n. A task names its
# counts in [counts]; the dry run recomputes each from data.csv.


def _complete(frame: Any, task: Task) -> Any:
    columns = list(task.args)
    if task.function == "withinstats":
        columns.append(task.kwargs["subject_id"])
    if task.function == "barstats" and task.kwargs.get("counts"):
        columns.append(task.kwargs["counts"])
    return frame[columns].dropna()


def _within_wide(frame: Any, task: Task) -> Any:
    x, y = task.args
    return _wide(frame, x, y, task.kwargs["subject_id"])


def count_observations(frame: Any, task: Task) -> int:
    """The values the test used: complete rows (complete subjects times the
    conditions in a within design; the summed counts of a table of counts)."""
    if task.function == "withinstats":
        wide = _within_wide(frame, task)
        return int(wide.shape[0] * wide.shape[1])
    clean = _complete(frame, task)
    if task.function == "barstats" and task.kwargs.get("counts"):
        return int(clean[task.kwargs["counts"]].sum())
    return len(clean)


def count_subjects(frame: Any, task: Task) -> int:
    """Subjects with every condition (within), or units with both answers (paired
    yes/no)."""
    if task.function == "withinstats":
        return int(_within_wide(frame, task).shape[0])
    if task.function == "barstats" and task.kwargs.get("paired"):
        return len(_complete(frame, task))
    raise ValueError("subjects: not a repeated-measures design")


def count_pairs(frame: Any, task: Task) -> int:
    """Complete pairs: subjects with both of two conditions."""
    if task.function == "withinstats" and _within_wide(frame, task).shape[1] != 2:
        raise ValueError("pairs: not two conditions")
    return count_subjects(frame, task)


def count_nonzero_differences(frame: Any, task: Task) -> int:
    """Pairs (or values against the test value) whose difference is not zero: the
    signed-rank and sign tests drop the others."""
    if task.function == "withinstats":
        wide = _within_wide(frame, task)
        if wide.shape[1] != 2:
            raise ValueError("nonzero_differences: not two conditions")
        d = wide.iloc[:, 0].to_numpy(dtype=float) - wide.iloc[:, 1].to_numpy(
            dtype=float
        )
    elif task.function == "histostats":
        values = frame[task.args[0]].dropna().to_numpy(dtype=float)
        d = values - float(task.kwargs.get("test_value", 0))
    else:
        raise ValueError("nonzero_differences: not a paired or one-sample design")
    return int((d != 0).sum())


def count_responses(frame: Any, task: Task) -> int:
    """Both answers of every complete unit (paired yes/no): twice the units."""
    if not (task.function == "barstats" and task.kwargs.get("paired")):
        raise ValueError("responses: not paired answers")
    return 2 * len(_complete(frame, task))


def count_discordant_pairs(frame: Any, task: Task) -> int:
    """Units whose two answers differ: the only ones McNemar's test uses."""
    if not (task.function == "barstats" and task.kwargs.get("paired")):
        raise ValueError("discordant_pairs: not paired answers")
    clean = _complete(frame, task)
    a, b = task.args
    return int((clean[a].astype(str) != clean[b].astype(str)).sum())


COUNTS: Mapping[str, Callable[[Any, Task], int]] = {
    "observations": count_observations,
    "subjects": count_subjects,
    "pairs": count_pairs,
    "nonzero_differences": count_nonzero_differences,
    "responses": count_responses,
    "discordant_pairs": count_discordant_pairs,
}


# --- the dry-run checks ------------------------------------------------------------


def within(name: str, expected: Expected, got: float | None, sign_free: bool) -> bool:
    if got is None or (isinstance(got, float) and math.isnan(got)):
        return False
    tol_abs, tol_rel = expected.tolerance(name)
    signed = name in ("statistic", "estimate", "conf_low", "conf_high")
    candidates = [expected.value, *expected.alt]
    for want in candidates:
        values = [got, -got] if (sign_free and signed) else [got]
        for value in values:
            if math.isinf(want) or math.isinf(value):
                if want == value:
                    return True
                continue
            if abs(value - want) <= max(tol_abs, tol_rel * abs(want)):
                return True
    return False


def interval_within(
    block: Block, low: float | None, high: float | None
) -> tuple[bool, bool]:
    """Both bounds, also as the mirrored interval when the block is sign-free."""
    want_low, want_high = block.numbers.get("conf_low"), block.numbers.get("conf_high")

    def ok(name: str, want: Expected | None, got: float | None) -> bool:
        if want is None:
            return True
        return within(
            name,
            Expected(want.value, want.alt, None, want.tol_abs, want.tol_rel),
            got,
            sign_free=False,
        )

    direct = (ok("conf_low", want_low, low), ok("conf_high", want_high, high))
    if block.sign_free and low is not None and high is not None:
        mirrored = (ok("conf_low", want_low, -high), ok("conf_high", want_high, -low))
        if sum(mirrored) > sum(direct):
            return mirrored
    return direct


def _fixture_value(fixture: Mapping[str, Any], key: str) -> Any:
    node: Any = fixture
    for part in key.split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    if node in ("Inf", "-Inf"):
        return math.inf if node == "Inf" else -math.inf
    return node


def result_numbers(result: Any) -> dict[str, float | None]:
    test, effect, bayes = result.test, result.effect_size, result.bayes
    numbers: dict[str, float | None] = {"n": float(result.n_obs)}
    if test is not None:
        numbers["statistic"] = test.statistic
        numbers["df"] = test.df if test.df is not None else test.df_error
        numbers["df_error"] = test.df_error if test.df is not None else None
        numbers["p_value"] = test.p_value
    if effect is not None:
        numbers |= {
            "estimate": effect.estimate,
            "conf_low": effect.conf_low,
            "conf_high": effect.conf_high,
        }
    if bayes is not None:
        numbers["bf10"] = bayes.bf10
    return numbers


def _analyze(task: Task, frame: Any, kwargs: Mapping[str, Any]) -> Any:
    import inferplot as ip

    function = getattr(ip.analyze, task.function)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(frame, *task.args, **kwargs)


def _source_frame(task: Task) -> Any:
    import pandas as pd

    assert task.data_source is not None
    source = pd.read_csv(REPO_ROOT / task.data_source, dtype=str, keep_default_na=False)
    columns = dict(task.data_columns) or {c: c for c in source.columns}
    frame: Any = source[list(columns)]
    frame = frame.rename(columns=columns)
    for column, mapping in task.data_levels.items():
        frame[column] = frame[column].map(lambda v, m=mapping: m.get(v, v))
    return frame


def validate_task(task: Task, fixtures_dir: Path | None = FIXTURES_DIR) -> list[str]:
    """Every problem with one task; an empty list when it is right."""
    import pandas as pd

    problems: list[str] = []
    if not task.data_file.is_file():
        return [f"{task.id}: dataset {task.data_file.name} is missing"]
    try:
        frame = pd.read_csv(task.data_file)
    except (OSError, ValueError) as error:
        return [f"{task.id}: dataset unreadable: {error}"]
    for column in task.args:
        if column not in frame.columns:
            problems.append(f"{task.id}: column '{column}' not in the dataset")
    if task.data_source is not None:
        if (REPO_ROOT / task.data_source).is_file():
            expected = _source_frame(task)
            actual = pd.read_csv(task.data_file, dtype=str, keep_default_na=False)
            if list(expected.columns) != list(actual.columns) or not expected.equals(
                actual
            ):
                problems.append(
                    f"{task.id}: dataset differs from {task.data_source} "
                    "with the stated renames"
                )
        else:
            problems.append(f"{task.id}: source {task.data_source} not found")
    if problems:
        return problems

    # Every count the task states for n, recomputed from the data.
    for name, value in task.counts.items():
        try:
            have = COUNTS[name](frame, task)
        except (ValueError, KeyError) as error:
            problems.append(f"{task.id}: count {name} does not apply: {error}")
            continue
        if have != value:
            problems.append(f"{task.id}: count {name} is {have}, task says {value:g}")

    # The reference call: it raises what the task says, or gives the check codes.
    if task.raises is not None:
        try:
            _analyze(task, frame, task.kwargs)
        except Exception as error:
            if type(error).__name__ != task.raises:
                problems.append(
                    f"{task.id}: raised {type(error).__name__}, not {task.raises}"
                )
        else:
            problems.append(
                f"{task.id}: the reference call did not raise {task.raises}"
            )
    else:
        result = _analyze(task, frame, task.kwargs)
        codes = sorted({issue.code for issue in result.check()})
        if codes != sorted(task.check_codes):
            problems.append(
                f"{task.id}: check() codes {codes}, "
                f"task says {sorted(task.check_codes)}"
            )

    for kind, blocks in (("accepted", task.accepted), ("traps", task.traps)):
        for i, block in enumerate(blocks):
            where = f"{task.id}.{kind}[{i}] ({block.label})"
            problems.extend(_validate_block(task, frame, block, where, fixtures_dir))
            n = block.numbers.get("n")
            if (
                kind == "accepted"
                and n is not None
                and n.value not in task.counts.values()
            ):
                problems.append(f"{where}: n {n.value:g} is none of the task's counts")
    return problems


def _same_values(want: Sequence[float], have: Sequence[float]) -> bool:
    if len(want) != len(have):
        return False
    return all(
        math.isclose(w, h, rel_tol=RECIPE_REL, abs_tol=1e-12)
        for w, h in zip(sorted(want), sorted(have), strict=True)
    )


def _validate_block(
    task: Task, frame: Any, block: Block, where: str, fixtures_dir: Path | None
) -> list[str]:
    problems: list[str] = []
    if block.source == "fixture":
        assert block.fixture is not None
        path = fixtures_dir / f"{block.fixture}.json" if fixtures_dir else None
        if path is None or not path.is_file():
            problems.append(f"{where}: fixture {block.fixture} not found")
        else:
            fixture = json.loads(path.read_text(encoding="utf-8"))
            for name, want in block.numbers.items():
                assert want.key is not None
                try:
                    have = float(_fixture_value(fixture, want.key))
                except (KeyError, IndexError, TypeError, ValueError):
                    problems.append(f"{where}: {name}: no '{want.key}' in the fixture")
                    continue
                if not math.isclose(
                    have, want.value, rel_tol=FIXTURE_REL, abs_tol=0.0
                ) and not (math.isinf(have) and have == want.value):
                    problems.append(
                        f"{where}: {name} {want.value!r} != fixture {have!r}"
                    )
        if block.kwargs is not None:
            result = _analyze(task, frame, block.kwargs)
            got = result_numbers(result)
            for name, want in block.numbers.items():
                if name in ("conf_low", "conf_high"):
                    continue
                if not within(name, want, got.get(name), block.sign_free):
                    problems.append(
                        f"{where}: inferplot gives {name} {got.get(name)!r}, "
                        f"expected {want.value!r} within {want.tolerance(name)}"
                    )
            low_ok, high_ok = interval_within(
                block, got.get("conf_low"), got.get("conf_high")
            )
            if not (low_ok and high_ok):
                problems.append(
                    f"{where}: inferplot's interval [{got.get('conf_low')!r}, "
                    f"{got.get('conf_high')!r}] is not the expected one"
                )
    elif block.source == "recipe":
        assert block.recipe is not None
        got = RECIPES[block.recipe](frame, **block.recipe_args)
        for name, want in block.numbers.items():
            have = got.get(name)
            if have is None or not math.isclose(
                have, want.value, rel_tol=RECIPE_REL, abs_tol=1e-12
            ):
                problems.append(
                    f"{where}: recipe {block.recipe} gives {name} {have!r}, "
                    f"task {want.value!r}"
                )
            # The other forms of a number (n's come from the task's counts).
            if name != "n" and not _same_values(want.alt, got.get(f"{name}_alt", [])):
                problems.append(
                    f"{where}: recipe {block.recipe} gives {name} alternatives "
                    f"{got.get(f'{name}_alt', [])!r}, task {list(want.alt)!r}"
                )
    for effect in block.effects:
        have = EFFECTS[effect.recipe](frame, **effect.recipe_args)
        alts = [EFFECT_ALT[effect.recipe](have)] if effect.recipe in EFFECT_ALT else []
        if not math.isclose(
            have, effect.estimate.value, rel_tol=RECIPE_REL, abs_tol=1e-12
        ) or not _same_values(effect.estimate.alt, alts):
            problems.append(
                f"{where}: effect {effect.recipe} gives {have!r} (other form "
                f"{alts!r}), task {effect.estimate.value!r} "
                f"{list(effect.estimate.alt)!r}"
            )
    return problems


# --- the two arms ------------------------------------------------------------------

PROMPT_SUFFIX = """

The data are in data.csv in the current directory. Work only in this directory and
save the figure you make here. When you are done, also write answer.json here, a
JSON object with these keys (null where a key does not apply):
- "test": the name of the main statistical test you report, or null if you decided
  not to run one
- "statistic", "df", "df_error" (the second degrees of freedom of an F test),
  "p_value"
- "effect_size" (its name), "estimate", "conf_low", "conf_high"
- "n": the number of observations (or subjects, or pairs) the test used
- "tests_run": the names of every statistical test you ran, including any you do
  not report
- "figure": the file name of the saved figure
- "check_issues": every problem your tools reported about the analysis or the
  figure, as a list of short strings (empty if none)
- "notes": what the researcher should know, in a few sentences
"""


def prompt_for(task: Task) -> str:
    return task.request + "\n" + PROMPT_SUFFIX


# What an agent can see is neutral: the two arm roots are "arm-a" and "arm-b" in an
# order drawn for each run, each holding its environment ("env") and its run
# directories ("run-01", ...); nothing in those paths names the package, the arm or
# the task. The map from names to arms and tasks, the prompts, the transcripts and
# the scores go to a separate results root that no run directory is under.
ARM_NAMES = ("arm-a", "arm-b")


def draw_arm_names() -> dict[str, str]:
    names = list(ARM_NAMES)
    random.SystemRandom().shuffle(names)
    return dict(zip(ARMS, names, strict=True))


@dataclass(frozen=True)
class Options:
    workdir: Path  # the agents' side: one root per arm
    results: Path  # the harness side: prompts, the name map, transcripts, scores
    claude: str = "claude"
    model: str | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    time_limit: int = DEFAULT_TIME_LIMIT
    repeats: int = 1
    arms: tuple[str, ...] = ARMS
    arm_names: Mapping[str, str] = field(
        default_factory=lambda: dict(zip(ARMS, ARM_NAMES, strict=True))
    )


def arm_root(options: Options, arm: str) -> Path:
    return options.workdir / options.arm_names[arm]


def env_dir(options: Options, arm: str) -> Path:
    return arm_root(options, arm) / "env"


def package_version() -> str:
    text = (REPO_ROOT / "src" / "inferplot" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError("no __version__ in src/inferplot/__init__.py")
    return match[1]


def wheel_path(options: Options) -> Path:
    """inferplot's wheel, built into the inferplot arm's root: the environment
    records this path, not the repository's."""
    name = f"inferplot-{package_version()}-py3-none-any.whl"
    return arm_root(options, "inferplot") / "wheels" / name


def env_commands(options: Options) -> dict[str, list[list[str]]]:
    """The commands that create each arm's Python environment (network, no model)."""
    commands: dict[str, list[list[str]]] = {}
    for arm in options.arms:
        venv = env_dir(options, arm)
        packages: list[str] = list(BASE_PACKAGES)
        steps: list[list[str]] = []
        if arm == "inferplot":
            wheel = wheel_path(options)
            steps.append(
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--out-dir",
                    str(wheel.parent),
                    str(REPO_ROOT),
                ]
            )
            packages.append(f"inferplot[pandas] @ {wheel.as_uri()}")
        commands[arm] = [
            *steps,
            ["uv", "venv", "--python", "3.12", str(venv)],
            ["uv", "pip", "install", "--python", str(venv), *packages],
        ]
    return commands


def run_names(tasks: Sequence[Task], options: Options) -> dict[tuple[str, int], str]:
    names: dict[tuple[str, int], str] = {}
    for i, task in enumerate(tasks, start=1):
        for repeat in range(options.repeats):
            suffix = f"-{repeat + 1}" if options.repeats > 1 else ""
            names[(task.id, repeat)] = f"run-{i:02d}{suffix}"
    return names


def result_dir(options: Options, arm: str, task: Task, repeat: int) -> Path:
    suffix = f"-r{repeat}" if options.repeats > 1 else ""
    return options.results / arm / f"{task.id}{suffix}"


def build_arms(
    tasks: Sequence[Task], options: Options
) -> dict[tuple[str, str, int], Path]:
    """Create each run's fresh project directory; return the dirs.

    A baseline directory holds data.csv only. An inferplot directory also holds the
    skill in .claude/skills/inferplot. The prompts and the map from the neutral
    names to arms and tasks go to the results root; nothing of the task file (the
    expected numbers) is copied.
    """
    dirs: dict[tuple[str, str, int], Path] = {}
    prompts = options.results / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    names = run_names(tasks, options)
    runs: dict[str, dict[str, Any]] = {}
    for task in tasks:
        (prompts / f"{task.id}.txt").write_text(prompt_for(task), encoding="utf-8")
        for arm in options.arms:
            for repeat in range(options.repeats):
                target = arm_root(options, arm) / names[(task.id, repeat)]
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True)
                shutil.copyfile(task.data_file, target / "data.csv")
                if arm == "inferplot":
                    shutil.copytree(
                        SKILL_DIR, target / ".claude" / "skills" / "inferplot"
                    )
                dirs[(arm, task.id, repeat)] = target
                runs[str(target.relative_to(options.workdir))] = {
                    "arm": arm,
                    "task": task.id,
                    "repeat": repeat,
                }
    (options.results / "runs.json").write_text(
        json.dumps(
            {"workdir": str(options.workdir), "arms": options.arm_names, "runs": runs},
            indent=2,
        ),
        encoding="utf-8",
    )
    return dirs


def claude_command(task: Task, options: Options) -> list[str]:
    prompt_file = options.results / "prompts" / f"{task.id}.txt"
    command = [
        options.claude,
        "-p",
        prompt_file.read_text(encoding="utf-8"),
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-turns",
        str(options.max_turns),
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        ALLOWED_TOOLS,
        # Only the run directory's own settings (the inferplot arm's skill), none of
        # the owner's user settings, memory or MCP servers.
        "--setting-sources",
        "project",
        "--strict-mcp-config",
    ]
    if options.model:
        command += ["--model", options.model]
    return command


def printable_command(task: Task, options: Options, cwd: Path, arm: str) -> str:
    command = claude_command(task, options)
    prompt_file = options.results / "prompts" / f"{task.id}.txt"
    shown = [shlex.quote(c) for c in command]
    shown[2] = f'"$(cat {shlex.quote(str(prompt_file))})"'
    venv_bin = env_dir(options, arm) / ("Scripts" if os.name == "nt" else "bin")
    # The harness enforces the time limit itself and removes from the environment
    # every variable and PATH entry that names the repository or the results root.
    return (
        f"cd {shlex.quote(str(cwd))} && PATH={shlex.quote(str(venv_bin))}:$PATH "
        + " ".join(shown)
    )


def estimate(n_tasks: int, options: Options) -> str:
    sessions = n_tasks * len(options.arms) * options.repeats
    return (
        f"{sessions} claude sessions ({n_tasks} tasks x {len(options.arms)} arms x "
        f"{options.repeats} repeat{'s' if options.repeats != 1 else ''}); each session "
        f"makes one model request per turn, at most --max-turns {options.max_turns}: "
        f"at most {sessions * options.max_turns} model requests (how many a session "
        f"really takes is not measured yet), each session limited to "
        f"{options.time_limit} s"
    )


# --- scoring -----------------------------------------------------------------------

INFO_CODES = frozenset(
    {
        "pairs_hidden",
        "rows_dropped",
        "points_subsampled",
        "pairwise_conf_level",
        "pairs_without_test",
    }
)
CHECK_CODES = frozenset(
    {
        "small_n",
        "small_expected",
        "pairs_without_test",
        "heavy_ties",
        "rows_dropped",
        "pairs_hidden",
        "infinite_effect_size",
        "estimate_outside_interval",
        "pairwise_conf_level",
        "label_overlap",
        "text_clipped",
        "points_subsampled",
        "path_hidden",
        "paths_crowded",
        "centrality_label_overlap",
        "centrality_label_ambiguous",
        "point_labels_crowded",
        "figure_oversized",
        "cell_labels_hidden",
        "line_label_ambiguous",
        "low_contrast",
        "glyph_missing",
        "axis_changed",
        "data_outside_view",
        "flat_data",
    }
)


@dataclass
class Score:
    task: str
    arm: str
    repeat: int = 0
    ran: bool = False
    failure: str | None = None
    test_reported: str | None = None
    family: str | None = None
    variant: str | None = None  # the accepted analysis the numbers were scored on
    correct_test: bool | None = None
    test_reason: str = ""
    numbers_ok: bool | None = None
    effect_ok: bool | None = None
    effect_matched: str | None = None
    numbers: dict[str, Any] = field(default_factory=dict)
    trap: str | None = None  # the trap analysis whose numbers were reported
    trap_scoring: str = ""
    trap_avoided: bool | None = None  # None: not scored (see read_for)
    read_for: str | None = None
    check_clean: bool | None = None
    check_reason: str = ""
    used_inferplot: bool = False
    contaminated: bool = False
    inferplot_uses: list[str] = field(default_factory=list)
    turns: int | None = None
    tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cost_usd: float | None = None
    duration_s: float | None = None


_SUPERSCRIPTS = str.maketrans(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a",
    "0123456789-+",
)
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?"
_SCIENTIFIC = re.compile(
    rf"({_NUMBER})\s*(?:x|\u00d7|\*|\u00b7|\u22c5|\\times)\s*10\s*(?:\^|\*\*)?\s*"
    r"\(?\s*([-+]?\d+)\s*\)?",
    re.IGNORECASE,
)
_COMPARATOR = re.compile(r"<=|>=|\u2264|\u2265|<|>|=|less than|greater than")


def parse_number(value: Any) -> tuple[float | None, str | None]:
    """A reported number as (value, bound): bound is "<" or ">" for an inequality
    ("p < .001", "p<.01"), else None. Text before "=" is a label ("t(9) = -4.06"),
    "2.83 x 10^-3" and the same with a multiplication sign and superscripts are
    0.00283, a one-element list
    is its element."""
    if isinstance(value, list | tuple) and len(value) == 1:
        return parse_number(value[0])
    if value is None or isinstance(value, bool):
        return None, None
    if isinstance(value, int | float):
        number = float(value)
        return (None, None) if math.isnan(number) else (number, None)
    if not isinstance(value, str):
        return None, None
    text = value.translate(_SUPERSCRIPTS)
    text = unicodedata.normalize("NFKC", text).replace("\u2212", "-")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)  # 2,201 (not 0,05)
    bound: str | None = None
    match = _COMPARATOR.search(text)
    if match:
        symbol = match[0]
        if symbol in ("<", "<=", "\u2264", "less than"):
            bound = "<"
        elif symbol in (">", ">=", "\u2265", "greater than"):
            bound = ">"
        text = text[match.end() :]
    scientific = _SCIENTIFIC.search(text)
    if scientific:
        return float(scientific[1]) * 10.0 ** int(scientific[2]), bound
    found = re.search(_NUMBER, text, re.IGNORECASE)
    if found:
        return float(found[0]), bound
    return None, None


def _dfs(answer: Mapping[str, Any]) -> tuple[float | None, float | None]:
    df, df_error = answer.get("df"), answer.get("df_error")
    if isinstance(df, list) and len(df) == 2:
        return parse_number(df[0])[0], parse_number(df[1])[0]
    if isinstance(df, str) and "," in df and df_error is None:
        first, second = df.split(",", 1)
        return parse_number(first)[0], parse_number(second)[0]
    return parse_number(df)[0], parse_number(df_error)[0]


def _p_bound_ok(want: Expected, bound: float) -> bool:
    """A p reported as "< bound" is right when the true p is below the bound and the
    bound is as tight as conventional reporting: .001 or below (APA reports exact p
    above .001), or within the p tolerance of the true p."""
    if want.value > bound * (1 + 1e-9):
        return False
    return bound <= P_FLOOR or within("p_value", want, bound, sign_free=False)


def compare_numbers(block: Block, answer: Mapping[str, Any]) -> dict[str, Any]:
    """Per expected number: (expected, reported, ok), and the bound if one was
    reported ("<" or ">")."""
    df, df_error = _dfs(answer)
    if (
        "df" in block.numbers
        and "df_error" not in block.numbers
        and df is None
        and df_error is not None
    ):
        df = df_error  # a test with one df reported in the second field
    reported: dict[str, float | None] = {"df": df, "df_error": df_error}
    bounds: dict[str, str | None] = {}
    for name in (
        "statistic",
        "p_value",
        "n",
        "estimate",
        "conf_low",
        "conf_high",
        "bf10",
    ):
        reported[name], bounds[name] = parse_number(answer.get(name))
    detail: dict[str, Any] = {}
    low_ok, high_ok = interval_within(
        block, reported["conf_low"], reported["conf_high"]
    )
    for name, want in block.numbers.items():
        got = reported.get(name)
        bound = bounds.get(name)
        if name == "conf_low":
            ok = low_ok and got is not None
        elif name == "conf_high":
            ok = high_ok and got is not None
        elif name == "p_value" and bound == "<" and got is not None:
            ok = _p_bound_ok(want, got)
        else:
            ok = within(name, want, got, block.sign_free)
        detail[name] = {"expected": want.value, "reported": got, "ok": ok}
        if bound:
            detail[name]["bound"] = bound
    return detail


def _core_ok(detail: Mapping[str, Any]) -> bool | None:
    core = [v["ok"] for k, v in detail.items() if k in CORE]
    return all(core) if core else None


def _rank(detail: Mapping[str, Any]) -> tuple[bool, bool, bool, int, int]:
    """Order of the variants: right as a whole, then p, then the statistic, then
    the count of right core numbers, then of right effect numbers."""
    core = {k: bool(v["ok"]) for k, v in detail.items() if k in CORE}
    effect = sum(1 for k, v in detail.items() if k in EFFECT and v["ok"])
    return (
        bool(core) and all(core.values()),
        core.get("p_value", False),
        core.get("statistic", False),
        sum(core.values()),
        effect,
    )


def effect_score(
    blocks: Sequence[Block], answer: Mapping[str, Any]
) -> tuple[bool | None, str | None]:
    """The effect size is right when it is the oracle's (estimate and interval) or
    one of the standard alternatives (estimate only) of the chosen variant or of
    another accepted variant of the same test (the effect size does not depend on
    how the p-value is computed). None when none of them has an effect size."""
    estimate = parse_number(answer.get("estimate"))[0]
    scored = False
    for block in blocks:
        detail = compare_numbers(block, answer)
        own = [v["ok"] for k, v in detail.items() if k in EFFECT]
        scored = scored or bool(own) or bool(block.effects)
        if own and all(own):
            return True, "the oracle's effect size"
        for alternative in block.effects:
            if within("estimate", alternative.estimate, estimate, sign_free=True):
                return True, alternative.label
    return (False, None) if scored else (None, None)


# --- contamination: did a run use inferplot? ---------------------------------------
# Real use only: importing it, installing it or running it, in a shell command, in
# a file the agent wrote or in a script left in its directory. A path that merely
# contains the word (a directory name) is not use.

_PACKAGE = r"(?<![\w./\\-])inferplot(?:\[[\w,\s]*\])?(?![\w./\\-])"
_USE_PATTERNS = (
    re.compile(
        r"(?:^|[\s;:(\"'`])(?:import\s+inferplot\b|from\s+inferplot(?:\.\w+)*\s+import\b)",
        re.MULTILINE,
    ),
    re.compile(r"(?:import_module|__import__)\(\s*[\"']inferplot\b"),
    re.compile(
        r"\b(?:pip3?|uv\s+pip|uv|pipx|poetry|conda|mamba|micromamba)\s+(?:install|add)"
        rf"\b[^\n;&|]*?(?:{_PACKAGE}|inferplot-[\w.+-]+\.whl|inferplot\s*@|"
        r"github\.com/[\w.-]+/inferplot\b|/inferplot(?:\.git)?(?=[\s\"']|$))",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"-m\s+inferplot\b"),
)
CODE_SUFFIXES = (".py", ".ipynb", ".sh")


def inferplot_uses(text: str) -> list[str]:
    """The snippets of ``text`` that import, install or run inferplot."""
    found: list[str] = []
    for pattern in _USE_PATTERNS:
        found.extend(m[0].strip() for m in pattern.finditer(text))
    return found


def _written_code(tool_input: Mapping[str, Any]) -> list[str]:
    """Shell commands and file contents in one tool call (not the paths it names)."""
    texts: list[str] = []
    for key in ("command", "content", "new_string", "new_source"):
        value = tool_input.get(key)
        if isinstance(value, str):
            texts.append(value)
    for edit in tool_input.get("edits") or []:
        if isinstance(edit, dict) and isinstance(edit.get("new_string"), str):
            texts.append(edit["new_string"])
    return texts


def parse_transcript(text: str) -> dict[str, Any]:
    """Facts from a `claude -p --output-format stream-json --verbose` transcript."""
    facts: dict[str, Any] = {
        "result": None,
        "tool_calls": 0,
        "commands": [],
        "code": [],
    }
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            facts["result"] = event
        elif event.get("type") == "assistant":
            content = (event.get("message") or {}).get("content") or []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_input = item.get("input") or {}
                    facts["tool_calls"] += 1
                    facts["commands"].append(json.dumps(tool_input))
                    if isinstance(tool_input, dict):
                        facts["code"].extend(_written_code(tool_input))
    return facts


def run_uses(facts: Mapping[str, Any], run_path: Path) -> list[str]:
    uses: list[str] = []
    for code in facts["code"]:
        uses.extend(inferplot_uses(code))
    if run_path.is_dir():
        for path in sorted(run_path.rglob("*")):
            if path.suffix in CODE_SUFFIXES and ".claude" not in path.parts:
                try:
                    uses.extend(inferplot_uses(path.read_text(encoding="utf-8")))
                except (OSError, UnicodeDecodeError):
                    continue
    return uses


def embedded_result_codes(path: Path) -> list[str] | None:
    """The statistical check() codes of a result inferplot embedded in a figure."""
    import inferplot as ip

    payload: str | None = None
    try:
        if path.suffix == ".svg":
            root = ET.parse(path).getroot()
            found = root.find(".//{http://purl.org/dc/elements/1.1/}description")
            payload = found.text if found is not None else None
        elif path.suffix == ".png":
            from PIL import Image

            with Image.open(path) as image:
                payload = image.info.get("Description")
    except (OSError, ET.ParseError):
        return None
    if not payload:
        return None
    try:
        result = ip.StatsResult.from_json(payload)
    except (ValueError, KeyError, TypeError):
        return None
    return sorted({issue.code for issue in result.check()})


def _reported_codes(answer: Mapping[str, Any]) -> list[str]:
    issues = answer.get("check_issues") or []
    if not isinstance(issues, list):
        issues = [issues]
    text = " ".join(str(i) for i in issues)
    return sorted(code for code in CHECK_CODES if code in text)


def _trap_avoided(task: Task, score: Score) -> bool | None:
    """Whether the run avoided the task's trap; None when it cannot be told from
    answer.json and the transcript (the trap is then read, not counted)."""
    if task.trap_scoring in ("none", "manual") or score.correct_test is None:
        return None
    fell = score.trap is not None or score.test_reason.startswith(("rejected", "trap"))
    if fell:
        return False
    if not score.correct_test:
        return None  # an unrecognized test: neither avoided nor fallen into
    if task.trap_scoring == "n":
        n = score.numbers.get("n")
        if n is None or n["reported"] is None:
            return None
        return bool(n["ok"])
    return True


def score_run(
    task: Task,
    arm: str,
    transcript: str,
    run_path: Path,
    *,
    exit_code: int | None,
    timed_out: bool = False,
    repeat: int = 0,
) -> Score:
    """Score one run from its transcript and the files in its directory."""
    score = Score(task=task.id, arm=arm, repeat=repeat)
    score.trap_scoring = task.trap_scoring
    score.read_for = task.read_for
    facts = parse_transcript(transcript)
    result = facts["result"] or {}
    usage = result.get("usage") or {}
    score.turns = result.get("num_turns")
    score.tool_calls = facts["tool_calls"]
    score.input_tokens = usage.get("input_tokens")
    score.output_tokens = usage.get("output_tokens")
    score.cache_read_tokens = usage.get("cache_read_input_tokens")
    score.cache_creation_tokens = usage.get("cache_creation_input_tokens")
    score.cost_usd = result.get("total_cost_usd")
    if result.get("duration_ms") is not None:
        score.duration_s = result["duration_ms"] / 1000
    score.inferplot_uses = run_uses(facts, run_path)
    score.used_inferplot = bool(score.inferplot_uses)
    score.contaminated = arm == "baseline" and score.used_inferplot

    answer_path = run_path / "answer.json"
    answer: dict[str, Any] | None = None
    if answer_path.is_file():
        try:
            loaded = json.loads(answer_path.read_text(encoding="utf-8"))
            answer = loaded if isinstance(loaded, dict) else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            answer = None
    if timed_out:
        score.failure = "time limit"
    elif exit_code != 0:
        score.failure = f"exit code {exit_code}"
    elif not result:
        score.failure = "no result in the transcript"
    elif result.get("is_error") or result.get("subtype") not in (None, "success"):
        score.failure = f"session ended with {result.get('subtype', 'an error')}"
    elif answer is None:
        score.failure = "no readable answer.json"
    score.ran = score.failure is None
    if answer is None:
        return score

    name = answer.get("test")
    score.test_reported = None if name is None else str(name)
    named = families_of(score.test_reported)
    text = normalize_name(score.test_reported or "")
    # Every accepted variant of a named family, scored as a whole; a variant with a
    # qualifier ("log") counts when the name says it or the numbers are its own.
    candidates: list[tuple[Block, dict[str, Any]]] = []
    for block in task.accepted:
        if block.family not in named:
            continue
        detail = compare_numbers(block, answer)
        if (
            block.qualifier
            and not any(_has(q, text) for q in block.qualifier)
            and _core_ok(detail) is not True
        ):
            continue
        candidates.append((block, detail))
    if candidates:
        best, detail = max(candidates, key=lambda c: _rank(c[1]))
        score.family = best.family
        score.variant = best.label
        score.correct_test = True
        score.test_reason = (
            f"accepted: {FAMILIES.get(best.family, Family('no test', ())).label}"
        )
        if detail:
            score.numbers = detail
            score.numbers_ok = _core_ok(detail)
        same_test = [b for b in task.accepted if b.family == best.family]
        score.effect_ok, score.effect_matched = effect_score(
            [best, *(b for b in same_test if b is not best)], answer
        )
    else:
        score.correct_test = False
        rejected = [f for f in task.rejected if f in named]
        trap_families = [b.family for b in task.traps if b.family in named]
        if rejected:
            score.test_reason = f"rejected: {FAMILIES[rejected[0]].label}"
        elif trap_families:
            score.test_reason = f"trap: {FAMILIES[trap_families[0]].label}"
        else:
            score.test_reason = "not recognized as an accepted test"

    # Traps: the reported statistic, df and p are those of a trap analysis (the df
    # tell a sphericity-corrected F from the uncorrected one with the same F), and
    # not those of an accepted variant.
    if score.numbers_ok is not True:
        for block in task.traps:
            trap_detail = compare_numbers(block, answer)
            core = [
                v["ok"]
                for k, v in trap_detail.items()
                if k in ("statistic", "df", "df_error", "p_value")
            ]
            if core and all(core):
                score.trap = block.label
                break
    score.trap_avoided = _trap_avoided(task, score)

    if arm == "inferplot" and task.raises is None:
        score.check_clean, score.check_reason = _check_clean(task, answer, run_path)
    return score


def _check_clean(
    task: Task, answer: Mapping[str, Any], run_path: Path
) -> tuple[bool, str]:
    allowed = set(task.check_codes) | INFO_CODES
    figure = answer.get("figure")
    if not figure or not (run_path / str(figure)).is_file():
        return False, "no saved figure"
    reported = _reported_codes(answer)
    extra = sorted(set(reported) - allowed)
    if extra:
        return False, f"figure problems left: {', '.join(extra)}"
    missing = sorted(set(task.check_codes) - INFO_CODES - set(reported))
    if missing:
        return False, f"not reported: {', '.join(missing)}"
    embedded = embedded_result_codes(run_path / str(figure))
    if embedded is None:
        return True, "self-reported (no inferplot result embedded in the figure)"
    unexpected = sorted(set(embedded) - allowed)
    if unexpected:
        return (
            False,
            f"result problems not expected for these data: {', '.join(unexpected)}",
        )
    return True, "clean"


def summarize(scores: Sequence[Score]) -> str:
    """One row per arm. "effect" is whether the effect size is the oracle's or a
    listed alternative: it mostly says which tool was used, not how good the
    analysis is. "trap avoided" counts only the runs whose trap can be scored;
    "to read" counts the runs whose trap must be read in the answer."""
    lines = [
        "arm        runs  ran  correct test  numbers  effect*  check clean  "
        "trap avoided  to read  used inferplot  turns (median)  output tokens (sum)"
    ]
    for arm in ARMS:
        mine = [s for s in scores if s.arm == arm]
        if not mine:
            continue

        def count(attr: str, rows: Sequence[Score] = mine) -> str:
            values = [getattr(s, attr) for s in rows if getattr(s, attr) is not None]
            return f"{sum(bool(v) for v in values)}/{len(values)}"

        turns = sorted(s.turns for s in mine if s.turns is not None)
        median = turns[len(turns) // 2] if turns else "-"
        tokens = sum(s.output_tokens or 0 for s in mine)
        to_read = sum(1 for s in mine if s.trap_scoring == "manual")
        used = sum(1 for s in mine if s.used_inferplot)
        lines.append(
            f"{arm:<10} {len(mine):>4}  {count('ran'):>4} {count('correct_test'):>13} "
            f"{count('numbers_ok'):>8} {count('effect_ok'):>8} "
            f"{count('check_clean'):>12} {count('trap_avoided'):>13} {to_read:>8} "
            f"{used:>15} {median!s:>15} {tokens:>20}"
        )
    lines.append(
        "* effect: the oracle's effect size or a listed alternative; it says more "
        "about the tool than about the analysis (README)."
    )
    manual = sorted(
        {(s.task, s.read_for) for s in scores if s.trap_scoring == "manual"}
    )
    if manual:
        lines.append("traps to read in answer.json (not scored automatically):")
        lines.extend(f"  {task}: {read_for}" for task, read_for in manual)
    contaminated = [f"{s.task} (repeat {s.repeat})" for s in scores if s.contaminated]
    if contaminated:
        lines.append("baseline runs that used inferplot: " + ", ".join(contaminated))
    return "\n".join(lines)


# --- running -----------------------------------------------------------------------


def _end_group(process: subprocess.Popen[Any]) -> None:
    """End the session's whole process group: the shells it started and anything
    they left running in the background (not a process that left the group)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        for _ in range(50):
            try:
                os.killpg(process.pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.1)


def _call_claude(
    command: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: int
) -> tuple[int | None, str, bool]:
    """The only place a model is called. Returns (exit code, stdout, timed out).

    The session starts its own process group, which is ended when the session
    exits or its time is up, so nothing it started outlives it. Its output goes to
    a file, so a background process holding the pipe cannot keep the run waiting.
    Process groups are POSIX; on Windows the run is refused rather than left
    without its time limit on the session's children.
    """
    if os.name == "nt":
        raise SystemExit("refused: the evaluation needs POSIX process groups")
    with tempfile.TemporaryFile("w+", encoding="utf-8") as out:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _end_group(process)
            process.wait()
        out.seek(0)
        text = out.read()
    return (None if timed_out else process.returncode), text, timed_out


def _build_envs(options: Options) -> None:
    for arm, commands in env_commands(options).items():
        for command in commands:
            subprocess.run(command, check=True)
            built = arm == "inferplot" and command[:2] == ["uv", "build"]
            if built and not wheel_path(options).is_file():
                raise SystemExit(f"the wheel {wheel_path(options).name} was not built")
    # Installed; the wheel goes, so the baseline's agents find no file named for
    # the package next to their own directories.
    if "inferplot" in options.arms:
        shutil.rmtree(wheel_path(options).parent)


# Variables an agent's environment never inherits from the harness (it runs under
# `uv run` in the repository).
DROPPED_ENV = frozenset(
    {
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PROJECT",
        "PWD",
        "OLDPWD",
    }
)


def _hidden_roots(options: Options) -> list[str]:
    roots = {str(p) for p in (REPO_ROOT, options.results)}
    roots |= {str(p.resolve()) for p in (REPO_ROOT, options.results)}
    return sorted(roots)


def _run_env(options: Options, arm: str, cwd: Path | None = None) -> dict[str, str]:
    """The harness's environment without anything that names the repository or the
    results root, with the arm's environment first on PATH."""
    hidden = _hidden_roots(options)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in DROPPED_ENV and not any(h in value for h in hidden)
    }
    path = [
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and not any(p.startswith(h) for h in hidden)
    ]
    venv = env_dir(options, arm)
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    env["PATH"] = os.pathsep.join([str(bin_dir), *path])
    env["VIRTUAL_ENV"] = str(venv)
    if cwd is not None:
        env["PWD"] = str(cwd)
    return env


def real_run(
    tasks: Sequence[Task], options: Options, out: Callable[[str], None]
) -> list[Score]:
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        raise SystemExit("refused: the evaluation does not run in CI")
    if shutil.which(options.claude) is None:
        raise SystemExit(f"refused: '{options.claude}' is not on PATH")
    out(f"real run: {estimate(len(tasks), options)}")
    _build_envs(options)
    dirs = build_arms(tasks, options)
    scores: list[Score] = []
    for task in tasks:
        for arm in options.arms:
            for repeat in range(options.repeats):
                cwd = dirs[(arm, task.id, repeat)]
                code, transcript, timed_out = _call_claude(
                    claude_command(task, options),
                    cwd,
                    _run_env(options, arm, cwd),
                    options.time_limit,
                )
                # Move the run out of the agents' side before the next session
                # starts, so no later agent finds this one's files.
                keep = result_dir(options, arm, task, repeat)
                if keep.exists():
                    shutil.rmtree(keep)
                shutil.copytree(cwd, keep, ignore=shutil.ignore_patterns(".claude"))
                shutil.rmtree(cwd)
                (keep / "transcript.jsonl").write_text(transcript, encoding="utf-8")
                score = score_run(
                    task,
                    arm,
                    transcript,
                    keep,
                    exit_code=code,
                    timed_out=timed_out,
                    repeat=repeat,
                )
                (keep / "score.json").write_text(
                    json.dumps(asdict(score), indent=2), encoding="utf-8"
                )
                scores.append(score)
                out(
                    f"{arm:<9} {task.id:<34} ran={score.ran} test={score.correct_test} "
                    f"numbers={score.numbers_ok} trap={score.trap}"
                )
    (options.results / "scores.json").write_text(
        json.dumps([asdict(s) for s in scores], indent=2), encoding="utf-8"
    )
    out(summarize(scores))
    return scores


def dry_run(tasks: Sequence[Task], options: Options, out: Callable[[str], None]) -> int:
    failures = 0
    for task in tasks:
        problems = validate_task(task)
        n_numbers = sum(len(b.numbers) for b in (*task.accepted, *task.traps))
        if problems:
            failures += 1
            out(f"FAIL {task.id}")
            for problem in problems:
                out(f"     {problem}")
        else:
            out(
                f"ok   {task.id:<34} {task.function:<12} {len(task.accepted):>2} "
                f"accepted {n_numbers:>3} numbers  trap ({task.trap_scoring}): "
                f"{task.trap}"
            )
    dirs = build_arms(tasks, options)
    out("")
    out(f"arms built in {options.workdir} ({len(dirs)} run directories)")
    out(f"results root {options.results} (arm and run names: runs.json)")
    out("commands a real run would execute (environments first; network, no model):")
    for arm, commands in env_commands(options).items():
        for command in commands:
            out(f"  [{arm}] " + shlex.join(command))
    if "inferplot" in options.arms:
        out(f"  [inferplot] then removes {wheel_path(options).parent} (installed)")
    out("then, once per task, arm and repeat (a model session each):")
    for (arm, task_id, _repeat), cwd in sorted(dirs.items()):
        task = next(t for t in tasks if t.id == task_id)
        out("  " + printable_command(task, options, cwd, arm))
    out("")
    out(f"a full evaluation would make {estimate(len(tasks), options)}")
    out(f"no model call was made. {len(tasks) - failures} of {len(tasks)} tasks valid.")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(
        description="inferplot agent evaluation. Dry run unless "
        f"{SPEND_FLAG} is given (see evals/README.md)."
    )
    parser.add_argument("--dry-run", action="store_true", help="the default")
    parser.add_argument(
        "--real", action="store_true", help=f"run the agents (needs {SPEND_FLAG})"
    )
    parser.add_argument(
        SPEND_FLAG,
        dest="spend",
        action="store_true",
        help="confirm that the run spends the owner's Claude usage",
    )
    parser.add_argument(
        "--tasks", default="", help="comma-separated task ids (default all)"
    )
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--claude", default="claude", help="the claude executable")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="where the agents' directories go (default: a new temporary one)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="where prompts, transcripts and scores go, outside --workdir "
        "(default: a new temporary one)",
    )
    args = parser.parse_args(argv)

    if args.real and not args.spend:
        parser.error(
            f"--real needs {SPEND_FLAG}: a real run spends the owner's Claude usage"
        )
    if args.dry_run and args.spend:
        parser.error(f"--dry-run and {SPEND_FLAG} contradict each other")
    arms = tuple(a for a in args.arms.split(",") if a)
    if not arms or any(a not in ARMS for a in arms):
        parser.error(f"--arms takes {', '.join(ARMS)}")
    workdir = (args.workdir or Path(tempfile.mkdtemp(prefix="agent-eval-"))).absolute()
    results = (
        args.results or Path(tempfile.mkdtemp(prefix="eval-results-"))
    ).absolute()
    repo = REPO_ROOT.resolve()
    for label, path in (("--workdir", workdir), ("--results", results)):
        if path.resolve().is_relative_to(repo):
            parser.error(
                f"{label} must be outside the repository, "
                "where agents cannot read the tasks"
            )
    if results.resolve().is_relative_to(
        workdir.resolve()
    ) or workdir.resolve().is_relative_to(results.resolve()):
        parser.error("--results and --workdir must not contain each other")
    options = Options(
        workdir=workdir,
        results=results,
        claude=args.claude,
        model=args.model,
        max_turns=args.max_turns,
        time_limit=args.time_limit,
        repeats=args.repeats,
        arms=arms,
        arm_names=draw_arm_names(),
    )
    if "inferplot" in str(workdir).lower():
        out(
            f"warning: the agents' paths start with {workdir}, which names the "
            "package; choose a --workdir that does not (README)"
        )
    try:
        tasks = load_tasks([t for t in args.tasks.split(",") if t])
    except TaskError as error:
        out(f"task error: {error}")
        return 1
    if args.spend:
        scores = real_run(tasks, options, out)
        return 0 if scores else 1
    return dry_run(tasks, options, out)


if __name__ == "__main__":
    sys.exit(main())
