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
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
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
    Phrases match at the start of a word of the normalized name.
    """

    label: str
    groups: tuple[tuple[str, ...], ...]
    none: tuple[str, ...] = ()


FAMILIES: Mapping[str, Family] = {
    "welch_t": Family(
        "Welch's t-test",
        (("welch",),),
        ("anova", "analysis of", "one way", "oneway", "f test", "paired"),
    ),
    "student_t": Family(
        "Student's t-test (equal variances)",
        (("student", "pooled", "equal var true", "var equal true"),),
        ("welch", "paired", "one sample", "anova"),
    ),
    "welch_anova": Family(
        "Welch's ANOVA",
        (("welch",), ("anova", "analysis of means", "one way", "oneway", "f test")),
    ),
    "oneway_anova": Family(
        "one-way ANOVA (equal variances)",
        (("anova", "f oneway", "f test"),),
        (
            "welch",
            "repeated",
            "rm ",
            "friedman",
            "kruskal",
            "two way",
            "factorial",
            "trimmed",
            "brown",
            "within",
        ),
    ),
    "kruskal_wallis": Family("Kruskal-Wallis test", (("kruskal",),)),
    "mann_whitney": Family(
        "Mann-Whitney U / Wilcoxon rank-sum test",
        (("mann", "whitney", "rank sum", "ranksum", "u test"),),
        ("signed",),
    ),
    "paired_t": Family(
        "paired t-test",
        (
            ("paired", "dependent", "related", "ttest rel", "repeated"),
            ("t test", "ttest", "t statistic"),
        ),
        ("wilcoxon", "yuen", "trimmed", "welch", "independent"),
    ),
    "wilcoxon_signed_rank": Family(
        "Wilcoxon signed-rank test",
        (("signed rank", "signedrank", "wilcoxon"),),
        ("rank sum", "mann"),
    ),
    "rm_anova": Family(
        "repeated-measures ANOVA",
        (
            ("repeated", "rm anova", "within subject", "anovarm", "anova rm"),
            ("anova", "anovarm", "f test"),
        ),
        ("friedman", "trimmed", "robust", "kruskal"),
    ),
    "friedman": Family("Friedman test", (("friedman",),)),
    "one_sample_t": Family(
        "one-sample t-test",
        (("one sample", "single sample", "1samp", "1 sample"), ("t test", "ttest")),
        ("wilcoxon", "signed", "bootstrap", "trimmed", "bayes"),
    ),
    "pearson": Family(
        "Pearson correlation", (("pearson",),), ("chi", "spearman", "winsor")
    ),
    "spearman": Family("Spearman correlation", (("spearman",),)),
    "winsorized": Family("winsorized correlation", (("winsor",),)),
    "yuen": Family("Yuen's trimmed-means test", (("yuen",),)),
    "trimmed_anova": Family(
        "trimmed-means ANOVA", (("trimmed", "t1way"), ("anova", "t1way", "f test"))
    ),
    "trimmed_bootstrap": Family(
        "bootstrap-t test of the trimmed mean", (("trimmed",), ("bootstrap", "test"))
    ),
    "mcnemar": Family("McNemar's test", (("mcnemar",),)),
    "chi_square_independence": Family(
        "chi-square test of independence",
        (("chi",),),
        ("goodness", "given prob", "gof", "mcnemar", "one way", "one sample"),
    ),
    "chi_square_gof": Family(
        "chi-square goodness-of-fit test",
        (("chi",),),
        ("contingency", "independence", "mcnemar"),
    ),
    "fisher": Family("Fisher's exact test", (("fisher",),)),
    "regression": Family(
        "linear regression with covariates",
        (
            (
                "regression",
                "ols",
                "linear model",
                "lm",
                "multiple",
                "least squares",
                "ancova",
            ),
        ),
    ),
    "two_way_anova": Family(
        "two-way (factorial) ANOVA",
        (("two way", "twoway", "2 way", "factorial", "interaction", "two factor"),),
    ),
    "bayes_factor": Family("a Bayes factor", (("bayes",),)),
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


def normalize_name(text: str) -> str:
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[-_'`.,()/=:;\[\]{}\"]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has(phrase: str, text: str) -> bool:
    return re.search(r"(?:^|\s)" + re.escape(phrase), text) is not None


def family_matches(family_id: str, name: str | None) -> bool:
    if family_id == NO_TEST:
        return name is None or normalize_name(name) in NO_TEST_WORDS
    if name is None:
        return False
    family = FAMILIES[family_id]
    text = normalize_name(name)
    if any(_has(p, text) for p in family.none):
        return False
    return all(any(_has(p, text) for p in group) for group in family.groups)


# --- recipes: alternative analyses computed with SciPy ------------------------------
# Some accepted answers (a variant R does not run by default, such as Yates'
# correction or the exact McNemar test) and some trap answers (the analysis a trap
# leads to) have no ggstatsplot fixture. Their numbers are computed here from the
# task's dataset; the dry run recomputes them and compares with the task file.


def _frame(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path)


def _two(df: Any, x: str, y: str, levels: Sequence[str]) -> tuple[Any, Any]:
    clean = df[[x, y]].dropna()
    keys = clean[x].astype(str)
    a = clean.loc[keys == levels[0], y].to_numpy(dtype=float)
    b = clean.loc[keys == levels[1], y].to_numpy(dtype=float)
    return a, b


def _welch_df(a: Any, b: Any) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)))


def recipe_student_t(
    df: Any, *, x: str, y: str, levels: Sequence[str]
) -> dict[str, float]:
    from scipy import stats

    a, b = _two(df, x, y, levels)
    res = stats.ttest_ind(a, b, equal_var=True)
    return {
        "statistic": float(res.statistic),
        "df": float(len(a) + len(b) - 2),
        "p_value": float(res.pvalue),
        "n": float(len(a) + len(b)),
    }


def recipe_welch_t(
    df: Any, *, x: str, y: str, levels: Sequence[str]
) -> dict[str, float]:
    from scipy import stats

    a, b = _two(df, x, y, levels)
    res = stats.ttest_ind(a, b, equal_var=False)
    return {
        "statistic": float(res.statistic),
        "df": _welch_df(a, b),
        "p_value": float(res.pvalue),
        "n": float(len(a) + len(b)),
    }


def recipe_paired_by_row_order(
    df: Any, *, x: str, y: str, levels: Sequence[str]
) -> dict[str, float]:
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


def recipe_oneway_anova(df: Any, *, x: str, y: str) -> dict[str, float]:
    from scipy import stats

    clean = df[[x, y]].dropna()
    groups = [g[y].to_numpy(dtype=float) for _, g in clean.groupby(x)]
    res = stats.f_oneway(*groups)
    return {
        "statistic": float(res.statistic),
        "df": float(len(groups) - 1),
        "df_error": float(len(clean) - len(groups)),
        "p_value": float(res.pvalue),
        "n": float(len(clean)),
    }


def recipe_rm_anova_uncorrected(
    df: Any, *, x: str, y: str, subject: str
) -> dict[str, float]:
    """Repeated-measures ANOVA without a sphericity correction, complete subjects."""
    import numpy as np
    from scipy import stats

    wide = df.pivot_table(index=subject, columns=x, values=y, aggfunc="first").dropna()
    data = wide.to_numpy(dtype=float)
    n, k = data.shape
    grand = data.mean()
    ss_cond = n * float(((data.mean(axis=0) - grand) ** 2).sum())
    ss_subj = k * float(((data.mean(axis=1) - grand) ** 2).sum())
    ss_total = float(((data - grand) ** 2).sum())
    ss_err = ss_total - ss_cond - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f = (ss_cond / df1) / (ss_err / df2)
    if not np.isfinite(f):
        raise ValueError("rm_anova_uncorrected: no error variance")
    return {
        "statistic": f,
        "df": float(df1),
        "df_error": float(df2),
        "p_value": float(stats.f.sf(f, df1, df2)),
        "n": float(n),
    }


def recipe_kruskal_untied(df: Any, *, x: str, y: str) -> dict[str, float]:
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


def recipe_mcnemar_corrected(df: Any, *, x: str, y: str) -> dict[str, float]:
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


def recipe_mcnemar_exact(df: Any, *, x: str, y: str) -> dict[str, float]:
    from scipy import stats

    t = _table(df, x, y)
    b, c = int(t[0, 1]), int(t[1, 0])
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return {"p_value": float(p), "n": float(t.sum())}


def recipe_chi2_yates(df: Any, *, x: str, y: str) -> dict[str, float]:
    from scipy import stats

    t = _table(df, x, y)
    res = stats.chi2_contingency(t, correction=True)
    return {
        "statistic": float(res.statistic),
        "df": float(res.dof),
        "p_value": float(res.pvalue),
        "n": float(t.sum()),
    }


def recipe_fisher_2x2(df: Any, *, x: str, y: str) -> dict[str, float]:
    from scipy import stats

    t = _table(df, x, y)
    res = stats.fisher_exact(t)
    return {"p_value": float(res.pvalue), "n": float(t.sum())}


def recipe_chi2_rows_unweighted(df: Any, *, x: str, y: str) -> dict[str, float]:
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


def recipe_ols_term(
    df: Any, *, y: str, predictors: Sequence[str], term: str
) -> dict[str, float]:
    """Ordinary least squares y ~ predictors; the t-test of one coefficient."""
    import numpy as np
    from scipy import stats

    clean = df[[y, *predictors]].dropna()
    design = np.column_stack(
        [np.ones(len(clean))] + [clean[p].to_numpy(dtype=float) for p in predictors]
    )
    target = clean[y].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    dof = len(clean) - design.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    j = 1 + list(predictors).index(term)
    t = float(coef[j] / math.sqrt(cov[j, j]))
    return {
        "statistic": t,
        "df": float(dof),
        "p_value": float(2 * stats.t.sf(abs(t), dof)),
        "estimate": float(coef[j]),
        "n": float(len(clean)),
    }


def recipe_two_way_interaction(df: Any, *, y: str, a: str, b: str) -> dict[str, float]:
    """F-test of the a x b interaction in a two-way ANOVA (balanced: any SS type)."""
    import numpy as np
    from scipy import stats

    clean = df[[y, a, b]].dropna()
    fa = clean[a].astype(str)
    fb = clean[b].astype(str)
    target = clean[y].to_numpy(dtype=float)

    def dummies(codes: Any) -> Any:
        levels = sorted(set(codes))
        return np.column_stack(
            [(codes == lv).to_numpy(dtype=float) for lv in levels[1:]]
        )

    da, db = dummies(fa), dummies(fb)
    inter = np.column_stack(
        [da[:, i] * db[:, j] for i in range(da.shape[1]) for j in range(db.shape[1])]
    )
    ones = np.ones((len(clean), 1))
    additive = np.hstack([ones, da, db])
    full = np.hstack([additive, inter])

    def rss(design: Any) -> float:
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        r = target - design @ coef
        return float(r @ r)

    df_int = inter.shape[1]
    df_res = len(clean) - full.shape[1]
    f = ((rss(additive) - rss(full)) / df_int) / (rss(full) / df_res)
    return {
        "statistic": f,
        "df": float(df_int),
        "df_error": float(df_res),
        "p_value": float(stats.f.sf(f, df_int, df_res)),
        "n": float(len(clean)),
    }


RECIPES: Mapping[str, Callable[..., dict[str, float]]] = {
    "student_t": recipe_student_t,
    "welch_t": recipe_welch_t,
    "paired_by_row_order": recipe_paired_by_row_order,
    "oneway_anova": recipe_oneway_anova,
    "rm_anova_uncorrected": recipe_rm_anova_uncorrected,
    "kruskal_untied": recipe_kruskal_untied,
    "mcnemar_corrected": recipe_mcnemar_corrected,
    "mcnemar_exact": recipe_mcnemar_exact,
    "chi2_yates": recipe_chi2_yates,
    "fisher_2x2": recipe_fisher_2x2,
    "chi2_rows_unweighted": recipe_chi2_rows_unweighted,
    "ols_term": recipe_ols_term,
    "two_way_interaction": recipe_two_way_interaction,
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


@dataclass(frozen=True)
class Task:
    id: str
    path: Path
    function: str
    trap: str
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


def _block(raw: Any, where: str) -> Block:
    if not isinstance(raw, dict):
        raise TaskError(f"{where}: must be a table")
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
    )


TASK_KEYS = {
    "id",
    "function",
    "trap",
    "request",
    "design",
    "correct_rule",
    "data",
    "reference",
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
    data = _need(raw, "data", dict, where)
    reference = _need(raw, "reference", dict, where)
    accepted = tuple(
        _block(b, f"{where}.accepted[{i}]")
        for i, b in enumerate(_need(raw, "accepted", list, where))
    )
    if not accepted:
        raise TaskError(f"{where}: no accepted analysis")
    traps = tuple(
        _block(b, f"{where}.traps[{i}]") for i, b in enumerate(raw.get("traps", []))
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
    return problems


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


@dataclass(frozen=True)
class Options:
    workdir: Path
    claude: str = "claude"
    model: str | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    time_limit: int = DEFAULT_TIME_LIMIT
    repeats: int = 1
    arms: tuple[str, ...] = ARMS


def env_dir(options: Options, arm: str) -> Path:
    return options.workdir / "envs" / arm


def env_commands(options: Options) -> dict[str, list[list[str]]]:
    """The commands that create each arm's Python environment (network, no model)."""
    commands: dict[str, list[list[str]]] = {}
    for arm in options.arms:
        venv = env_dir(options, arm)
        packages: list[str] = list(BASE_PACKAGES)
        if arm == "inferplot":
            packages.append(f"inferplot[pandas] @ {REPO_ROOT.as_uri()}")
        commands[arm] = [
            ["uv", "venv", "--python", "3.12", str(venv)],
            ["uv", "pip", "install", "--python", str(venv), *packages],
        ]
    return commands


def run_dir(options: Options, arm: str, task: Task, repeat: int) -> Path:
    suffix = f"-r{repeat}" if options.repeats > 1 else ""
    return options.workdir / arm / f"{task.id}{suffix}"


def build_arms(
    tasks: Sequence[Task], options: Options
) -> dict[tuple[str, str, int], Path]:
    """Create each run's fresh project directory and prompt file; return the dirs.

    A baseline directory holds data.csv only. An inferplot directory also holds the
    skill in .claude/skills/inferplot. The prompts go to workdir/prompts, outside
    the directories the agents see; nothing of the task file (expected numbers) is
    copied.
    """
    dirs: dict[tuple[str, str, int], Path] = {}
    prompts = options.workdir / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        (prompts / f"{task.id}.txt").write_text(prompt_for(task), encoding="utf-8")
        for arm in options.arms:
            for repeat in range(options.repeats):
                target = run_dir(options, arm, task, repeat)
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True)
                shutil.copyfile(task.data_file, target / "data.csv")
                if arm == "inferplot":
                    shutil.copytree(
                        SKILL_DIR, target / ".claude" / "skills" / "inferplot"
                    )
                dirs[(arm, task.id, repeat)] = target
    return dirs


def claude_command(task: Task, options: Options) -> list[str]:
    prompt_file = options.workdir / "prompts" / f"{task.id}.txt"
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
    prompt_file = options.workdir / "prompts" / f"{task.id}.txt"
    shown = [shlex.quote(c) for c in command]
    shown[2] = f'"$(cat {shlex.quote(str(prompt_file))})"'
    venv_bin = env_dir(options, arm) / ("Scripts" if os.name == "nt" else "bin")
    # The harness enforces the time limit itself (subprocess timeout).
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
    correct_test: bool | None = None
    test_reason: str = ""
    numbers_ok: bool | None = None
    effect_ok: bool | None = None
    numbers: dict[str, Any] = field(default_factory=dict)
    trap: str | None = None
    check_clean: bool | None = None
    check_reason: str = ""
    contaminated: bool = False
    turns: int | None = None
    tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cost_usd: float | None = None
    duration_s: float | None = None


def parse_number(value: Any) -> tuple[float | None, bool]:
    """A reported number as (value, is_upper_bound); strings like '< .001' allowed."""
    if value is None or isinstance(value, bool):
        return None, False
    if isinstance(value, int | float):
        return float(value), False
    if isinstance(value, str):
        text = value.strip().replace("\u2212", "-").replace(",", "")
        bound = text.startswith("<")
        match = re.search(r"-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?", text, re.IGNORECASE)
        if match:
            return float(match[0]), bound
    return None, False


def _dfs(answer: Mapping[str, Any]) -> tuple[float | None, float | None]:
    df, df_error = answer.get("df"), answer.get("df_error")
    if isinstance(df, list) and len(df) == 2:
        return parse_number(df[0])[0], parse_number(df[1])[0]
    if isinstance(df, str) and "," in df and df_error is None:
        first, second = df.split(",", 1)
        return parse_number(first)[0], parse_number(second)[0]
    return parse_number(df)[0], parse_number(df_error)[0]


def compare_numbers(block: Block, answer: Mapping[str, Any]) -> dict[str, Any]:
    """Per expected number: (expected, reported, ok)."""
    df, df_error = _dfs(answer)
    reported = {"df": df, "df_error": df_error}
    for name in (
        "statistic",
        "p_value",
        "n",
        "estimate",
        "conf_low",
        "conf_high",
        "bf10",
    ):
        reported[name] = parse_number(answer.get(name))[0]
    p_bound = parse_number(answer.get("p_value"))[1]
    detail: dict[str, Any] = {}
    low_ok, high_ok = interval_within(
        block, reported["conf_low"], reported["conf_high"]
    )
    for name, want in block.numbers.items():
        got = reported.get(name)
        if name == "conf_low":
            ok = low_ok and got is not None
        elif name == "conf_high":
            ok = high_ok and got is not None
        elif name == "p_value" and p_bound and got is not None:
            ok = want.value <= got <= P_FLOOR
        else:
            ok = within(name, want, got, block.sign_free)
        detail[name] = {"expected": want.value, "reported": got, "ok": ok}
    return detail


def _score_of(detail: Mapping[str, Any]) -> int:
    return sum(1 for v in detail.values() if v["ok"])


def _core_ok(detail: Mapping[str, Any]) -> bool | None:
    core = [v["ok"] for k, v in detail.items() if k in CORE]
    return all(core) if core else None


def _effect_ok(detail: Mapping[str, Any]) -> bool | None:
    effect = [v["ok"] for k, v in detail.items() if k in EFFECT]
    return all(effect) if effect else None


def parse_transcript(text: str) -> dict[str, Any]:
    """Facts from a `claude -p --output-format stream-json --verbose` transcript."""
    facts: dict[str, Any] = {"result": None, "tool_calls": 0, "commands": []}
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
                    facts["tool_calls"] += 1
                    facts["commands"].append(json.dumps(item.get("input", {})))
    return facts


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
    if arm == "baseline":
        score.contaminated = "inferplot" in " ".join(facts["commands"])

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
    for block in task.accepted:
        if family_matches(block.family, score.test_reported):
            score.family = block.family
            break
    if score.family is not None:
        score.correct_test = True
        score.test_reason = (
            f"accepted: {FAMILIES.get(score.family, Family('no test', ())).label}"
        )
    else:
        score.correct_test = False
        rejected = [f for f in task.rejected if family_matches(f, score.test_reported)]
        trap_families = [
            b.family
            for b in task.traps
            if family_matches(b.family, score.test_reported)
        ]
        if rejected:
            score.test_reason = f"rejected: {FAMILIES[rejected[0]].label}"
        elif trap_families:
            score.test_reason = f"trap: {FAMILIES[trap_families[0]].label}"
        else:
            score.test_reason = "not recognized as an accepted test"

    # Numbers: against the accepted blocks of the matched family (the best variant).
    candidates = [b for b in task.accepted if b.family == score.family and b.numbers]
    if candidates:
        details = [compare_numbers(b, answer) for b in candidates]
        best = max(details, key=_score_of)
        score.numbers = best
        score.numbers_ok = _core_ok(best)
        score.effect_ok = _effect_ok(best)
    # Traps: the reported statistic, df and p are those of a trap analysis (the df
    # tell a sphericity-corrected F from the uncorrected one with the same F).
    for block in task.traps:
        detail = compare_numbers(block, answer)
        core = [
            v["ok"]
            for k, v in detail.items()
            if k in ("statistic", "df", "df_error", "p_value")
        ]
        if core and all(core):
            score.trap = block.label
            break

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
    lines = [
        "arm        runs  ran  correct test  numbers  effect  check clean  traps  "
        "turns (median)  output tokens (sum)"
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
        traps = sum(1 for s in mine if s.trap)
        lines.append(
            f"{arm:<10} {len(mine):>4}  {count('ran'):>4} {count('correct_test'):>13} "
            f"{count('numbers_ok'):>8} {count('effect_ok'):>7} "
            f"{count('check_clean'):>12} "
            f"{traps:>6} {median!s:>15} {tokens:>20}"
        )
    return "\n".join(lines)


# --- running -----------------------------------------------------------------------


def _call_claude(
    command: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: int
) -> tuple[int | None, str, bool]:
    """The only place a model is called. Returns (exit code, stdout, timed out)."""
    try:
        done = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        out = expired.stdout or ""
        return None, out.decode() if isinstance(out, bytes) else out, True
    return done.returncode, done.stdout, False


def _build_envs(options: Options) -> None:
    for commands in env_commands(options).values():
        for command in commands:
            subprocess.run(command, check=True)


def _run_env(options: Options, arm: str) -> dict[str, str]:
    env = dict(os.environ)
    venv = env_dir(options, arm)
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv)
    env.pop("PYTHONPATH", None)
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
    results = options.workdir / "results"
    scores: list[Score] = []
    for task in tasks:
        for arm in options.arms:
            for repeat in range(options.repeats):
                cwd = dirs[(arm, task.id, repeat)]
                code, transcript, timed_out = _call_claude(
                    claude_command(task, options),
                    cwd,
                    _run_env(options, arm),
                    options.time_limit,
                )
                keep = results / arm / cwd.name
                if keep.exists():
                    shutil.rmtree(keep)
                shutil.copytree(cwd, keep, ignore=shutil.ignore_patterns(".claude"))
                (keep / "transcript.jsonl").write_text(transcript, encoding="utf-8")
                score = score_run(
                    task,
                    arm,
                    transcript,
                    cwd,
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
    (results / "scores.json").write_text(
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
                f"ok   {task.id:<34} {task.function:<12} {n_numbers:>3} numbers  "
                f"trap: {task.trap}"
            )
    dirs = build_arms(tasks, options)
    out("")
    out(f"arms built in {options.workdir} ({len(dirs)} run directories)")
    out("commands a real run would execute (environments first; network, no model):")
    for arm, commands in env_commands(options).items():
        for command in commands:
            out(f"  [{arm}] " + shlex.join(command))
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
        help="where run directories go (default: a new temporary one)",
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
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="inferplot-evals-"))
    if workdir.resolve().is_relative_to(REPO_ROOT.resolve()):
        parser.error(
            "--workdir must be outside the repository, "
            "where agents cannot read the tasks"
        )
    options = Options(
        workdir=workdir,
        claude=args.claude,
        model=args.model,
        max_turns=args.max_turns,
        time_limit=args.time_limit,
        repeats=args.repeats,
        arms=arms,
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
