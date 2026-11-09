"""Fixture data and arguments for the statistics-level parity tests.

Each oracle case is read the way R read it (``tests.oracle_fixtures.read_dataset``),
split into groups by the case's own ``analysis_levels``, and its arguments are
honoured with ggstatsplot's defaults where the case leaves them out, so new fixtures
are picked up without new code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from inferplot.stats import Alternative, PAdjustMethod
from tests.oracle_fixtures import (
    OracleCase,
    cases_for,
    number,
    r_level_label,
    read_dataset,
    require_cases,
)

FloatArray = NDArray[np.float64]

# ggstatsplot 1.1.4 defaults for the arguments the statistics depend on.
DEFAULT_CONF_LEVEL = 0.95
DEFAULT_BF_PRIOR = 0.707
DEFAULT_P_ADJUST = "holm"
DEFAULT_TR = 0.2
# Every pairwise Bayes factor uses this prior, whatever bf.prior says (SPEC.md §4).
PAIRWISE_BF_PRIOR = 0.707

R_ALTERNATIVE: dict[str, Alternative] = {
    "two.sided": "two-sided",
    "greater": "greater",
    "less": "less",
}
P_ADJUST: dict[str, PAdjustMethod] = {
    "holm": "holm",
    "hochberg": "hochberg",
    "hommel": "hommel",
    "bonferroni": "bonferroni",
    "BH": "BH",
    "BY": "BY",
    "fdr": "fdr",
    "none": "none",
}


def groups(case: OracleCase) -> list[FloatArray]:
    """The y values of each analysis level, complete rows only, in level order."""
    frame = read_dataset(case)
    x, y = case.args["x"], case.args["y"]
    used = frame[[x, y]].dropna()
    labels = used[x].astype(object).map(r_level_label)
    out = [
        used.loc[labels == level, y].to_numpy(dtype=np.float64)
        for level in case.analysis_levels
    ]
    for level, values in zip(case.analysis_levels, out, strict=True):
        assert values.size == case.n_per_level[level], f"{case.name}: {level}"
    return out


def alternative(case: OracleCase) -> Alternative:
    return R_ALTERNATIVE[str(case.args.get("alternative", "two.sided"))]


def conf_level(case: OracleCase) -> float:
    return float(case.args.get("conf.level", DEFAULT_CONF_LEVEL))


def tr(case: OracleCase) -> float:
    return float(case.args.get("tr", DEFAULT_TR))


def bf_prior(case: OracleCase) -> float:
    return float(case.args.get("bf.prior", DEFAULT_BF_PRIOR))


def p_adjust_method(case: OracleCase) -> PAdjustMethod:
    return P_ADJUST[str(case.args.get("p.adjust.method", DEFAULT_P_ADJUST))]


def value(row: Any, column: str) -> float:
    assert column in row, f"no {column!r} column; columns are {sorted(row)}"
    return number(row[column])


def between_cases(type_: str, *, two_groups: bool | None = None) -> list[OracleCase]:
    """ggbetweenstats cases of one type that ran, optionally by number of groups."""
    return [
        case
        for case in cases_for("ggbetweenstats")
        if case.type == type_
        and case.error is None
        and (two_groups is None or (len(case.analysis_levels) == 2) == two_groups)
    ]


def params(
    cases: list[OracleCase], what: str, *, where: Callable[[OracleCase], bool] = bool
) -> list[Any]:
    """``pytest.param`` per selected case (never an empty parametrization)."""
    chosen = require_cases([c for c in cases if where(c)], what)
    return [pytest.param(case, id=case.name) for case in chosen]
