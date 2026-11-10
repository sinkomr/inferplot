"""The ggwithinstats fixtures, and how inferplot replays each.

The parity tests of types parametric and bayes, and the text and drawing tests of
every type (``tests/test_withinstats_texts.py``), read them from here. Every
fixture is replayed as the inferplot call ``tests/oracle_args.translate`` builds
from its R arguments, without the drawing keywords (``point_path``,
``centrality_path``), which belong to ``ip.withinstats`` and
``ip.render.withinstats``, not to the analysis. Where inferplot refuses a call
ggstatsplot answers, by design (SPEC.md sections 2 and 5), :func:`refusal` says how
it refuses and which changed call reproduces ggstatsplot's output:

* a one-sided ``alternative`` with type bayes or robust raises ``ValueError``
  (ggstatsplot ignores it), so the replay leaves it out and must give
  ggstatsplot's output;
* type bayes with three or more conditions raises ``NotImplementedError`` unless
  ``results_subtitle=False`` (the Bayesian repeated-measures ANOVA arrives in 0.2,
  D41); the replay passes it and must give ggstatsplot's output without its
  subtitle (MAP labels and a Bayes factor bracket per pair).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import inferplot as ip
from inferplot.result import StatsResult
from tests.oracle_args import translate
from tests.oracle_fixtures import (
    Backend,
    OracleCase,
    cases_for,
    r_level_label,
    read_dataset,
)

FloatArray = npt.NDArray[np.float64]
BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")
# Keywords of the drawing, not of the analysis.
DRAWING_KEYWORDS = frozenset({"point_path", "centrality_path"})


def cases_of(type_name: str) -> list[OracleCase]:
    """The completed ggwithinstats fixtures of one type (none of these types has a
    failed fixture: ``test_no_failed_cases`` keeps it so)."""
    cases = [c for c in cases_for("ggwithinstats") if c.type == type_name]
    if not cases:
        raise LookupError(f"no ggwithinstats {type_name} fixtures")
    return cases


@dataclasses.dataclass(frozen=True)
class Refusal:
    """inferplot refuses the fixture's call with ``error`` matching ``match``;
    ``replay`` (keyword changes, None drops one) reproduces ggstatsplot's output."""

    error: type[Exception]
    match: str
    replay: Mapping[str, Any]


def call_arguments(case: OracleCase) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The positional and keyword arguments of the fixture's analysis call."""
    call = translate(case.function, case.args)
    assert call.function == "withinstats"
    kwargs = {k: v for k, v in call.kwargs.items() if k not in DRAWING_KEYWORDS}
    return call.args, kwargs


def refusal(case: OracleCase) -> Refusal | None:
    """SPEC.md's refusals (module docstring), decided from the case's arguments.
    A one-sided ``alternative`` with type robust is refused like type bayes's
    (sleep-greater-robust; ``tests/test_withinstats_parity_robust.py``)."""
    _, kwargs = call_arguments(case)
    k = len(case.analysis_levels)
    one_sided = kwargs.get("alternative", "two-sided") != "two-sided"
    if case.type in ("bayes", "robust") and one_sided:
        return Refusal(ValueError, "alternative=", {"alternative": None})
    if case.type == "bayes" and k >= 3 and kwargs.get("results_subtitle", True):
        return Refusal(
            NotImplementedError, "results_subtitle=False", {"results_subtitle": False}
        )
    return None


def replay_arguments(case: OracleCase) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The fixture's call, changed as its refusal prescribes."""
    args, kwargs = call_arguments(case)
    found = refusal(case)
    if found is not None:
        for key, value in found.replay.items():
            if value is None:
                kwargs.pop(key, None)
            else:
                kwargs[key] = value
    return args, kwargs


TYPES: tuple[str, ...] = ("parametric", "nonparametric", "robust", "bayes")
# Every ggwithinstats fixture, of every type (the text tests replay them all).
BY_NAME: dict[str, OracleCase] = {c.name: c for t in TYPES for c in cases_of(t)}
ALL_CASES: list[OracleCase] = list(BY_NAME.values())


@functools.cache
def analyze(name: str, backend: Backend) -> StatsResult:
    """The fixture's call, or its replay when inferplot refuses the call."""
    case = BY_NAME[name]
    args, kwargs = replay_arguments(case)
    return ip.analyze.withinstats(read_dataset(case, backend), *args, **kwargs)


def params(cases: list[OracleCase]) -> list[Any]:
    return [
        pytest.param(case.name, backend, id=f"{case.name}-{backend}")
        for case in cases
        for backend in BACKENDS
    ]


def names(cases: list[OracleCase]) -> list[Any]:
    return [pytest.param(case.name, id=case.name) for case in cases]


def conditions(case: OracleCase) -> list[FloatArray]:
    """One array per condition in R's level order, subjects in R's order (the
    subjects-by-conditions table the fixture records)."""
    columns = case.within_columns()
    assert tuple(columns) == case.analysis_levels
    return [np.array(columns[level], dtype=np.float64) for level in columns]


def centrality_layers(case: OracleCase) -> list[dict[str, Any]]:
    """The centrality rows by level: the centrality points and their labels."""
    x = str(case.args["x"])
    layers = [
        {r_level_label(row[x]): row for row in rows}
        for geom in ("GeomPoint", "GeomLabelRepel")
        for rows in case.layer_data(geom)
    ]
    assert len(layers) == 2, f"{case.name}: expected two centrality layers"
    return layers


def drawn(case: OracleCase) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for layer in case.layer_params("GeomSignif"):
        out.extend((str(a), str(b)) for a, b in layer["stat"]["comparisons"])
    return out


def pair_rows(case: OracleCase, result: StatsResult) -> list[Mapping[str, Any]]:
    """ggstatsplot's pairwise rows, in inferplot's order (they must agree)."""
    rows = case.pairwise
    assert [(p.group1, p.group2) for p in result.pairwise] == [
        (str(r["group1"]), str(r["group2"])) for r in rows
    ]
    return rows
