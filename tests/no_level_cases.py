"""The gghistostats and ggscatterstats fixtures, and how inferplot replays each.

``prepared(case)`` runs the analysis' own argument checks and data reading
(``_prepare``) on a fixture's dataset with the fixture's arguments;
``analyzed(name, backend)`` runs the fixture's call through ``ip.analyze`` on its
CSV read as pandas, polars or pyarrow (cached: the parity, contract and text tests
share them). A ``label.expression`` case gets the labels of the other points set
to missing (``tests/oracle_args.apply_label_filter``), inferplot's way to label
some points (D82).

Where inferplot refuses a call ggstatsplot answers, by design (SPEC.md sections 2
and 5), ``REFUSALS`` says how it refuses and which changed call, if any, gives
ggstatsplot's output:

* a one-sided ``alternative`` with type robust or bayes raises ``ValueError``
  (ggstatsplot ignores it: ToothGrowth-greater-robust and -bayes equal the
  two-sided cases), so the replay leaves it out and must give ggstatsplot's output;
* a constant x raises ``ValueError`` for every type (ggstatsplot prints NA for
  every number and drops the Bayes factor; edge_corr_constant_x), with no replay.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
from collections.abc import Mapping
from typing import Any

import pytest

import inferplot as ip
from inferplot.result import StatsResult
from tests.oracle_args import apply_label_filter, split
from tests.oracle_fixtures import Backend, OracleCase, cases_for, read_dataset

histo: Any = importlib.import_module("inferplot.analyses.histostats")
scatter: Any = importlib.import_module("inferplot.analyses.scatterstats")

BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")
HISTO: tuple[OracleCase, ...] = tuple(cases_for("gghistostats"))
SCATTER: tuple[OracleCase, ...] = tuple(cases_for("ggscatterstats"))
BY_NAME: dict[str, OracleCase] = {c.name: c for c in (*HISTO, *SCATTER)}


@dataclasses.dataclass(frozen=True)
class Refusal:
    """inferplot refuses the fixture's call with ``error`` whose message contains
    ``match``; ``replay`` (keyword changes, None drops one) reproduces
    ggstatsplot's output, or is None where no call does."""

    error: type[Exception]
    match: str
    replay: Mapping[str, Any] | None


_ONE_SIDED = Refusal(
    ValueError,
    "applies to type='parametric' and type='nonparametric' only",
    {"alternative": None},
)
_CONSTANT_X = Refusal(
    ValueError,
    "Every value of x='x' is 5, so its standard deviation is 0 and a correlation "
    "with it is undefined for every type (ggstatsplot prints NA for every number).",
    None,
)
REFUSALS: Mapping[str, Refusal] = {
    "gghistostats-ToothGrowth-greater-robust": _ONE_SIDED,
    "gghistostats-ToothGrowth-greater-bayes": _ONE_SIDED,
    **{
        f"ggscatterstats-edge_corr_constant_x-{t}": _CONSTANT_X
        for t in ("parametric", "nonparametric", "robust", "bayes")
    },
}


def module(case: OracleCase) -> Any:
    return histo if case.function == "gghistostats" else scatter


def frame_of(case: OracleCase, backend: Backend = "pandas") -> Any:
    """The case's dataset as ``backend`` reads it, with a ``label.expression``
    applied as missing labels (pandas only)."""
    call = split(case.function, case.args)
    frame = read_dataset(case, backend)
    if call.label_filter is not None:
        assert backend == "pandas", "label filters are applied with pandas"
        frame = apply_label_filter(
            frame, str(call.analysis["label_var"]), call.label_filter
        )
    return frame


def backends(case: OracleCase) -> tuple[Backend, ...]:
    """The backends a case is replayed on (a label filter is applied in pandas)."""
    return ("pandas",) if "label.expression" in case.args else BACKENDS


def call_arguments(case: OracleCase) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The positional and keyword arguments of the fixture's analysis call."""
    call = split(case.function, case.args)
    return call.args, dict(call.analysis)


def prepared(case: OracleCase, backend: Backend = "pandas") -> tuple[Any, Any, int]:
    """``_prepare`` on the case's dataset with its analysis arguments: the data
    read, the validated arguments and digits."""
    args, kwargs = call_arguments(case)
    return module(case)._prepare(frame_of(case, backend), *args, **kwargs)


def analyze(case: OracleCase, frame: Any, **changes: Any) -> StatsResult:
    """``ip.analyze`` of the case's function on ``frame`` with the case's
    arguments, ``changes`` applied (None drops an argument)."""
    args, kwargs = call_arguments(case)
    for key, value in changes.items():
        if value is None:
            kwargs.pop(key, None)
        else:
            kwargs[key] = value
    analysis: Any = (
        ip.analyze.histostats
        if case.function == "gghistostats"
        else ip.analyze.scatterstats
    )
    return analysis(frame, *args, **kwargs)


@functools.cache
def analyzed(name: str, backend: Backend = "pandas") -> StatsResult | None:
    """The fixture's call, or its replay where inferplot refuses the call; None
    for a refusal without a replay."""
    case = BY_NAME[name]
    refusal = REFUSALS.get(name)
    if refusal is None:
        return analyze(case, frame_of(case, backend))
    if refusal.replay is None:
        return None
    return analyze(case, frame_of(case, backend), **refusal.replay)


def params(cases: tuple[OracleCase, ...], *, every_backend: bool = True) -> list[Any]:
    """``pytest.param(name, backend)`` for each case answered (directly or by its
    replay) and backend."""
    out: list[Any] = []
    for case in cases:
        refusal = REFUSALS.get(case.name)
        if refusal is not None and refusal.replay is None:
            continue
        for backend in backends(case) if every_backend else ("pandas",):
            out.append(pytest.param(case.name, backend, id=f"{case.name}-{backend}"))
    return out
