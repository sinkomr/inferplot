"""The nonparametric ggwithinstats fixtures, and how inferplot replays each.

Every fixture is replayed as the inferplot call ``tests/oracle_args.translate``
builds from its R arguments, without the drawing keywords
(``tests/within_pb_cases.py``). inferplot refuses none of these calls; where
ggstatsplot swallows an error and leaves out an output, :data:`HIDDEN` names the
case and what inferplot reports instead.
"""

from __future__ import annotations

import functools
from typing import Any

import inferplot as ip
from inferplot.result import StatsResult
from tests import within_pb_cases as wc
from tests.oracle_fixtures import Backend, OracleCase, read_dataset

CASES: list[OracleCase] = wc.cases_of("nonparametric")
TWO = [c for c in CASES if len(c.analysis_levels) == 2]
MORE = [c for c in CASES if len(c.analysis_levels) >= 3]
BY_NAME: dict[str, OracleCase] = {c.name: c for c in CASES}

# ggstatsplot swallows an error and drops an output: the case, the component it
# drops, and what inferplot reports there instead (see test_hidden_errors).
HIDDEN: dict[str, str] = {
    # Every subject ranks the three conditions in the same order, so Kendall's W
    # is 1 in every bootstrap resample; effectsize's bootstrap fails on that
    # degenerate distribution and ggstatsplot drops the subtitle without a word.
    # inferplot reports Friedman's test and W = 1 with the interval [1, 1] and a
    # warnings entry.
    "ggwithinstats-edge_within_small-nonparametric": "subtitle",
}


@functools.cache
def analyze(name: str, backend: Backend) -> StatsResult:
    """The fixture's call through ``ip.analyze.withinstats``."""
    case = BY_NAME[name]
    args, kwargs = wc.call_arguments(case)
    return ip.analyze.withinstats(read_dataset(case, backend), *args, **kwargs)


def params(cases: list[OracleCase]) -> list[Any]:
    return wc.params(cases)


def names(cases: list[OracleCase]) -> list[Any]:
    return wc.names(cases)
