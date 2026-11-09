"""The nonparametric, robust and bayes ggbetweenstats fixtures, and how inferplot
replays each one.

Every fixture of these types is replayed as the inferplot call
``tests/oracle_args.translate`` builds from its R arguments, so a new fixture is
picked up without new code. Where inferplot refuses a call ggstatsplot answers, by
design (SPEC.md sections 2 and 5), :func:`refusal` says how it refuses and which
changed call reproduces ggstatsplot's output:

* a one-sided ``alternative`` with type robust or bayes raises ``ValueError``
  (ggstatsplot ignores it), so the replay leaves it out and must give ggstatsplot's
  output;
* type bayes with three or more groups raises ``NotImplementedError`` unless
  ``results_subtitle=False`` (D41); the replay passes it and must give ggstatsplot's
  output without its subtitle (MAP labels and a Bayes factor bracket per pair);
* type robust with three or more groups at a ``conf_level`` other than 0.95 raises
  ``NotImplementedError``: lincon's critical values are known here only at 95%.
  There is no replay.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from tests.oracle_args import translate
from tests.oracle_fixtures import OracleCase, cases_for, require_cases

TYPES: tuple[str, ...] = ("nonparametric", "robust", "bayes")

ALL = require_cases(
    [c for c in cases_for("ggbetweenstats") if c.type in TYPES],
    "nonparametric, robust and bayes ggbetweenstats",
)
# Cases where ggstatsplot itself fails (none among the Phase 0 cases).
FAILED = [c for c in cases_for("ggbetweenstats", failed=True) if c.type in TYPES]
BY_NAME = {case.name: case for case in (*ALL, *FAILED)}


@dataclasses.dataclass(frozen=True)
class Refusal:
    """inferplot refuses the fixture's call with ``error`` matching ``match``;
    ``replay`` (keyword changes) then reproduces ggstatsplot's output, or None."""

    error: type[Exception]
    match: str
    replay: Mapping[str, Any] | None


def call_arguments(case: OracleCase) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The positional and keyword arguments of the fixture's call."""
    call = translate(case.function, case.args)
    assert call.function == "betweenstats"
    return call.args, dict(call.kwargs)


def refusal(case: OracleCase) -> Refusal | None:
    """SPEC.md's refusals (module docstring), decided from the case's arguments."""
    _, kwargs = call_arguments(case)
    k = len(case.analysis_levels)
    side = kwargs.get("alternative", "two-sided")
    if case.type in ("robust", "bayes") and side != "two-sided":
        return Refusal(ValueError, "alternative=", {"alternative": None})
    if case.type == "bayes" and k >= 3 and kwargs.get("results_subtitle", True):
        return Refusal(
            NotImplementedError, "results_subtitle=False", {"results_subtitle": False}
        )
    if case.type == "robust" and k >= 3 and kwargs.get("conf_level", 0.95) != 0.95:
        return Refusal(NotImplementedError, "conf_level", None)
    return None


def replay_arguments(case: OracleCase) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The fixture's call, changed as its refusal prescribes (a None value drops
    the keyword)."""
    args, kwargs = call_arguments(case)
    found = refusal(case)
    if found is not None:
        assert found.replay is not None, f"{case.name} has no replay"
        for key, value in found.replay.items():
            if value is None:
                kwargs.pop(key, None)
            else:
                kwargs[key] = value
    return args, kwargs


def _replayable(case: OracleCase) -> bool:
    found = refusal(case)
    return found is None or found.replay is not None


CASES = require_cases(
    [case for case in ALL if _replayable(case)],
    "replayable fixtures of the types nonparametric, robust and bayes",
)
REFUSED = [case for case in ALL if refusal(case) is not None]
