"""Every public function has a runnable example in its docstring, and it runs.

The examples run with doctest in a temporary folder, with whitespace normalized
(``NORMALIZE_WHITESPACE``) so that a long printed sentence may wrap in the
docstring. They print the package's formatted texts only, so they give the same
output on every platform and on the oldest supported versions.

A test, not ``--doctest-modules``: it also asserts that each public function has
an example, and it leaves the collection of ``tests/`` unchanged.
"""

from __future__ import annotations

import doctest
from typing import Any

import pytest

import inferplot as ip
from inferplot.render.figure import StatsFigure
from inferplot.result import StatsResult

FUNCTIONS = ("betweenstats", "withinstats", "histostats", "scatterstats", "barstats")

TARGETS: dict[str, Any] = {
    **{f"ip.{n}": getattr(ip, n) for n in FUNCTIONS},
    **{f"ip.analyze.{n}": getattr(ip.analyze, n) for n in FUNCTIONS},
    **{f"ip.render.{n}": getattr(ip.render, n) for n in FUNCTIONS},
    "StatsResult.to_json": StatsResult.to_json,
    "StatsResult.from_json": StatsResult.from_json,
    "StatsResult.to_frame": StatsResult.to_frame,
    "StatsResult.check": StatsResult.check,
    "StatsFigure.check": StatsFigure.check,
    "StatsFigure.save": StatsFigure.save,
}


@pytest.mark.parametrize("name", TARGETS)
def test_docstring_example_runs(
    name: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # nothing may be written elsewhere
    found = doctest.DocTestFinder(recurse=False).find(TARGETS[name], name, globs={})
    examples = [example for test in found for example in test.examples]
    assert len(examples) >= 3, f"{name} has no runnable example in its docstring"
    report: list[str] = []
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    for test in found:
        runner.run(test, out=report.append)
    assert runner.failures == 0, "".join(report)
    assert list(tmp_path.iterdir()) == []
