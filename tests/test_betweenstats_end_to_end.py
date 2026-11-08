"""ip.betweenstats end to end: analyze, draw, check and save in one call.

The analysis parity lives in ``tests/test_betweenstats_parity.py`` and the drawing
in ``tests/render``; these tests check the top-level function that joins them, on
every completed parametric ggbetweenstats fixture.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot.render import StatsFigure
from tests.oracle_args import translate
from tests.oracle_fixtures import OracleCase, cases_for, read_dataset, require_cases

FIGURE_CODES = frozenset({"label_overlap", "text_clipped", "points_subsampled"})
CASES = require_cases(
    [c for c in cases_for("ggbetweenstats") if c.type == "parametric"],
    "parametric ggbetweenstats",
)


def _call(case: OracleCase) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    call = translate(case.function, case.args)
    return read_dataset(case), call.args, dict(call.kwargs)


def _teeth() -> Any:
    (case,) = [c for c in CASES if c.name == "ggbetweenstats-ToothGrowth-parametric"]
    return read_dataset(case)


def test_signature_is_the_analysis_then_the_drawing() -> None:
    top = inspect.signature(ip.betweenstats).parameters
    analysis = inspect.signature(ip.analyze.betweenstats).parameters
    drawing = dict(inspect.signature(ip.render.betweenstats).parameters)
    drawing.pop("result")
    assert list(top) == [*analysis, *drawing]
    for name, parameter in top.items():
        source = analysis.get(name) or drawing[name]
        assert parameter.kind == source.kind, name
        assert parameter.default == source.default, name


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_every_parametric_fixture(case: OracleCase, tmp_path: Path) -> None:
    data, args, kwargs = _call(case)
    fig = ip.betweenstats(data, *args, **kwargs)
    assert isinstance(fig, StatsFigure)
    assert fig.result == ip.analyze.betweenstats(data, *args, **kwargs)
    issues = fig.check()
    assert not [i for i in issues if i.code in FIGURE_CODES], issues
    assert issues == fig.result.check()
    fig.save(tmp_path / "figure.svg")
    assert (tmp_path / "figure.svg").stat().st_size > 0


@pytest.mark.parametrize(
    ("name", "x", "y"),
    [("ToothGrowth", "supp", "len"), ("iris", "Species", "Sepal.Length")],
)
def test_slice_examples_check_clean(name: str, x: str, y: str, tmp_path: Path) -> None:
    # CLAUDE.md's definition of done: the example renders to SVG and fig.check()
    # is empty (two groups; three groups with every pair drawn).
    (case,) = [c for c in CASES if c.name == f"ggbetweenstats-{name}-parametric"]
    fig = ip.betweenstats(read_dataset(case), x, y)
    assert fig.check() == ()
    path = tmp_path / f"{name}.svg"
    fig.save(path)
    text = path.read_text(encoding="utf-8")
    assert "<text" in text  # svg.fonttype none keeps text editable
    assert all(f"{level}" in text for level in fig.result.levels)


def test_drawing_arguments_reach_the_renderer() -> None:
    teeth = _teeth()
    fig = ip.betweenstats(
        teeth,
        "supp",
        "len",
        layers=("box", "points"),
        centrality_plotting=False,
        title="Tooth growth",
        caption="Source: R datasets",
        xlab="Supplement",
    )
    texts = [t.get_text() for t in fig.figure.texts]
    assert "Tooth growth" in texts
    assert "Source: R datasets" in texts
    assert fig.axes["main"].get_xlabel() == "Supplement"


def test_subtitle_needs_results_subtitle_false() -> None:
    teeth = _teeth()
    with pytest.raises(ValueError, match="results_subtitle=False"):
        ip.betweenstats(teeth, "supp", "len", subtitle="My subtitle")
    fig = ip.betweenstats(
        teeth, "supp", "len", results_subtitle=False, subtitle="My subtitle"
    )
    assert "My subtitle" in [t.get_text() for t in fig.figure.texts]
    assert fig.result.text.subtitle is None


def test_unknown_keyword_names_the_top_level_function() -> None:
    frame = pd.DataFrame({"g": list("aabb"), "y": [1.0, 2.0, 4.0, 3.0]})
    with pytest.raises(TypeError, match=r"betweenstats\(\).*point_args"):
        ip.betweenstats(frame, "g", "y", pointargs={})  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError, match=r"var\.equal"):
        ip.betweenstats(frame, "g", "y", **{"var.equal": True})  # pyright: ignore[reportArgumentType]
