"""Every ggwithinstats fixture inferplot computes, drawn by ``ip.withinstats``.

Each fixture's call (or its replay, ``tests/within_pb_cases.refusal``) is drawn with
the case's own drawing arguments (``point.path``, ``centrality.path``). For each
figure: every text drawn equals ``figure_texts`` (tick labels, subtitle, centrality
labels, the bracket labels of the displayed pairs, caption lines), which
``tests/test_withinstats_texts.py`` compares with ggstatsplot's; the layers drawn;
and the subject and centrality paths drawn exactly where ggwithinstats draws them
(its ``GeomPath`` layers: the subject paths use the plot's data, the centrality path
its own); ``fig.check()`` reports only what the data warrant; and ``save()`` writes
SVG, PDF and PNG, each embedding a result whose ``check()`` is the figure's.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import inferplot as ip
from inferplot.render import StatsFigure
from inferplot.result import StatsResult
from inferplot.text import figure_texts
from tests import within_pb_cases as wc
from tests.oracle_args import translate
from tests.oracle_fixtures import OracleCase, read_dataset
from tests.render._helpers import (
    gids,
    numbered_count,
    numbered_texts,
    pdf_info_string,
    svg_description,
    text_of,
)

CASES = wc.ALL_CASES


def drawing_arguments(case: OracleCase) -> dict[str, Any]:
    """The case's drawing keywords (``point_path``, ``centrality_path``)."""
    call = translate(case.function, case.args)
    return {k: v for k, v in call.kwargs.items() if k in wc.DRAWING_KEYWORDS}


@functools.cache
def figure(name: str) -> StatsFigure:
    """``ip.withinstats`` on the fixture's call (or replay), drawn once."""
    case = wc.BY_NAME[name]
    args, kwargs = wc.replay_arguments(case)
    return ip.withinstats(
        read_dataset(case, "pandas"), *args, **kwargs, **drawing_arguments(case)
    )


def _params() -> list[Any]:
    return [pytest.param(case.name, id=case.name) for case in CASES]


def r_paths(case: OracleCase) -> tuple[bool, bool]:
    """(subject paths, centrality path) as ggwithinstats drew them: its subject
    paths are a ``GeomPath`` on the plot's data, the centrality path one on its
    own data (the centrality values)."""
    paths = [layer for layer in case.plot_layers if layer["geom"] == "GeomPath"]
    subjects = any(not layer["data"] for layer in paths)
    centrality = any(layer["data"] for layer in paths)
    assert len(paths) == subjects + centrality
    return subjects, centrality


@pytest.mark.parametrize("name", _params())
def test_fixture_figure_draws_figure_texts(name: str) -> None:
    case = wc.BY_NAME[name]
    fig = figure(name)
    result = fig.result
    texts = figure_texts(result)
    ax = fig.axes["main"]
    assert set(fig.axes) == {"main"}
    assert [t.get_text() for t in ax.get_xticklabels()] == list(texts.tick_labels)
    assert text_of(fig, "subtitle") == texts.subtitle
    labelled = [
        i for i, d in enumerate(result.descriptives) if d.value is not None
    ]  # a MAP without a density has no label (D55)
    assert numbered_texts(fig, "centrality-label") == [
        texts.centrality_labels[i] for i in labelled
    ]
    displayed = [n for n, pair in enumerate(result.pairwise) if pair.displayed]
    assert numbered_texts(fig, "bracket-label") == [
        texts.pairwise_labels[n] for n in displayed
    ]
    assert numbered_count(fig, "bracket") == len(displayed) == len(wc.drawn(case))
    assert numbered_texts(fig, "caption") == list(texts.caption_lines)
    assert ax.get_xlabel() == result.x
    assert ax.get_ylabel() == result.y


def _figure_issues(case: OracleCase) -> frozenset[str]:
    """The rendering codes a fixture's own figure warrants (FIGURE_ISSUES)."""
    return FIGURE_ISSUES.get(case.name.rsplit("-", 1)[0], frozenset())


# The rendering issues the fixtures' own figures report, by data set (every
# type). edge_within2_large_n: 1,500 subjects at two conditions, 3,000 values,
# fewer than the 5,000 that start subsampling, so all 1,500 subject paths are
# drawn and cover 48% of the area between the conditions at half opacity or
# more (D57's limit: 15%, about 250 subjects).
FIGURE_ISSUES: dict[str, frozenset[str]] = {
    "ggwithinstats-edge_within2_large_n": frozenset({"paths_crowded"}),
}


@pytest.mark.parametrize("name", _params())
def test_fixture_figure_checks_and_saves(name: str, tmp_path: Path) -> None:
    """``fig.check()`` reports the result's issues and only the rendering issues
    the data warrant; SVG, PDF and PNG are written, each embedding the result,
    whose ``check()`` read back from it is the figure result's."""
    case = wc.BY_NAME[name]
    fig = figure(name)
    result_issues = fig.result.check()
    issues = fig.check()
    assert issues[: len(result_issues)] == result_issues
    rendering = {issue.code for issue in issues[len(result_issues) :]}
    assert rendering == _figure_issues(case)
    for suffix in ("svg", "pdf", "png"):
        path = tmp_path / f"figure.{suffix}"
        fig.save(path, dpi=72)
        if suffix == "svg":
            embedded = svg_description(path)
        elif suffix == "pdf":
            embedded = pdf_info_string(path.read_bytes(), "Subject")
        else:
            with Image.open(path) as image:
                embedded = image.text["Description"]  # pyright: ignore[reportAttributeAccessIssue]
        assert embedded == fig.result.to_json(indent=None, include_values=False)
        back = StatsResult.from_json(embedded)
        assert back.values == ()
        assert back.subjects == ()
        assert back.check() == result_issues
        assert back.text.alt_text == fig.result.text.alt_text


@pytest.mark.parametrize("name", _params())
def test_fixture_figure_layers_and_paths(name: str) -> None:
    case = wc.BY_NAME[name]
    fig = figure(name)
    present = gids(fig)
    for i in range(len(fig.result.levels)):
        for layer in ("box", "points", "centrality"):
            assert f"{layer}-{i}" in present, (layer, i)
    subjects, centrality = r_paths(case)
    assert ("subject-paths" in present) == subjects
    assert ("centrality-path" in present) == centrality
