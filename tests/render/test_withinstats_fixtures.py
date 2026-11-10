"""Every ggwithinstats fixture inferplot computes, drawn by ``ip.withinstats``.

Each fixture's call (or its replay, ``tests/within_pb_cases.refusal``) is drawn with
the case's own drawing arguments (``point.path``, ``centrality.path``). For each
figure: every text drawn equals ``figure_texts`` (tick labels, subtitle, centrality
labels, the bracket labels of the displayed pairs, caption lines), which
``tests/test_withinstats_texts.py`` compares with ggstatsplot's; the layers drawn;
and the subject and centrality paths drawn exactly where ggwithinstats draws them
(its ``GeomPath`` layers: the subject paths use the plot's data, the centrality path
its own).
"""

from __future__ import annotations

import functools
from typing import Any

import pytest

import inferplot as ip
from inferplot.render import StatsFigure
from inferplot.text import figure_texts
from tests import within_pb_cases as wc
from tests.oracle_args import translate
from tests.oracle_fixtures import OracleCase, read_dataset
from tests.render._helpers import gids, numbered_count, numbered_texts, text_of

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
