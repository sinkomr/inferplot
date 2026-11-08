"""Every completed parametric ggbetweenstats fixture, drawn and checked.

For each fixture: the layers drawn, every text equal to ``figure_texts`` (tick
labels, subtitle, centrality labels, bracket labels of the displayed pairs, caption
lines), no overlapping or clipped text in ``fig.check()``, bracket geometry, and the
SVG written by ``save()`` (text kept as text, layer groups present).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from matplotlib.lines import Line2D

from inferplot.render import betweenstats
from inferplot.render.brackets import LEVEL_STEP
from inferplot.text import figure_texts
from tests.oracle_fixtures import OracleCase
from tests.render._helpers import (
    case_params,
    default_figure,
    figure_codes,
    gids,
    numbered_count,
    numbered_texts,
    short,
    svg_ids,
    svg_texts,
    text_of,
)
from tests.render._results import fixture_result

CASES = case_params()


def test_every_completed_parametric_fixture_is_drawn() -> None:
    # 28 new parametric cases plus the Phase 0 parametric ones, minus edge_n2,
    # which ggstatsplot itself fails on.
    assert len(CASES) == 40


@pytest.mark.parametrize("case", CASES)
def test_fixture_figure_matches_figure_texts(case: OracleCase) -> None:
    result = fixture_result(case)
    fig = default_figure(short(case))
    texts = figure_texts(result)
    k = len(result.levels)

    # Nothing about these fixtures needs a rendering warning.
    assert fig.result == result
    ax = fig.axes["main"]
    assert set(fig.axes) == {"main"}

    present = gids(fig)
    for i in range(k):
        for layer in ("violin", "box", "points", "centrality", "centrality-label"):
            assert f"{layer}-{i}" in present, (layer, i)
    assert not any(g.startswith("density-") for g in present)

    assert [t.get_text() for t in ax.get_xticklabels()] == list(texts.tick_labels)
    assert text_of(fig, "subtitle") == texts.subtitle
    assert numbered_texts(fig, "centrality-label") == list(texts.centrality_labels)
    displayed = [n for n, pair in enumerate(result.pairwise) if pair.displayed]
    assert numbered_texts(fig, "bracket-label") == [
        texts.pairwise_labels[n] for n in displayed
    ]
    assert numbered_count(fig, "bracket") == len(displayed)
    assert numbered_texts(fig, "caption") == list(texts.caption_lines)
    assert text_of(fig, "title") is None
    assert ax.get_xlabel() == result.x
    assert ax.get_ylabel() == result.y

    issues = fig.check()
    assert issues[: len(result.check())] == result.check()
    assert figure_codes(fig) == []


@pytest.mark.parametrize("case", CASES)
def test_fixture_brackets_are_stacked_by_span(case: OracleCase) -> None:
    result = fixture_result(case)
    fig = default_figure(short(case))
    lines = [fig.figure.findobj(lambda a, m=m: a.get_gid() == f"bracket-{m}")[0]
             for m in range(numbered_count(fig, "bracket"))]  # fmt: skip
    data = [v for group in result.values for v in group]
    span = max(data) - min(data)
    spans: list[tuple[float, float, float]] = []
    for line in lines:
        assert isinstance(line, Line2D)
        xs = np.asarray(line.get_xdata(), dtype=float).tolist()
        ys = np.asarray(line.get_ydata(), dtype=float).tolist()
        assert ys[1] == ys[2] > max(data)
        spans.append((xs[0], xs[3], ys[1]))
    heights = sorted({y for _, _, y in spans})
    for low, high in itertools.pairwise(heights):
        steps = (high - low) / (LEVEL_STEP * span)
        assert steps == pytest.approx(round(steps), abs=1e-9)
        assert round(steps) >= 1
    for a_lo, a_hi, a_y in spans:
        for b_lo, b_hi, b_y in spans:
            if (a_lo, a_hi) != (b_lo, b_hi) and a_lo <= b_lo and b_hi <= a_hi:
                assert a_y > b_y, "an enclosing bracket must sit above"


@pytest.mark.parametrize("case", CASES)
def test_fixture_without_violin_and_centrality(case: OracleCase) -> None:
    result = fixture_result(case)
    fig = betweenstats(result, layers=("box", "points"), centrality_plotting=False)
    present = gids(fig)
    assert not any(g.startswith(("violin-", "centrality", "density-")) for g in present)
    for i in range(len(result.levels)):
        assert {f"box-{i}", f"points-{i}"} <= present
    assert figure_codes(fig) == []


@pytest.mark.parametrize("case", CASES)
def test_fixture_svg_keeps_text_as_text(case: OracleCase, tmp_path: Path) -> None:
    result = fixture_result(case)
    fig = default_figure(short(case))
    texts = figure_texts(result)
    path = tmp_path / "figure.svg"
    fig.save(path)
    written = svg_texts(path)
    # svg.fonttype "none": each line of a plain text is one <text> element.
    for label in texts.tick_labels:
        for line in label.split("\n"):
            assert line in written, line
    plain = [line for line in texts.caption_lines if not line.startswith("$")]
    for line in plain:
        assert line in written
    assert result.y in written
    assert result.x in written
    ids = svg_ids(path)
    assert not any(i.startswith("DejaVuSans") for i in ids), "glyphs drawn as paths"
    for i in range(len(result.levels)):
        assert {f"violin-{i}", f"box-{i}", f"points-{i}"} <= ids
