"""The alt text describes what the figure draws (D70; review of phase 3, finding
7): a centrality path only when a line is drawn, no label text for a level that
draws no label, and no centrality values when centrality_plotting=False."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import inferplot as ip


def _long(columns: list[NDArray[np.float64]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (f"s{i:02d}", f"c{j + 1}", float(col[i]))
            for i in range(len(columns[0]))
            for j, col in enumerate(columns)
        ],
        columns=["id", "cond", "y"],
    )


def _no_map() -> list[NDArray[np.float64]]:
    # More than half of c1's values equal: no MAP estimate there (D55).
    rng = np.random.default_rng(0)
    a = np.array([5, 5, 5, 5, 5, 5, 5, 5, 6, 4.0])
    b = a + rng.normal(1, 1, 10)
    return [a, b, b + rng.normal(0, 1, 10)]


def test_no_path_is_named_where_no_line_is_drawn() -> None:
    cols = _no_map()
    fig = ip.withinstats(_long(cols[:2]), "cond", "y", subject_id="id", type="bayes")
    result = fig.result
    assert result.descriptives[0].value is None
    assert result.text.centrality_labels[0] == ""
    assert result.text.centrality_labels[1].startswith("μ_MAP = ")
    alt = result.text.alt_text
    assert "c1 (no MAP estimate)" in alt
    assert "lines join each subject's values." in alt
    assert "MAP estimates." not in alt
    # The analysis' own alt text (the default figure) says the same.
    analyzed = ip.analyze.withinstats(
        _long(cols[:2]), "cond", "y", subject_id="id", type="bayes"
    )
    assert analyzed.text.alt_text == alt


def test_a_broken_path_with_a_drawn_piece_is_named() -> None:
    fig = ip.withinstats(
        _long(_no_map()), "cond", "y", subject_id="id", type="bayes",
        results_subtitle=False,
    )  # fmt: skip
    assert "; a line joins the MAP estimates." in fig.result.text.alt_text


def test_without_centrality_points_no_centrality_values() -> None:
    rng = np.random.default_rng(5)
    cols = [rng.normal(10 + j, 1, 12) for j in range(3)]
    for k in (2, 3):
        fig = ip.withinstats(
            _long(cols[:k]), "cond", "y", subject_id="id", centrality_plotting=False
        )
        alt = fig.result.text.alt_text
        assert "mean" not in alt.split(".")[0]
        names = ", ".join(f"c{j + 1}" for j in range(k))
        assert f"measured on the same 12 subjects: {names}" in alt
        assert "joins the means" not in alt
        assert "and the means" not in alt
    frame = _long(cols).drop(columns="id")
    fig = ip.betweenstats(frame, "cond", "y", centrality_plotting=False)
    alt = fig.result.text.alt_text
    assert alt.startswith(
        "Plot comparing y across the 3 groups of cond: c1 (n = 12), c2 (n = 12), "
        "c3 (n = 12)."
    )
    drawn = ip.betweenstats(frame, "cond", "y")
    assert "mean" in drawn.result.text.alt_text.split(".")[0]
