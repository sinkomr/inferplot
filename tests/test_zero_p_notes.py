"""A p-value of exactly 0 printed as 2.23e-308 is explained in ``warnings``.

ggstatsplot prints a p-value of 0 as .Machine$double.xmin, 2.23e-308, and so
does inferplot (D108, the owner's call). Wherever a text prints it so, the
result's ``warnings`` say what it means (review B15), as the one-sample
bootstrap case and the exact correlation already did.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import inferplot as ip


def _zero_notes(result: ip.StatsResult) -> list[str]:
    return [w for w in result.warnings if "is 0 as computed" in w]


def test_barstats_main_test_and_bars() -> None:
    frame = pd.DataFrame(
        {"x": ["x1", "x2", "x1", "x2"], "y": ["y1", "y1", "y2", "y2"]}
    ).assign(n=[100000, 5, 5, 100000])
    result = ip.analyze.barstats(frame, "x", "y", counts="n", bf_message=False)
    assert result.test is not None
    assert result.test.p_value == 0.0
    assert "p = 2.23e-308" in (result.text.subtitle or "")
    main, bars = _zero_notes(result)
    assert "Pearson's chi-square test" in main
    assert "The subtitle prints p = 2.23e-308" in main
    assert "'y1', 'y2'" in bars
    assert "The labels above those bars print" in bars
    # Without the subtitle and the bars' labels nothing prints it.
    quiet = ip.analyze.barstats(
        frame, "x", "y", counts="n", results_subtitle=False, proportion_test=False
    )
    assert _zero_notes(quiet) == []


def test_betweenstats_main_test() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "g": ["a"] * 50 + ["b"] * 50,
            "v": np.r_[rng.normal(0, 0.01, 50), rng.normal(100, 0.01, 50)],
        }
    )
    result = ip.analyze.betweenstats(frame, "g", "v")
    assert result.test is not None
    assert result.test.p_value == 0.0
    (note,) = _zero_notes(result)
    assert "Welch" in note


def test_a_p_above_zero_gets_no_note() -> None:
    frame = pd.DataFrame({"x": list("aabbab"), "y": list("uvuvvu")})
    assert _zero_notes(ip.analyze.barstats(frame, "x", "y", bf_message=False)) == []
