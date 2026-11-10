"""inferplot's rank tests and WRS2 p-values against probe-ranks and probe-wrs2p.

tests/test_reference_vs_oracle.py checks these probes against SciPy and inline
reference code only; here they go through inferplot's own functions:

* probe-ranks: ``wilcox.test(exact = FALSE)``, ``effectsize::rank_biserial``,
  ``kruskal.test`` and ``PMCMRplus::kwAllPairsDunnTest`` on tied data, against
  ``mann_whitney``, ``rank_biserial``, ``kruskal_wallis`` and ``dunn`` (tolerance
  ``closed_form``). ggstatsplot runs ``wilcox.test`` with the continuity
  correction, the only variant ``mann_whitney`` has; the uncorrected rows are a
  negative control.
* probe-wrs2p: WRS2's yuen, t1way and lincon p-values at R's own t, F and df,
  against robust.py's accurate upper tails (``wrs2_p``: R computes 1 - cdf), and
  R's exact upper tail at the same points (``closed_form``).
"""

from __future__ import annotations

import numpy as np
import pytest

from inferplot.stats import dunn, kruskal_wallis, mann_whitney, rank_biserial
from inferplot.stats.parametric import Alternative
from inferplot.stats.robust import (
    f_upper_p,
    two_sided_t_p,
)
from tests.oracle_fixtures import Row, probe, r_double
from tests.tolerances import assert_close, is_close

ALTERNATIVES: dict[str, Alternative] = {
    "two.sided": "two-sided",
    "less": "less",
    "greater": "greater",
}


def _x(text: object) -> float:
    value = r_double(str(text))
    assert value is not None, f"unexpected NA: {text!r}"
    return value


def _sample(hexes: list[str]) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return np.array([_x(h) for h in hexes], dtype=np.float64)


def _rows(probe_name: str, section: str) -> list[Row]:
    return probe(probe_name).rows(section)


@pytest.mark.parametrize(
    "entry", _rows("probe-ranks", "two_sample"), ids=lambda e: str(e["name"])
)
def test_mann_whitney_against_wilcox_test(entry: Row) -> None:
    a, b = _sample(entry["a_hex"]), _sample(entry["b_hex"])
    corrected = uncorrected_differ = 0
    for test in entry["wilcox"]:
        what = f"{entry['name']} {test['alternative']} correct={test['correct']}"
        assert test["error"] is None, what
        got = mann_whitney(a, b, alternative=ALTERNATIVES[str(test["alternative"])])
        assert_close("closed_form", got.statistic, _x(test["statistic_hex"]), what=what)
        if test["correct"]:
            corrected += 1
            assert_close("closed_form", got.p_value, _x(test["p_value_hex"]), what=what)
        else:
            uncorrected_differ += not is_close(
                "closed_form", got.p_value, _x(test["p_value_hex"])
            )
    assert corrected == 3
    # Negative control: without the correction R's p is another number.
    assert uncorrected_differ == 3


@pytest.mark.parametrize(
    "entry", _rows("probe-ranks", "two_sample"), ids=lambda e: str(e["name"])
)
def test_rank_biserial_against_effectsize(entry: Row) -> None:
    a, b = _sample(entry["a_hex"]), _sample(entry["b_hex"])
    assert len(entry["rank_biserial"]) == 3
    for out in entry["rank_biserial"]:
        side = ALTERNATIVES[str(out["alternative"])]
        what = f"{entry['name']} rank-biserial {side}"
        assert out["error"] is None, what
        got = rank_biserial(a, b, conf_level=0.95, alternative=side)
        assert_close("closed_form", got.estimate, _x(out["estimate_hex"]), what=what)
        assert_close("closed_form", got.conf_low, _x(out["conf_low_hex"]), what=what)
        assert_close("closed_form", got.conf_high, _x(out["conf_high_hex"]), what=what)


@pytest.mark.parametrize(
    "entry", _rows("probe-ranks", "k_sample"), ids=lambda e: str(e["name"])
)
def test_kruskal_wallis_and_dunn_against_r(entry: Row) -> None:
    levels = [str(v) for v in entry["levels"]]
    groups = [_sample(g) for g in entry["groups_hex"]]
    assert len(groups) == len(levels)
    name = str(entry["name"])
    kw = kruskal_wallis(groups)
    assert kw.df == float(entry["kruskal_parameter"])
    assert_close(
        "closed_form", kw.statistic, _x(entry["kruskal_statistic_hex"]), what=name
    )
    assert_close("closed_form", kw.p_value, _x(entry["kruskal_p_value_hex"]), what=name)
    pairs = dunn(groups)
    assert len(pairs) == len(entry["dunn_pairs"])
    for pair, row in zip(pairs, entry["dunn_pairs"], strict=True):
        what = f"{name} Dunn {row['group1']}-{row['group2']}"
        assert (levels[pair.i], levels[pair.j]) == (row["group1"], row["group2"])
        # ggstatsplot reports |z|; kwAllPairsDunnTest's sign follows its own order.
        assert_close(
            "closed_form", pair.statistic, abs(_x(row["statistic_hex"])), what=what
        )
        assert_close("closed_form", pair.p_value, _x(row["p_value_hex"]), what=what)


def test_yuen_p_against_wrs2() -> None:
    rows = _rows("probe-wrs2p", "yuen")
    assert len(rows) == 84
    for row in rows:
        what = f"yuen n={row['n']} shift={row['shift']}"
        t, df = abs(_x(row["test_hex"])), _x(row["df_hex"])
        got = two_sided_t_p(t, df)
        # WRS2's 2 (1 - pt): the tail rounded to 2^-53, doubled.
        assert_close("wrs2_p", got, _x(row["p_value_hex"]), what=what, atol_scale=2)
        # R's own accurate tail at the same point.
        expected = min(1.0, 2 * _x(row["upper_tail_hex"]))
        assert_close("closed_form", got, expected, what=what)


def test_t1way_p_against_wrs2() -> None:
    rows = _rows("probe-wrs2p", "t1way")
    assert len(rows) == 84
    for row in rows:
        what = f"t1way n={row['n']} shift={row['shift']}"
        f, df1, df2 = (_x(row[k]) for k in ("test_hex", "df1_hex", "df2_hex"))
        got = f_upper_p(f, df1, df2)
        assert_close("wrs2_p", got, _x(row["p_value_hex"]), what=what)
        assert_close("closed_form", got, _x(row["upper_tail_hex"]), what=what)


def test_lincon_p_against_wrs2() -> None:
    entries = _rows("probe-wrs2p", "lincon")
    assert len(entries) == 84
    checked = 0
    for entry in entries:
        for pair in entry["pairs"]:
            what = f"lincon n={entry['n']} shift={entry['shift']} {pair['group1']}"
            t, df = abs(_x(pair["yuen_test_hex"])), _x(pair["yuen_df_hex"])
            got = two_sided_t_p(t, df)
            # lincon's unadjusted p (what ggstatsplot adjusts) is the pair's Yuen p.
            assert_close(
                "wrs2_p", got, _x(pair["p_value_hex"]), what=what, atol_scale=2
            )
            expected = min(1.0, 2 * _x(pair["upper_tail_hex"]))
            assert_close("closed_form", got, expected, what=what)
            checked += 1
    assert checked > 84
