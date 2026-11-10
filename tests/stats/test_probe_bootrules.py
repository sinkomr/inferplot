"""Bootstrap percentile rules read from R's own replicates (probe-bootrules).

For each bootstrapped interval ggstatsplot 1.1.4 uses, the probe traced the vector of
replicate values the R function sorted (base R's ``sort.int``) or bootstrapped
(``boot::boot``'s ``t``), next to the bounds it returned, at B = 39, 40, 99, 100, 199
and 200 resamples and, where the function takes a level, 95% and 90%. The candidate
rules pick different order statistics at these B, so a rule that reproduces R's
bounds from R's own replicates at all of them is R's rule (review of Phases 0-3, T6).
The 4 Monte Carlo SD parity rule cannot tell such rules apart (about 0.14 SD).

Then the T7 data sets whose resamples include undefined (0/0) values: what each
function does with them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from inferplot.stats.nonparametric import (
    _percentile_lower,  # pyright: ignore[reportPrivateUsage]
)
from inferplot.stats.robust import (
    _rounded_percentile_bounds,  # pyright: ignore[reportPrivateUsage]
    floor_percentile_bounds,
)
from tests.oracle_fixtures import Row, probe, r_double

FloatArray = NDArray[np.float64]
B_GRID = (39, 40, 99, 100, 199, 200)
# inferplot's resample count (SPEC.md section 5).
INFERPLOT_RESAMPLES = 2000


def _value(text: str) -> float:
    value = r_double(text)
    return math.nan if value is None else value


def _values(texts: Sequence[str]) -> FloatArray:
    return np.array([_value(t) for t in texts], dtype=np.float64)


def _rule(name: str) -> Row:
    (rule,) = [r for r in probe("probe-bootrules").rows("rules") if r["name"] == name]
    return rule


def _rows(name: str) -> list[Row]:
    rows = list(_rule(name)["rows"])
    assert sorted({row["B"] for row in rows}) == list(B_GRID)
    for row in rows:
        assert row["error"] is None, (name, row["B"], row["error"])
        assert row["unbalanced_traces"] == 0
    return rows


def _bounds(row: Row) -> tuple[float, float, float]:
    values = _values(row["values_hex"])
    return float(values[0]), float(values[1]), float(values[2])


def _sorted_inputs(row: Row) -> list[FloatArray]:
    """Every vector of length B the function sorted, sorted."""
    return [np.sort(_values(call["input_hex"])) for call in row["traced"]]


def _conf(row: Row) -> float:
    return 0.95 if row["conf"] is None else float(row["conf"])


def _ids(name: str) -> list[Any]:
    return [
        pytest.param(row, id=f"B{row['B']}-conf{_conf(row):g}") for row in _rows(name)
    ]


@pytest.mark.parametrize("row", _ids("akp.effect"))
def test_akp_effect_takes_the_floor_rule(row: Row) -> None:
    # WRS2::akp.effect sorts its B replicates once; inferplot's floor rule applied to
    # them gives R's bounds exactly: the (floor(alpha/2 B) + 1)-th and
    # floor((1 - alpha/2) B)-th smallest.
    (ordered,) = _sorted_inputs(row)
    _, low, high = _bounds(row)
    conf = _conf(row)
    if math.floor((1.0 - conf) / 2.0 * ordered.size) < 1:
        # B = 39 at 95%: the rule's lower bound is the smallest resample, which R
        # returns and inferplot refuses (review S4: the interval would stop widening
        # with the level).
        assert (low, high) == _floor_rule(ordered, conf)
        assert low == ordered[0]
        with pytest.raises(ValueError, match="too close to 1"):
            floor_percentile_bounds(ordered, conf)
        return
    assert floor_percentile_bounds(ordered, conf) == (low, high)


def _floor_rule(ordered: FloatArray, conf: float) -> tuple[float, float]:
    """The floor rule's order statistics, also where inferplot refuses them."""
    alpha = 1.0 - conf
    low = math.floor(alpha / 2.0 * ordered.size)
    high = math.floor((1.0 - alpha / 2.0) * ordered.size)
    return float(ordered[low]), float(ordered[high - 1])


@pytest.mark.parametrize("row", _ids("t1way"))
def test_t1way_takes_the_rounded_rule(row: Row) -> None:
    # t1way sorts the same replicate vector twice (once per bound); the rounded rule
    # with alpha = 1 - conf.level in floating point gives R's bounds exactly.
    first, second = _sorted_inputs(row)
    assert np.array_equal(first, second)
    _, low, high = _bounds(row)
    assert _rounded_percentile_bounds(first, _conf(row)) == (low, high)


def _boot_t(row: Row) -> FloatArray:
    (boot,) = row["boots"]
    return _values(boot["t_hex"])


def _assert_percentile_lower(t: FloatArray, conf: float, low: float) -> None:
    """boot.ci's percentile bound: exact where (R + 1) alpha is a whole order
    statistic; where it interpolates on the normal scale, within 3e-15 relative
    (R's qnorm and SciPy's ndtri, and the order of the arithmetic)."""
    alpha = 1.0 - conf
    mine = _percentile_lower(t, alpha)
    position = (t.size + 1) * alpha
    if position == math.floor(position):
        assert mine == low
    else:
        assert math.isclose(mine, low, rel_tol=3e-15, abs_tol=0.0), (mine, low)


@pytest.mark.parametrize("name", ["rank_epsilon_squared", "kendalls_w"])
def test_effectsize_takes_boot_ci_percentile_rule(name: str) -> None:
    # effectsize bootstraps with boot::boot and takes boot.ci's one-sided percentile
    # bound, the (R + 1) alpha order statistic interpolated on the normal scale, as
    # inferplot's _percentile_lower does; the upper bound is 1.
    for row in _rows(name):
        t = _boot_t(row)
        assert t.size == row["B"]
        assert not np.isnan(t).any()
        _, low, high = _bounds(row)
        assert high == 1.0
        _assert_percentile_lower(t, _conf(row), low)


def _round_rule(ordered: FloatArray, alpha: float) -> tuple[float, float]:
    """l = round(alpha / 2 B), R's rounding half to even; the (l + 1)-th and
    (B - l)-th smallest."""
    count = ordered.size
    low = round(alpha / 2.0 * count)
    return float(ordered[low]), float(ordered[count - low - 1])


# B at which dep.effect's rule (the rounded rule at alpha exactly 0.05) and the floor
# rule inferplot applies to it pick different order statistics: which bound differs.
DEP_EFFECT_FLOOR_DIFFERS = {39: "low", 99: "high", 100: "high", 199: "low"}


@pytest.mark.parametrize("row", _ids("dep.effect"))
def test_dep_effect_takes_the_rounded_rule_at_alpha_005(row: Row) -> None:
    # WRS2::dep.effect takes no level: its AKP interval is the (l + 1)-th and
    # (B - l)-th smallest with l = round(0.025 B), alpha exactly 0.05 (B = 100: the 3rd
    # and 98th). It sorts three replicate vectors; the first is AKP's.
    ordered = _sorted_inputs(row)[0]
    _, low, high = _bounds(row)
    assert _round_rule(ordered, 0.05) == (low, high)
    # inferplot applies the floor rule to dep.effect (D66). The two pick the same
    # order statistics at inferplot's 2,000 resamples, not at every B.
    floor = _floor_rule(ordered, 0.95)
    differs = DEP_EFFECT_FLOOR_DIFFERS.get(row["B"])
    assert (floor[0] != low) == (differs == "low")
    assert (floor[1] != high) == (differs == "high")


def test_dep_effect_rules_agree_at_inferplots_resample_count() -> None:
    ordered = np.arange(INFERPLOT_RESAMPLES, dtype=np.float64)
    assert floor_percentile_bounds(ordered, 0.95) == _round_rule(ordered, 0.05)


@pytest.mark.parametrize("row", _ids("wmcpAKP"))
def test_wmcpakp_averages_pair_bounds_from_separate_runs(row: Row) -> None:
    # WRS2::wmcpAKP sorts 27 vectors for three conditions (three pairs): per pair,
    # three bootstrap runs of dep.effect's kind, each sorting three vectors, AKP's
    # first. The lower bound is the mean over pairs of the second run's lower bound,
    # the upper bound the mean of the third run's upper bound, each by dep.effect's
    # rule: the two bounds come from different resamples.
    ordered = _sorted_inputs(row)
    assert len(ordered) == 27
    _, low, high = _bounds(row)
    lows = [_round_rule(ordered[9 * p + 3], 0.05)[0] for p in range(3)]
    highs = [_round_rule(ordered[9 * p + 6], 0.05)[1] for p in range(3)]
    assert math.isclose(math.fsum(lows) / 3, low, rel_tol=1e-15, abs_tol=1e-15)
    assert math.isclose(math.fsum(highs) / 3, high, rel_tol=1e-15, abs_tol=1e-15)


@pytest.mark.parametrize("row", _ids("trimcibt"))
def test_trimcibt_is_a_symmetric_bootstrap_t(row: Row) -> None:
    # WRS2::trimcibt sorts |T*| = |trimmed mean* - trimmed mean| / se* over B
    # resamples (all n B values drawn by one sample() call, resample b taking draws
    # b, b + B, ...). The interval is trimmed mean -/+ c se with c the
    # round((1 - alpha) B)-th smallest |T*| (alpha = 1 - conf.level as computed), and
    # p is the share of |T*| at least |T|.
    (abs_t,) = _sorted_inputs(row)
    assert (abs_t >= 0).all()
    values = _values(row["values_hex"])
    estimate, low, high, p_value, statistic = (float(v) for v in values)
    se = (estimate - 0.5) / statistic  # nv = 0.5
    alpha = 1.0 - _conf(row)
    c = float(abs_t[round((1.0 - alpha) * abs_t.size) - 1])
    assert math.isclose(estimate - c * se, low, rel_tol=1e-14)
    assert math.isclose(estimate + c * se, high, rel_tol=1e-14)
    assert p_value == float(np.mean(abs_t >= abs(statistic)))


def _undefined(name: str) -> list[Row]:
    (entry,) = [
        u for u in probe("probe-bootrules").rows("undefined") if u["name"] == name
    ]
    rows = list(entry["rows"])
    assert {(row["B"], row["seed"]) for row in rows} == {
        (b, s) for b in (100, 2000) for s in (20261107, 1, 2)
    }
    return rows


@pytest.mark.parametrize("name", ["edge_akp_undefined", "edge_akp_undefined2"])
def test_akp_effect_drops_undefined_resamples_but_counts_positions_from_b(
    name: str,
) -> None:
    # A 0/0 resample (group a winsorized to a constant equal to b's) is NaN. sort()
    # drops it, while the bound positions stay floor(alpha/2 B) + 1 and
    # floor((1 - alpha/2) B) of all B: the upper bound moves up by the number dropped,
    # and is NA once more than alpha/2 of the resamples are undefined. That is the
    # floor rule with the undefined resamples sorted last; inferplot's D65 (positions
    # counted over the defined resamples) is another rule.
    differs_from_d65 = 0
    for row in _undefined(name):
        (call,) = row["traced"]
        given = _values(call["input_hex"])
        kept = _values(call["output_hex"])
        assert np.array_equal(np.sort(given[~np.isnan(given)]), kept)
        dropped = int(np.isnan(given).sum())
        assert dropped > 0
        b = row["B"]
        last = np.concatenate([kept, np.full(dropped, math.nan)])
        low, high = floor_percentile_bounds(last, 0.95)
        _, r_low, r_high = _bounds(row)
        assert low == r_low
        assert (math.isnan(high) and math.isnan(r_high)) or high == r_high
        assert math.isnan(r_high) == (dropped > b - math.floor(0.975 * b))
        d65 = floor_percentile_bounds(kept, 0.95)
        differs_from_d65 += d65 != (r_low, r_high)
    assert differs_from_d65 == 6


@pytest.mark.parametrize("name", ["edge_tied3_small", "edge_rm_some_tied"])
def test_effectsize_does_not_count_undefined_resamples_as_zero(name: str) -> None:
    # Resamples with every value tied (rank epsilon squared) or of constant subjects
    # only (Kendall's W) are NaN in boot's t. R's lower bound is the percentile of the
    # defined values (D54); counting the undefined ones as 0 would give 0 wherever
    # more than alpha of them are undefined (edge_tied3_small: 6% to 9%).
    for row in _undefined(name):
        t = _boot_t(row)
        undefined = int(np.isnan(t).sum())
        defined = t[~np.isnan(t)]
        _, low, high = _bounds(row)
        assert high == 1.0
        assert low > 0.0
        _assert_percentile_lower(defined, 0.95, low)
        if undefined > 0.05 * t.size:
            as_zero = np.where(np.isnan(t), 0.0, t)
            assert _percentile_lower(as_zero, 0.05) == 0.0
