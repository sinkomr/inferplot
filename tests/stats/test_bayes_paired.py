"""The paired JZS Bayes factor (Rouder et al. 2009, eq. 1 with N = n, nu = n - 1)
against high-precision references, BayesFactor's probe-pairedbf and the
ggwithinstats fixtures.

tests/data/pairedbf_reference.json holds 40-digit mpmath quadratures of log BF10
for the 152 rows of probe-pairedbf (``tools/reference/pairedbf_mpmath.py``).
inferplot is compared with them across the whole grid (n 5 to 200, |t| 0 to 1,000,
r 0.707 and 0.8). ``BayesFactor::ttestBF(paired = TRUE)``, which ggstatsplot
calls, misses them at large |t|; following DECISIONS.md D31 and D50, inferplot
reports the accurate value and both sides are pinned: R within
``ttest_bayes_factor`` of the reference everywhere except the rows of
``BAYESFACTOR_MISSES``, where R's relative miss is the recorded one (within 1% of
it), so a stale entry fails instead of hiding a difference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest
from scipy import stats
from tools.reference import stats_reference as ref

from inferplot.stats import jzs_bf10_paired, jzs_log_bf10, paired_t
from tests.oracle_fixtures import cases_for, number, probe, r_double
from tests.tolerances import TOLERANCES, assert_close

REFERENCE: Final = (
    Path(__file__).resolve().parents[1] / "data" / "pairedbf_reference.json"
)
ROWS = probe("probe-pairedbf").rows("rows")
REFERENCE_ROWS = json.loads(REFERENCE.read_text(encoding="utf-8"))["rows"]
RTOL: Final = TOLERANCES["ttest_bayes_factor"].rtol

# probe-pairedbf rows where BayesFactor's log BF10 misses the 40-digit reference
# by more than ``ttest_bayes_factor`` (rtol 1e-8): row -> R's relative miss in BF10,
# exp(R's log BF10 - reference) - 1. Every row with |t| >= 20 at n 5, 10 and 30, and
# with |t| >= 30 at n 200 (plus n 200, t 12, r 0.8); up to |t| 12 R is within 2.6e-6
# (the n 200, t 12, r 0.8 row) and otherwise within 2.3e-10.
BAYESFACTOR_MISSES: Final[dict[int, float]] = {
    16: -0.000634,  # n 5, t 20, r 0.707
    17: -0.000637,  # n 5, t 20, r 0.8
    18: 0.0058,  # n 5, t 30, r 0.707
    19: 0.00582,  # n 5, t 30, r 0.8
    20: 0.00303,  # n 5, t 40, r 0.707
    21: 0.00304,  # n 5, t 40, r 0.8
    22: -0.0163,  # n 5, t 50, r 0.707
    23: -0.0163,  # n 5, t 50, r 0.8
    24: -0.0365,  # n 5, t 70, r 0.707
    25: -0.0365,  # n 5, t 70, r 0.8
    26: -0.0147,  # n 5, t 100, r 0.707
    27: -0.0147,  # n 5, t 100, r 0.8
    28: 0.05,  # n 5, t 150, r 0.707
    29: 0.05,  # n 5, t 150, r 0.8
    30: 0.108,  # n 5, t 200, r 0.707
    31: 0.108,  # n 5, t 200, r 0.8
    32: 0.203,  # n 5, t 300, r 0.707
    33: 0.203,  # n 5, t 300, r 0.8
    34: 0.355,  # n 5, t 500, r 0.707
    35: 0.355,  # n 5, t 500, r 0.8
    36: 0.681,  # n 5, t 1000, r 0.707
    37: 0.68,  # n 5, t 1000, r 0.8
    54: -8.42e-05,  # n 10, t 20, r 0.707
    55: -8.43e-05,  # n 10, t 20, r 0.8
    56: 0.00237,  # n 10, t 30, r 0.707
    57: 0.00238,  # n 10, t 30, r 0.8
    58: 0.00471,  # n 10, t 40, r 0.707
    59: 0.00471,  # n 10, t 40, r 0.8
    60: -0.0166,  # n 10, t 50, r 0.707
    61: -0.0166,  # n 10, t 50, r 0.8
    62: -0.0132,  # n 10, t 70, r 0.707
    63: -0.0132,  # n 10, t 70, r 0.8
    64: 0.0149,  # n 10, t 100, r 0.707
    65: 0.0149,  # n 10, t 100, r 0.8
    66: 0.0329,  # n 10, t 150, r 0.707
    67: 0.0329,  # n 10, t 150, r 0.8
    68: 0.0389,  # n 10, t 200, r 0.707
    69: 0.0389,  # n 10, t 200, r 0.8
    70: 0.0437,  # n 10, t 300, r 0.707
    71: 0.0437,  # n 10, t 300, r 0.8
    72: 0.0478,  # n 10, t 500, r 0.707
    73: 0.0478,  # n 10, t 500, r 0.8
    74: 0.0529,  # n 10, t 1000, r 0.707
    75: 0.0529,  # n 10, t 1000, r 0.8
    94: -0.00131,  # n 30, t 30, r 0.707
    95: -0.00131,  # n 30, t 30, r 0.8
    96: 0.003,  # n 30, t 40, r 0.707
    97: 0.00301,  # n 30, t 40, r 0.8
    98: -0.00779,  # n 30, t 50, r 0.707
    99: -0.00779,  # n 30, t 50, r 0.8
    100: 0.00216,  # n 30, t 70, r 0.707
    101: 0.00215,  # n 30, t 70, r 0.8
    102: 0.00273,  # n 30, t 100, r 0.707
    103: 0.00273,  # n 30, t 100, r 0.8
    104: 0.00275,  # n 30, t 150, r 0.707
    105: 0.00274,  # n 30, t 150, r 0.8
    106: 0.00275,  # n 30, t 200, r 0.707
    107: 0.00275,  # n 30, t 200, r 0.8
    108: 0.00275,  # n 30, t 300, r 0.707
    109: 0.00275,  # n 30, t 300, r 0.8
    110: 0.00275,  # n 30, t 500, r 0.707
    111: 0.00275,  # n 30, t 500, r 0.8
    112: 0.00275,  # n 30, t 1000, r 0.707
    113: 0.00275,  # n 30, t 1000, r 0.8
    129: -2.56e-06,  # n 200, t 12, r 0.8
    132: -3.76e-06,  # n 200, t 30, r 0.707
    133: -3.8e-06,  # n 200, t 30, r 0.8
    134: -0.000679,  # n 200, t 40, r 0.707
    135: -0.000674,  # n 200, t 40, r 0.8
    136: 4.56e-05,  # n 200, t 50, r 0.707
    137: 4.42e-05,  # n 200, t 50, r 0.8
    138: 4.92e-05,  # n 200, t 70, r 0.707
    139: 4.84e-05,  # n 200, t 70, r 0.8
    140: 5.07e-05,  # n 200, t 100, r 0.707
    141: 5.03e-05,  # n 200, t 100, r 0.8
    142: 5.15e-05,  # n 200, t 150, r 0.707
    143: 5.13e-05,  # n 200, t 150, r 0.8
    144: 5.18e-05,  # n 200, t 200, r 0.707
    145: 5.17e-05,  # n 200, t 200, r 0.8
    146: 5.2e-05,  # n 200, t 300, r 0.707
    147: 5.2e-05,  # n 200, t 300, r 0.8
    148: 5.21e-05,  # n 200, t 500, r 0.707
    149: 5.21e-05,  # n 200, t 500, r 0.8
    150: 5.22e-05,  # n 200, t 1000, r 0.707
    151: 5.22e-05,  # n 200, t 1000, r 0.8
}


def _probe_t(row: Mapping[str, Any]) -> float:
    t = r_double(row["t_hex"])
    assert t is not None
    return t


def _probe_data(
    row: Mapping[str, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    a = np.array([r_double(v) for v in row["a_hex"]], dtype=np.float64)
    b = np.array([r_double(v) for v in row["b_hex"]], dtype=np.float64)
    return a, b


def _miss(log_value: float, log_reference: float) -> float:
    """Relative difference in BF10 from the difference of the logs (BF10 itself
    overflows a double beyond log 709)."""
    return math.expm1(log_value - log_reference)


def test_reference_covers_the_probe() -> None:
    assert len(ROWS) == len(REFERENCE_ROWS) == 152
    for index, (row, entry) in enumerate(zip(ROWS, REFERENCE_ROWS, strict=True)):
        assert entry["row"] == index
        assert (entry["n"], entry["rscale_hex"], entry["t_hex"]) == (
            row["n"],
            row["rscale_hex"],
            row["t_hex"],
        )
        assert row["call_error"] is None
    assert {row["n"] for row in ROWS} == {5, 10, 30, 200}
    assert max(row["t_target"] for row in ROWS) == 1000


def test_inferplot_matches_the_reference_across_the_grid() -> None:
    worst = 0.0
    for row, entry in zip(ROWS, REFERENCE_ROWS, strict=True):
        n = int(row["n"])
        scale = r_double(row["rscale_hex"])
        assert scale is not None
        accurate = float(entry["log_bf10"])
        # At R's own t, and from the data through the exact paired t.
        at_t = jzs_log_bf10(_probe_t(row), n, n - 1.0, prior_scale=scale)
        a, b = _probe_data(row)
        bf10, log_bf01 = jzs_bf10_paired(a, b, prior_scale=scale)
        for value in (at_t, -log_bf01):
            miss = _miss(value, accurate)
            worst = max(worst, abs(miss))
            assert abs(miss) <= RTOL, (entry["row"], miss)
        # BF10 overflows a double beyond log 709.78 (n 200, t 500 and 1,000).
        assert bf10 == (math.exp(-log_bf01) if -log_bf01 < 709.0 else math.inf)
    # Measured: 1.1e-13 at most (n 200, t 1,000).
    assert worst < 1e-12


def test_paired_t_of_the_probe_data_is_the_exact_t() -> None:
    # inferplot's t from the exact differences is the exact t to a unit in the last
    # place; R's (rounded differences) is up to 1e-14 off at t 1,000.
    for row, entry in zip(ROWS, REFERENCE_ROWS, strict=True):
        a, b = _probe_data(row)
        t = paired_t(a, b).t
        exact = float(entry["t_exact"])
        if abs(exact) > 1e-10:
            assert abs(t / exact - 1.0) <= 2.3e-16, entry["row"]


def test_bayesfactor_misses_are_pinned() -> None:
    found: dict[int, float] = {}
    for row, entry in zip(ROWS, REFERENCE_ROWS, strict=True):
        r_log = r_double(row["log_bf10_hex"])
        assert r_log is not None
        miss = _miss(r_log, float(entry["log_bf10"]))
        if abs(miss) > RTOL:
            found[int(entry["row"])] = miss
    assert set(found) == set(BAYESFACTOR_MISSES)
    for index, miss in found.items():
        recorded = BAYESFACTOR_MISSES[index]
        assert abs(miss - recorded) <= 0.01 * abs(recorded), (index, miss, recorded)
    # The brief's landmarks: +68% at n 5 and t 1,000, -0.78% at n 30 and t 50.
    assert BAYESFACTOR_MISSES[36] == pytest.approx(0.681, rel=1e-3)
    assert BAYESFACTOR_MISSES[98] == pytest.approx(-0.00779, rel=1e-3)
    small = [
        abs(_miss(r_double(row["log_bf10_hex"]) or 0.0, float(entry["log_bf10"])))
        for row, entry in zip(ROWS, REFERENCE_ROWS, strict=True)
        if row["t_target"] <= 12
    ]
    assert max(small) < 2.6e-6


# --------------------------------------------------------------------------- #
# definition and properties
# --------------------------------------------------------------------------- #

RNG = np.random.Generator(np.random.PCG64(41))
A = RNG.normal(0.0, 1.0, 16)
B = A + RNG.normal(0.4, 0.9, 16)


@pytest.mark.parametrize("prior", [0.5, 0.707, 0.8, 1.0])
def test_is_the_one_sample_bayes_factor_of_the_differences(prior: float) -> None:
    bf10, log_bf01 = jzs_bf10_paired(A, B, prior_scale=prior)
    t = float(stats.ttest_1samp(A - B, 0.0).statistic)
    expected = ref.jzs_bf10_t(t, 16.0, 15.0, r=prior)
    assert_close("ttest_bayes_factor", bf10, expected, what="BF10")
    assert log_bf01 == pytest.approx(-math.log(bf10), rel=1e-15)
    # Not the two-sample (Student's t) Bayes factor of the same values.
    student = float(stats.ttest_ind(A, B).statistic)
    other = ref.jzs_bf10_t(student, 8.0, 30.0, r=prior)
    assert abs(other / bf10 - 1.0) > 1e-3


def test_symmetry_order_and_scale() -> None:
    forward = jzs_bf10_paired(A, B, prior_scale=0.707)
    assert jzs_bf10_paired(B, A, prior_scale=0.707) == forward
    order = RNG.permutation(16)
    shuffled = jzs_bf10_paired(A[order], B[order], prior_scale=0.707)
    assert shuffled[1] == pytest.approx(forward[1], rel=1e-14)
    for power in (-900, -40, 40, 900):
        factor = 2.0**power
        assert jzs_bf10_paired(A * factor, B * factor, prior_scale=0.707) == forward
    # The prior is the number 0.707, not sqrt(2)/2.
    medium = jzs_bf10_paired(A, B, prior_scale=math.sqrt(2.0) / 2.0)
    assert abs(medium[0] / forward[0] - 1.0) > 1e-6


def test_huge_t_stays_finite_in_log_space() -> None:
    a = np.linspace(0.0, 1.0, 300)
    b = a - 50.0 - 1e-3 * np.sin(np.arange(300.0))
    bf10, log_bf01 = jzs_bf10_paired(a, b, prior_scale=0.707)
    assert bf10 == math.inf
    assert math.isfinite(log_bf01)
    assert log_bf01 < -709.0


def test_refuses_constant_differences() -> None:
    a = np.array([1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="every difference"):
        jzs_bf10_paired(a, a + 1.0, prior_scale=0.707)
    with pytest.raises(ValueError, match="prior_scale"):
        jzs_bf10_paired(A, B, prior_scale=0.0)


# --------------------------------------------------------------------------- #
# the ggwithinstats fixtures
# --------------------------------------------------------------------------- #


def _paired_bayes_factors() -> list[Any]:
    """Every paired Bayes factor the ggwithinstats fixtures hold: the type bayes
    subtitles and the parametric captions with two conditions, and the type bayes
    pairs, with their data and prior."""
    out: list[Any] = []
    for case in cases_for("ggwithinstats"):
        columns = {
            level: np.array(v, dtype=np.float64)
            for level, v in case.within_columns().items()
        }
        levels = case.analysis_levels
        prior = float(case.args.get("bf.prior", 0.707))
        if len(levels) == 2:
            row = None
            if case.type == "bayes" and case.subtitle_rows:
                row = case.subtitle_data
            elif case.type == "parametric" and case.caption_data is not None:
                row = case.caption_data
            if row is not None:
                out.append(
                    pytest.param(
                        columns[levels[0]],
                        columns[levels[1]],
                        prior,
                        row,
                        id=case.name,
                    )
                )
        elif case.type == "bayes":
            for row in case.pairwise:
                g1, g2 = str(row["group1"]), str(row["group2"])
                out.append(
                    pytest.param(
                        columns[g1],
                        columns[g2],
                        0.707,
                        row,
                        id=f"{case.name}-{g1}-{g2}",
                    )
                )
    return out


@pytest.mark.parametrize(("a", "b", "prior", "row"), _paired_bayes_factors())
def test_fixture_bayes_factors(
    a: np.ndarray[Any, Any], b: np.ndarray[Any, Any], prior: float, row: Any
) -> None:
    assert number(row["prior.scale"]) == prior
    bf10, log_bf01 = jzs_bf10_paired(a, b, prior_scale=prior)
    assert_close("ttest_bayes_factor", bf10, number(row["bf10"]), what="BF10")
    assert_close(
        "ttest_bayes_factor", -log_bf01, number(row["log_e_bf10"]), what="log BF10"
    )
