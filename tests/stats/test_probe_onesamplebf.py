"""One-sample JZS Bayes factors: BayesFactor::ttestBF(x, mu) against inferplot.

probe-onesamplebf holds 304 calls of ``ttestBF(x = x, mu = mu, rscale = r)``: n = 5,
10, 30 and 200 values, |t| from 0 to 1,000, mu 0 and 3, r 0.707 and 0.8, with R's
one-sample t and log BF10. inferplot's JZS integral (``jzs_log_bf10``, within 1.1e-13
of 40-digit references on probe-pairedbf, D61) evaluated at R's own t agrees with
BayesFactor wherever BayesFactor's quadrature holds: every row with |t| up to 8 is
within ``ttest_bayes_factor`` (at most 5.2e-9). Beyond that BayesFactor misses as it
does for paired data (D61): 2.6e-6 at |t| 12, up to 68% at n 5 and |t| 1,000, while
its own error estimate stays near 1e-20 or below.
"""

from __future__ import annotations

import math

from inferplot.stats.bayes import jzs_log_bf10
from tests.oracle_fixtures import probe, r_double
from tests.tolerances import assert_close

ROWS = probe("probe-onesamplebf").rows("rows")


def _exact(text: str) -> float:
    value = r_double(text)
    assert value is not None
    return value


def test_grid() -> None:
    assert len(ROWS) == 4 * 2 * 19 * 2
    assert {row["n"] for row in ROWS} == {5, 10, 30, 200}
    assert {row["mu"] for row in ROWS} == {0, 3}
    assert all(row["call_error"] is None for row in ROWS)


def test_bayesfactor_agrees_where_its_quadrature_holds() -> None:
    worst = 0.0
    for row in ROWS:
        if row["t_target"] > 8:
            continue
        t = _exact(row["t_hex"])
        n = float(row["n"])
        mine = jzs_log_bf10(t, n, n - 1.0, prior_scale=float(row["rscale"]))
        theirs = _exact(row["log_bf10_hex"])
        assert_close(
            "ttest_bayes_factor",
            math.exp(mine),
            math.exp(theirs),
            what=f"n {row['n']} mu {row['mu']} t {row['t_target']} r {row['rscale']}",
        )
        worst = max(worst, abs(math.expm1(theirs - mine)))
    assert worst < 6e-9


def test_bayesfactor_misses_at_large_t() -> None:
    # The largest miss over the grid, at n 5 and |t| 1,000 (paired data: +68% too).
    misses = []
    for row in ROWS:
        t = _exact(row["t_hex"])
        n = float(row["n"])
        mine = jzs_log_bf10(t, n, n - 1.0, prior_scale=float(row["rscale"]))
        misses.append(math.expm1(_exact(row["log_bf10_hex"]) - mine))
    worst = max(misses, key=abs)
    assert 0.6 < worst < 0.75, worst
