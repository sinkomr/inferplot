"""JZS Bayes factors of t statistics (Rouder et al. 2009)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from tools.reference import stats_reference as ref

from inferplot.stats import bayes, jzs_bf10_two_sample, jzs_log_bf10
from tests.tolerances import assert_close

CASES = [
    (3.5, 10.0, 38.0),  # two samples of 20 (pingouin's documented example)
    (0.0, 7.5, 28.0),
    (-1.2, 4.0, 14.0),
    (2.0, 50.0, 198.0),
    (15.0, 25.0, 98.0),
    (0.5, 1e4, 4e4),
]


@pytest.mark.parametrize(("t", "n_eff", "df"), CASES)
@pytest.mark.parametrize("prior", [0.5, 0.707, 0.8, 1.0, 2.0])
def test_agrees_with_reference_quadrature(
    t: float, n_eff: float, df: float, prior: float
) -> None:
    # tools/reference integrates over g with SciPy's adaptive quadrature;
    # inferplot uses the trapezoidal rule over log g. Independent routes.
    ours = math.exp(jzs_log_bf10(t, n_eff, df, prior_scale=prior))
    assert_close(
        "ttest_bayes_factor", ours, ref.jzs_bf10_t(t, n_eff, df, r=prior), what="BF10"
    )


def test_published_example() -> None:
    # pingouin's documentation (validated there against JASP and BayesFactor):
    # t = 3.5, two samples of 20, r = 0.707 gives BF10 = 26.743 (3 decimals).
    bf10 = math.exp(jzs_log_bf10(3.5, 10.0, 38.0, prior_scale=0.707))
    assert round(bf10, 3) == 26.743


def test_symmetric_in_t_and_prior_dependent() -> None:
    plus = jzs_log_bf10(2.4, 12.0, 46.0, prior_scale=0.707)
    minus = jzs_log_bf10(-2.4, 12.0, 46.0, prior_scale=0.707)
    assert plus == minus
    # BayesFactor's "medium" sqrt(2)/2 is not 0.707: the difference shows.
    medium = jzs_log_bf10(2.4, 12.0, 46.0, prior_scale=math.sqrt(2.0) / 2.0)
    assert abs(math.exp(plus - medium) - 1.0) > 1e-6


def test_huge_t_stays_finite_in_log_space() -> None:
    log_bf = jzs_log_bf10(400.0, 2500.0, 9998.0, prior_scale=0.707)
    assert 700.0 < log_bf < 1e6
    a = np.linspace(0.0, 1.0, 200)
    bf10, log_bf01 = jzs_bf10_two_sample(a, a + 50.0, prior_scale=0.707)
    assert bf10 == math.inf
    assert math.isfinite(log_bf01)
    assert log_bf01 < -709.0


@pytest.mark.parametrize("t", [1e100, 1e150, 2.0**512, 1e200, 1e300, -1.7e308])
@pytest.mark.parametrize(("df", "n_eff"), [(2.0, 1.0), (4.0, 1.5), (1e5, 2e4)])
def test_huge_t_follows_the_large_t_limit(t: float, df: float, n_eff: float) -> None:
    # Beyond |t| = 2^512 t^2 / df overflowed (RuntimeWarning, then a wrong or
    # infinite log). For large |t| the integral over g is dominated by
    # g ~ t^2, where it has the closed form
    # BF10 = (n_eff r^2 / (2 pi))^(1/2) 2 / (df - 1) (t^2 / df)^((df - 1) / 2).
    r = 0.707
    limit = (
        0.5 * (df - 1.0) * (2.0 * math.log(abs(t)) - math.log(df))
        + math.log(2.0 / (df - 1.0))
        + 0.5 * math.log(n_eff * r * r / (2.0 * math.pi))
    )
    assert jzs_log_bf10(t, n_eff, df, prior_scale=r) == pytest.approx(limit, rel=1e-13)


def test_evidence_for_the_null_grows_with_n() -> None:
    values = [jzs_log_bf10(0.0, n, 2 * n - 2, prior_scale=0.707) for n in (5, 50, 500)]
    assert values == sorted(values, reverse=True)
    assert values[-1] < 0.0


def test_two_sample_uses_students_t() -> None:
    rng = np.random.Generator(np.random.PCG64(2))
    a, b = rng.normal(0.0, 1.0, 12), rng.normal(0.8, 3.0, 30)
    bf10, log_bf01 = jzs_bf10_two_sample(a, b, prior_scale=0.707)
    student = float(stats.ttest_ind(a, b, equal_var=True).statistic)
    expected = ref.jzs_bf10_t(student, 12 * 30 / 42, 40.0, r=0.707)
    assert_close("ttest_bayes_factor", bf10, expected, what="BF10")
    assert log_bf01 == pytest.approx(-math.log(bf10), rel=1e-15)
    # Welch's t would give a different value for unequal sizes.
    welch = float(stats.ttest_ind(a, b, equal_var=False).statistic)
    assert abs(ref.jzs_bf10_t(welch, 12 * 30 / 42, 40.0, r=0.707) / bf10 - 1) > 1e-4


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((math.nan, 10.0, 18.0, 0.707), "t must"),
        ((1.0, 0.0, 18.0, 0.707), "n_eff"),
        ((1.0, 10.0, -1.0, 0.707), "df"),
        ((1.0, 10.0, 18.0, 0.0), "prior_scale"),
        ((1.0, 10.0, 18.0, math.inf), "prior_scale"),
    ],
)
def test_validation(args: tuple[float, float, float, float], message: str) -> None:
    t, n_eff, df, prior = args
    with pytest.raises(ValueError, match=message):
        jzs_log_bf10(t, n_eff, df, prior_scale=prior)


@pytest.mark.parametrize(("t", "n_eff", "df"), CASES)
def test_refining_the_step_does_not_change_the_result(
    t: float, n_eff: float, df: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Self-convergence: four more halvings of the trapezoid step (16 times the
    # nodes) agree with the default to 1e-13 in the log.
    default = jzs_log_bf10(t, n_eff, df, prior_scale=0.707)
    monkeypatch.setattr(bayes, "_MIN_HALVINGS", 5)
    refined = jzs_log_bf10(t, n_eff, df, prior_scale=0.707)
    assert refined == pytest.approx(default, rel=1e-13, abs=1e-13)
