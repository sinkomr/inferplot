"""Fast checks of the simulation settings' exact rates, true values and reference.

The slow simulation tests centre a kind-2 band on an exact rate or on the rate in
``tests/data/simulation_reference.json``; these tests check that the exact rates
are those of inferplot's own p-values, that the derived true values agree with
the independent Monte Carlo values the reference script recorded, and that the
reference matches the settings it was made for.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from inferplot import stats as st
from tests import simulation_settings as ss


def test_mann_whitney_exact_rate_counts_inferplots_p_values() -> None:
    n1, n2 = 4, 7
    ranks = np.arange(1.0, n1 + n2 + 1.0)
    rejected = total = 0
    for chosen in itertools.combinations(range(n1 + n2), n1):
        mask = np.zeros(n1 + n2, dtype=bool)
        mask[list(chosen)] = True
        p = st.mann_whitney(ranks[mask], ranks[~mask]).p_value
        rejected += p < ss.ALPHA
        total += 1
    assert ss.mann_whitney_exact_rate(n1, n2) == pytest.approx(rejected / total)


SIDES: tuple[st.Alternative, ...] = ("two-sided", "less", "greater")


@pytest.mark.parametrize("side", SIDES)
def test_signed_rank_exact_rate_counts_inferplots_p_values(
    side: st.Alternative,
) -> None:
    n = 9
    magnitudes = np.arange(1.0, n + 1.0)
    rejected = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        x = magnitudes * np.asarray(signs)
        p = st.one_sample_signed_rank(x, alternative=side).p_value
        rejected += p < ss.ALPHA
    assert ss.signed_rank_exact_rate(n, side) == pytest.approx(rejected / 2**n)


@pytest.mark.parametrize(("t", "df"), [(0.5, 14.0), (2.3, 14.0), (3.1, 19.0)])
def test_noncentral_t_quadrature_inverts_inferplots_interval(
    t: float, df: float
) -> None:
    low, high = st.t_ncp_interval(t, df, conf_level=0.95, alternative="two-sided")
    assert ss.noncentral_t_cdf(t, df, low) == pytest.approx(0.975, abs=1e-10)
    assert ss.noncentral_t_cdf(t, df, high) == pytest.approx(0.025, abs=1e-10)


def test_hedges_g_coverage_is_nominal_at_no_effect() -> None:
    assert ss.hedges_g_coverage(15, 0.0, "two-sided") == pytest.approx(0.95, abs=1e-9)
    assert ss.hedges_g_coverage(15, 0.0, "greater") == pytest.approx(0.95, abs=1e-9)


def test_winsorized_normal_variance_is_inferplots() -> None:
    assert ss.winsorized_normal_variance(0.2) == pytest.approx(
        st.winsorized_normal_variance(0.2), rel=1e-12
    )


def test_true_values_agree_with_the_recorded_monte_carlo() -> None:
    # Each Monte Carlo value comes from 2,000,000 draws (400,000 subjects for W);
    # their standard errors are below 1e-3.
    recorded = ss.load_reference()["_true_values"]
    formulas = {
        "winsorized_rho_0.3": ss.winsorized_rho_of_normal(0.3),
        "winsorized_rho_0.8": ss.winsorized_rho_of_normal(0.8),
        "spearman_rho_0.3": ss.spearman_rho_of_normal(0.3),
        "spearman_rho_0.8": ss.spearman_rho_of_normal(0.8),
        "xi": ss.XI_TRUE,
        "epsilon_squared": ss.EPS_TRUE,
        "kendall_w": ss.W_TRUE,
        "lognormal_trimmed_mean": ss.LOGNORMAL_TM,
        "cramers_v_weak": ss.V_WEAK_TRUE,
        "cramers_v_strong": ss.V_STRONG_TRUE,
        "pearsons_c": ss.GOF_C_TRUE,
    }
    assert set(formulas) == set(recorded)
    for name, value in formulas.items():
        assert recorded[name]["formula"] == pytest.approx(value, rel=1e-12), name
        assert recorded[name]["monte_carlo"] == pytest.approx(value, abs=2e-3), name


def test_winsorized_rho_limits() -> None:
    assert ss.winsorized_rho_of_normal(0.0) == 0.0
    assert ss.winsorized_rho_of_normal(0.999999) == pytest.approx(1.0, abs=1e-4)
    assert ss.winsorized_rho_of_normal(-0.5) == pytest.approx(
        -ss.winsorized_rho_of_normal(0.5), rel=1e-10
    )


def test_every_kind_two_rate_has_a_matching_reference() -> None:
    reference = ss.load_reference()
    for setting in ss.ALL:
        assert len({q.key for q in setting.quantities}) == len(setting.quantities)
        needs = [
            q for q in setting.quantities if q.kind == 2 and q.key not in setting.exact
        ]
        if not needs:
            continue
        entry = reference[setting.name]
        assert entry["params"] == setting.params, setting.name
        assert entry["reps"] == setting.reference_reps >= 10_000, setting.name
        assert entry["reps"] >= 5 * setting.reps, setting.name
        assert entry["seed"] == setting.reference_seed != setting.seed, setting.name
        assert set(entry["hits"]) == {q.key for q in setting.quantities}


def test_exact_rates_agree_with_the_reference_simulations() -> None:
    # Where a setting knows its exact rate, the reference simulation of 10,000 or
    # more replications must agree with it within its 99% binomial band.
    reference = ss.load_reference()
    checked = 0
    for setting in ss.ALL:
        if setting.name not in reference:
            continue
        entry = reference[setting.name]
        for key, rate in setting.exact.items():
            low, high = ss.binomial_band(rate, entry["reps"])
            observed = entry["hits"][key] / entry["reps"]
            assert low <= observed <= high, (setting.name, key, observed, rate)
            assert entry["exact"][key] == pytest.approx(rate, abs=1e-9)
            checked += 1
    assert checked >= 3
    assert math.isfinite(ss.OMEGA_TRUE)
