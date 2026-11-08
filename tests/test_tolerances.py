"""The tolerance registry is well formed and its helpers decide correctly."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.tolerances import (
    BOOTSTRAP,
    MAP_CENTRALITY,
    TOLERANCES,
    assert_close,
    assert_same_text,
    assert_within_grid,
    assert_within_monte_carlo,
    is_close,
)

EXPECTED_KINDS = {"closed_form", "games_howell_p", "noncentral_ci", "bayes_factor"}


def test_registry_has_the_spec_entries() -> None:
    assert set(TOLERANCES) == EXPECTED_KINDS


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_every_entry_is_bounded_and_explained(kind: str) -> None:
    tol = TOLERANCES[kind]
    assert 0.0 <= tol.rtol <= 1e-3
    assert 0.0 <= tol.atol <= 1e-3
    assert len(tol.reason) > 40


def test_spec_values() -> None:
    # SPEC.md section 10 starting values; a change needs a measured reason. Both rtol
    # and atol are pinned: a nonzero atol would make every small p-value "close".
    expected = {
        "closed_form": (1e-10, 0.0),
        "games_howell_p": (1e-6, 0.0),
        "noncentral_ci": (1e-6, 0.0),
        "bayes_factor": (1e-4, 0.0),
    }
    for kind, (rtol, atol) in expected.items():
        assert (TOLERANCES[kind].rtol, TOLERANCES[kind].atol) == (rtol, atol), kind
    assert BOOTSTRAP.n_sd == 4.0
    assert MAP_CENTRALITY.steps == 1.0


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        TOLERANCES["closed_form"] = TOLERANCES["bayes_factor"]  # type: ignore[index]


finite = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e300, max_value=1e300
)


@given(finite)
def test_a_value_is_close_to_itself(x: float) -> None:
    for kind in EXPECTED_KINDS:
        assert is_close(kind, x, x)


@given(
    st.floats(min_value=1e-300, max_value=1e300),
    st.sampled_from(sorted(EXPECTED_KINDS)),
)
def test_relative_boundary(x: float, kind: str) -> None:
    rtol = TOLERANCES[kind].rtol
    assert is_close(kind, x * (1 + rtol / 2), x)
    assert not is_close(kind, x * (1 + rtol * 2), x)


def test_special_values() -> None:
    assert is_close("closed_form", math.inf, math.inf)
    assert not is_close("closed_form", math.inf, -math.inf)
    assert not is_close("closed_form", 1e308, math.inf)
    assert is_close("closed_form", math.nan, math.nan)
    assert not is_close("closed_form", 0.0, math.nan)


def test_assert_close_reports_the_measured_difference() -> None:
    with pytest.raises(AssertionError, match=r"Welch F: .*rel diff 1e-08.*closed_form"):
        assert_close("closed_form", 1.00000001, 1.0, what="Welch F")


def test_monte_carlo_band() -> None:
    assert_within_monte_carlo(0.53, 0.50, 0.01, what="AKP lower bound")
    with pytest.raises(AssertionError, match="Monte Carlo SD"):
        assert_within_monte_carlo(0.55, 0.50, 0.01, what="AKP lower bound")


def test_grid_band() -> None:
    assert_within_grid(1.04, 1.0, 0.05, what="MAP")
    with pytest.raises(AssertionError, match="grid step"):
        assert_within_grid(1.11, 1.0, 0.05, what="MAP")


def test_text_is_exact() -> None:
    assert_same_text("0.04", "0.04", what="p")
    with pytest.raises(AssertionError):
        assert_same_text("0.040", "0.04", what="p")
