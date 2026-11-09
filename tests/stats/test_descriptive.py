"""The correctly rounded mean and the sample variance."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inferplot.stats import mean, variance
from tests.oracle_fixtures import cases_for
from tests.stats.fixture_data import groups


def _exact_mean(values: list[float]) -> float:
    # Fraction -> float is correctly rounded (integer true division).
    return float(sum((Fraction(v) for v in values), Fraction(0)) / len(values))


def _exact_variance(values: list[float]) -> float:
    exact = [Fraction(v) for v in values]
    m = sum(exact, Fraction(0)) / len(exact)
    return float(sum(((v - m) ** 2 for v in exact), Fraction(0)) / (len(exact) - 1))


finite = st.floats(allow_nan=False, allow_infinity=False, width=64)


@given(st.lists(finite, min_size=1, max_size=60))
def test_mean_is_correctly_rounded(values: list[float]) -> None:
    exact = sum((Fraction(v) for v in values), Fraction(0)) / len(values)
    if abs(exact) > Fraction(1.7976931348623157e308):
        return  # the mean itself is not a finite double
    assert mean(np.array(values)) == _exact_mean(values)


def test_mean_decimal_tie_that_numpy_misses() -> None:
    # The fixtures' "12.22" centrality label needs the correctly rounded mean; the
    # sum in another order lands one ulp away and prints "12.23".
    values = [12.1, 12.2, 12.225, 12.25, 12.35]
    assert mean(values) == _exact_mean(values)
    rng = np.random.Generator(np.random.PCG64(3))
    misses = 0
    for _ in range(300):
        x = np.round(rng.normal(10.0, 3.0, int(rng.integers(3, 40))), 2)
        assert mean(x) == _exact_mean(x.tolist())
        misses += float(np.mean(x)) != _exact_mean(x.tolist())
    assert misses > 0  # NumPy's pairwise sum is not correctly rounded


def test_mean_extremes() -> None:
    big = 1.7976931348623157e308
    assert mean([big, big]) == big  # the exact sum overflows; the mean does not
    assert mean([5e-324, 5e-324, 0.0]) == _exact_mean([5e-324, 5e-324, 0.0])
    assert mean([1e300, 1.0, -1e300]) == pytest.approx(1.0 / 3.0, rel=1e-16)
    assert mean([2.5]) == 2.5
    assert mean([-1.0, -2.0]) == -1.5
    assert mean(np.full(1000, 0.1)) == 0.1


def test_mean_of_every_fixture_group_is_correctly_rounded() -> None:
    for case in cases_for("ggbetweenstats"):
        if case.error is None:
            for values in groups(case):
                assert mean(values) == _exact_mean(values.tolist()), case.name


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=2, max_size=40))
def test_variance_is_accurate(values: list[float]) -> None:
    exact = _exact_variance(values)
    assert variance(values) == pytest.approx(exact, rel=1e-13, abs=1e-290)


def test_variance_cases() -> None:
    assert variance([3.0, 3.0, 3.0]) == 0.0
    assert variance([1.0, 2.0]) == 0.5
    # Cancellation-prone data: a large offset does not change the variance.
    x = np.array([1e9 + 0.1, 1e9 + 0.2, 1e9 + 0.3, 1e9 + 0.4])
    assert variance(x) == pytest.approx(_exact_variance(x.tolist()), rel=1e-12)
    np.testing.assert_allclose(variance(x), np.var(x, ddof=1), rtol=1e-6)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "at least 1"),
        ([1.0, math.nan], "finite"),
        ([1.0, math.inf], "finite"),
        ([[1.0, 2.0]], "one-dimensional"),
    ],
)
def test_mean_rejects_bad_input(values: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        mean(values)  # type: ignore[arg-type]


def test_variance_needs_two_values() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        variance([1.0])


def test_results_are_python_floats() -> None:
    assert type(mean(np.arange(5.0))) is float
    assert type(variance(np.arange(5.0))) is float
