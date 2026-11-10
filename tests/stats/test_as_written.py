"""Numbers that are the same as written are the same (DECISIONS.md D74), in the
robust bootstraps: the same data written in whole numbers, in tenths, shifted by a
tenth, or in another decimal unit give the same classification of every resample
(0 / 0, infinite, T* at least |t|), the same counts and p, and the same bounds
after rescaling; an exact-decimal replay of every resample
(tools/reference/decimal_replay.py) agrees with each (Phase 4 review, Z1)."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from tools.reference import decimal_replay

from inferplot.stats.one_sample import trimmed_mean_bootstrap_t
from inferplot.stats.written import written_value, written_values

# ------------------------------------------------------------------ readings


@pytest.mark.parametrize(
    ("value", "reading"),
    [
        (0.3, Fraction(3, 10)),
        (0.1 + 0.2, Fraction(3, 10)),  # one unit above 0.3's double
        (3 * 0.3, Fraction(9, 10)),  # 0.8999999999999999
        (3 * 1.1, Fraction(33, 10)),  # 3.3000000000000003
        (-0.7, Fraction(-7, 10)),
        (36.8, Fraction(368, 10)),
        (123456789.1, Fraction(1234567891, 10)),
        (2.5, Fraction(5, 2)),  # exact in binary
        (0.0, Fraction(0)),
    ],
)
def test_written_value_reads_short_decimals(value: float, reading: Fraction) -> None:
    assert written_value(value) == reading


@pytest.mark.parametrize(
    "value",
    [
        math.pi,
        1 / 3,
        1.7e9 + 0.123,  # 13 significant digits: too fine next to the rounding
        0.3 / 2**30,  # 24 significant digits
        float(np.nextafter(0.3, 1.0) + 2 * math.ulp(0.3)),  # three units off 0.3
        5e-324,
    ],
)
def test_values_without_a_short_decimal_are_read_as_stored(value: float) -> None:
    assert written_value(value) == Fraction(value)


def test_measured_data_are_read_as_stored() -> None:
    # Values measured or computed to many digits are not taken for decimals: of
    # 100,000 standard normal draws none is.
    x = np.random.default_rng(1).normal(size=100_000)
    assert written_values(x) == [Fraction(v) for v in x.tolist()]


def test_readings_are_symmetric_and_follow_decimal_units() -> None:
    k = np.arange(-60, 61, dtype=np.float64)
    tenths = written_values(k / 10)
    assert tenths == [Fraction(int(v), 10) for v in k.tolist()]
    assert written_values(k * 2.54) == [Fraction(int(v) * 254, 100) for v in k.tolist()]
    assert written_values(-k / 10) == [-r for r in tenths]


# ------------------------------------------------------------------ one sample

_RATINGS = [1.0, 2, 2, 3, 3, 3, 3, 4, 4, 5, 3, 2, 4, 3, 3]


def _text(value: Decimal) -> float:
    return float(value)


# The same whole numbers written another way: (map of a whole number, unit, shift).
_WRITING_MAPS = {
    "whole": (lambda v: v, 1.0, 0.0),
    "tenths": (lambda v: v / 10, 0.1, 0.0),
    "shifted": (lambda v: _text(Decimal(repr(v)) / 10 + Decimal("0.1")), 0.1, 0.1),
    "shifted in binary": (lambda v: v / 10 + 0.1, 0.1, 0.1),
    "times 0.3 in binary": (lambda v: v * 0.3, 0.3, 0.0),
    "times 2.54": (lambda v: _text(Decimal(repr(v)) * Decimal("2.54")), 2.54, 0.0),
}


def _writings(
    values: list[float], mu: float = 0.0
) -> dict[str, tuple[list[float], float, float, float]]:
    """The same whole numbers written as (values, mu, unit, shift)."""
    return {
        name: ([f(float(v)) for v in values], f(mu), unit, shift)
        for name, (f, unit, shift) in _WRITING_MAPS.items()
    }


def test_one_sample_zero_over_zero_as_written() -> None:
    # Phase 4 review Z1 (high): in tenths, the 250 resamples whose winsorized values
    # are all the trimmed mean 0.3 were counted infinite (the trimmed mean of 0.2,
    # 0.3 and 0.4 is not the double 0.3): p .227 and CI [-Inf, Inf] instead of
    # p .117 and CI [2.35, 3.65].
    whole = trimmed_mean_bootstrap_t(np.array(_RATINGS), mu=2.5)
    assert (whole.undefined_resamples, whole.infinite_resamples) == (250, 0)
    for name, (x, mu, unit, shift) in _writings(_RATINGS, 2.5).items():
        got = trimmed_mean_bootstrap_t(np.array(x), mu=mu)
        counts = (got.undefined_resamples, got.infinite_resamples)
        assert counts == (250, 0), name
        assert got.p_value == whole.p_value, name
        for bound, want in (
            (got.conf_low, whole.conf_low),
            (got.conf_high, whole.conf_high),
        ):
            assert (bound - shift) / unit == pytest.approx(want, rel=1e-12), name


@pytest.mark.parametrize(
    ("x", "mu"),
    [
        ([0.1, 0.2, 0.3], 0.2),
        ([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5], 0.3),
        ([1.1, 2.2, 2.2, 2.2, 3.3, 3.3, 3.3, 3.3, 4.4, 4.4, 4.4, 5.5], 3.3),
        ([v * 1.1 for v in [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5]], 3.3),
    ],
)
def test_one_sample_t_is_zero_when_the_trimmed_mean_is_mu_as_written(
    x: list[float], mu: float
) -> None:
    # Review Z1 (a4f): t was about 2e-16 and p .773 to .883, where the same data
    # in whole numbers give t = 0 and p = 1.
    result = trimmed_mean_bootstrap_t(np.array(x), mu=mu)
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_one_sample_ties_with_t_as_written() -> None:
    # Resamples whose T* equals |t| as written count toward p (T* >= |t|): 1.3% of
    # all resamples for these ratings against 3 (review a4e). In tenths the tie
    # was broken by rounding, p .584 instead of .6025.
    x = [1.0, 2, 2, 3, 4, 4, 5, 5]
    whole = trimmed_mean_bootstrap_t(np.array(x), mu=3.0)
    for name, (values, mu, _, _) in _writings(x, 3.0).items():
        got = trimmed_mean_bootstrap_t(np.array(values), mu=mu)
        assert got.p_value == whole.p_value, name
    replay = decimal_replay.one_sample(x, 3.0)
    assert whole.p_value == replay["p"]


def test_one_sample_draw_order_follows_the_values_as_written() -> None:
    # Review Z6 (6): the deviations from the trimmed mean are drawn in the order
    # (|d|, d) of the deviations as written, so a decimal unit or shift does not
    # change which resamples are drawn (in tenths, -0.1 and +0.1 from the trimmed
    # mean are 0.09999999999999998 and 0.10000000000000003 in binary and swapped).
    x = [2.0, 1, 4, 4, 1, 1, 3, 2, 5, 3]
    whole = trimmed_mean_bootstrap_t(np.array(x), mu=2.5, seed=3)
    for name, (values, mu, _, _) in _writings(x, 2.5).items():
        got = trimmed_mean_bootstrap_t(np.array(values), mu=mu, seed=3)
        assert (got.undefined_resamples, got.infinite_resamples) == (
            whole.undefined_resamples,
            whole.infinite_resamples,
        ), name
        assert got.p_value == whole.p_value, name
        assert got.critical_value == pytest.approx(whole.critical_value, rel=1e-12), (
            name
        )


_ONE_SAMPLE_CASES = [
    ([1.0, 2, 3, 4, 5], 2.0, 0.2),
    ([2.0, 3, 3, 3, 3, 4, 4, 5, 3, 3], 2.5, 0.2),
    (_RATINGS, 2.5, 0.2),
    ([1.0, 2, 2, 3, 4, 4, 5, 5], 3.0, 0.2),
    ([1.0, 2, 2, 3, 4, 4, 5, 5], 3.0, 0.0),
    ([368.0, 371, 366, 368, 368, 372, 369, 368, 365, 368, 370], 368.0, 0.2),
    ([31.0, 47, 22, 59, 44], 30.0, 0.2),
]


@pytest.mark.parametrize(("x", "mu", "tr"), _ONE_SAMPLE_CASES)
@pytest.mark.parametrize("writing", ["whole", "tenths", "shifted"])
def test_one_sample_against_the_exact_decimal_replay(
    x: list[float], mu: float, tr: float, writing: str
) -> None:
    values, mu_w, _, _ = _writings(x, mu)[writing]
    got = trimmed_mean_bootstrap_t(np.array(values), mu=mu_w, tr=tr)
    want = decimal_replay.one_sample(values, mu_w, tr=tr)
    assert got.undefined_resamples == want["undefined"]
    assert got.infinite_resamples == want["infinite"]
    assert got.p_value == want["p"]
    assert (got.statistic == 0.0) == want["t_zero"]
    for bound, reference in (
        (got.conf_low, want["conf_low"]),
        (got.conf_high, want["conf_high"]),
    ):
        assert bound == pytest.approx(reference, rel=1e-12, abs=1e-12)
