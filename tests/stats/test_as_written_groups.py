"""Numbers that are the same as written are the same (DECISIONS.md D74) in the
robust bootstraps of two or more groups and of paired data: the AKP effect size,
the explanatory measure of effect size, the paired AKP and its average over pairs.
Every resample is classified as an exact-decimal replay of it is
(tools/reference/decimal_replay.py), whatever decimal unit or shift the data are
written in (Phase 4 review, Z1)."""

from __future__ import annotations

import numpy as np
import pytest
from tools.reference import decimal_replay

from inferplot.stats.robust import akp_effect, explanatory_effect
from inferplot.stats.within_robust import average_dependent_akp, dependent_akp
from tests.stats.test_as_written import _writings

# ------------------------------------------------------------------ between groups


def _between_cases() -> list[list[list[float]]]:
    rng = np.random.default_rng(12)
    return [
        [rng.integers(1, 6, 8).astype(float).tolist() for _ in range(2)]
        for _ in range(12)
    ]


@pytest.mark.parametrize("groups", _between_cases())
@pytest.mark.parametrize("writing", ["whole", "tenths", "shifted"])
def test_akp_against_the_exact_decimal_replay(
    groups: list[list[float]], writing: str
) -> None:
    # Review Z1 (b): the AKP resamples were classified on rounded doubles: 22 of
    # 60 rating data sets in tenths counted 0 / 0 and infinite resamples
    # differently from the whole numbers (a bound of 1.20 became 1.24).
    a, b = (_writings(g)[writing][0] for g in groups)
    try:
        got = akp_effect(np.array(a), np.array(b))
    except ValueError:
        return
    want = decimal_replay.akp(a, b)
    assert (got.undefined_resamples, got.infinite_resamples) == (
        want["undefined"],
        want["infinite"],
    )
    assert got.conf_low == pytest.approx(want["conf_low"], rel=1e-9, abs=1e-12)
    assert got.conf_high == pytest.approx(want["conf_high"], rel=1e-9, abs=1e-12)


def test_akp_numerator_zero_as_written() -> None:
    # A resample whose trimmed means are equal as written has AKP 0, as the whole
    # numbers give; in tenths it was -7.9e-17 and became a lower bound printed
    # "-0.00" (lead's a4g).
    rng = np.random.default_rng(12)
    for _ in range(60):
        a = rng.integers(1, 6, 8).astype(float)
        b = rng.integers(1, 6, 8).astype(float)
        try:
            whole = akp_effect(a, b)
        except ValueError:
            continue
        for name, (x, _, _, _) in _writings(a.tolist()).items():
            y = _writings(b.tolist())[name][0]
            got = akp_effect(np.array(x), np.array(y))
            assert (got.undefined_resamples, got.infinite_resamples) == (
                whole.undefined_resamples,
                whole.infinite_resamples,
            ), name
            assert got.conf_low == pytest.approx(whole.conf_low, rel=1e-12, abs=0.0), (
                name
            )
            assert got.conf_high == pytest.approx(
                whole.conf_high, rel=1e-12, abs=0.0
            ), name
            assert got.estimate == pytest.approx(whole.estimate, rel=1e-12, abs=0.0), (
                name
            )


def _xi_cases() -> list[list[list[float]]]:
    rng = np.random.default_rng(5)
    return [
        [
            rng.integers(1, 4, int(rng.integers(6, 10))).astype(float).tolist()
            for _ in range(3)
        ]
        for _ in range(8)
    ]


@pytest.mark.parametrize("groups", _xi_cases())
@pytest.mark.parametrize("writing", ["whole", "tenths", "shifted"])
def test_explanatory_effect_against_the_exact_decimal_replay(
    groups: list[list[float]], writing: str
) -> None:
    data = [_writings(g)[writing][0] for g in groups]
    try:
        got = explanatory_effect([np.array(d) for d in data])
    except ValueError:
        return
    want = decimal_replay.xi(data)
    assert (got.undefined_resamples, got.infinite_resamples) == (
        want["undefined"],
        want["infinite"],
    )
    assert got.conf_low == pytest.approx(want["conf_low"], rel=1e-9, abs=1e-12)
    assert got.conf_high == pytest.approx(want["conf_high"], rel=1e-9, abs=1e-12)


# ------------------------------------------------------------------ within subjects


def _within_cases(k: int) -> list[list[list[float]]]:
    rng = np.random.default_rng(77 + k)
    out = []
    for _ in range(8):
        n = int(rng.integers(6, 15))
        a = rng.integers(1, 6, n).astype(float)
        columns = [a]
        for _ in range(k - 1):
            columns.append(np.clip(columns[-1] + rng.choice([-1.0, 0.0, 1.0], n), 1, 5))
        out.append([c.tolist() for c in columns])
    return out


@pytest.mark.parametrize("conditions", _within_cases(2) + _within_cases(3))
@pytest.mark.parametrize("writing", ["whole", "tenths", "shifted"])
def test_within_akp_against_the_exact_decimal_replay(
    conditions: list[list[float]], writing: str
) -> None:
    data = [np.array(_writings(c)[writing][0]) for c in conditions]
    try:
        got = dependent_akp(*data) if len(data) == 2 else average_dependent_akp(data)
    except ValueError:
        return
    want = decimal_replay.dependent([d.tolist() for d in data])
    assert (got.pair_undefined or (got.undefined_resamples,)) == want["undefined"]
    assert (got.pair_infinite or (got.infinite_resamples,)) == want["infinite"]
    assert got.conf_low == pytest.approx(want["conf_low"], rel=1e-9, abs=1e-12)
    assert got.conf_high == pytest.approx(want["conf_high"], rel=1e-9, abs=1e-12)


def test_within_differences_computed_in_binary_are_read_as_written() -> None:
    # Ratings times 0.3 computed in binary (0.8999999999999999 for 0.9) are up to
    # 0.8 units in the last place off the decimals; the paired differences equal as
    # written were tied only within half a unit per value, and 5 of 2,000
    # resamples were counted differently from the whole numbers. Differences are
    # now tied on the values as written (``written_values``).
    a = np.array([3.0, 3, 3, 4, 4, 1, 5])
    b = np.array([2.0, 5, 3, 5, 5, 3, 5])
    whole = dependent_akp(a, b)
    assert (whole.undefined_resamples, whole.infinite_resamples) == (17, 32)
    for factor in (0.3, 2.54, 1.1):
        got = dependent_akp(a * factor, b * factor)
        assert (got.undefined_resamples, got.infinite_resamples) == (17, 32), factor
        assert got.conf_low == pytest.approx(whole.conf_low, rel=1e-12)
        assert got.conf_high == pytest.approx(whole.conf_high, rel=1e-12)
