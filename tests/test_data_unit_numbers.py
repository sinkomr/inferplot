"""Quantities in the data's units are printed with the subtitle's number rule.

A review found the robust histostats APA sentence printing the trimmed mean and
its interval with two fixed decimals: "trimmed mean = 0.00, 95% CI [0.00, 0.00]"
for data in 1e-9 units, 13-digit numbers at 1e12. The trimmed mean is the first
effect size in the data's units; it, its bounds, the histogram's range in the
alt text and the check messages that print them follow SPEC section 7's number
rule (``format_value`` with ``digits``), which leaves ordinary magnitudes as
they were.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot.result import StatsResult
from inferplot.text import data_number, format_value


def _robust(scale: float, test_value: float, **kwargs: Any) -> StatsResult:
    values = np.random.default_rng(5).normal(3.0, 1.0, 20) * scale
    return ip.analyze.histostats(
        pd.DataFrame({"x": values}), "x", type="robust", test_value=test_value, **kwargs
    )


@pytest.mark.parametrize(
    ("scale", "test_value", "expected"),
    [
        (1e-9, 2.5e-9, "trimmed mean = 2.70e-09, 95% CI [2.23e-09, 3.18e-09]"),
        (1e12, 2.5e12, "trimmed mean = 2.70e+12, 95% CI [2.23e+12, 3.18e+12]"),
        (1.0, 2.5, "trimmed mean = 2.70, 95% CI [2.23, 3.18]"),
    ],
)
def test_trimmed_mean_in_the_data_units(
    scale: float, test_value: float, expected: str
) -> None:
    result = _robust(scale, test_value)
    assert expected in result.text.apa
    assert expected in result.text.alt_text
    es = result.effect_size
    assert es is not None
    assert format_value(es.estimate, 2) in expected


def test_digits_apply_to_the_trimmed_mean() -> None:
    result = _robust(1.0, 2.5, digits=3)
    es = result.effect_size
    assert es is not None
    assert f"trimmed mean = {es.estimate:.3f}, " in result.text.apa


def test_histogram_range_in_the_alt_text() -> None:
    constant = ip.analyze.histostats(
        pd.DataFrame({"x": [5.0] * 10}), "x", type="nonparametric", test_value=3
    )
    assert constant.text.alt_text.startswith(
        "Histogram of x (10 values, all equal to 5.00"
    )
    # Ends that would print alike get the digits that tell them apart.
    seconds = ip.analyze.histostats(
        pd.DataFrame({"x": 1.7e9 + np.arange(10.0)}), "x", type="nonparametric"
    )
    assert "from 1.70000000e+09 to 1.70000001e+09" in seconds.text.alt_text
    tiny = ip.analyze.histostats(
        pd.DataFrame({"x": np.linspace(1e-9, 4e-9, 12)}), "x", test_value=0
    )
    assert "from 1.00e-09 to 4.00e-09" in tiny.text.alt_text
    ordinary = ip.analyze.histostats(
        pd.DataFrame({"x": [4.2, 10.0, 20.0, 33.9]}), "x", test_value=18
    )
    assert "from 4.20 to 33.90" in ordinary.text.alt_text


def test_infinite_bound_message_in_the_data_units() -> None:
    result = _robust(1e-9, 2.5e-9)
    es = result.effect_size
    assert es is not None
    broken = dataclasses.replace(
        result,
        effect_size=dataclasses.replace(
            es, infinite_resamples=3, conf_high=float("inf")
        ),
    )
    (issue,) = [i for i in broken.check() if i.code == "infinite_effect_size"]
    assert f"95% CI [{data_number(es.conf_low, 2)}, Inf]" in issue.message
    assert "e-09" in issue.message


def test_data_number_rule() -> None:
    assert data_number(19.2449, 2) == "19.24"
    assert data_number(-0.0001, 2) == "-1.00e-04"
    assert data_number(-0.004, 2) == "-4.00e-03"
    assert data_number(-0.001, 0) == "-1e-03"
    assert data_number(-1e-12 * 0, 2) == "0.00"
    assert data_number(123456.0, 2) == "1.23e+05"
    assert data_number(float("inf"), 2) == "Inf"
    assert data_number(float("nan"), 2) == "NA"
