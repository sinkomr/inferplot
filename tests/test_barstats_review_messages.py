"""barstats' messages after the review (visible half): each says what barstats
does, never another function's rule, and every call it advises works."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import inferplot as ip

FRAME = pd.DataFrame({"x": list("aabbab"), "y": list("uvuvvu")})
anything: Any = ip.barstats  # for calls with keywords barstats does not take
analysis: Any = ip.analyze.barstats


def _message(error: type[Exception], call: Any) -> str:
    with pytest.raises(error) as caught:
        call()
    return str(caught.value)


# -- B8: a direction ---------------------------------------------------------------


@pytest.mark.parametrize(
    "keyword",
    ["tail", "tails", "sided", "sides", "one_sided", "alternative_hypothesis"],
)
@pytest.mark.parametrize("function", [anything, analysis])
def test_direction_keywords_get_barstats_wording(keyword: str, function: Any) -> None:
    text = _message(TypeError, lambda: function(FRAME, "x", "y", **{keyword: 1}))
    assert f"{keyword}= is not an argument of barstats()" in text
    assert "barstats' tests have no direction" in text
    assert "only makes the effect-size intervals one-sided" in text
    assert "barstats has no order=" in text
    assert "first in order=" not in text
    assert "first level of x is greater" not in text


@pytest.mark.parametrize("value", ["one-sided", "one.sided", "one-tailed", "onesided"])
def test_one_sided_values_get_barstats_wording(value: str) -> None:
    text = _message(ValueError, lambda: ip.barstats(FRAME, "x", "y", alternative=value))
    assert "barstats' tests have no direction" in text
    assert "first in order=" not in text
    # The values it names work.
    for side in ("greater", "less", "two-sided"):
        ip.analyze.barstats(FRAME, "x", "y", alternative=side, bf_message=False)
