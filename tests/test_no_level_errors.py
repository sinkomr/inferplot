"""Data histostats and scatterstats cannot use, at each type's boundary, and the
hints that lead elsewhere (SPEC.md section 2).

Each refusal is a plain ``ValueError`` naming the column, raised before the
method runs, and every remedy it names is followed here to a call that works
(review rule: no hint may lead to another error). Each type's minimum is tested
on both sides of its boundary:

* histostats: 2 values for every type (``read_one``); a standard deviation for
  types parametric, bayes and robust (also to within the rounding of the values,
  D74); a value other than ``test_value`` for type nonparametric; at least 2
  values after trimming and a winsorized standard deviation for type robust, and
  a ``conf_level`` the bootstrap-t rule can resolve;
* scatterstats: 3 complete pairs (``read_pair``); 4 for the types with a Fisher z
  interval; a standard deviation in both columns for every type; for type robust,
  both columns varying after winsorizing.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot.result import StatsResult

H: Any = ip.analyze.histostats
S: Any = ip.analyze.scatterstats
TOP_H: Any = ip.histostats
TOP_S: Any = ip.scatterstats
DRAW_H: Any = ip.render.histostats
DRAW_S: Any = ip.render.scatterstats
TYPES = ("parametric", "nonparametric", "robust", "bayes")


def one(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"v": values})


def pair(a: list[float], b: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"a": a, "b": b})


def message(error: type[Exception], thunk: Any) -> str:
    with pytest.raises(error) as info:
        thunk()
    assert type(info.value) is error
    return str(info.value)


def follow(text: str) -> dict[str, Any]:
    """The keyword changes a message advises ("Pass a smaller tr (tr=0.1)", "use
    type='nonparametric'"), each as one alternative call."""
    out: dict[str, Any] = {}
    found = re.search(r"\(tr=([0-9.]+)\)", text)
    if found:
        out["tr"] = float(found.group(1))
    found = re.search(r"use type='(\w+)'", text, re.IGNORECASE)
    if found:
        out["type"] = found.group(1)
    return out


# --------------------------------------------------------------------------- #
# histostats
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type_", TYPES)
def test_histostats_minimum_is_two_values(type_: str) -> None:
    assert "needs at least 2 values of x='v'" in message(
        ValueError, lambda: H(one([1.0]), "v", type=type_)
    )
    result = H(one([1.0, 3.0]), "v", type=type_)
    assert result.n_obs == 2
    assert [i.code for i in result.check()][:1] == ["small_n"]


def test_two_values_give_hedges_g_of_zero_with_a_warning() -> None:
    result = H(one([1.0, 3.0]), "v")
    assert result.effect_size is not None
    assert (result.effect_size.estimate, result.effect_size.conf_low) == (0.0, 0.0)
    assert result.warnings[0].startswith("Hedges' g is 0 by construction with 2 values")
    assert "Hedges' g is 0" not in " ".join(H(one([1.0, 3.0, 4.0]), "v").warnings)


@pytest.mark.parametrize("type_", ["parametric", "robust", "bayes"])
@pytest.mark.parametrize(
    ("values", "start"),
    [
        ([5.0, 5.0, 5.0], "Every value of x='v' is 5, so "),
        (
            [0.1 + 0.2, 0.3, 0.3],
            "The values of x='v' differ only by the rounding of double precision "
            "(from 0.3 to 0.30000000000000004): as written they are one number, so ",
        ),
    ],
)
def test_no_spread_is_refused_and_the_signed_rank_test_advised(
    type_: str, values: list[float], start: str
) -> None:
    text = message(ValueError, lambda: H(one(values), "v", type=type_))
    assert text.startswith(start)
    assert "has no standard deviation" in text
    assert text.endswith(
        "Use type='nonparametric': the signed-rank test needs only values that "
        "differ from test_value."
    )
    result = H(one(values), "v", **follow(text))
    assert result.type == "nonparametric"
    assert result.effect_size is not None
    assert result.effect_size.estimate == 1.0
    assert "rank-biserial r is at its bound (1)" in result.warnings[0]


@pytest.mark.parametrize("type_", TYPES)
def test_every_value_equal_to_test_value_advises_nothing_else(type_: str) -> None:
    text = message(
        ValueError, lambda: H(one([5.0, 5.0]), "v", type=type_, test_value=5)
    )
    # No other type is offered: each would refuse these values too.
    assert "type=" not in text
    assert "there is no difference to test" in text.lower()
    assert "check the data for entry errors" in text


def test_the_signed_rank_test_needs_one_value_other_than_test_value() -> None:
    result = H(one([2.0, 3.0]), "v", type="nonparametric", test_value=2)
    assert result.test is not None
    assert result.test.statistic == 1.0  # one value ranked, the other dropped


def test_robust_needs_two_values_after_trimming() -> None:
    # n = 3, tr = 0.4: floor(1.2) = 1 trimmed at each end leaves 1.
    text = message(ValueError, lambda: H(one([1.0, 2.0, 4.0]), "v", type="r", tr=0.4))
    assert text == (
        "For x='v', trimming 40% at each end (tr=0.4) leaves 1 of the 3 values; the "
        "trimmed mean's bootstrap-t test needs at least 2. Pass a smaller tr "
        "(tr=0.2), or use type='nonparametric'."
    )
    assert H(one([1.0, 2.0, 4.0]), "v", type="r", tr=0.2).n_obs == 3
    assert H(one([1.0, 2.0, 4.0]), "v", type="np").n_obs == 3
    # n = 4, tr = 0.4: 1 trimmed at each end leaves 2, the boundary.
    result = H(one([1.0, 2.0, 4.0, 7.0]), "v", type="r", tr=0.4)
    assert result.effect_size is not None
    assert result.effect_size.estimate == 3.0


def test_robust_winsorized_to_one_number_offers_a_smaller_tr_that_works() -> None:
    values = [1.0] * 8 + [2.0, 3.0]
    text = message(ValueError, lambda: H(one(values), "v", type="r"))
    assert text == (
        "For x='v', after winsorizing 20% at each end (tr=0.2) every value is the "
        "same number, so the winsorized standard deviation is 0 and the bootstrap-t "
        "test of the trimmed mean is undefined. Pass a smaller tr (tr=0.1), or use "
        "type='nonparametric'."
    )
    assert H(one(values), "v", type="robust", tr=0.1).n_obs == 10
    assert H(one(values), "v", type="nonparametric").n_obs == 10
    # One value apart from the rest: only tr=0 (no winsorizing) works.
    almost = [1.0] * 9 + [2.0]
    text = message(ValueError, lambda: H(one(almost), "v", type="r", tr=0.1))
    assert text.endswith("Pass a smaller tr (tr=0.0), or use type='nonparametric'.")
    assert H(one(almost), "v", type="r", tr=0.0).n_obs == 10


@pytest.mark.parametrize(
    ("level", "ok"),
    [(0.9998, False), (0.999, True), (0.0002, False), (0.001, True)],
)
def test_robust_conf_level_must_be_resolvable(level: float, ok: bool) -> None:
    def call() -> StatsResult:
        return H(one([1.0, 2.0, 4.0, 7.0, 9.0, 3.0]), "v", type="r", conf_level=level)

    if ok:
        assert call().settings.conf_level == level
        return
    text = message(ValueError, call)
    end = "1" if level > 0.5 else "0"
    assert text == (
        f"conf_level={level!r} is too close to {end} for the bootstrap-t interval "
        "from 2,000 resamples: its bounds would come from the most extreme resample "
        "whatever the level. Pass a conf_level between 0.001 and 0.999, or use "
        "type='parametric', whose interval is not resampled."
    )


def test_infinite_bootstrap_t_is_kept_and_reported() -> None:
    result = H(one([1.0, 3.0]), "v", type="r")
    es = result.effect_size
    assert es is not None
    assert es.infinite_resamples is not None
    assert es.infinite_resamples > 100
    assert (es.conf_low, es.conf_high) == (-math.inf, math.inf)
    assert (
        "so the trimmed mean's interval is unbounded: [-Inf, Inf]"
        in (result.warnings[0])
    )
    issue = next(i for i in result.check() if i.code == "infinite_effect_size")
    assert issue.message.startswith(f"{es.infinite_resamples:,} of 2,000 bootstrap")
    assert "is unbounded" in issue.message
    assert StatsResult.from_json(result.to_json()) == result


# --------------------------------------------------------------------------- #
# scatterstats
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type_", TYPES)
def test_scatterstats_minimum_pairs(type_: str) -> None:
    assert "at least 3 complete pairs" in message(
        ValueError, lambda: S(pair([1.0, 2], [2.0, 1]), "a", "b", type=type_)
    )
    three = pair([1.0, 2, 4], [2.0, 1, 5])
    four = pair([1.0, 2, 4, 3], [2.0, 1, 5, 3])
    assert S(four, "a", "b", type=type_).n_obs == 4
    if type_ == "bayes":
        assert S(three, "a", "b", type=type_).n_obs == 3
        return
    text = message(ValueError, lambda: S(three, "a", "b", type=type_))
    assert text.startswith("x='a' and y='b' have 3 complete pairs; the Fisher z")
    assert text.endswith(
        "Collect more pairs, or use type='bayes', whose Bayes factor needs 3."
    )
    assert S(three, "a", "b", **follow(text)).bayes is not None


@pytest.mark.parametrize("type_", TYPES)
@pytest.mark.parametrize(
    ("a", "b", "start"),
    [
        ([5.0, 5, 5, 5], [2.0, 1, 5, 3], "Every value of x='a' is 5, so "),
        ([1.0, 2, 4, 3], [2.0, 2, 2, 2], "Every value of y='b' is 2, so "),
        (
            [0.1 + 0.2, 0.3, 0.3, 0.3],
            [2.0, 1, 5, 3],
            "The values of x='a' differ only by the rounding of double precision",
        ),
    ],
)
def test_a_column_without_spread_is_refused_for_every_type(
    type_: str, a: list[float], b: list[float], start: str
) -> None:
    text = message(ValueError, lambda: S(pair(a, b), "a", "b", type=type_))
    assert text.startswith(start)
    assert text.endswith(
        "its standard deviation is 0 and a correlation with it is undefined for "
        "every type (ggstatsplot prints NA for every number). Check the data for "
        "entry errors."
    )


def test_robust_winsorized_column_offers_what_works() -> None:
    frame = pair([1.0] * 8 + [2.0, 3.0], [float(v) for v in range(10)])
    text = message(ValueError, lambda: S(frame, "a", "b", type="r"))
    assert text == (
        "After winsorizing 20% at each end (tr=0.2), every value of x='a' is the "
        "same number, so its winsorized standard deviation is 0 and the winsorized "
        "correlation is undefined. Pass a smaller tr (tr=0.1), or use "
        "type='nonparametric'."
    )
    assert S(frame, "a", "b", type="r", tr=0.1).n_obs == 10
    assert S(frame, "a", "b", type="np").n_obs == 10


@pytest.mark.parametrize("type_", TYPES)
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_exactly_collinear_pairs(type_: str, sign: float) -> None:
    x = [1.0, 2, 3, 4, 5, 6]
    frame = pair(x, [3 + sign * 0.5 * v for v in x])
    result = S(frame, "a", "b", type=type_)
    bound = "1" if sign > 0 else "-1"
    (warning,) = [w for w in result.warnings if "is exactly" in w]
    assert f"is exactly {bound}." in warning
    if type_ == "bayes":
        assert result.bayes is not None
        assert result.bayes.bf10 == math.inf
    else:
        assert result.effect_size is not None
        assert result.test is not None
        es = result.effect_size
        assert (es.estimate, es.conf_low, es.conf_high) == (sign, sign, sign)
        assert result.test.p_value == 0.0
    back = StatsResult.from_json(result.to_json())
    assert back == result
    assert back.check() == result.check()
    assert not [i for i in result.check() if i.code != "small_n"]


def test_collinear_with_few_pairs_has_a_finite_bayes_factor() -> None:
    # (n - 1) / 2 < 1 / 0.707 + 1/2 at n = 3: the limit is finite.
    result = S(pair([1.0, 2, 3], [2.0, 4, 6]), "a", "b", type="bayes")
    assert result.bayes is not None
    assert math.isfinite(result.bayes.bf10)
    assert f"(BF10 {result.bayes.bf10:.2f})" in result.warnings[0]


# --------------------------------------------------------------------------- #
# hints: mu=, popmean= and method=
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("keyword", "habit"),
    [
        ("mu", "R's t.test() and wilcox.test() call it mu"),
        ("popmean", "SciPy's ttest_1samp() calls it popmean"),
    ],
)
def test_test_value_habits(keyword: str, habit: str) -> None:
    frame = one([1.0, 2.0, 4.0])
    text = message(TypeError, lambda: H(frame, "v", **{keyword: 2.5}))
    assert text == (
        f"{keyword}= is not an argument of histostats(): the value x is tested "
        f"against is test_value= ({habit}; ggstatsplot calls it test.value). Write "
        f"test_value=2.5. Remove {keyword}=."
    )
    assert H(frame, "v", test_value=2.5).settings.test_value == 2.5
    # The top-level form says the same; a value that is not a number is not
    # repeated (it would fail): the example is test_value=0.
    assert message(TypeError, lambda: TOP_H(frame, "v", **{keyword: 1})) == (
        text.replace("2.5", "1")
    )
    assert "Write test_value=0." in message(
        TypeError, lambda: H(frame, "v", **{keyword: "two"})
    )
    # The drawing function points to the analysis.
    result = H(frame, "v")
    assert "calls it test_value=" not in message(
        TypeError, lambda: DRAW_H(result, **{keyword: 1})
    )
    assert "an argument of the analysis: pass it to ip.analyze.histostats()" in (
        message(TypeError, lambda: DRAW_H(result, **{keyword: 1}))
    )
    # scatterstats has no test value at all.
    assert message(
        TypeError, lambda: S(pair([1.0, 2, 4], [2.0, 1, 5]), "a", "b", **{keyword: 1})
    ) == (
        f"{keyword}= is not an argument of scatterstats(): it names the value a "
        f"one-sample test compares a column with ({habit}): histostats(data, x, "
        "test_value=...) tests one column against a value; scatterstats() tests "
        f"whether the correlation of x and y is 0. Remove {keyword}=."
    )


@pytest.mark.parametrize(
    ("value", "advice", "type_"),
    [
        ("pearson", "Pearson's r is type='parametric'. Write type='parametric'.", "p"),
        (
            "Spearman",
            "Spearman's rho is type='nonparametric'. Write type='nonparametric'.",
            "np",
        ),
    ],
)
def test_method_points_to_type(value: str, advice: str, type_: str) -> None:
    frame = pair([1.0, 2, 4, 3], [2.0, 1, 5, 3])
    text = message(TypeError, lambda: S(frame, "a", "b", method=value))
    assert text == (
        "method= is not an argument of scatterstats(): scatterstats() chooses the "
        f"correlation with type=; {advice} Remove method=."
    )
    written = re.search(r"Write type='(\w+)'", text)
    assert written is not None
    assert (
        S(frame, "a", "b", type=written.group(1)).type
        == S(frame, "a", "b", type=type_).type
    )
    assert message(TypeError, lambda: TOP_S(frame, "a", "b", method=value))
    result = S(frame, "a", "b")
    assert "an argument of the analysis: pass it to ip.analyze.scatterstats()" in (
        message(TypeError, lambda: DRAW_S(result, method=value))
    )


def test_method_kendall_does_not_exist() -> None:
    frame = pair([1.0, 2, 4, 3], [2.0, 1, 5, 3])
    text = message(TypeError, lambda: S(frame, "a", "b", method="kendall"))
    assert text == (
        "method= is not an argument of scatterstats(): Kendall's tau is not "
        "available: ggscatterstats has no Kendall correlation either. scatterstats() "
        "chooses the correlation with type=: 'parametric' (Pearson's r), "
        "'nonparametric' (Spearman's rho), 'robust' (the winsorized correlation) or "
        "'bayes' (the Bayes factor of Pearson's r). For a rank correlation write "
        "type='nonparametric' (Spearman's rho). Remove method=."
    )
    assert S(frame, "a", "b", type="nonparametric").test is not None
    other = message(TypeError, lambda: S(frame, "a", "b", method=3))
    assert other.endswith("'bayes' (the Bayes factor of Pearson's r). Remove method=.")


def test_render_names_the_analysis_with_its_own_columns() -> None:
    result = H(one([1.0, 2.0, 4.0]), "v")
    assert message(TypeError, lambda: DRAW_H(result, test_value=3)) == (
        "test_value= is an argument of the analysis, ip.analyze.histostats(data, x, "
        "...); render.histostats() draws a result that already holds it. Pass "
        "test_value= to ip.analyze.histostats() or ip.histostats() and draw that "
        "result. Remove test_value=."
    )


def test_undefined_bootstrap_t_is_left_out_and_counted() -> None:
    # Trimmed mean 5; a resample whose third smallest and third largest values
    # are both 5 has every winsorized value 5: its bootstrap t is 0 / 0 (D65).
    frame = one([1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 7.0, 8.0, 9.0])
    result = H(frame, "v", type="robust", test_value=4)
    es = result.effect_size
    assert es is not None
    assert es.undefined_resamples is not None
    assert es.undefined_resamples > 50
    assert es.infinite_resamples == 0
    defined = 2000 - es.undefined_resamples
    assert result.warnings == (
        f"{es.undefined_resamples:,} of 2,000 bootstrap resamples give an undefined "
        "(0/0) bootstrap t, because every winsorized value of the resample equals the "
        "trimmed mean of the data. They are left out: the critical value, the "
        f"interval and the p-value come from the other {defined:,}.",
    )
    assert es.conf_low < es.estimate < es.conf_high
    assert StatsResult.from_json(result.to_json()) == result
    assert result.check() == ()
