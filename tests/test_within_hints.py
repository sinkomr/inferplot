"""withinstats' words and hints (review of phase 3, findings 8 and 9): each names
what exists in withinstats, and a hint followed literally leads to a call that
works or says that nothing will."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot.check import _named  # pyright: ignore[reportPrivateUsage]

_any: Any = ip  # unknown keywords on purpose


def _long(m: Any) -> pd.DataFrame:
    m = np.asarray(m, dtype=float)
    n, k = m.shape
    return pd.DataFrame(
        [
            (f"s{i:02d}", f"c{j + 1}", float(m[i, j]))
            for i in range(n)
            for j in range(k)
        ],
        columns=["id", "cond", "y"],
    )


RNG = np.random.default_rng(0)
THREE = _long(RNG.normal(50, 5, (12, 3)))
TWO = THREE[THREE["cond"] != "c3"]


def _raised(call: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises((TypeError, ValueError, NotImplementedError)) as raised:
        call(*args, **kwargs)
    return str(raised.value)


def test_to_frame_names_the_paired_bayes_row() -> None:
    result = ip.analyze.withinstats(TWO, "cond", "y", subject_id="id", type="bayes")
    assert result.to_frame()["label"].tolist() == ["Bayesian paired t-test"]
    between = ip.analyze.betweenstats(TWO, "cond", "y", type="bayes")
    assert between.to_frame()["label"].tolist() == ["Bayesian t-test"]


def test_var_equal_in_withinstats_names_no_welch_test() -> None:
    result = ip.analyze.withinstats(THREE, "cond", "y", subject_id="id")
    for message in (
        _raised(_any.withinstats, THREE, "cond", "y", subject_id="id",
                **{"var.equal": True}),
        _raised(_any.analyze.withinstats, THREE, "cond", "y", subject_id="id",
                **{"var.equal": True}),
        _raised(_any.render.withinstats, result, **{"var.equal": True}),
    ):  # fmt: skip
        assert "Welch" not in message
        assert "paired t-test" in message
        assert message.endswith("Remove var.equal=.")


def test_pairwise_comparisons_says_how_pairs_are_controlled() -> None:
    within = ip.analyze.withinstats(THREE, "cond", "y", subject_id="id")
    between = ip.analyze.betweenstats(THREE, "cond", "y")
    for render, result, levels in (
        (_any.render.withinstats, within, "conditions"),
        (_any.render.betweenstats, between, "groups"),
    ):
        message = _raised(render, result, **{"pairwise.comparisons": False})
        assert "has no pairwise comparisons" not in message
        assert f"three or more {levels}" in message
        assert "pass pairwise_display='none' or 'all' to ip.analyze." in message
    message = _raised(
        _any.withinstats, THREE, "cond", "y", subject_id="id",
        **{"pairwise.comparisons": False},
    )  # fmt: skip
    assert "three or more conditions" in message
    assert "groups" not in message


def test_estimate_outside_interval_names_kendalls_w() -> None:
    assert _named("Kendall's W") == "Kendall's W"
    assert _named("Hedges' g") == "Hedges' g"
    assert _named("average AKP robust standardized difference") == (
        "The average AKP robust standardized difference"
    )
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n, k = int(rng.integers(5, 9)), int(rng.integers(6, 13))
        result = ip.analyze.withinstats(
            _long(rng.normal(0, 1, (n, k))), "cond", "y", subject_id="id",
            type="nonparametric", pairwise_display="none",
        )  # fmt: skip
        issues = [c for c in result.check() if c.code == "estimate_outside_interval"]
        if issues:
            break
    else:  # pragma: no cover
        pytest.fail("no Kendall's W outside its interval in 40 noise data sets")
    (issue,) = issues
    assert issue.message.startswith("Kendall's W, ")
    assert "The Kendall" not in issue.message
    assert "between the conditions" in issue.message
    assert "between the groups" not in issue.message


@pytest.mark.parametrize("data", [TWO, THREE], ids=["k2", "k3"])
@pytest.mark.parametrize("type_", ["parametric", "nonparametric", "robust", "bayes"])
def test_bf_message_without_subtitle_leads_somewhere(
    data: pd.DataFrame, type_: str
) -> None:
    kwargs: dict[str, Any] = {"subject_id": "id", "type": type_}
    message = _raised(
        ip.analyze.withinstats, data, "cond", "y", bf_message=True,
        results_subtitle=False, **kwargs,
    )  # fmt: skip
    if "Pass results_subtitle=True" in message:
        result = ip.analyze.withinstats(
            data, "cond", "y", bf_message=True, results_subtitle=True, **kwargs
        )
        assert result.text.caption is not None
    else:
        assert "Remove bf_message=True" in message
        result = ip.analyze.withinstats(
            data, "cond", "y", results_subtitle=False, **kwargs
        )
        assert result.text.subtitle is None


@pytest.mark.parametrize("type_", ["robust", "bayes"])
def test_one_sided_with_three_conditions_names_the_two_condition_rule(
    type_: str,
) -> None:
    extra = {"results_subtitle": False} if type_ == "bayes" else {}
    message = _raised(
        ip.analyze.withinstats, THREE, "cond", "y", subject_id="id", type=type_,
        alternative="less", **extra,
    )  # fmt: skip
    assert "applies only to two conditions" in message
    assert "signed-rank" not in message
    # With two conditions the hint names types that take it, and they do.
    message = _raised(
        ip.analyze.withinstats, TWO, "cond", "y", subject_id="id", type=type_,
        alternative="less",
    )  # fmt: skip
    assert "type='nonparametric' (the signed-rank test)" in message
    for other in ("parametric", "nonparametric"):
        result = ip.analyze.withinstats(
            TWO, "cond", "y", subject_id="id", type=other, alternative="less"
        )
        assert result.test is not None
        assert result.test.alternative == "less"


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("type_", ["parametric", "robust", "bayes"])
def test_fully_constant_data_advise_no_type(k: int, type_: str) -> None:
    extra = {"results_subtitle": False} if type_ == "bayes" and k == 3 else {}
    message = _raised(
        ip.analyze.withinstats, _long(np.full((8, k), 5.0)), "cond", "y",
        subject_id="id", type=type_, **extra,
    )  # fmt: skip
    assert "type='nonparametric'" not in message
    assert "smaller tr" not in message
    assert "no difference between the conditions for any type to test" in message
    with pytest.raises(ValueError, match=r"no difference to test|nothing to rank"):
        ip.analyze.withinstats(
            _long(np.full((8, k), 5.0)), "cond", "y", subject_id="id",
            type="nonparametric",
        )  # fmt: skip


def test_equal_differences_advise_what_works() -> None:
    shifted = _long(np.column_stack([np.arange(8.0), np.arange(8.0) + 1]))
    for type_ in ("parametric", "robust", "bayes"):
        message = _raised(
            ip.analyze.withinstats, shifted, "cond", "y", subject_id="id", type=type_
        )
        assert "type='nonparametric'" in message
        # Every difference is equal before winsorizing: trimming less cannot help.
        assert "smaller tr" not in message
    result = ip.analyze.withinstats(
        shifted, "cond", "y", subject_id="id", type="nonparametric"
    )
    assert result.test is not None


def test_subject_next_to_subject_id() -> None:
    for call in (ip.withinstats, ip.analyze.withinstats):
        message = _raised(call, THREE, "cond", "y", subject_id="id", subject="id")
        assert message == (
            "subject= is not an argument of withinstats(), and subject_id= (the "
            "column that identifies each subject) is already given. Remove subject=."
        )


def test_point_args_s_names_sizes() -> None:
    message = _raised(
        ip.withinstats, THREE, "cond", "y", subject_id="id", point_args={"s": 30}
    )
    assert "'sizes'" in message
    assert "point_args={'sizes': [30]}" in message
    fig = ip.withinstats(
        THREE, "cond", "y", subject_id="id", point_args={"sizes": [30]}
    )
    assert fig.result.n_obs == 12
