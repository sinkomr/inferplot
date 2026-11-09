"""Property tests of ``ip.analyze.betweenstats`` (SPEC.md section 11).

* the order of the rows does not matter;
* renaming the levels (keeping their order) changes only the names;
* pandas, polars and pyarrow frames give the identical result;
* Hedges' g and partial omega squared do not change when y is shifted or multiplied
  by a positive number.

Groups are drawn as distinct multiples of 1/8, so every group varies and every value
is exact in binary. Profiles (deadline, derandomization in CI) are in
``tests/conftest.py``.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
from hypothesis import given, settings
from hypothesis import strategies as st

import inferplot as ip
from inferplot import text as tx
from inferplot.result import StatsResult
from tests.tolerances import assert_close

betweenstats: Any = ip.analyze.betweenstats

Groups = list[list[float]]

_group = st.lists(
    st.integers(min_value=-400, max_value=400), min_size=3, max_size=10, unique=True
).map(lambda values: [v / 8 for v in values])


def _groups(min_k: int = 2, max_k: int = 4) -> st.SearchStrategy[Groups]:
    return st.lists(_group, min_size=min_k, max_size=max_k)


def _frame(groups: Groups, names: list[str] | None = None) -> pd.DataFrame:
    names = names or [f"g{i}" for i in range(len(groups))]
    rows = [
        (name, v) for name, values in zip(names, groups, strict=True) for v in values
    ]
    return pd.DataFrame(rows, columns=["group", "score"])


def _close_numbers(actual: StatsResult, expected: StatsResult, what: str) -> None:
    """Every number of two results of the same data, up to summation order."""
    for field in ("statistic", "df", "df_error", "p_value"):
        a, b = getattr(actual.test, field), getattr(expected.test, field)
        if b is None:
            assert a is None
        else:
            assert_close("closed_form", a, b, what=f"{what} {field}")
    assert_close(
        "closed_form",
        actual.effect_size.estimate,
        expected.effect_size.estimate,
        what=f"{what} estimate",
    )
    for bound in ("conf_low", "conf_high"):
        assert_close(
            "noncentral_ci",
            getattr(actual.effect_size, bound),
            getattr(expected.effect_size, bound),
            what=f"{what} {bound}",
        )
    assert (actual.bayes is None) == (expected.bayes is None)
    if actual.bayes is not None and expected.bayes is not None:
        assert_close(
            "closed_form",
            actual.bayes.log_bf01,
            expected.bayes.log_bf01,
            what=f"{what} log_bf01",
        )


@settings(max_examples=60)
@given(groups=_groups(), data=st.data())
def test_row_order_invariance(groups: Groups, data: st.DataObject) -> None:
    frame = _frame(groups)
    permutation = data.draw(st.permutations(list(range(len(frame)))))
    shuffled = frame.iloc[permutation].reset_index(drop=True)
    original = betweenstats(frame, "group", "score")
    result = betweenstats(shuffled, "group", "score")
    assert result.levels == original.levels
    assert result.descriptives == original.descriptives  # correctly rounded means
    assert [sorted(v) for v in result.values] == [sorted(v) for v in original.values]
    _close_numbers(result, original, "shuffled")
    assert [(p.group1, p.group2) for p in result.pairwise] == [
        (p.group1, p.group2) for p in original.pairwise
    ]
    for pair, reference in zip(result.pairwise, original.pairwise, strict=True):
        assert_close("closed_form", pair.statistic, reference.statistic, what="q")
        assert pair.df is not None
        assert reference.df is not None
        assert_close("closed_form", pair.df, reference.df, what="pair df")
        assert_close("closed_form", pair.p_adjusted, reference.p_adjusted, what="p")
    assert result.text == original.text


_names = st.lists(
    st.text(alphabet="abcXYZ09 _-", min_size=1, max_size=6),
    min_size=4,
    max_size=4,
    unique=True,
)


@settings(max_examples=60)
@given(groups=_groups(min_k=2, max_k=4), names=_names)
def test_level_relabelling_invariance(groups: Groups, names: list[str]) -> None:
    """New names, same declared order: same numbers per level; ggstatsplot's pair
    rows follow the new names' text order."""
    k = len(groups)
    old = [f"g{i}" for i in range(k)]
    new = names[:k]
    original = betweenstats(_frame(groups, old), "group", "score")
    relabelled_frame = _frame(groups, new)
    relabelled_frame["group"] = pd.Categorical(
        relabelled_frame["group"], categories=new
    )
    result = betweenstats(relabelled_frame, "group", "score")
    rename = dict(zip(old, new, strict=True))
    assert result.levels == tuple(new)
    assert result.values == original.values
    assert result.test == original.test
    assert result.effect_size == original.effect_size
    assert result.bayes == original.bayes
    assert [(d.n, d.value) for d in result.descriptives] == [
        (d.n, d.value) for d in original.descriptives
    ]
    assert result.text.subtitle == original.text.subtitle
    assert result.text.apa == original.text.apa
    assert result.text.centrality_labels == original.text.centrality_labels
    assert result.text.tick_labels == tuple(
        tx.tick_label(name, d.n)
        for name, d in zip(new, original.descriptives, strict=True)
    )
    expected = sorted(
        (
            dataclasses.replace(
                pair, group1=rename[pair.group1], group2=rename[pair.group2]
            )
            for pair in original.pairwise
        ),
        key=lambda pair: (pair.group1, pair.group2),
    )
    assert list(result.pairwise) == expected


@settings(max_examples=40)
@given(groups=_groups(), missing=st.booleans())
def test_backends_give_identical_json(groups: Groups, missing: bool) -> None:
    frame = _frame(groups)
    if missing:
        frame.loc[0, "score"] = math.nan
        frame.loc[len(frame) - 1, "group"] = None
    if min(frame.dropna().groupby("group").size()) < 2:  # a group lost its values
        frame = _frame(groups)
    expected = betweenstats(frame, "group", "score").to_json()
    polars_frame = pl.from_pandas(frame.astype({"group": "string"}))
    arrow_table = pa.Table.from_pandas(frame, preserve_index=False)
    assert betweenstats(polars_frame, "group", "score").to_json() == expected
    assert betweenstats(arrow_table, "group", "score").to_json() == expected


@settings(max_examples=60)
@given(
    groups=_groups(),
    shift=st.integers(min_value=-800, max_value=800).map(lambda v: v / 8),
    scale=st.integers(min_value=1, max_value=80).map(lambda v: v / 8),
)
def test_effect_sizes_invariant_to_location_and_scale(
    groups: Groups, shift: float, scale: float
) -> None:
    # Multiples of 1/8 shifted by a multiple of 1/8 and scaled by one: every moved
    # value is exact, so the only differences are the statistics' own roundings
    # (differences of means are exact, DECISIONS.md, exact differences). With
    # arbitrary floats the rounding of the moved data is amplified without bound
    # when the effect is near 0.
    original = betweenstats(_frame(groups), "group", "score")
    moved = [[shift + scale * v for v in values] for values in groups]
    for values, new in zip(groups, moved, strict=True):
        assert [Fraction(m) for m in new] == [
            Fraction(shift) + Fraction(scale) * Fraction(v) for v in values
        ]
    result = betweenstats(_frame(moved), "group", "score")
    es, reference = result.effect_size, original.effect_size
    assert es.name == reference.name
    assert es.name in {"hedges_g", "partial_omega_squared"}
    if es.name == "partial_omega_squared":
        # omega = (F - 1) df1 / (F df1 + df2 + 1): F is accurate to a few units of
        # 2^-53 relative, and F - 1 cancels when F is near 1, so the allowance
        # adds 64 units of F's share to closed_form's relative part.
        f, df1, df2 = (
            original.test.statistic,
            original.test.df,
            original.test.df_error,
        )
        assert df1 is not None
        assert df2 is not None
        slack = 64 * 2.0**-53 * f * df1 / (f * df1 + df2 + 1.0)
        assert abs(es.estimate - reference.estimate) <= slack + 1e-10 * abs(
            reference.estimate
        )
    else:
        assert_close("closed_form", es.estimate, reference.estimate, what="estimate")
    for bound in ("conf_low", "conf_high"):
        assert_close(
            "noncentral_ci",
            getattr(es, bound),
            getattr(reference, bound),
            what=bound,
        )
