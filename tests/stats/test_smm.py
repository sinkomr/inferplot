"""The Studentized maximum modulus distribution (inferplot.stats.smm) against
high-precision references and closed forms."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scipy import stats

from inferplot.stats import smm_cdf, smm_quantile
from tests.tolerances import assert_close

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "smm_reference.json"


def _rows() -> list[dict[str, object]]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))["rows"]


def test_reference_covers_the_range() -> None:
    rows = _rows()
    assert len(rows) == 148
    assert {r["comparisons"] for r in rows} >= {1, 2, 28, 29, 36, 45, 300}
    assert {r["df"] for r in rows} >= {"1", "1.5", "1e5", "inf"}


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(r, id=f"C{r['comparisons']}-df{r['df']}-{r['conf_level']}")
        for r in _rows()
    ],
)
def test_quantile_against_mpmath(row: dict[str, object]) -> None:
    df = math.inf if row["df"] == "inf" else float(str(row["df"]))
    got = smm_quantile(
        int(str(row["comparisons"])), df, conf_level=float(str(row["conf_level"]))
    )
    assert_close("smm_quantile", got, float(str(row["quantile"])), what=str(row))


@pytest.mark.parametrize("df", [1.0, 2.0, 4.69, 10.0, 60.0, 1490.0, 1e7])
def test_one_comparison_is_students_t(df: float) -> None:
    # max |Z| / S with one Z is |T_df|.
    expected = float(stats.t.isf(0.025, df))
    assert_close("smm_quantile", smm_quantile(1, df), expected, what=f"df {df}")
    assert smm_cdf(expected, 1, df) == pytest.approx(0.95, abs=1e-14)


def test_cdf_is_a_distribution_function() -> None:
    assert smm_cdf(0.0, 5, 10.0) == 0.0
    assert smm_cdf(-1.0, 5, 10.0) == 0.0
    assert smm_cdf(200.0, 5, 10.0) == pytest.approx(1.0, abs=1e-14)
    values = [smm_cdf(c, 36, 5.5) for c in (1.0, 2.0, 3.0, 4.0, 6.0, 10.0)]
    assert values == sorted(values)


def test_quantile_falls_with_df_and_rises_with_comparisons() -> None:
    by_df = [smm_quantile(36, df) for df in (1.0, 2.0, 5.0, 20.0, 1e3, math.inf)]
    assert by_df == sorted(by_df, reverse=True)
    by_c = [smm_quantile(c, 7.5) for c in (1, 2, 10, 29, 100)]
    assert by_c == sorted(by_c)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_comparisons": 0, "df": 10.0}, "n_comparisons"),
        ({"n_comparisons": 3, "df": 0.5}, "at least 1"),
        ({"n_comparisons": 3, "df": math.nan}, "at least 1"),
        ({"n_comparisons": 3, "df": 10.0, "conf_level": 1.0}, "conf_level"),
    ],
)
def test_refusals(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        smm_quantile(**kwargs)  # type: ignore[arg-type]
