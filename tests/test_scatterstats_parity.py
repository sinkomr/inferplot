"""``ip.analyze.scatterstats`` against every ggscatterstats fixture, all four types.

Each fixture is replayed through ``ip.analyze.scatterstats`` on its CSV read as
pandas, polars and pyarrow (``tests/no_level_cases.py``; the label filter case in
pandas) with the fixture's own arguments, and every number the fixture holds is
compared under its ``tests/tolerances.py`` kind: the coefficient with its interval
and level, the statistic (t, or S for Spearman's rho), df, p and method, the
Bayes factor and its prior (subtitle of type bayes, caption of type parametric),
n, the rows dropped and the point labels. The correlation::correlation row each
subtitle came from says the same.

Fixtures inferplot answers differently from ggstatsplot, by name:

* edge_corr_constant_x (4 types): x is constant, so every number is undefined;
  ggstatsplot prints NA for each (and its Bayesian correlation stops), inferplot
  raises a plain error naming x (R's NA rows are pinned);
* edge_corr_perfect and edge_corr_perfect-neg (4 types each, exactly collinear
  data): inferplot's exact sums give r = +/-1, t = +/-Inf, p = 0, the interval
  [r, r], S = 0 or (n^3 - n) / 3, and the Bayes factor's limit, +Inf, with a
  ``warnings`` entry (D85); R prints its rounding noise (t = 2^27, p = 1.06e-62)
  and an empty Bayes factor. R's side is pinned as the fixture holds it
  (``R_DIFFERS``, tests/stats/test_correlation_parity.py).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import inferplot as ip
from inferplot.result import StatsResult
from tests import no_level_cases as nl
from tests.oracle_fixtures import OracleCase, number
from tests.stats.test_correlation_parity import R_DIFFERS
from tests.tolerances import assert_close

CASES = nl.SCATTER
CONSTANT_X = sorted(n for n in nl.REFUSALS if n.startswith("ggscatterstats-"))
COLLINEAR = sorted(c.name for c in CASES if "edge_corr_perfect" in c.name)
_METHODS = {
    "pearson_correlation": "Pearson correlation",
    "spearman_correlation": "Spearman correlation",
    "winsorized_correlation": "Winsorized Pearson correlation",
}
_EFFECT_SIZES = {
    "pearson_r": ("parametric", "Pearson correlation"),
    "spearman_rho": ("nonparametric", "Spearman correlation"),
    "winsorized_r": ("robust", "Winsorized Pearson correlation"),
}


def _result(name: str, backend: nl.Backend) -> StatsResult:
    result = nl.analyzed(name, backend)
    assert result is not None
    return result


def _row(case: OracleCase) -> Any:
    """The case's subtitle row, or that of the same call with the subtitle (the
    numbers do not depend on it: mtcars-nosubtitle-* against mtcars-*)."""
    if case.subtitle_rows:
        return case.subtitle_data
    assert case.args.get("results.subtitle") is False
    return nl.BY_NAME[case.name.replace("-nosubtitle-", "-")].subtitle_data


def _compare(case: OracleCase, column: str, mine: float, row: Any, key: str) -> None:
    """``closed_form``, or for exactly collinear data inferplot's exact value and
    R's as the fixture holds it."""
    differs = R_DIFFERS.get(key, {})
    if column in differs:
        theirs, expected = differs[column]
        oracle = row[column]
        assert (oracle is None) if theirs is None else number(oracle) == theirs
        assert mine == expected, (case.name, column, mine)
        return
    assert_close("closed_form", mine, number(row[column]), what=f"{key} {column}")


def test_fixture_counts() -> None:
    assert len(CASES) == 64
    counts = {
        t: sum(c.type == t for c in CASES)
        for t in ("parametric", "nonparametric", "robust", "bayes")
    }
    assert counts == {"parametric": 20, "nonparametric": 14, "robust": 15, "bayes": 15}
    assert len(CONSTANT_X) == 4
    assert len(COLLINEAR) == 8
    assert not [c.name for c in CASES if c.error is not None]
    # 60 answered cases on three backends, the label filter case on pandas only.
    assert len(nl.params(CASES)) == 59 * 3 + 1


@pytest.mark.parametrize("name", CONSTANT_X)
@pytest.mark.parametrize("backend", nl.BACKENDS)
def test_constant_x_is_refused(name: str, backend: nl.Backend) -> None:
    case = nl.BY_NAME[name]
    refusal = nl.REFUSALS[name]
    with pytest.raises(refusal.error) as info:
        nl.analyze(case, nl.frame_of(case, backend))
    assert str(info.value) == refusal.match + " Check the data for entry errors."
    assert nl.analyzed(name, backend) is None
    # R's side: every number NA (the Bayesian correlation stops, so type bayes
    # and the parametric caption are empty).
    if case.type == "bayes":
        assert "subtitle" in case.expected_hidden_errors
        return
    row = case.subtitle_data
    for column in ("estimate", "statistic", "p.value", "conf.low", "conf.high"):
        assert row[column] is None, column
    assert case.correlation["warnings"] == ["the standard deviation is zero"]


@pytest.mark.parametrize(("name", "backend"), nl.params(CASES))
def test_pairs_counts_and_labels(name: str, backend: nl.Backend) -> None:
    case, result = nl.BY_NAME[name], _result(name, backend)
    data = case.analyzed_data
    assert (result.analysis, result.design, result.type) == (
        "scatterstats",
        "correlation",
        case.type,
    )
    assert (result.x, result.y) == (case.args["x"], case.args["y"])
    assert (result.levels, result.descriptives, result.pairwise) == ((), (), ())
    assert result.values == (
        tuple(case.analyzed_values("x")),
        tuple(case.analyzed_values("y")),
    )
    assert result.n_obs == data["n"]
    assert result.n_dropped == data["rows_dropped"]
    assert result.n_obs + result.n_dropped == case.raw["dataset"]["rows"]
    assert result.settings.alternative == "two-sided"
    assert result.settings.test_value is None
    label_var = case.args.get("label.var")
    assert result.label_var == label_var
    if label_var is None:
        assert result.point_labels == ()
    else:
        ((drawn),) = case.layer_data("GeomLabelRepel")
        assert [label for label in result.point_labels if label] == [
            str(row[label_var]) for row in drawn
        ]


@pytest.mark.parametrize(("name", "backend"), nl.params(CASES))
def test_correlation(name: str, backend: nl.Backend) -> None:
    case, result = nl.BY_NAME[name], _result(name, backend)
    test, es = result.test, result.effect_size
    if case.type == "bayes":
        assert (test, es) == (None, None)  # D45
        return
    assert test is not None
    assert es is not None
    row = _row(case)
    key = case.name.replace("-nosubtitle-", "-")
    assert test.method == row["method"] == _METHODS[test.name]
    assert (case.type, row["effectsize"]) == _EFFECT_SIZES[es.name]
    assert int(number(row["n.obs"])) == result.n_obs
    level = float(case.args.get("conf.level", 0.95))
    assert es.conf_level == number(row["conf.level"]) == level
    assert (es.sidedness, es.bootstrap_resamples) == ("two-sided", None)
    assert test.alternative == "two-sided"
    for column, mine in (
        ("estimate", es.estimate),
        ("statistic", test.statistic),
        ("p.value", test.p_value),
        ("conf.low", es.conf_low),
        ("conf.high", es.conf_high),
    ):
        _compare(case, column, mine, row, key)
    if case.type == "nonparametric":
        assert (test.statistic_name, test.df, test.df_error) == ("S", None, None)
    else:
        assert (test.statistic_name, test.df) == ("t", None)
        assert test.df_error == number(row["df.error"]) == result.n_obs - 2
    # The correlation::correlation row the subtitle came from says the same.
    (corr,) = case.correlation["value"] if case.subtitle_rows else ({},)
    if corr:
        coefficient = number(corr["rho" if case.type == "nonparametric" else "r"])
        assert_close("closed_form", es.estimate, coefficient, what="correlation row")


@pytest.mark.parametrize(("name", "backend"), nl.params(CASES))
def test_bayes_factor(name: str, backend: nl.Backend) -> None:
    case, result = nl.BY_NAME[name], _result(name, backend)
    bayes = result.bayes
    if case.type == "bayes" and not case.subtitle_rows:
        assert bayes is None  # no subtitle: nothing is computed
        return
    row = case.subtitle_data if case.type == "bayes" else case.caption_data
    collinear = name in COLLINEAR and case.type in ("parametric", "bayes")
    if row is None and not collinear:
        # No caption: the type has none, or bf.message = FALSE, or no subtitle.
        assert bayes is None
        return
    assert bayes is not None
    prior = float(case.args.get("bf.prior", 0.707))
    assert (bayes.prior, bayes.test) == ("beta", "pearson_correlation")
    # The beta prior's parameter, 1 / bf.prior, as R's row and ggstatsplot print it.
    assert bayes.prior_scale == 1.0 / prior
    assert result.settings.bf_prior == prior
    if collinear:
        # R: correlationBF returns NaN; the subtitle prints an empty log BF01 and
        # the parametric caption is NA. inferplot: the limit as |r| -> 1.
        key = name + ("" if case.type == "bayes" else "-caption")
        assert R_DIFFERS[key]["bf10"] == (None, math.inf)
        if row is not None:
            assert row["bf10"] is None
        else:
            assert "caption" in case.raw["expected_na_values"]
        assert (bayes.bf10, bayes.log_bf01) == (math.inf, -math.inf)
        return
    assert row is not None
    assert row["prior.distribution"] == "beta"
    assert_close("closed_form", bayes.prior_scale, number(row["prior.scale"]), what="")
    assert math.isclose(bayes.log_bf01, -math.log(bayes.bf10), rel_tol=1e-15)
    assert_close(
        "correlation_bayes_factor", bayes.bf10, number(row["bf10"]), what=f"{name} BF"
    )


@pytest.mark.parametrize("name", COLLINEAR)
def test_exactly_collinear_data(name: str) -> None:
    case, result = nl.BY_NAME[name], _result(name, "pandas")
    sign = -1.0 if "-neg-" in name else 1.0
    (warning,) = [w for w in result.warnings if "is exactly" in w]
    assert f"is exactly {'-1' if sign < 0 else '1'}." in warning
    assert warning.endswith(
        "A rank correlation needs no straight line for this: y only has to rise or "
        "fall with x."
        if case.type == "nonparametric"
        else "Check that neither column is computed from the other."
    )
    if case.type != "bayes":
        assert "ggstatsplot prints the rounding of R's arithmetic" in warning
    if case.type in ("parametric", "bayes"):
        assert "where ggstatsplot leaves it empty" in warning
    if case.type == "bayes":
        assert result.test is None
        assert "BF10 infinite" in warning
    else:
        test, es = result.test, result.effect_size
        assert test is not None
        assert es is not None
        assert (es.estimate, es.conf_low, es.conf_high) == (sign, sign, sign)
        assert test.p_value == 0.0
        if case.type == "nonparametric":
            assert test.statistic == (0.0 if sign > 0 else 330.0)
        else:
            assert test.statistic == sign * math.inf
        assert f"Its t is {'-' if sign < 0 else ''}Inf and its p is 0" in warning
    # Every special value survives the JSON round trip, the frame and check().
    back = StatsResult.from_json(result.to_json())
    assert back == result
    if case.type != "nonparametric":  # S is finite; t and the Bayes factor are not
        assert ('"-Inf"' if sign < 0 and case.type != "bayes" else '"Inf"') in (
            result.to_json()
        )
    assert back.check() == result.check() == ()
    assert len(result.to_frame()) == 1
    assert str(result)


def test_results_do_not_depend_on_the_backend() -> None:
    for case in CASES:
        if nl.analyzed(case.name, "pandas") is None:
            continue
        first = nl.analyzed(case.name, "pandas")
        assert first is not None
        for backend in nl.backends(case):
            result = nl.analyzed(case.name, backend)
            assert result is not None
            assert result.to_json() == first.to_json()


def test_every_fixture_is_answered_or_refused_by_name() -> None:
    answered = {p.values[0] for p in nl.params(CASES)}
    assert answered | set(CONSTANT_X) == {c.name for c in CASES}
    assert not answered & set(CONSTANT_X)
    frame = nl.frame_of(nl.BY_NAME[CONSTANT_X[0]])
    with pytest.raises(ValueError, match="Every value of x='x' is 5"):
        ip.analyze.scatterstats(frame, "x", "y", type="bayes")
