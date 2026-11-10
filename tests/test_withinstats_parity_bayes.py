"""``ip.analyze.withinstats(type="bayes")`` against every bayes ggwithinstats
fixture.

Each fixture is replayed through ``ip.analyze.withinstats`` on its CSV read as
pandas, polars and pyarrow (``tests/within_pb_cases.py``); where inferplot refuses
the call by design (a one-sided ``alternative``; three or more conditions with the
subtitle, a Bayesian repeated-measures ANOVA, v0.2) the refusal is checked and the
replay compared. Numbers are compared under their ``tests/tolerances.py`` kinds: the
paired Bayes factor of the subtitle with its prior, every pair's Bayes factor at
prior 0.707, the MAP centrality on R's density grid, the drawn pairs and n. Each
pair's t and df, which R's rows do not carry, are compared with SciPy's
``ttest_rel``. Texts are not tested here.
"""

from __future__ import annotations

import pytest
from scipy import stats as scipy_stats

import inferplot as ip
from inferplot.result import StatsResult
from inferplot.stats.density import GRID_POINTS
from tests import within_pb_cases as wc
from tests.oracle_fixtures import number, read_dataset
from tests.tolerances import assert_close, assert_within_grid

CASES = wc.cases_of("bayes")
TWO = [c for c in CASES if len(c.analysis_levels) == 2]
MORE = [c for c in CASES if len(c.analysis_levels) >= 3]
REFUSED = [c for c in CASES if wc.refusal(c) is not None]


def test_fixture_counts() -> None:
    assert (len(TWO), len(MORE)) == (15, 19)
    # Refused: sleep-greater (alternative) and the 18 cases with three or more
    # conditions and a subtitle.
    assert len(REFUSED) == 19


@pytest.mark.parametrize("name", wc.names(REFUSED))
def test_refused_calls_raise(name: str) -> None:
    case = wc.BY_NAME[name]
    found = wc.refusal(case)
    assert found is not None
    args, kwargs = wc.call_arguments(case)
    with pytest.raises(found.error, match=found.match) as raised:
        ip.analyze.withinstats(read_dataset(case), *args, **kwargs)
    if found.error is NotImplementedError:
        k = len(case.analysis_levels)
        assert str(raised.value) == (
            f"type='bayes' with {k} conditions: the subtitle ggstatsplot draws here "
            "is a Bayesian repeated-measures ANOVA, which arrives in inferplot 0.2. "
            "Pass results_subtitle=False for the rest of ggstatsplot's output (MAP "
            "estimates and a Bayes factor for every pair), or use type='parametric'."
        )


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_levels_counts_and_values(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    assert (result.analysis, result.type) == ("withinstats", "bayes")
    n = len(case.within_data["subjects"])
    assert result.levels == case.analysis_levels
    assert result.n_obs == n
    assert [d.n for d in result.descriptives] == [n] * len(result.levels)
    assert [list(v) for v in result.values] == [
        list(column) for column in wc.conditions(case)
    ]
    assert result.test is None
    assert result.effect_size is None


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_map_centrality(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    y = str(case.args["y"])
    columns = dict(zip(case.analysis_levels, wc.conditions(case), strict=True))
    for layer in wc.centrality_layers(case):
        assert list(layer) == list(result.levels)
        for d in result.descriptives:
            row = layer[d.level]
            assert d.centrality == "map"
            assert int(row["n.obs"]) == d.n
            assert d.value is not None
            values = columns[d.level]  # the complete subjects' values
            low, high = float(values.min()), float(values.max())
            step = (high - low) / (GRID_POINTS - 1)
            assert_within_grid(
                d.value, number(row[y]), origin=low, step=step, what=f"{name} MAP"
            )


@pytest.mark.parametrize(("name", "backend"), wc.params(TWO))
def test_subtitle_bayes_factor(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    assert result.pairwise == ()
    if not case.subtitle_rows:
        assert case.args.get("results.subtitle") is False
        assert result.bayes is None
        return
    row = case.subtitle_data
    bayes = result.bayes
    assert bayes is not None
    assert bayes.test == "jzs_paired"
    assert bayes.prior == row["prior.distribution"] == "cauchy"
    assert bayes.prior_scale == number(row["prior.scale"])
    assert bayes.prior_scale == float(case.args.get("bf.prior", 0.707))
    assert_close("ttest_bayes_factor", bayes.bf10, number(row["bf10"]), what="bf10")
    assert_close(
        "ttest_bayes_factor",
        bayes.log_bf01,
        -number(row["log_e_bf10"]),
        what="log_bf01",
    )


@pytest.mark.parametrize(("name", "backend"), wc.params(MORE))
def test_pairwise_bayes_factors(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    assert result.bayes is None
    k = len(result.levels)
    assert len(result.pairwise) == k * (k - 1) // 2
    if case.args.get("pairwise.display") == "none":
        assert case.pairwise == []
        assert not any(pair.displayed for pair in result.pairwise)
        return
    rows = wc.pair_rows(case, result)
    columns = dict(zip(case.analysis_levels, wc.conditions(case), strict=True))
    for pair, row in zip(result.pairwise, rows, strict=True):
        what = f"{name} {pair.group1}-{pair.group2}"
        assert pair.test_label == row["test"] == "Student's t"
        assert (pair.p_value, pair.p_adjusted, pair.significant) == (None, None, None)
        assert pair.adjust_method == "none"
        assert pair.displayed
        assert pair.bayes is not None
        # 0.707 whatever bf_prior says (bugs_long-prior08-bayes).
        assert pair.bayes.prior_scale == number(row["prior.scale"]) == 0.707
        assert_close(
            "ttest_bayes_factor", pair.bayes.bf10, number(row["bf10"]), what=what
        )
        reference = scipy_stats.ttest_rel(columns[pair.group1], columns[pair.group2])
        assert_close(
            "closed_form", pair.statistic, float(reference.statistic), what=f"{what} t"
        )
        assert pair.df == len(columns[pair.group1]) - 1.0


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_drawn_pairs(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == wc.drawn(case)


# The warnings of the bayes fixtures beyond the data notes, by case.
_POSTERIOR = (
    "The posterior difference and its credible interval, which ggstatsplot adds to "
    "the Bayes factor subtitle, arrive in inferplot 0.2."
)
_CL90 = (
    "conf_level=0.9 has no effect: type='bayes' shows no interval in this version "
    "(the posterior credible interval arrives in inferplot 0.2)."
)
_EXPECTED_WARNINGS: dict[str, tuple[str, ...]] = {
    "ggwithinstats-sleep-cl90-bayes": (_POSTERIOR, _CL90),
    "ggwithinstats-bugs_long-cl90-bayes": (_CL90,),
    "ggwithinstats-bugs_long-prior08-bayes": (
        "bf_prior=0.8 has no effect: the pairwise Bayes factors always use the prior "
        "scale 0.707, as in ggstatsplot; the Bayesian repeated-measures ANOVA "
        "arrives in inferplot 0.2.",
    ),
    "ggwithinstats-bugs_long-display-ns-bayes": (
        "pairwise_display='non-significant' has no effect: Bayes factors have no "
        "significance, so every pair is drawn (as in ggstatsplot); pass "
        "pairwise_display='none' to draw none.",
    ),
}


@pytest.mark.parametrize("name", wc.names(CASES))
def test_warnings(name: str) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, "pandas")
    found = tuple(w for w in result.warnings if "left out of the analysis" not in w)
    default = (
        (_POSTERIOR,) if len(case.analysis_levels) == 2 and case.subtitle_rows else ()
    )
    assert found == _EXPECTED_WARNINGS.get(name, default)


@pytest.mark.parametrize("name", wc.names(CASES))
def test_backends_give_identical_results(name: str) -> None:
    first = wc.analyze(name, "pandas").to_json()
    for backend in wc.BACKENDS[1:]:
        assert wc.analyze(name, backend).to_json() == first


@pytest.mark.parametrize("name", wc.names(CASES))
def test_json_round_trip(name: str) -> None:
    result = wc.analyze(name, "pandas")
    assert StatsResult.from_json(result.to_json()) == result
    assert len(result.to_frame()) == len(result.frame_rows())
