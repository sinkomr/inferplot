"""``ip.barstats``, ``ip.analyze.barstats`` and ``ip.render.barstats``: signatures,
argument checks, the design checks that need only the table, and the result a
contingency design returns (``tests/bar_cases.py``; the numbers against the
oracle are tests/test_barstats_parity.py's).

The drawing raises ``NotImplementedError`` after checking its arguments.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot import _validate as v
from inferplot.result import StatsResult
from tests import bar_cases as bc
from tests.oracle_fixtures import Backend, OracleCase, read_csv

analyze: Any = ip.analyze.barstats
draw: Any = ip.render.barstats
top: Any = ip.barstats
module: Any = bc.bars


def message(error: type[Exception], thunk: Callable[[], object]) -> str:
    """The message of the ``error`` that ``thunk()`` raises (exactly that type)."""
    with pytest.raises(error) as info:
        thunk()
    assert type(info.value) is error
    return str(info.value)


@pytest.fixture(scope="module")
def mtcars() -> pd.DataFrame:
    return read_csv("mtcars.csv")


def prepare(data: object, x: str, y: str | None = None, **kwargs: Any) -> Any:
    return module._prepare(data, x, y, **kwargs)


def warnings_of(data: object, x: str, y: str | None = None, **kwargs: Any) -> list[str]:
    """The warnings of the analysis' result."""
    return list(analyze(data, x, y, **kwargs).warnings)


def _stand_in(table: Any, args: Any) -> Any:
    case = bc.BY_NAME["ggbarstats-mtcars-cyl-am-nosubtitle-parametric"]
    bar_tests = bc._bar_tests(case, table)
    test, effect = bc._main(case, args)
    return module._Parts(
        test=test,
        effect_size=effect,
        bar_tests=bar_tests,
        pairwise=(),
        warnings=(),
        citations=(),
    )


# --------------------------------------------------------------------------- #
# signatures and the three forms
# --------------------------------------------------------------------------- #

ANALYSIS_DEFAULTS = {
    "counts": None,
    "paired": False,
    "ratio": None,
    "label": "percentage",
    "proportion_test": None,
    "p_adjust_method": "holm",
    "digits_perc": 0,
    "type": "parametric",
    "conf_level": 0.95,
    "digits": 2,
    "alternative": "two-sided",
    "bf_message": None,
    "results_subtitle": True,
    "seed": 0,
}
DRAWING_DEFAULTS = {
    "xlab": None,
    "ylab": None,
    "title": None,
    "subtitle": None,
    "caption": None,
    "legend_title": None,
    "palette": None,
    "bar_args": None,
    "ax": None,
}


def _defaults(fn: Callable[..., object]) -> dict[str, object]:
    return {
        p.name: p.default
        for p in inspect.signature(fn).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }


def test_signatures() -> None:
    kind = inspect.Parameter
    params = list(inspect.signature(analyze).parameters.values())
    assert [p.name for p in params[:3]] == ["data", "x", "y"]
    assert all(p.kind is kind.POSITIONAL_OR_KEYWORD for p in params[:3])
    assert params[2].default is None
    assert _defaults(analyze) == ANALYSIS_DEFAULTS
    assert next(iter(inspect.signature(draw).parameters)) == "result"
    assert _defaults(draw) == DRAWING_DEFAULTS
    assert _defaults(top) == {**ANALYSIS_DEFAULTS, **DRAWING_DEFAULTS}
    assert set(DRAWING_DEFAULTS) == v.DRAWING_KEYWORDS["barstats"]
    # label and digits_perc are told apart by barstats' own messages.
    assert set(ANALYSIS_DEFAULTS) - {"label", "digits_perc"} <= v.ANALYSIS_KEYWORDS


@pytest.mark.parametrize(
    ("y", "kwargs", "test_name"),
    [
        ("am", {}, "pearson_chi_square"),
        (None, {}, "chi_square_gof"),
        ("am", {"type": "nonparametric"}, "pearson_chi_square"),
    ],
)
def test_each_design_is_analyzed_and_drawn(
    mtcars: pd.DataFrame, y: str | None, kwargs: dict[str, Any], test_name: str
) -> None:
    result = analyze(mtcars, "cyl", y, **kwargs)
    assert result.test is not None
    assert result.test.name == test_name
    # The one-call form analyzes and draws: the same numbers and texts, with the
    # drawing's own warnings appended (none for these figures).
    figure = top(mtcars, "cyl", y, **kwargs)
    assert figure.result.to_json() == result.to_json()
    assert set(figure.axes) == {"main"}


def test_the_paired_design_is_analyzed() -> None:
    data = pd.DataFrame({"a": ["no", "yes", "yes"], "b": ["yes", "yes", "no"]})
    result = analyze(data, "a", "b", paired=True)
    assert result.test is not None
    assert (result.test.name, result.test.statistic, result.n_obs) == (
        "mcnemar",
        0.0,
        3,
    )


def test_render_checks_its_arguments() -> None:
    result = bc.assembled(bc.BY_NAME["ggbarstats-mtcars-cyl-am-parametric"])
    # A copy without observations keeps the counts: still a whole result, and
    # drawn (tests/render/test_barstats_render.py draws every fixture).
    copy = StatsResult.from_json(result.to_json(include_values=False))
    assert copy.counts == result.counts
    assert draw(copy).result.counts == result.counts
    message(TypeError, lambda: draw(result, xlab=1))
    message(TypeError, lambda: draw(result, legend_title=2))
    message(TypeError, lambda: draw(result, bar_args=[1]))
    message(TypeError, lambda: draw(result, ax="axes"))
    assert "subtitle= would replace" in message(
        ValueError, lambda: draw(result, subtitle="mine")
    )
    other = bc.assembled(bc.BY_NAME["ggbarstats-mtcars-cyl-parametric"])
    assert other.analysis == "barstats"
    histo = ip.analyze.histostats(pd.DataFrame({"v": [1.0, 2.0, 4.0]}), "v")
    assert "this result comes from ip.analyze.histostats()" in message(
        TypeError, lambda: draw(histo)
    )


def test_the_top_level_form_checks_drawing_arguments_first(
    mtcars: pd.DataFrame,
) -> None:
    # A bad title is reported before the data are read (x is not a column).
    text = message(TypeError, lambda: top(mtcars, "nope", title=3))
    assert text.startswith("title must be a string or None")


def test_analysis_imports_no_matplotlib() -> None:
    code = textwrap.dedent(
        """
        import sys
        import pandas as pd
        import inferplot
        frame = pd.DataFrame({"a": list("xxyyx"), "b": list("ppqqq")})
        for y in ("b", None):
            inferplot.analyze.barstats(frame, "a", y)
        inferplot.analyze.barstats(frame, "a", "b", results_subtitle=False)
        assert "matplotlib" not in sys.modules
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


# --------------------------------------------------------------------------- #
# argument values
# --------------------------------------------------------------------------- #


def test_type(mtcars: pd.DataFrame) -> None:
    for given in ("robust", "r"):
        assert message(
            ValueError, lambda given=given: analyze(mtcars, "cyl", "am", type=given)
        ) == (
            f"type={given!r}: there is no robust method for counts. barstats' tests "
            "(Pearson's chi-square, McNemar's, goodness of fit) use the counts "
            "themselves, and ggstatsplot silently runs the same test for "
            "type='robust'. Use type='parametric' (type='nonparametric' runs the "
            "same test)."
        )
    assert message(
        NotImplementedError, lambda: analyze(mtcars, "cyl", type="bayes")
    ).startswith("type='bayes': the Bayes factor of counts")
    _, args, _ = prepare(mtcars, "cyl", "am", type="np")
    assert args.type == "nonparametric"
    assert "Did you mean type='parametric'" in message(
        ValueError, lambda: analyze(mtcars, "cyl", type="parametrc")
    )


def test_ratio_is_never_rescaled(mtcars: pd.DataFrame) -> None:
    _, args, _ = prepare(mtcars, "cyl", ratio=np.array([0.5, 0.25, 0.25]))
    assert args.ratio == (0.5, 0.25, 0.25)
    _, args, _ = prepare(mtcars, "cyl", ratio=[np.float32(0.6), 0.2, 0.2])
    assert args.ratio == (0.6, 0.2, 0.2)
    assert message(ValueError, lambda: analyze(mtcars, "cyl", ratio=(2, 1, 1))) == (
        "ratio must sum to 1: it holds the expected share of each level of x, and "
        "[2.0, 1.0, 1.0] sums to 4. inferplot does not rescale it (R's chisq.test "
        "refuses it, and ggstatsplot then leaves out its subtitle silently). If "
        "these are relative sizes, write ratio=[0.5, 0.25, 0.25]."
    )
    thirds = message(ValueError, lambda: analyze(mtcars, "cyl", ratio=[1, 1, 1]))
    assert thirds.endswith(
        "write ratio=[0.333333333333, 0.333333333333, 0.333333333333]."
    )
    # The values the message suggests are accepted.
    prepare(mtcars, "cyl", ratio=[0.333333333333] * 3)
    assert message(ValueError, lambda: analyze(mtcars, "cyl", ratio=[0.5, 0.5])) == (
        "ratio has 2 shares but x='cyl' has 3 levels: '4', '6', '8'. Give one share "
        "per level, in that order (R's chisq.test refuses it too, and ggstatsplot "
        "then leaves out its subtitle silently)."
    )
    assert "ratio must hold positive numbers" in message(
        ValueError, lambda: analyze(mtcars, "cyl", ratio=[1.0, 0.0, 0.0])
    )
    for bad in ("0.5", {0.5, 0.25}, {"4": 0.5}, 0.5, [True, 0.5]):
        message(TypeError, lambda bad=bad: analyze(mtcars, "cyl", ratio=bad))
    declared = mtcars.assign(cyl=pd.Categorical(mtcars["cyl"], categories=[4, 5, 6, 8]))
    assert "x='cyl' has 4 levels, the declared empty one included ('5')" in message(
        ValueError, lambda: analyze(declared, "cyl", ratio=[0.5, 0.25, 0.25])
    )


def test_label_digits_perc_and_flags(mtcars: pd.DataFrame) -> None:
    for label in ("percentage", "counts", "both"):
        _, args, _ = prepare(mtcars, "cyl", "am", label=label)
        assert args.label == label
    assert message(ValueError, lambda: analyze(mtcars, "cyl", label="percent")) == (
        "label='percent' is not allowed; use one of 'percentage', 'counts', 'both'. "
        "Did you mean label='percentage'?"
    )
    message(TypeError, lambda: analyze(mtcars, "cyl", label=None))
    message(ValueError, lambda: analyze(mtcars, "cyl", digits_perc=16))
    message(TypeError, lambda: analyze(mtcars, "cyl", digits_perc=1.0))
    message(TypeError, lambda: analyze(mtcars, "cyl", "am", paired=1))
    message(TypeError, lambda: analyze(mtcars, "cyl", "am", proportion_test="yes"))
    assert message(TypeError, lambda: analyze(mtcars, "cyl", counts=["n"])).startswith(
        "counts must be a column name (a string)"
    )
    with pytest.raises(KeyError, match="counts='nope' is not a column"):
        analyze(mtcars, "cyl", counts="nope")


def test_proportion_test_follows_results_subtitle(mtcars: pd.DataFrame) -> None:
    def resolved(**kwargs: Any) -> bool:
        return bool(prepare(mtcars, "cyl", "am", **kwargs)[1].proportion_test)

    assert resolved() is True
    assert resolved(results_subtitle=False) is False
    assert resolved(results_subtitle=False, proportion_test=True) is True
    assert resolved(proportion_test=False) is False
    assert prepare(mtcars, "cyl")[1].proportion_test is False
    assert message(
        ValueError, lambda: analyze(mtcars, "cyl", proportion_test=True)
    ).startswith("proportion_test=True asks for a proportion test above each bar")
    assert (
        "proportion_test=False has no effect: without y there is a single bar, whose "
        "test is the main test."
    ) in warnings_of(mtcars, "cyl", proportion_test=False)


def test_paired_needs_y(mtcars: pd.DataFrame) -> None:
    assert message(ValueError, lambda: analyze(mtcars, "am", paired=True)).startswith(
        "paired=True needs y: x='am' and y are the two measurements"
    )


def test_bf_message(mtcars: pd.DataFrame) -> None:
    assert message(
        NotImplementedError, lambda: analyze(mtcars, "cyl", "am", bf_message=True)
    ).startswith("bf_message=True: the Bayes factor caption of counts")
    assert message(
        ValueError,
        lambda: analyze(mtcars, "cyl", "am", bf_message=True, results_subtitle=False),
    ).startswith("bf_message=True asks for the Bayes factor caption, which is drawn")
    assert message(
        ValueError, lambda: analyze(mtcars, "am", "vs", bf_message=True, paired=True)
    ).startswith("bf_message=True asks for a Bayes factor caption, which ggstatsplot")
    note = (
        "ggstatsplot draws a Bayes factor caption here; the Bayes factor of counts "
        "(a Bayesian contingency-table test) arrives in inferplot 0.2, so there is "
        "none. Pass bf_message=False to leave it out without this note."
    )
    assert note in warnings_of(mtcars, "cyl", "am")
    assert note not in warnings_of(mtcars, "cyl", "am", bf_message=False)
    assert any(
        "(a Bayesian goodness-of-fit test)" in w for w in warnings_of(mtcars, "cyl")
    )
    assert (
        "bf_message=False has no effect: results_subtitle=False leaves the caption out."
    ) in warnings_of(mtcars, "cyl", "am", bf_message=False, results_subtitle=False)
    assert not any(
        "Bayes" in w for w in warnings_of(mtcars, "cyl", "am", results_subtitle=False)
    )


def test_arguments_without_effect_warn(mtcars: pd.DataFrame) -> None:
    two_levels = warnings_of(mtcars, "am", "cyl", p_adjust_method="BH")
    assert (
        "p_adjust_method='BH' has no effect: it adjusts the pairwise Fisher tests "
        "between the levels of x, and x='am' has 2 levels, and they are computed "
        "for three or more."
    ) in two_levels
    assert not any(
        "p_adjust_method" in w
        for w in warnings_of(mtcars, "cyl", "am", p_adjust_method="BH")
    )
    assert any(
        "results_subtitle=False leaves them out" in w
        for w in warnings_of(
            mtcars, "cyl", "am", p_adjust_method="none", results_subtitle=False
        )
    )
    assert (
        "The proportion test of each bar reports Pearson's C with a 95% interval "
        "whatever conf_level says (as in ggstatsplot): conf_level=0.9 applies to "
        "the main effect size and the pairs only."
    ) in warnings_of(mtcars, "cyl", "am", conf_level=0.9)
    assert not any(
        "conf_level" in w for w in warnings_of(mtcars, "cyl", conf_level=0.9)
    )
    assert (
        "conf_level=0.9 has no effect: results_subtitle=False computes no main test "
        "and no pairs; the proportion test of each bar reports a 95% interval "
        "whatever conf_level says (as in ggstatsplot)."
    ) in warnings_of(mtcars, "cyl", "am", conf_level=0.9, results_subtitle=False)
    assert (
        "conf_level=0.9 has no effect: results_subtitle=False computes no main test "
        "and no pairs."
    ) in warnings_of(mtcars, "cyl", conf_level=0.9, results_subtitle=False)
    assert (
        "digits_perc=1 has no effect: label='counts' shows no percentages."
    ) in warnings_of(mtcars, "cyl", "am", label="counts", digits_perc=1)
    assert (
        "ratio= sets the expected shares of the proportion test of each bar; the "
        "chi-square test of independence does not use it (as in ggstatsplot)."
    ) in warnings_of(mtcars, "cyl", "am", ratio=[0.25, 0.25, 0.5])
    greater = warnings_of(mtcars, "cyl", "am", alternative="greater")
    assert (
        "alternative='greater' makes the Cramér's V interval one-sided, [low, 1], "
        "also for each pair, and makes the Fisher test of each pair whose table is "
        "2 x 2 one-sided; the chi-square test has no direction, so its p-value is "
        "the same, as are the proportion tests of the bars (as in ggstatsplot)."
    ) in greater
    assert any(
        "alternative='less' has no effect: results_subtitle=False" in w
        for w in warnings_of(mtcars, "cyl", alternative="less", results_subtitle=False)
    )


# --------------------------------------------------------------------------- #
# the design checks that need only the table (D101)
# --------------------------------------------------------------------------- #


def test_tables_no_test_can_use_raise() -> None:
    data = pd.DataFrame(
        {"one": ["a"] * 4, "x": ["a", "b", "a", "b"], "y": ["p", "p", "q", "q"]}
    )
    assert message(ValueError, lambda: analyze(data, "one")) == (
        "x='one' has a single category ('a'), and the goodness-of-fit test compares "
        "the counts of two or more (ggstatsplot leaves out its subtitle silently "
        "here). If other categories are possible but have no rows, declare them, for "
        "example pd.Categorical(data['one'], categories=['a', ...]), and they count "
        "as 0."
    )
    assert message(ValueError, lambda: analyze(data, "one", "y")) == (
        "The chi-square test of independence needs observations in two or more "
        "levels of x='one' and of y='y', and 'one' has observations only in 'a'. "
        "(ggstatsplot turns such a table into a goodness-of-fit test, finds no "
        "effect size for it and leaves out its subtitle silently.) To test how the "
        "observations divide among the levels of 'y', count that column alone: "
        "barstats(data, x='y')."
    )
    assert message(ValueError, lambda: analyze(data, "x", "one")).endswith(
        "barstats(data, x='x')."
    )
    assert "Every observation is in one cell" in message(
        ValueError, lambda: analyze(data.assign(z="u"), "one", "z")
    )
    # A declared empty category makes a one-way table of two categories.
    declared = data.assign(one=pd.Categorical(data["one"], categories=["a", "b"]))
    one_way = analyze(declared, "one")
    assert one_way.counts == ((4,), (0,))
    assert one_way.test is not None
    assert one_way.test.statistic == 4.0  # (4 - 2)^2 / 2 + (0 - 2)^2 / 2
    # ... but not a two-way table: its empty row is left out.
    message(ValueError, lambda: analyze(declared, "one", "y"))
    single = pd.DataFrame(
        {
            "a": pd.Categorical(["no"] * 3, categories=["no"]),
            "b": pd.Categorical(["no"] * 3, categories=["no"]),
        }
    )
    assert "have a single level ('no')" in message(
        ValueError, lambda: analyze(single, "a", "b", paired=True)
    )


def test_empty_declared_levels_are_worded_by_design() -> None:
    data = pd.DataFrame(
        {
            "x": pd.Categorical(["a", "b", "a", "b"], categories=["a", "e", "b"]),
            "y": pd.Categorical(["p", "p", "q", "q"], categories=["p", "q", "none"]),
        }
    )
    two_way = warnings_of(data, "x", "y")
    (empty_x,) = [w for w in two_way if w.startswith("The declared level 'e'")]
    assert empty_x.startswith(
        "The declared level 'e' of 'x' has no observations; it stays in the legend "
        "with no share of any bar, and the chi-square test leaves the empty row out "
        "(as ggstatsplot does). Each bar's proportion test, though, counts it as a "
        "category expected to hold 1/3 of the bar (3 categories in all)"
    )
    assert (
        "The declared level 'none' of 'y' has no observations, so it has no bar, and "
        "the chi-square test leaves the empty column out (as ggstatsplot does)."
    ) in two_way
    assert (
        "The declared level 'e' of 'x' has no observations; it is a category with a "
        "count of 0 in the goodness-of-fit test, as ggstatsplot counts it."
    ) in warnings_of(data, "x")


# --------------------------------------------------------------------------- #
# keywords
# --------------------------------------------------------------------------- #

TARGETED = {
    "hue": "hue= is a seaborn argument. barstats() colours each bar by x",
    "stacked": "the bars are always stacked to 100%",
    "normalize": "label= chooses what the cells say",
    "stat": "label= chooses what the cells say",
    "dodge": "the bars are always stacked, not side by side",
    "orient": "the bars are vertical",
    "weights": "is passed as counts=",
    "wt": "is passed as counts=",
    "kind": "it is an argument of pandas' plot()",
    "index": "Pass the column as x=",
    "columns": "Pass the column as y=",
    "values": "counts='<column>'",
    "margins": "the totals of the table",
    "legend": "its title is legend_title=",
    "label.args": "the labels inside the bars are not styled",
    "sample.size.label.args": "the n below each bar is not styled",
    "label.repel": "an argument of ggstatsplot's ggpiestats",
    "perc.k": "perc.k was renamed digits.perc in ggstatsplot 0.12.0",
    "labels": "label= chooses what the cells say",
    "order": "barstats has no order=",
    "levels": "barstats has no order=",
    "subject_id": "there are no subjects to name",
    "test_value": "tests the shares of the levels of x against ratio=",
    "p": "are passed as ratio=",
    "f_exp": "are passed as ratio=",
    "correct": "without a continuity correction",
    "simulate.p.value": "there is no simulated p-value",
    "tr": "barstats has no trimmed methods",
    "bf.prior": "ggbarstats has no bf.prior",
    "pairwise_display": "never draws them",
    "pairwise.comparisons": "never draws them",
    "var.equal": "the tests of counts have no equal-variance option",
    "effsize.type": "the effect size follows the design",
    "method": "the test follows the design",
    "color": "colours are set with palette=",
    "alpha": "bar_args={'alpha': 0.5}",
    "point_args": "barstats draws stacked bars",
    "centrality.type": "barstats counts categories and has no centrality",
    "sampling.plan": "removed from ggbarstats in ggstatsplot 1.0.0",
    "ggtheme": "there are no ggplot2 themes in matplotlib",
    "xlabel": "matplotlib's name for the x axis label, which inferplot calls xlab=",
    "ylabel": "which inferplot calls ylab=",
}


@pytest.mark.parametrize(("keyword", "expected"), sorted(TARGETED.items()))
def test_targeted_keyword_messages(
    mtcars: pd.DataFrame, keyword: str, expected: str
) -> None:
    for fn in (analyze, top):
        text = message(TypeError, lambda fn=fn: fn(mtcars, "cyl", **{keyword: 1}))
        assert expected in text, text


def test_snake_case_of_ggstatsplot_names(mtcars: pd.DataFrame) -> None:
    for r_name, name in (
        ("digits.perc", "digits_perc"),
        ("proportion.test", "proportion_test"),
        ("legend.title", "legend_title"),
    ):
        text = message(
            TypeError, lambda r_name=r_name: top(mtcars, "cyl", **{r_name: 1})
        )
        assert f"Write {name}=1." in text


def test_drawing_and_analysis_keywords_point_to_their_function(
    mtcars: pd.DataFrame,
) -> None:
    assert message(TypeError, lambda: analyze(mtcars, "cyl", legend_title="a")) == (
        "legend_title= is a drawing option of ip.barstats(...) and "
        "ip.render.barstats(result, ...); ip.analyze.barstats() only computes the "
        "statistics. Pass legend_title= when drawing, or remove it."
    )
    result = bc.assembled(bc.BY_NAME["ggbarstats-mtcars-cyl-am-parametric"])
    for keyword in ("label", "digits_perc", "paired", "counts", "ratio"):
        text = message(TypeError, lambda keyword=keyword: draw(result, **{keyword: 1}))
        assert "argument of the analysis" in text, text
    assert "is passed as counts=" in message(
        TypeError, lambda: draw(result, weights="n")
    )
    assert "pass it to ip.analyze.barstats() or ip.barstats()" in message(
        TypeError, lambda: draw(result, weights="n")
    )


def test_an_extra_positional_value_is_likely_counts(mtcars: pd.DataFrame) -> None:
    text = message(TypeError, lambda: analyze(mtcars, "cyl", "am", "n"))
    assert text.endswith(
        "If 'n' is the column of counts of a table (how many observations each row "
        "stands for), write barstats(..., counts='n')."
    )


# --------------------------------------------------------------------------- #
# the fixtures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "case", [bc.BY_NAME[name] for name in sorted(bc.REFUSALS)], ids=sorted(bc.REFUSALS)
)
def test_refusals(case: OracleCase) -> None:
    refusal = bc.REFUSALS[case.name]
    with pytest.raises(refusal.error) as info:
        bc.prepared(case)
    assert refusal.match in str(info.value)
    if refusal.replay is not None:
        replayed = bc.assembled(case, changes=refusal.replay)
        assert replayed.counts == tuple(tuple(r) for r in case.counts_table())


def test_the_answered_and_refused_fixtures() -> None:
    # 22 refused: 7 robust, 7 bayes, 4 degenerate tables, paired with different
    # levels, paired without a changing pair, 2 wrong ratios.
    # Review round 3 added 10, all answered.
    assert len(bc.CASES) == 89 + 10
    assert len(bc.ANSWERED) == 67 + 10
    assert len(bc.REFUSALS) == 22


@pytest.mark.parametrize("case", bc.ANSWERED, ids=[c.name for c in bc.ANSWERED])
def test_every_answered_fixture_gives_a_whole_result(case: OracleCase) -> None:
    result = bc.assembled(case)
    data = case.contingency_data
    assert (result.analysis, result.design) == ("barstats", "contingency")
    assert list(result.levels) == data["x_levels"]
    assert list(result.y_levels) == data["y_levels"]
    assert [list(r) for r in result.counts] == case.counts_table()
    assert (result.n_obs, result.n_dropped) == (data["n"], data["rows_dropped"])
    assert StatsResult.from_json(result.to_json()) == result
    assert StatsResult.from_json(result.to_json(include_values=False)) == result
    totals = [sum(c) for c in zip(*case.counts_table(), strict=True)]
    assert [bar.n for bar in result.bars] == [t for t in totals if t]
    if result.y is not None:
        drawn = {str(row[result.y]): int(row["counts"]) for row in case.one_sample}
        assert {bar.level: bar.n for bar in result.bars} == drawn
    rows = result.frame_rows()
    assert len(rows) == (result.test is not None) + sum(
        bar.test is not None for bar in result.bars
    ) + len(result.pairwise)
    result.check()
    assert result.text.apa
    assert len(result.text.alt_text) <= 400


@pytest.mark.parametrize("backend", ["polars", "pyarrow"])
def test_results_agree_across_backends(backend: Backend) -> None:
    for case in bc.ANSWERED:
        pandas = bc.assembled(case)
        other = bc.assembled(case, backend)
        assert other.counts == pandas.counts
        assert other.cells == pandas.cells
        assert other.bars == pandas.bars
        # The recipes are written in each frame's own library (review B14).
        assert [_unlibrary(w) for w in other.warnings] == [
            _unlibrary(w) for w in pandas.warnings
        ]


def _unlibrary(warning: str) -> str:
    warning = re.sub(r"(in their names), .*", r"\1, ...", warning)
    return re.sub(r"(leave (?:it|them) out): .*", r"\1: ...", warning)


def test_cell_percentages_are_shares_of_their_bar() -> None:
    result = bc.assembled(bc.BY_NAME["ggbarstats-edge_bar_rounding-parametric"])
    first = [c for c in result.cells if c.bar == "g16"]
    assert [(c.level, c.count, c.percent) for c in first] == [
        ("a", 1, 6.25),
        ("b", 5, 31.25),
        ("c", 10, 62.5),
    ]
    thirds = [c.percent for c in result.cells if c.bar == "g3"]
    # (count / n) * 100, as ggstatsplot computes perc: 33.33333333333333, not
    # 100 / 3 = 33.333333333333336.
    assert thirds == [1 / 3 * 100] * 3


def test_texts_are_built_with_the_result() -> None:
    # Every text against ggstatsplot's: tests/test_barstats_texts.py.
    two_way = bc.assembled(bc.BY_NAME["ggbarstats-mtcars-cyl-am-parametric"])
    texts = two_way.text
    assert texts.subtitle is not None
    assert texts.subtitle.startswith("\u03c7_Pearson^2(2) = 8.74, p = 0.01, ")
    assert texts.caption is None  # the Bayes caption arrives in 0.2
    assert texts.cell_labels == ("16%", "21%", "63%", "62%", "23%", "15%")
    assert texts.bar_labels == ("p = 0.02", "p = 0.09")
    assert texts.n_labels == ("(n = 19)", "(n = 13)")
    quiet = bc.assembled(bc.BY_NAME["ggbarstats-mtcars-cyl-am-nosubtitle-parametric"])
    assert quiet.text.subtitle is None
    assert quiet.text.apa.startswith(
        "No test of independence was computed (results_subtitle=False), n = 32. "
    )
    long_names = module._assemble(*_long_names(), digits=2)
    assert len(long_names.text.alt_text) <= 400
    assert long_names.text.alt_text.endswith(
        "No test of independence was computed (results_subtitle=False), n = 4."
    )


def _long_names() -> tuple[Any, Any, Any]:
    data = pd.DataFrame({"x" * 300: list("aabb"), "y" * 300: list("pqpq")})
    table, args, _ = prepare(data, "x" * 300, "y" * 300, results_subtitle=False)
    return table, args, _stand_in(table, args)
