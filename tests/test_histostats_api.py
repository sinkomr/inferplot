"""``ip.histostats``, ``ip.analyze.histostats`` and ``ip.render.histostats``:
signatures, argument checks and the result a one-sample design returns.

Every type's analysis returns a complete result; the drawing (``ip.render`` and
the top-level form after its analysis) raises ``NotImplementedError`` until the
renderer lands. The result contract is checked on every fixture's real result
(``tests/no_level_cases.py``); the numbers are compared with R in
``tests/test_histostats_parity.py``.
"""

from __future__ import annotations

import importlib
import inspect
import math
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
from tests import no_level_cases as nl
from tests.oracle_args import split
from tests.oracle_fixtures import (
    Backend,
    OracleCase,
    cases_for,
    oracle_params,
    read_csv,
    read_dataset,
)

analyze: Any = ip.analyze.histostats
draw: Any = ip.render.histostats
top: Any = ip.histostats
module: Any = importlib.import_module("inferplot.analyses.histostats")

BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")
# ggstatsplot ignores alternative for these types; inferplot refuses it (SPEC §2).
REFUSED_ONE_SIDED = frozenset(
    {
        "gghistostats-ToothGrowth-greater-robust",
        "gghistostats-ToothGrowth-greater-bayes",
    }
)


def message(error: type[Exception], thunk: Callable[[], object]) -> str:
    """The message of the ``error`` that ``thunk()`` raises (exactly that type)."""
    with pytest.raises(error) as info:
        thunk()
    assert type(info.value) is error
    return str(info.value)


@pytest.fixture(scope="module")
def tooth() -> pd.DataFrame:
    return read_csv("ToothGrowth.csv")


def case_named(name: str) -> OracleCase:
    (case,) = [c for c in cases_for("gghistostats") if c.name == f"gghistostats-{name}"]
    return case


def made(name: str, **changes: Any) -> StatsResult:
    """The result of a fixture's call, with ``changes`` to its arguments."""
    case = case_named(name)
    return nl.analyze(case, nl.frame_of(case), **changes)


# --------------------------------------------------------------------------- #
# signatures and the three forms
# --------------------------------------------------------------------------- #

ANALYSIS_DEFAULTS = {
    "type": "parametric",
    "test_value": 0.0,
    "conf_level": 0.95,
    "digits": 2,
    "tr": 0.2,
    "bf_prior": 0.707,
    "bf_message": None,
    "results_subtitle": True,
    "alternative": "two-sided",
    "seed": 0,
}
DRAWING_DEFAULTS = {
    "binwidth": None,
    "centrality_plotting": True,
    "xlab": None,
    "ylab": None,
    "title": None,
    "subtitle": None,
    "caption": None,
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
    assert [p.name for p in params[:2]] == ["data", "x"]
    assert all(p.kind is kind.POSITIONAL_OR_KEYWORD for p in params[:2])
    assert _defaults(analyze) == ANALYSIS_DEFAULTS
    assert next(iter(inspect.signature(draw).parameters)) == "result"
    assert _defaults(draw) == DRAWING_DEFAULTS
    assert _defaults(top) == {**ANALYSIS_DEFAULTS, **DRAWING_DEFAULTS}
    assert set(DRAWING_DEFAULTS) == v.DRAWING_KEYWORDS["histostats"]
    assert set(ANALYSIS_DEFAULTS) <= v.ANALYSIS_KEYWORDS


@pytest.mark.parametrize(
    "type_name", ["parametric", "nonparametric", "robust", "bayes"]
)
def test_each_type_returns_a_result_and_draws(
    tooth: pd.DataFrame, type_name: str
) -> None:
    result = analyze(tooth, "len", type=type_name, test_value=18)
    assert (result.analysis, result.type, result.n_obs) == (
        "histostats",
        type_name,
        60,
    )
    assert result.text.apa.endswith("n = 60.")
    # ip.histostats analyzes, then draws the same result.
    fig = top(tooth, "len", type=type_name, test_value=18)
    assert fig.result.text.apa == result.text.apa


def test_aliases_are_canonical(tooth: pd.DataFrame) -> None:
    for alias, canonical in (("p", "parametric"), ("np", "nonparametric")):
        _, args, _ = module._prepare(tooth, "len", type=alias)
        assert args.type == canonical
    _, args, _ = module._prepare(tooth, "len", type="r")
    assert args.type == "robust"
    _, args, _ = module._prepare(tooth, "len", type="bf")
    assert args.type == "bayes"


def test_render_checks_its_arguments_then_draws() -> None:
    result = made("ToothGrowth-parametric")
    assert isinstance(draw(result), ip.render.StatsFigure)
    assert "binwidth must be a finite number greater than 0" in message(
        ValueError, lambda: draw(result, binwidth=0)
    )
    assert message(TypeError, lambda: draw(result, centrality_plotting=1)) == (
        "centrality_plotting must be True or False; got 1."
    )
    assert "bar_args must be a dict" in message(
        TypeError, lambda: draw(result, bar_args=["alpha"])
    )
    assert message(TypeError, lambda: draw(result, test_value=3)).startswith(
        "test_value= is an argument of the analysis, ip.analyze.histostats("
    )
    between = ip.analyze.betweenstats(read_csv("ToothGrowth.csv"), "supp", "len")
    assert "Draw it with ip.render.betweenstats(result)" in message(
        TypeError, lambda: ip.render.histostats(between)
    )


def test_render_refuses_data(tooth: pd.DataFrame) -> None:
    assert message(TypeError, lambda: draw(tooth)) == (
        "render.histostats() draws the result of ip.analyze.histostats(data, x); "
        "got a pandas.DataFrame. Compute it first with result = "
        "ip.analyze.histostats(data, x, ...) and pass result, or use "
        "ip.histostats(data, x, ...) to analyze and draw in one call."
    )


def test_top_level_checks_drawing_arguments_before_the_analysis(
    tooth: pd.DataFrame,
) -> None:
    assert "binwidth must be a finite number greater than 0" in message(
        ValueError, lambda: top(tooth, "len", binwidth=-1)
    )
    assert message(TypeError, lambda: top(tooth, "len", xlab=3)).startswith(
        "xlab must be a string or None"
    )


def test_analysis_imports_no_matplotlib() -> None:
    code = textwrap.dedent(
        """
        import sys
        import pandas as pd
        import inferplot
        frame = pd.DataFrame({"v": [1.0, 2.0, 4.0, 3.0, 8.0]})
        for kind in ("parametric", "nonparametric", "robust", "bayes"):
            result = inferplot.analyze.histostats(frame, "v", type=kind, test_value=2)
            result.to_json()
            result.check()
            str(result)
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


def test_test_value_must_be_a_finite_number(tooth: pd.DataFrame) -> None:
    for bad, error in ((True, TypeError), ("18", TypeError), (None, TypeError)):
        text = message(error, lambda bad=bad: analyze(tooth, "len", test_value=bad))
        assert text.startswith("test_value must be a number, for example test_value=0")
    for bad in (math.nan, math.inf, -math.inf):
        text = message(
            ValueError, lambda bad=bad: analyze(tooth, "len", test_value=bad)
        )
        assert text == (
            "test_value must be a finite number, for example test_value=0; got "
            f"{bad!r}."
        )
    for good, expected in (
        (18, 18.0),
        (np.int64(18), 18.0),
        (np.float32(0.9), 0.9),
        (-2.5, -2.5),
    ):
        _, args, _ = module._prepare(tooth, "len", test_value=good)
        assert args.test_value == expected
        assert type(args.test_value) is float


def test_common_arguments(tooth: pd.DataFrame) -> None:
    assert "It is a proportion, not a percentage: write conf_level=0.95." in message(
        ValueError, lambda: analyze(tooth, "len", conf_level=95)
    )
    assert "digits is a number of decimal places and must be at most 15" in message(
        ValueError, lambda: analyze(tooth, "len", digits=16)
    )
    assert message(TypeError, lambda: analyze(tooth, "len", digits=True)).startswith(
        "digits must be a whole number"
    )
    assert "bf_prior must be a finite number greater than 0" in message(
        ValueError, lambda: analyze(tooth, "len", bf_prior=0)
    )
    assert message(TypeError, lambda: analyze(tooth, "len", bf_message=1)) == (
        "bf_message must be None, True or False; got 1."
    )
    assert message(TypeError, lambda: analyze(tooth, "len", results_subtitle=None)) == (
        "results_subtitle must be True or False; got None."
    )
    assert "Did you mean type='nonparametric'?" in message(
        ValueError, lambda: analyze(tooth, "len", type="nonparam")
    )
    assert "R spells it 'two.sided'" in message(
        ValueError, lambda: analyze(tooth, "len", alternative="two.sided")
    )
    assert message(ValueError, lambda: analyze(tooth, "len", seed=-1)).startswith(
        "seed must be a whole number >= 0"
    )
    _, args, _ = module._prepare(tooth, "len", conf_level=np.float32(0.9))
    assert args.conf_level == 0.9


def test_tr_only_with_type_robust(tooth: pd.DataFrame) -> None:
    text = message(ValueError, lambda: analyze(tooth, "len", tr=0.1))
    assert text.startswith(
        "tr=0.1 sets the trimming of the robust methods (type='robust'), but "
        "type='parametric' trims nothing"
    )
    _, args, _ = module._prepare(tooth, "len", type="robust", tr=0.1)
    assert args.tr == 0.1


def test_bf_message_rules(tooth: pd.DataFrame) -> None:
    text = message(
        ValueError, lambda: analyze(tooth, "len", type="robust", bf_message=True)
    )
    assert text.startswith("bf_message=True asks for a Bayes factor caption")
    text = message(
        ValueError, lambda: analyze(tooth, "len", type="bayes", bf_message=True)
    )
    assert text.endswith("type='bayes' reports the Bayes factor in the subtitle.")
    text = message(
        ValueError,
        lambda: analyze(tooth, "len", bf_message=True, results_subtitle=False),
    )
    assert "results_subtitle=False turns both off" in text


@pytest.mark.parametrize("side", ["less", "greater"])
def test_one_sided_only_for_the_t_and_signed_rank_tests(
    tooth: pd.DataFrame, side: str
) -> None:
    for type_name in ("parametric", "nonparametric"):
        _, args, _ = module._prepare(tooth, "len", type=type_name, alternative=side)
        assert args.alternative == side
    for type_name, why in (
        ("robust", "the bootstrap-t test of the trimmed mean is always two-sided"),
        ("bayes", "the Bayes factor is always two-sided"),
    ):
        text = message(
            ValueError,
            lambda type_name=type_name: analyze(
                tooth, "len", type=type_name, alternative=side
            ),
        )
        assert text == (
            f"alternative={side!r} applies to type='parametric' and "
            f"type='nonparametric' only: with type={type_name!r} {why} "
            "(ggstatsplot silently ignores alternative there). Remove alternative=, "
            "or use type='parametric' (the one-sample t-test) or "
            "type='nonparametric' (the signed-rank test) for a one-sided test."
        )


# --------------------------------------------------------------------------- #
# keywords: targeted messages
# --------------------------------------------------------------------------- #


# Phrases of the comparison functions' messages that are wrong for a design without
# groups.
WRONG_WORDS = (
    "levels of x",
    "level of x",
    "independent groups",
    "colors already follow x",
    "first level",
    "Welch",
    "boxplot_args",
    "violin_args",
)


@pytest.mark.parametrize(
    ("keyword", "start"),
    [
        ("test.value", "histostats() has no argument 'test.value': inferplot spells"),
        ("conf.level", "histostats() has no argument 'conf.level'"),
        ("bf.prior", "histostats() has no argument 'bf.prior'"),
        ("bf.message", "histostats() has no argument 'bf.message'"),
        ("results.subtitle", "histostats() has no argument 'results.subtitle'"),
        ("binwidth", "binwidth= is a drawing option of ip.histostats(...)"),
        ("centrality.plotting", "centrality.plotting= is a drawing option of"),
        ("bar_args", "bar_args= is a drawing option of ip.histostats(...)"),
        ("bin.args", "bin.args= is not an argument of histostats(): the bars are"),
        ("centrality.line.args", "centrality.line.args= is not an argument of"),
        ("centrality.type", "centrality.type= is a ggstatsplot argument that"),
        ("centrality.point.args", "centrality.point.args= is a ggstatsplot argument"),
        ("ggsignif.args", "ggsignif.args= is a ggstatsplot argument that inferplot"),
        ("normal.curve", "normal.curve= is not an argument of ggstatsplot 1.1.4's"),
        ("normal.curve.args", "normal.curve.args= is not an argument of ggstatsplot"),
        ("label.var", "label.var= is not an argument of histostats(): it is an"),
        ("label.expression", "label.expression= is a ggstatsplot argument (a dplyr"),
        ("ggtheme", "ggtheme= is a ggstatsplot argument that inferplot does not"),
        ("ggplot.component", "ggplot.component= is a ggstatsplot argument"),
        ("y", "y= is not an argument of histostats(): histostats() tests one"),
        ("subject_id", "subject_id= is not an argument of histostats(): "),
        ("pairwise_display", "pairwise_display= is not an argument of histostats()"),
        ("p.adjust.method", "p.adjust.method= is not an argument of histostats()"),
        ("order", "order= is not an argument of histostats(): histostats() tests"),
        ("paired", "paired= is not an argument of histostats(): histostats() tests"),
        ("bins", "bins= is not an argument of histostats(): the width of the bars"),
        ("kde", "kde= is not an argument of histostats(): it is a seaborn argument"),
        ("stat", "stat= is not an argument of histostats(): the left axis counts"),
        ("density", "density= is not an argument of histostats(): the left axis"),
        ("color", "color= is not an argument of histostats(): colours are set"),
        ("hue", "hue= is a seaborn argument; histostats() has no grouping"),
        ("alpha", "alpha= is not an argument of histostats(): conf_level= sets"),
        ("legend", "legend= is not an argument of histostats(): there is no legend."),
        ("var.equal", "var.equal= was removed in ggstatsplot 1.0.0; the one-sample"),
        ("effsize.type", "effsize.type= was removed in ggstatsplot 1.0.0; Hedges' g"),
        ("tail", "tail= is not an argument of histostats(); the direction of the"),
        ("point_path", "point_path= is not an argument of histostats(): it is an"),
        ("nboot", "nboot= was removed in ggstatsplot 1.0.0"),
        ("k", "k= was renamed digits in ggstatsplot 0.12.3"),
    ],
)
def test_targeted_keyword_messages(
    tooth: pd.DataFrame, keyword: str, start: str
) -> None:
    text = message(TypeError, lambda: analyze(tooth, "len", **{keyword: 1}))
    assert text.startswith(start), text
    # No words of the comparison functions, whose x holds groups or levels.
    for phrase in WRONG_WORDS:
        assert phrase not in text, (phrase, text)


def test_messages_name_the_function_that_takes_the_argument(
    tooth: pd.DataFrame,
) -> None:
    assert message(TypeError, lambda: analyze(tooth, "len", y="dose")) == (
        "y= is not an argument of histostats(): histostats() tests one numeric "
        "column, x, against test_value. To relate x to another column use "
        "scatterstats(data, x, y); to compare a column across the groups of x use "
        "betweenstats(data, x, y). Remove y=."
    )
    assert message(TypeError, lambda: analyze(tooth, "len", subject_id="id")) == (
        "subject_id= is not an argument of histostats(): histostats() tests one "
        "numeric column, x, against test_value and has no subjects. For repeated "
        "measures of the same subjects use withinstats(data, x, y, subject_id=...). "
        "Remove subject_id=."
    )
    assert "label_var='label' to scatterstats()" in message(
        TypeError, lambda: analyze(tooth, "len", label_expression="len > 3")
    )
    # A column name given as a third positional argument.
    assert message(TypeError, lambda: analyze(tooth, "len", "dose")) == (
        "histostats() takes 2 positional arguments (data, x) but 3 were given. "
        "Everything after data, x is keyword-only; name each extra argument. Extra "
        "positional values: 'dose'. histostats() tests one column against "
        "test_value; to relate x to 'dose' use scatterstats(data, x, 'dose'), and to "
        "compare 'dose' across the groups of x use betweenstats(data, x, 'dose')."
    )
    # The drawing forms say the same for their own keywords.
    assert message(TypeError, lambda: top(tooth, "len", bins=10)).startswith(
        "bins= is not an argument of histostats(): the width of the bars is set by "
        "binwidth=, a drawing option of histostats()"
    )


def test_grouping_columns_point_to_the_function_that_fits(tooth: pd.DataFrame) -> None:
    assert message(TypeError, lambda: analyze(tooth, "supp")).startswith(
        "x='supp' holds text or categories, not numbers. histostats() compares one "
        "numeric column with test_value. To compare a numeric column across the "
        "groups of 'supp', use betweenstats(data, x='supp', y='<numeric column>')"
    )


# --------------------------------------------------------------------------- #
# warnings without effect
# --------------------------------------------------------------------------- #


def test_arguments_without_effect_warn() -> None:
    result = made("ToothGrowth-nonparametric", bf_prior=0.8, bf_message=False)
    assert result.warnings == (
        "bf_message=False has no effect: the Bayes factor caption is drawn only for "
        "type='parametric' with the results subtitle (as in ggstatsplot), and "
        "type='nonparametric' draws none.",
        "bf_prior=0.8 has no effect: no Bayes factor is computed "
        "(type='nonparametric' has no Bayes factor).",
    )
    result = made("ToothGrowth-bayes", conf_level=0.9)
    assert result.warnings == (
        "The posterior difference and its credible interval, which ggstatsplot adds "
        "to the Bayes factor subtitle, arrive in inferplot 0.2.",
        "conf_level=0.9 has no effect: type='bayes' shows no interval in this "
        "version (the posterior credible interval arrives in inferplot 0.2).",
    )
    result = made("ToothGrowth-parametric", seed=3)
    assert result.warnings == (
        "seed=3 has no effect: type='parametric' draws no random numbers (only "
        "type='robust' resamples, for its bootstrap-t test), and the histogram is "
        "drawn from every value.",
    )
    result = made("ToothGrowth-greater-parametric")
    assert result.warnings == (
        "The Bayes factor in the caption is two-sided: it weighs any difference from "
        "test_value, while alternative='greater' applies to the test and its effect "
        "size interval only (as in ggstatsplot).",
    )
    # Defaults and arguments that act add nothing.
    assert made("ToothGrowth-parametric").warnings == ()
    assert made("ToothGrowth-prior08-parametric").warnings == ()


# --------------------------------------------------------------------------- #
# fixture replays: the analyzed values, the counts and the assembled result
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", oracle_params("gghistostats"))
def test_fixture_values_and_counts(case: OracleCase, backend: Backend) -> None:
    call = split(case.function, case.args)
    frame = read_dataset(case, backend)
    if case.name in REFUSED_ONE_SIDED:
        assert "applies to type='parametric' and type='nonparametric' only" in message(
            ValueError, lambda: analyze(frame, *call.args, **call.analysis)
        )
        return
    data, args, _ = module._prepare(frame, *call.args, **call.analysis)
    assert data.values.tolist() == case.analyzed_values("x")
    assert data.n_dropped == case.analyzed_data["rows_dropped"]
    assert args.test_value == float(case.args.get("test.value", 0))


# Words of the comparison designs that a one-sample result never uses.
GROUP_WORDS = re.compile(r"\b(groups?|levels?|conditions?|pairs?|subjects?)\b", re.I)


@pytest.mark.parametrize("case", oracle_params("gghistostats"))
def test_result_contract_of_each_fixture(case: OracleCase) -> None:
    result = nl.analyzed(case.name)
    if result is None:
        # Constant data with types parametric, bayes and robust (edge_one_constant,
        # barstats oracle run): refused (tests/test_histostats_parity.py). The two
        # one-sided refusals are answered by their replay.
        assert nl.REFUSALS[case.name].replay is None
        assert "edge_one_constant" in case.name
        return
    n = case.analyzed_data["n"]
    assert (result.analysis, result.design) == ("histostats", "one-sample")
    assert (result.x, result.y, result.levels) == (case.args["x"], None, ())
    assert result.values == (tuple(case.analyzed_values("x")),)
    assert result.n_obs == n
    assert result.n_dropped == case.analyzed_data["rows_dropped"]
    (descriptive,) = result.descriptives
    assert (descriptive.level, descriptive.n) == (case.args["x"], n)
    settings = result.settings
    assert settings.test_value == float(case.args.get("test.value", 0))
    assert settings.conf_level == float(case.args.get("conf.level", 0.95))
    assert settings.digits == int(case.args.get("digits", 2))
    assert settings.results_subtitle is case.args.get("results.subtitle", True)
    assert settings.tr == float(case.args.get("tr", 0.2))
    assert result.point_labels == ()
    assert result.label_var is None
    # A subtitle exactly where the call asks for one; the texts themselves are
    # compared with ggstatsplot's in tests/test_no_level_texts.py.
    assert (result.text.subtitle is None) == (not settings.results_subtitle)
    assert (result.text.subtitle is None) == (case.subtitle is None)
    assert (result.text.caption is None) == (case.caption is None)
    assert result.text.tick_labels == result.text.pairwise_labels == ()
    assert result.text.apa.endswith(f"n = {n:,}.")
    assert len(result.text.alt_text) <= 400
    for text in (result.text.apa, result.text.alt_text, *result.warnings):
        assert not GROUP_WORDS.search(text), text
    assert str(result).startswith(result.text.apa)
    back = StatsResult.from_json(result.to_json())
    assert back == result
    assert back.to_json() == result.to_json()
    embedded = StatsResult.from_json(result.to_json(include_values=False))
    assert embedded.values == ()
    assert embedded.check() == result.check()
    frame = result.to_frame()
    assert len(frame) == (0 if result.test is None and result.bayes is None else 1)
    if result.test is not None:
        assert frame["test"][0] == result.test.name
        assert frame["statistic"][0] == result.test.statistic
        assert frame["ci_sidedness"][0] == (
            result.effect_size.sidedness if result.effect_size else None
        )
    elif result.bayes is not None:
        assert frame["bf10"][0] == result.bayes.bf10
    assert len(result.to_frame(backend="polars")) == len(frame)


def test_one_far_value_gives_no_map_estimate_with_a_warning() -> None:
    # Review Z3: a bare ZeroDivisionError from the MAP bandwidth, in histostats and
    # betweenstats; now no centrality value and a warning naming the value.
    one = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 1e50]})
    result = ip.analyze.histostats(one, "x", type="bayes")
    assert result.descriptives[0].value is None
    assert any("no MAP estimate" in w and "1e+50" in w for w in result.warnings)
    groups = pd.DataFrame(
        {"g": ["a"] * 6 + ["b"] * 6, "y": [1.0, 2, 3, 4, 5, 1e60, 2, 3, 4, 5, 6, 7]}
    )
    between = ip.analyze.betweenstats(groups, "g", "y", type="bayes")
    assert between.descriptives[0].value is None
    assert between.descriptives[1].value is not None
    assert any("no MAP estimate" in w and "1e+60" in w for w in between.warnings)
