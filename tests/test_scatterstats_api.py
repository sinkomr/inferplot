"""``ip.scatterstats``, ``ip.analyze.scatterstats`` and ``ip.render.scatterstats``:
signatures, argument checks and the result a correlation returns.

Every type's analysis returns a complete result; the drawing (``ip.render`` and
the top-level form after its analysis) raises ``NotImplementedError`` until the
renderer lands. The result contract is checked on every fixture's real result
(``tests/no_level_cases.py``); the numbers are compared with R in
``tests/test_scatterstats_parity.py``.
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
from inferplot._data import ColumnNotFoundError
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

analyze: Any = ip.analyze.scatterstats
draw: Any = ip.render.scatterstats
top: Any = ip.scatterstats
module: Any = importlib.import_module("inferplot.analyses.scatterstats")

BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")
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


def message(error: type[Exception], thunk: Callable[[], object]) -> str:
    """The message of the ``error`` that ``thunk()`` raises (exactly that type)."""
    with pytest.raises(error) as info:
        thunk()
    assert type(info.value) is error
    return str(info.value)


@pytest.fixture(scope="module")
def cars() -> pd.DataFrame:
    return read_csv("mtcars.csv")


def case_named(name: str) -> OracleCase:
    (case,) = [
        c for c in cases_for("ggscatterstats") if c.name == f"ggscatterstats-{name}"
    ]
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
    "conf_level": 0.95,
    "digits": 2,
    "tr": 0.2,
    "bf_prior": 0.707,
    "bf_message": None,
    "results_subtitle": True,
    "label_var": None,
    "seed": 0,
}
DRAWING_DEFAULTS = {
    "marginal": True,
    "smooth": True,
    "xlab": None,
    "ylab": None,
    "title": None,
    "subtitle": None,
    "caption": None,
    "palette": None,
    "point_args": None,
    "smooth_args": None,
    "marginal_args": None,
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
    assert _defaults(analyze) == ANALYSIS_DEFAULTS
    assert next(iter(inspect.signature(draw).parameters)) == "result"
    assert _defaults(draw) == DRAWING_DEFAULTS
    assert _defaults(top) == {**ANALYSIS_DEFAULTS, **DRAWING_DEFAULTS}
    assert set(DRAWING_DEFAULTS) == v.DRAWING_KEYWORDS["scatterstats"]
    assert set(ANALYSIS_DEFAULTS) <= v.ANALYSIS_KEYWORDS


@pytest.mark.parametrize(
    "type_name", ["parametric", "nonparametric", "robust", "bayes"]
)
def test_each_type_returns_a_result_and_drawing_is_not_implemented(
    cars: pd.DataFrame, type_name: str
) -> None:
    result = analyze(cars, "wt", "mpg", type=type_name)
    assert (result.analysis, result.type, result.n_obs) == (
        "scatterstats",
        type_name,
        32,
    )
    assert result.text.apa.endswith("n = 32 pairs.")
    text = message(
        NotImplementedError,
        lambda: top(cars, "wt", "mpg", type=type_name, marginal=False),
    )
    assert text.startswith("render.scatterstats(): drawing a scatterstats result")


def test_render_checks_its_arguments_then_says_it_is_not_implemented() -> None:
    result = made("mtcars-parametric")
    text = message(NotImplementedError, lambda: draw(result))
    assert text.startswith("render.scatterstats(): drawing a scatterstats result")
    assert message(TypeError, lambda: draw(result, marginal="yes")) == (
        "marginal must be True or False; got 'yes'."
    )
    assert message(TypeError, lambda: draw(result, smooth=None)) == (
        "smooth must be True or False; got None."
    )
    for name in ("point_args", "smooth_args", "marginal_args"):
        assert f"{name} must be a dict" in message(
            TypeError, lambda name=name: draw(result, **{name: 1})
        )
    assert message(TypeError, lambda: draw(result, label_var="model")).startswith(
        "label_var= is an argument of the analysis, ip.analyze.scatterstats("
    )
    between = ip.analyze.betweenstats(read_csv("ToothGrowth.csv"), "supp", "len")
    assert "Draw it with ip.render.betweenstats(result)" in message(
        TypeError, lambda: draw(between)
    )


def test_ax_needs_marginal_false(cars: pd.DataFrame) -> None:
    result = made("mtcars-parametric")
    text = message(ValueError, lambda: draw(result, ax=object()))
    assert text == (
        "ax= draws into an existing Axes, which has no room for the marginal "
        "histograms (marginal=True draws them in panels of their own). Pass "
        "marginal=False to draw into ax, or leave ax out."
    )
    # ip.scatterstats says so before analyzing.
    assert message(ValueError, lambda: top(cars, "wt", "mpg", ax=object())) == text
    assert message(
        NotImplementedError, lambda: draw(result, ax=object(), marginal=False)
    ).startswith("render.scatterstats()")


def test_analysis_imports_no_matplotlib() -> None:
    code = textwrap.dedent(
        """
        import sys
        import pandas as pd
        import inferplot
        frame = pd.DataFrame({"a": [1.0, 2.0, 4.0, 3.0], "b": [2.0, 1.0, 5.0, 3.0]})
        for kind in ("parametric", "nonparametric", "robust", "bayes"):
            result = inferplot.analyze.scatterstats(frame, "a", "b", type=kind)
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


def test_common_arguments(cars: pd.DataFrame) -> None:
    assert "write conf_level=0.9." in message(
        ValueError, lambda: analyze(cars, "wt", "mpg", conf_level=90)
    )
    assert message(TypeError, lambda: analyze(cars, "wt", "mpg", conf_level=True))
    assert "digits is a number of decimal places and must be >= 0" in message(
        ValueError, lambda: analyze(cars, "wt", "mpg", digits=-1)
    )
    assert "tr=0.1 sets the trimming of the robust methods" in message(
        ValueError, lambda: analyze(cars, "wt", "mpg", tr=0.1)
    )
    _, args, _ = module._prepare(cars, "wt", "mpg", type="r", tr=np.float32(0.1))
    assert (args.type, args.tr) == ("robust", 0.1)
    assert "bf_message=True asks for a Bayes factor caption" in message(
        ValueError, lambda: analyze(cars, "wt", "mpg", type="np", bf_message=True)
    )
    assert "results_subtitle=False turns both off" in message(
        ValueError,
        lambda: analyze(cars, "wt", "mpg", bf_message=True, results_subtitle=False),
    )
    assert "bf_prior must be a finite number greater than 0" in message(
        ValueError, lambda: analyze(cars, "wt", "mpg", bf_prior=math.inf)
    )


def test_label_var_reads_any_column(cars: pd.DataFrame) -> None:
    data, args, _ = module._prepare(cars, "wt", "mpg", label_var="model")
    assert args.label_var == "model"
    assert data.labels[:2] == ("Mazda RX4", "Mazda RX4 Wag")
    data, _, _ = module._prepare(cars, "wt", "mpg", label_var="cyl")
    assert data.labels[:3] == ("6", "6", "4")
    assert "Did you mean 'model'?" in message(
        ColumnNotFoundError, lambda: analyze(cars, "wt", "mpg", label_var="Model")
    )


# --------------------------------------------------------------------------- #
# keywords: targeted messages
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("keyword", "start"),
    [
        ("alternative", "alternative= is not an argument of scatterstats(): "),
        ("tail", "tail= is not an argument of scatterstats(): correlations are"),
        ("label.var", "scatterstats() has no argument 'label.var': inferplot spells"),
        ("label.expression", "label.expression= is a ggstatsplot argument (a dplyr"),
        ("conf.level", "scatterstats() has no argument 'conf.level'"),
        ("bf.prior", "scatterstats() has no argument 'bf.prior'"),
        ("marginal", "marginal= is a drawing option of ip.scatterstats(...)"),
        ("smooth", "smooth= is a drawing option of ip.scatterstats(...)"),
        ("point.args", "point.args= is a drawing option of ip.scatterstats(...)"),
        ("point.label.args", "point.label.args= is not an argument of scatterstats()"),
        ("smooth.line.args", "smooth.line.args= is not an argument of scatterstats()"),
        ("xsidehistogram.args", "xsidehistogram.args= is not an argument of"),
        ("ysidehistogram.args", "ysidehistogram.args= is not an argument of"),
        ("test.value", "test.value= is not an argument of scatterstats(): it is an"),
        ("ggtheme", "ggtheme= is a ggstatsplot argument that inferplot does not"),
        ("centrality.type", "centrality.type= is a ggstatsplot argument that"),
        ("centrality.label.args", "centrality.label.args= is a ggstatsplot"),
        ("ggplot.component", "ggplot.component= is a ggstatsplot argument"),
        ("subject_id", "subject_id= is not an argument of scatterstats(): "),
        ("pairwise_alpha", "pairwise_alpha= is not an argument of scatterstats()"),
        ("paired", "paired= is not an argument of scatterstats(): scatterstats()"),
        ("fit_reg", "fit_reg= is not an argument of scatterstats(): it is a seaborn"),
        ("ci", "ci= is a seaborn argument; intervals in inferplot are always"),
        ("x_jitter", "x_jitter= is not an argument of scatterstats(): every point"),
        ("line_kws", "line_kws= is not an argument of scatterstats(): it is a"),
        ("scatter_kws", "scatter_kws= is not an argument of scatterstats(): it is"),
        ("color", "color= is not an argument of scatterstats(): colours are set"),
        ("hue", "hue= is a seaborn argument; scatterstats() has no grouping"),
        ("bins", "bins= is not an argument of scatterstats(): the marginal"),
        ("kde", "kde= is not an argument of scatterstats(): it is a seaborn"),
        ("label", "label= is not an argument of scatterstats(): point labels come"),
        ("legend", "legend= is not an argument of scatterstats(): there is no"),
        ("var.equal", "var.equal= was removed in ggstatsplot 1.0.0; a correlation"),
        ("effsize.type", "effsize.type= was removed in ggstatsplot 1.0.0; the"),
        ("order", "order= is not an argument of scatterstats(): scatterstats()"),
        ("centrality_path", "centrality_path= is not an argument of scatterstats()"),
    ],
)
def test_targeted_keyword_messages(
    cars: pd.DataFrame, keyword: str, start: str
) -> None:
    text = message(TypeError, lambda: analyze(cars, "wt", "mpg", **{keyword: 1}))
    assert text.startswith(start), text
    for phrase in WRONG_WORDS:
        assert phrase not in text, (phrase, text)


def test_alternative_message(cars: pd.DataFrame) -> None:
    assert message(
        TypeError, lambda: analyze(cars, "wt", "mpg", alternative="less")
    ) == (
        "alternative= is not an argument of scatterstats(): correlations are tested "
        "two-sided in ggscatterstats and in inferplot, so there is no direction to "
        "choose. Remove alternative=."
    )
    assert message(
        TypeError, lambda: top(cars, "wt", "mpg", alternative="less")
    ).startswith("alternative= is not an argument of scatterstats()")


def test_label_expression_message(cars: pd.DataFrame) -> None:
    assert message(
        TypeError, lambda: analyze(cars, "wt", "mpg", label_expression="mpg > 25")
    ) == (
        "label_expression= is a ggstatsplot argument (a dplyr filter) that inferplot "
        "does not have: inferplot labels every point whose label_var value is "
        "present. To label only some points, set the labels of the others to "
        "missing first, for example data.assign(label=data['model'].where((data["
        "'mpg'] > 25) | (data['wt'] > 5))), and pass label_var='label'. Remove "
        "label_expression=."
    )


def test_numeric_columns_only(cars: pd.DataFrame) -> None:
    assert message(TypeError, lambda: analyze(cars, "model", "mpg")).startswith(
        "x='model' holds text or categories, not numbers. scatterstats() relates two "
        "numeric columns. To compare 'mpg' across the groups of 'model', use "
        "betweenstats(data, x='model', y='mpg')"
    )
    assert "both are 'wt'" in message(ValueError, lambda: analyze(cars, "wt", "wt"))


# --------------------------------------------------------------------------- #
# warnings without effect
# --------------------------------------------------------------------------- #


def test_arguments_without_effect_warn() -> None:
    assert made("mtcars-robust", bf_prior=0.8).warnings == (
        "bf_prior=0.8 has no effect: no Bayes factor is computed (type='robust' has "
        "no Bayes factor).",
    )
    assert made("mtcars-nosubtitle-parametric", bf_prior=0.8).warnings == (
        "bf_prior=0.8 has no effect: no Bayes factor is computed "
        "(results_subtitle=False leaves the Bayes factor out).",
    )
    assert made("mtcars-bayes", bf_message=False).warnings == (
        "The posterior correlation and its credible interval, which ggstatsplot adds "
        "to the Bayes factor subtitle, arrive in inferplot 0.2.",
        "bf_message=False has no effect: the Bayes factor caption is drawn only for "
        "type='parametric' with the results subtitle (as in ggstatsplot), and "
        "type='bayes' draws none; type='bayes' reports its Bayes factor in the "
        "subtitle.",
    )
    assert made("mtcars-parametric").warnings == ()
    assert made("mtcars-prior08-bayes").warnings == (
        "The posterior correlation and its credible interval, which ggstatsplot adds "
        "to the Bayes factor subtitle, arrive in inferplot 0.2.",
    )


# --------------------------------------------------------------------------- #
# fixture replays: the analyzed pairs, the counts and the assembled result
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", oracle_params("ggscatterstats"))
def test_fixture_pairs_and_counts(case: OracleCase, backend: Backend) -> None:
    call = split(case.function, case.args)
    frame = read_dataset(case, backend)
    data, args, _ = module._prepare(frame, *call.args, **call.analysis)
    assert data.x_values.tolist() == case.analyzed_values("x")
    assert data.y_values.tolist() == case.analyzed_values("y")
    assert data.n_dropped == case.analyzed_data["rows_dropped"]
    assert args.type == case.type


# Words of the comparison designs that a correlation result never uses.
GROUP_WORDS = re.compile(r"\b(groups?|levels?|conditions?|subjects?)\b", re.I)


@pytest.mark.parametrize("case", oracle_params("ggscatterstats"))
def test_result_contract_of_each_fixture(case: OracleCase) -> None:
    result = nl.analyzed(case.name)
    if result is None:  # constant x: refused (tests/test_scatterstats_parity.py)
        assert case.name in nl.REFUSALS
        return
    n = case.analyzed_data["n"]
    assert (result.analysis, result.design) == ("scatterstats", "correlation")
    assert (result.x, result.y) == (case.args["x"], case.args["y"])
    assert (result.levels, result.descriptives, result.pairwise) == ((), (), ())
    assert result.values == (
        tuple(case.analyzed_values("x")),
        tuple(case.analyzed_values("y")),
    )
    assert result.n_obs == n
    assert result.n_obs + result.n_dropped == case.raw["dataset"]["rows"]
    settings = result.settings
    assert settings.test_value is None
    assert settings.alternative == "two-sided"
    assert settings.conf_level == float(case.args.get("conf.level", 0.95))
    assert settings.digits == int(case.args.get("digits", 2))
    assert settings.tr == float(case.args.get("tr", 0.2))
    assert settings.bf_prior == float(case.args.get("bf.prior", 0.707))
    label_var = case.args.get("label.var")
    assert result.label_var == label_var
    if label_var is None:
        assert result.point_labels == ()
    # A subtitle exactly where the call asks for one; the texts themselves are
    # compared with ggstatsplot's in tests/test_no_level_texts.py.
    assert (result.text.subtitle is None) == (not settings.results_subtitle)
    assert (result.text.subtitle is None) == (case.subtitle is None)
    assert (result.text.caption is None) == (case.caption is None)
    assert result.text.tick_labels == result.text.pairwise_labels == ()
    assert result.text.apa.endswith(f"n = {n:,} pairs.")
    assert len(result.text.alt_text) <= 400
    for text in (result.text.apa, result.text.alt_text, *result.warnings):
        assert not GROUP_WORDS.search(text), text
    assert str(result).startswith(result.text.apa)
    back = StatsResult.from_json(result.to_json())
    assert back == result
    assert back.to_json() == result.to_json()
    embedded = StatsResult.from_json(result.to_json(include_values=False))
    assert (embedded.values, embedded.point_labels) == ((), ())
    assert embedded.label_var == label_var
    assert embedded.check() == result.check()
    frame = result.to_frame()
    assert len(frame) == (0 if result.test is None and result.bayes is None else 1)
    if result.test is not None:
        assert frame["test"][0] == result.test.name
        assert frame["ci_sidedness"][0] == "two-sided"
    elif result.bayes is not None:
        assert frame["bf10"][0] == result.bayes.bf10
    assert len(result.to_frame(backend="polars")) == len(frame)
