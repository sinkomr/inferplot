"""``ip.withinstats``, ``ip.analyze.withinstats`` and ``ip.render.withinstats``:
signatures, argument checks and the result a within-subjects design returns.

No within-subjects statistics exist yet: every analysis call that passes the checks
and reads the data raises ``NotImplementedError`` naming the type and the number of
conditions. The result contract is checked through ``_assemble`` with parts made up
here, so it is ready for the statistics; ``ip.render.withinstats`` draws such a
result (tests/render/test_withinstats_render.py).
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
from collections.abc import Callable
from typing import Any

import pandas as pd
import polars as pl
import pytest

import inferplot as ip
from inferplot import _validate as v
from inferplot._data import read_within
from inferplot.result import StatsResult
from tests.oracle_fixtures import read_csv

analyze: Any = ip.analyze.withinstats
draw: Any = ip.render.withinstats
top: Any = ip.withinstats

# The analysis module (the package attribute of the same name is the function).
module: Any = importlib.import_module("inferplot.analyses.withinstats")

REQUIRED_MESSAGE = (
    "withinstats() needs subject_id=, the column that identifies the subject of each "
    "row, for example withinstats(data, x, y, subject_id='subject'). Measurements "
    "are paired by subject and never by row order, which silently pairs the wrong "
    "values once rows are sorted, filtered or missing. Put the data in long format: "
    "one row per subject and level of x, with a column naming the subject."
)


def message(error: type[Exception], thunk: Callable[[], object]) -> str:
    """The message of the ``error`` that ``thunk()`` raises (exactly that type)."""
    with pytest.raises(error) as info:
        thunk()
    assert type(info.value) is error
    return str(info.value)


@pytest.fixture(scope="module")
def two() -> pd.DataFrame:
    return read_csv("sleep.csv")


@pytest.fixture(scope="module")
def four() -> pd.DataFrame:
    return read_csv("bugs_long.csv")


def call(frame: Any, **kwargs: Any) -> Any:
    return analyze(frame, "group", "extra", subject_id="ID", **kwargs)


# --------------------------------------------------------------------------- #
# signatures
# --------------------------------------------------------------------------- #

ANALYSIS_DEFAULTS = {
    "type": "parametric",
    "pairwise_display": "significant",
    "p_adjust_method": "holm",
    "pairwise_alpha": 0.05,
    "conf_level": 0.95,
    "digits": 2,
    "tr": 0.2,
    "bf_prior": 0.707,
    "bf_message": None,
    "results_subtitle": True,
    "alternative": "two-sided",
    "seed": 0,
    "order": None,
}
DRAWING_DEFAULTS = {
    "point_path": True,
    "centrality_path": True,
    "centrality_plotting": True,
    "layers": ("violin", "box", "points"),
    "xlab": None,
    "ylab": None,
    "title": None,
    "subtitle": None,
    "caption": None,
    "palette": None,
    "point_args": None,
    "boxplot_args": None,
    "violin_args": None,
    "ax": None,
}


def _defaults(fn: Callable[..., object]) -> dict[str, object]:
    return {
        p.name: p.default
        for p in inspect.signature(fn).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }


def test_analysis_signature() -> None:
    params = list(inspect.signature(ip.analyze.withinstats).parameters.values())
    kind = inspect.Parameter
    assert [p.name for p in params[:3]] == ["data", "x", "y"]
    assert all(p.kind is kind.POSITIONAL_OR_KEYWORD for p in params[:3])
    subject = params[3]
    assert (subject.name, subject.kind, subject.default) == (
        "subject_id",
        kind.KEYWORD_ONLY,
        kind.empty,
    )
    assert _defaults(ip.analyze.withinstats) == {
        "subject_id": kind.empty,
        **ANALYSIS_DEFAULTS,
    }
    # The same analysis arguments, in the same order, as betweenstats.
    between = list(_defaults(ip.analyze.betweenstats))
    assert list(_defaults(ip.analyze.withinstats))[1:] == between
    assert ip.analyze.withinstats.__doc__


def test_drawing_and_top_level_signatures() -> None:
    params = list(inspect.signature(ip.render.withinstats).parameters)
    assert params[0] == "result"
    assert _defaults(ip.render.withinstats) == DRAWING_DEFAULTS
    # betweenstats' drawing keywords plus the two paths.
    assert set(DRAWING_DEFAULTS) == set(_defaults(ip.render.betweenstats)) | {
        "point_path",
        "centrality_path",
    }
    assert _defaults(ip.withinstats) == {
        "subject_id": inspect.Parameter.empty,
        **ANALYSIS_DEFAULTS,
        **DRAWING_DEFAULTS,
    }
    assert list(inspect.signature(ip.withinstats).parameters)[:3] == ["data", "x", "y"]
    analysis = set(inspect.signature(ip.analyze.withinstats).parameters)
    drawing = set(params) - {"result"}
    assert drawing - analysis <= v.DRAWING_KEYWORDS["withinstats"]
    assert analysis - drawing <= v.ANALYSIS_KEYWORDS


# --------------------------------------------------------------------------- #
# subject_id is required
# --------------------------------------------------------------------------- #


def test_missing_subject_id_explains_pairing(two: pd.DataFrame) -> None:
    for fn in (analyze, top):
        assert message(TypeError, lambda fn=fn: fn(two, "group", "extra")) == (
            REQUIRED_MESSAGE
        )
        text = message(
            TypeError, lambda fn=fn: fn(two, "group", "extra", subject_id=None)
        )
        assert text == "subject_id=None is not a column name: " + REQUIRED_MESSAGE
    # A misspelled keyword is reported before the missing subject_id.
    text = message(TypeError, lambda: analyze(two, "group", "extra", conf_levle=0.9))
    assert text.endswith("Did you mean conf_level=0.9?")


def test_subject_spellings_point_to_subject_id(two: pd.DataFrame) -> None:
    for fn in (analyze, top):
        for name in ("subject", "id", "ID", "Subject"):
            bad = {name: "ID"}
            text = message(
                TypeError, lambda fn=fn, bad=bad: fn(two, "group", "extra", **bad)
            )
            assert text == (
                f"{name}= is not an argument of withinstats(); use subject_id='ID', "
                "the column that identifies each subject."
            )
        for name in ("subject.id", "subject_ID", "subjectId"):
            bad = {name: "ID"}
            text = message(
                TypeError, lambda fn=fn, bad=bad: fn(two, "group", "extra", **bad)
            )
            assert text == (
                f"withinstats() has no argument {name!r}: inferplot spells every "
                "argument in lowercase snake_case. Write subject_id='ID'."
            )
        text = message(
            TypeError, lambda fn=fn: fn(two, "group", "extra", subjectid="ID")
        )
        assert text.endswith("Did you mean subject_id='ID'?")


def test_paired_is_not_needed(two: pd.DataFrame, four: pd.DataFrame) -> None:
    expected = (
        "paired= is not needed: withinstats() always pairs the measurements of each "
        "subject, matched by subject_id. Remove paired=."
    )
    for fn in (analyze, top):
        assert (
            message(
                TypeError,
                lambda fn=fn: fn(two, "group", "extra", subject_id="ID", paired=True),
            )
            == expected
        )
    result = _fake_result(four)
    assert message(TypeError, lambda: draw(result, paired=True)) == expected


def test_extra_positional_argument_hints_subject_id(two: pd.DataFrame) -> None:
    text = message(TypeError, lambda: analyze(two, "group", "extra", "ID"))
    assert text == (
        "withinstats() takes 3 positional arguments (data, x, y) but 4 were given. "
        "Everything after data, x, y is keyword-only; name each extra argument. "
        "Extra positional values: 'ID'. If 'ID' is the column that identifies each "
        "subject, write withinstats(..., subject_id='ID')."
    )
    # Without subject_id= the subject column comes first, whatever the value: a
    # column may be called 's' or 'np' (STATUS U9).
    two = two.assign(s=two["ID"])
    text = message(TypeError, lambda: analyze(two, "group", "extra", "s"))
    assert text.endswith(
        "Extra positional values: 's'. If 's' is the column that identifies each "
        "subject, write withinstats(..., subject_id='s'); if it is "
        "pairwise_display, write withinstats(..., pairwise_display='s')."
    )
    text = message(TypeError, lambda: analyze(two, "group", "extra", "np"))
    assert text.endswith(
        "If 'np' is the column that identifies each subject, write "
        "withinstats(..., subject_id='np'); if it is type, write "
        "withinstats(..., type='np')."
    )
    text = message(TypeError, lambda: analyze(two, "group", "extra", two["ID"]))
    assert text.endswith(
        "Extra positional values: a pandas.Series. withinstats() takes the column "
        "that identifies each subject by its name: write "
        "withinstats(..., subject_id='ID')."
    )
    text = message(TypeError, lambda: analyze(two, "group", "extra", 3))
    assert text.endswith(
        "withinstats() takes the column that identifies each subject by its name: "
        "write withinstats(..., subject_id='subject')."
    )
    # With subject_id= given, the other hints are as before.
    text = message(
        TypeError, lambda: analyze(two, "group", "extra", "np", subject_id="ID")
    )
    assert text.endswith("For example withinstats(..., type='np').")


# --------------------------------------------------------------------------- #
# other keyword names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("keyword", "error", "phrase"),
    [
        ("var.equal", TypeError, "removed in ggstatsplot 1.0.0"),
        ("nboot", TypeError, "2,000 seeded"),
        ("effsize.type", TypeError, "always used"),
        ("k", TypeError, "renamed digits"),
        ("pairwise.comparisons", TypeError, "pairwise_display='none'"),
        ("output", TypeError, "ip.analyze.withinstats"),
        ("messages", TypeError, "never prints"),
        ("hue", TypeError, "seaborn"),
        ("ci", TypeError, "conf_level"),
        ("grouping.var", NotImplementedError, "v0.2"),
        ("centrality.type", TypeError, "always follows type"),
        ("ggsignif.args", TypeError, "brackets are not styled"),
        ("centrality.path.args", TypeError, "centrality_path=False leaves it out"),
        ("centrality_path_args", TypeError, "centrality_path=False leaves it out"),
        ("point.path.args", TypeError, "point_path=False leaves them out"),
    ],
)
def test_removed_and_borrowed_keywords(
    two: pd.DataFrame, keyword: str, error: type[Exception], phrase: str
) -> None:
    text = message(error, lambda: call(two, **{keyword: True}))
    assert phrase in text
    assert keyword in text
    assert "Did you mean" not in text


def test_dotted_drawing_keywords(two: pd.DataFrame) -> None:
    # Spelled as in ggwithinstats, given to the drawing-capable top-level function.
    for name in ("point.path", "centrality.path"):
        text = message(
            TypeError,
            lambda name=name: top(
                two, "group", "extra", subject_id="ID", **{name: False}
            ),
        )
        assert text.endswith(f"Write {name.replace('.', '_')}=False.")


def test_drawing_keywords_given_to_the_analysis(two: pd.DataFrame) -> None:
    for name in (
        "point_path",
        "centrality_path",
        "ax",
        "layers",
        "centrality_plotting",
    ):
        text = message(TypeError, lambda name=name: call(two, **{name: None}))
        assert text == (
            f"{name}= is a drawing option of ip.withinstats(...) and "
            "ip.render.withinstats(result, ...); ip.analyze.withinstats() only "
            f"computes the statistics. Pass {name}= when drawing, or remove it."
        )
    text = message(TypeError, lambda: call(two, **{"plot.type": "box"}))
    assert "ip.render.withinstats(result, layers=('box', 'points'))" in text


# --------------------------------------------------------------------------- #
# value validators run before the data are read
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("keyword", "value", "error", "phrase"),
    [
        ("conf_level", 1.5, ValueError, "strictly between 0 and 1"),
        ("digits", -1, ValueError, ">= 0"),
        ("alternative", "two.sided", ValueError, "two-sided"),
        ("p_adjust_method", "Holm", ValueError, "holm"),
        ("pairwise_display", None, TypeError, "pairwise_display='none'"),
        ("pairwise_alpha", 0.0, ValueError, "strictly between 0 and 1"),
        ("bf_prior", 0.0, ValueError, "greater than 0"),
        ("bf_message", "yes", TypeError, "None, True or False"),
        ("results_subtitle", None, TypeError, "True or False"),
        ("seed", -1, ValueError, ">= 0"),
        ("tr", 0.5, ValueError, "0 <= tr < 0.5"),
        ("type", "anova", ValueError, "use one of"),
        ("order", "12", TypeError, "list of level names"),
        ("order", ["1"], ValueError, "missing from order"),
    ],
)
def test_value_validators(
    two: pd.DataFrame,
    keyword: str,
    value: object,
    error: type[Exception],
    phrase: str,
) -> None:
    assert phrase in message(error, lambda: call(two, **{keyword: value}))


def test_arguments_a_type_cannot_use(two: pd.DataFrame, four: pd.DataFrame) -> None:
    text = message(ValueError, lambda: call(two, tr=0.1))
    assert text.startswith("tr=0.1 sets the trimming of the robust methods")
    for kind in ("robust", "bayes"):
        text = message(
            ValueError, lambda kind=kind: call(two, type=kind, alternative="less")
        )
        assert text.startswith(
            "alternative='less' applies to type='parametric' and "
            f"type='nonparametric' only: with type={kind!r}"
        )
    text = message(ValueError, lambda: call(two, type="np", bf_message=True))
    assert "draws only for type='parametric'" in text
    text = message(
        ValueError, lambda: call(two, bf_message=True, results_subtitle=False)
    )
    assert text.startswith("bf_message=True asks for the Bayes factor caption")
    # Rules that depend on the number of conditions, checked after reading.
    bugs = {"x": "condition", "y": "desire", "subject_id": "subject"}
    text = message(ValueError, lambda: analyze(four, alternative="greater", **bugs))
    assert text == (
        "alternative='greater' applies only to two conditions; x='condition' has 4, "
        "where the omnibus test and the pairwise comparisons are always two-sided. "
        "Remove alternative= or compare two conditions."
    )
    text = message(NotImplementedError, lambda: analyze(four, bf_message=True, **bugs))
    assert "arrives in inferplot 0.2" in text


# --------------------------------------------------------------------------- #
# the analysis: data are read, then NotImplementedError per type
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "k", "method", "later"),
    [
        (
            "robust",
            2,
            "Yuen's test for dependent trimmed means",
            "is not implemented yet",
        ),
        ("nonparametric", 4, "the Friedman test", "is not implemented yet"),
        (
            "robust",
            4,
            "the repeated-measures trimmed-means ANOVA",
            "is not implemented yet",
        ),
    ],
)
def test_each_type_raises_not_implemented(
    two: pd.DataFrame, four: pd.DataFrame, kind: str, k: int, method: str, later: str
) -> None:
    frame, x, y, subject_id, n = (
        (two, "group", "extra", "ID", 10)
        if k == 2
        else (four, "condition", "desire", "subject", 88)
    )
    for fn in (analyze, top):  # the top-level function analyzes before drawing
        text = message(
            NotImplementedError,
            lambda fn=fn: fn(frame, x, y, subject_id=subject_id, type=kind),
        )
        assert text == (
            f"withinstats(type={kind!r}) with {k} conditions ({method}) {later}"
            f"{' in this version of inferplot' if later.startswith('is') else ''}. "
            f"The data were read: {n} subjects with a value at each of the {k} levels "
            f"of x={x!r}."
        )


def test_type_aliases_reach_the_same_type(two: pd.DataFrame) -> None:
    for alias, kind in (("p", "parametric"), ("bf", "bayes"), ("np", "nonparametric")):
        assert call(two, type=alias).type == kind
    for alias, kind in (("r", "robust"),):
        text = message(NotImplementedError, lambda alias=alias: call(two, type=alias))
        assert text.startswith(f"withinstats(type={kind!r})")


def test_data_errors_come_before_not_implemented(two: pd.DataFrame) -> None:
    doubled = pd.concat([two, two.iloc[:1]], ignore_index=True)
    text = message(ValueError, lambda: call(doubled))
    assert "subject '1' at '1' (row positions 0, 20)" in text
    with pytest.raises(KeyError, match="Did you mean 'ID'"):
        analyze(two, "group", "extra", subject_id="id")
    with pytest.raises(TypeError, match="subject_id must be a column name"):
        analyze(two, "group", "extra", subject_id=3)


def test_any_backend(two: pd.DataFrame) -> None:
    assert call(pl.from_pandas(two)).n_obs == 10


# --------------------------------------------------------------------------- #
# the drawing function
# --------------------------------------------------------------------------- #


def _fake_result(frame: pd.DataFrame) -> StatsResult:
    """A within result built from made-up parts (no statistics exist yet)."""
    data = read_within(frame, "condition", "desire", "subject")
    args = _args()
    parts = module._Parts(  # pyright: ignore[reportPrivateUsage]
        test=None,
        effect_size=None,
        bayes=None,
        pairwise=(),
        centrality="mean",
        centers=tuple(float(c.mean()) for c in data.by_level()),
        warnings=("A note from the type's analysis.",),
        citations=("A citation.",),
    )
    return module._assemble(data, args, parts, digits=2)  # pyright: ignore[reportPrivateUsage]


def _args() -> Any:
    return module._Args(  # pyright: ignore[reportPrivateUsage]
        x="condition",
        subject_id="subject",
        type="parametric",
        conf_level=0.95,
        alternative="two-sided",
        p_adjust_method="holm",
        pairwise_alpha=0.05,
        pairwise_display="significant",
        prior_scale=0.707,
        bf_message=None,
        results_subtitle=False,
        tr=0.2,
        seed=0,
    )


def test_render_checks_then_draws(four: pd.DataFrame) -> None:
    result = _fake_result(four)
    fig = draw(result)
    assert fig.result.analysis == "withinstats"
    assert fig.result.warnings == result.warnings
    assert "point_path must be True or False" in message(
        TypeError, lambda: draw(result, point_path=1)
    )
    assert "centrality_path must be True or False" in message(
        TypeError, lambda: draw(result, centrality_path="no")
    )
    assert "not a layer" in message(ValueError, lambda: draw(result, layers=("dots",)))
    between = ip.analyze.betweenstats(read_csv("PlantGrowth.csv"), "group", "weight")
    assert message(TypeError, lambda: draw(between)) == (
        "render.withinstats() draws results of ip.analyze.withinstats(); this result "
        "comes from ip.analyze.betweenstats(). Draw it with "
        "ip.render.betweenstats(result)."
    )
    assert message(TypeError, lambda: ip.render.betweenstats(result)).startswith(
        "render.betweenstats() draws results of ip.analyze.betweenstats(); this "
        "result comes from ip.analyze.withinstats()."
    )


def test_render_keyword_messages(four: pd.DataFrame) -> None:
    result = _fake_result(four)
    text = message(TypeError, lambda: draw(four, "condition", "desire"))
    assert text.startswith(
        "render.withinstats() draws the result of ip.analyze.withinstats(data, x, y)"
    )
    for name in ("subject_id", "subject", "id", "subject.id"):
        text = message(TypeError, lambda name=name: draw(result, **{name: "subject"}))
        assert text == (
            f"{name}= is not an argument of render.withinstats(): the subject column "
            "is an argument of the analysis, subject_id=, and the result already "
            "holds it (result.subject_id). Pass subject_id= to "
            "ip.analyze.withinstats(data, x, y, subject_id=...) or "
            f"ip.withinstats(...) and draw that result. Remove {name}=."
        )
    assert message(TypeError, lambda: draw(result, pointpath=True)).endswith(
        "Did you mean point_path=True?"
    )


# --------------------------------------------------------------------------- #
# the result of a within design
# --------------------------------------------------------------------------- #


def test_result_contract(four: pd.DataFrame) -> None:
    result = _fake_result(four)
    data = read_within(four, "condition", "desire", "subject")
    assert (result.analysis, result.design, result.type) == (
        "withinstats",
        "within",
        "parametric",
    )
    assert (result.x, result.y, result.subject_id) == ("condition", "desire", "subject")
    assert result.levels == ("HDHF", "HDLF", "LDHF", "LDLF")
    assert result.subjects == data.subjects
    # values[level][i] belongs to subjects[i]: the matrix's columns.
    assert len(result.values) == 4
    assert all(len(level) == 88 for level in result.values)
    assert tuple(zip(*result.values, strict=True)) == tuple(
        map(tuple, data.values.tolist())
    )
    assert [d.n for d in result.descriptives] == [88] * 4
    assert (result.n_obs, result.n_dropped, result.dropped_subjects) == (88, 5, 5)
    assert result.warnings == (
        "5 subjects were left out of the analysis and the plot because they lack a "
        "value of 'desire' at some level of 'condition'; 88 complete subjects remain.",
        "A note from the type's analysis.",
    )
    assert result.text.apa == "No test was computed (results_subtitle=False), n = 88."


def test_result_json_round_trip_and_embedded_copy(four: pd.DataFrame) -> None:
    result = _fake_result(four)
    assert StatsResult.from_json(result.to_json()) == result
    assert StatsResult.from_dict(result.to_dict()) == result
    embedded = json.loads(result.to_json(include_values=False))
    assert embedded["values"] == []
    assert embedded["subjects"] == []  # subject labels are the user's data (D36)
    for label in ("'2'", "'10'", "'42'"):  # dropped subjects are never named
        assert all(label not in w for w in embedded["warnings"])
    back = StatsResult.from_json(result.to_json(include_values=False))
    assert (back.values, back.subjects, back.subject_id) == ((), (), "subject")
    assert back.n_obs == 88


def test_result_frames_and_check(four: pd.DataFrame) -> None:
    result = _fake_result(four)
    assert list(result.to_frame().columns)[:2] == ["test", "label"]
    assert result.to_frame("polars").height == 0  # no test computed
    issues = {issue.code: issue for issue in result.check()}
    assert set(issues) == {"rows_dropped"}
    assert issues["rows_dropped"].message == (
        "5 rows were left out because 'condition', 'desire' or 'subject' is missing; "
        "5 subjects were left out because they lack a value at some level of "
        "'condition'."
    )


def test_small_n_names_subjects_once() -> None:
    frame = pd.DataFrame(
        {
            "subject": [1, 1, 2, 2, 3, 3],
            "condition": list("ababab"),
            "desire": [1.0, 2.0, 3.0, 5.0, 4.0, 4.5],
        }
    )
    result = _fake_result(frame)
    issues = {issue.code: issue for issue in result.check()}
    assert set(issues) == {"small_n"}
    assert issues["small_n"].message == (
        "Only 3 subjects have a value at every level of 'condition', fewer than 5. "
        "Tests, intervals and plotted distributions are unreliable with so few "
        "subjects."
    )
    assert math.isclose(result.descriptives[1].value or 0.0, 23 / 6)
    assert result.subjects == ("1", "2", "3")
