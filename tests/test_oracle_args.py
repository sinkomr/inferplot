"""The R-to-inferplot argument table translates fixtures and refuses unknowns."""

import pytest

from tests.oracle_args import (
    FUNCTIONS,
    KEYWORDS,
    POSITIONAL,
    apply_label_filter,
    split,
    translate,
)


def test_translate_betweenstats() -> None:
    call = translate(
        "ggbetweenstats",
        {
            "x": "supp",
            "y": "len",
            "type": "np",
            "conf.level": 0.9,
            "alternative": "two.sided",
            "bf.message": False,
            "results.subtitle": True,
        },
    )
    assert call.function == "betweenstats"
    assert call.args == ("supp", "len")
    assert call.kwargs == {
        "type": "np",
        "conf_level": 0.9,
        "alternative": "two-sided",
        "bf_message": False,
        "results_subtitle": True,
    }


def test_translate_histostats_has_one_positional_column() -> None:
    call = translate("gghistostats", {"x": "len", "test.value": 10})
    assert call.args == ("len",)
    assert call.kwargs == {"test_value": 10}


def test_vector_arguments_pass_through() -> None:
    # jsonlite writes an R vector such as ratio = c(0.5, 0.5) as a JSON array.
    call = translate("ggbarstats", {"x": "a", "ratio": [0.5, 0.5]})
    assert call.kwargs == {"ratio": [0.5, 0.5]}
    # A vector of a respelled argument is respelled element by element.
    call = translate(
        "ggbetweenstats", {"x": "a", "y": "b", "alternative": ["two.sided", "less"]}
    )
    assert call.kwargs == {"alternative": ["two-sided", "less"]}


def test_one_sided_alternatives_keep_their_names() -> None:
    for side in ("less", "greater"):
        call = translate("ggbetweenstats", {"x": "a", "y": "b", "alternative": side})
        assert call.kwargs == {"alternative": side}


def test_unknown_argument_fails_loudly() -> None:
    with pytest.raises(KeyError, match=r"var\.equal"):
        translate("ggbetweenstats", {"x": "a", "y": "b", "var.equal": True})


def test_unknown_function_fails_loudly() -> None:
    with pytest.raises(KeyError, match="ggdotplotstats"):
        translate("ggdotplotstats", {"x": "a", "y": "b"})


def test_piestats_is_barstats() -> None:
    # The oracle records one-way cases from ggpiestats too; they are barstats calls.
    call = translate("ggpiestats", {"x": "cyl", "type": "parametric"})
    assert call.function == "barstats"
    assert call.args == ("cyl",)


def test_keywords_are_r_names_with_dots_as_underscores() -> None:
    for r_name, py_name in KEYWORDS.items():
        assert py_name == r_name.replace(".", "_")


def test_every_function_has_positional_columns() -> None:
    assert set(POSITIONAL) == set(FUNCTIONS)


def test_split_separates_analysis_and_drawing() -> None:
    call = split(
        "gghistostats",
        {"x": "len", "type": "parametric", "test.value": 18, "binwidth": 2},
    )
    assert (call.function, call.args) == ("histostats", ("len",))
    assert call.analysis == {"type": "parametric", "test_value": 18}
    assert call.drawing == {"binwidth": 2}
    call = split(
        "ggscatterstats",
        {"x": "wt", "y": "mpg", "type": "p", "marginal": False, "conf.level": 0.9},
    )
    assert call.analysis == {"type": "p", "conf_level": 0.9}
    assert call.drawing == {"marginal": False}
    call = split("ggwithinstats", {"x": "a", "y": "b", "point.path": False})
    assert call.drawing == {"point_path": False}


def test_split_handles_the_arguments_without_a_keyword() -> None:
    import pandas as pd

    call = split(
        "ggscatterstats",
        {
            "x": "wt",
            "y": "mpg",
            "label.var": "model",
            "label.expression": "mpg > 25 | wt > 5",
            "smooth.line.args": {"method": "loess", "formula": "y ~ x"},
        },
    )
    assert call.analysis == {"label_var": "model"}
    assert call.label_filter == "mpg > 25 | wt > 5"
    assert call.unsupported == {
        "smooth.line.args": {"method": "loess", "formula": "y ~ x"}
    }
    frame = pd.DataFrame(
        {"model": ["a", "b", "c"], "mpg": [30.0, 20.0, 10.0], "wt": [2.0, 3.0, 6.0]}
    )
    labelled = apply_label_filter(frame, "model", call.label_filter)
    assert labelled["model"].tolist()[0] == "a"
    assert pd.isna(labelled["model"][1])
    assert labelled["model"].tolist()[2] == "c"
    # translate() itself still refuses them, so a call cannot drop them silently.
    with pytest.raises(KeyError, match=r"label\.expression"):
        translate("ggscatterstats", {"x": "a", "y": "b", "label.expression": "a"})


def test_every_bar_fixture_splits_into_barstats_keywords() -> None:
    import inspect

    import inferplot as ip
    from tests.oracle_fixtures import cases_for

    analysis = set(inspect.signature(ip.analyze.barstats).parameters)
    drawing = set(inspect.signature(ip.render.barstats).parameters)
    cases = [
        *cases_for("ggbarstats"),
        *cases_for("ggpiestats"),
        *cases_for("ggbarstats", failed=True),
    ]
    assert len(cases) == 90
    for case in cases:
        call = split(case.function, case.args)
        assert call.function == "barstats"
        assert call.args == tuple(
            case.args[name] for name in ("x", "y") if name in case.args
        )
        assert set(call.analysis) <= analysis, case.name
        assert set(call.drawing) <= drawing, case.name
        assert not call.drawing  # every recorded argument is the analysis'
    call = split(
        "ggbarstats",
        {"x": "cyl", "y": "am", "label": "both", "digits.perc": 1, "counts": "n"},
    )
    assert call.analysis == {"label": "both", "digits_perc": 1, "counts": "n"}
