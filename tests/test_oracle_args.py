"""The R-to-inferplot argument table translates fixtures and refuses unknowns."""

import pytest

from tests.oracle_args import FUNCTIONS, KEYWORDS, POSITIONAL, translate


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
    with pytest.raises(KeyError, match="ggpiestats"):
        translate("ggpiestats", {"x": "a"})


def test_keywords_are_r_names_with_dots_as_underscores() -> None:
    for r_name, py_name in KEYWORDS.items():
        assert py_name == r_name.replace(".", "_")


def test_every_function_has_positional_columns() -> None:
    assert set(POSITIONAL) == set(FUNCTIONS)
