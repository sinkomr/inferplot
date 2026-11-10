"""Argument checks (``inferplot._validate``, SPEC.md section 2 "API errors").

The decorator is tested on a stand-in with exactly ``betweenstats``' signature from
the Phase 1 contract, so every message is checked against the real argument list.
"""

from __future__ import annotations

import difflib
import fractions
import inspect
import math
import types
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any, assert_type, cast

import numpy as np
import pandas as pd
import pytest

from inferplot import _validate as v
from inferplot._validate import checked


def betweenstats(
    data: object,
    x: str,
    y: str,
    *,
    type: str = "parametric",
    pairwise_display: str = "significant",
    p_adjust_method: str = "holm",
    pairwise_alpha: float = 0.05,
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    alternative: str = "two-sided",
    seed: int = 0,
    order: Sequence[str] | None = None,
    centrality_plotting: bool = True,
    layers: Sequence[str] = ("violin", "box", "points"),
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    palette: Sequence[str] | str | None = None,
    point_args: dict[str, Any] | None = None,
    boxplot_args: dict[str, Any] | None = None,
    violin_args: dict[str, Any] | None = None,
    ax: object = None,
) -> dict[str, object]:
    """Stand-in for ``inferplot.betweenstats``: returns the arguments it received."""
    return dict(locals())


STAND_IN = checked(betweenstats)
KEYWORDS = tuple(inspect.signature(betweenstats).parameters)
DEFAULTS = {
    name: p.default
    for name, p in inspect.signature(betweenstats).parameters.items()
    if p.default is not inspect.Parameter.empty
}


def call(**kwargs: Any) -> dict[str, object]:
    return STAND_IN("frame", "group", "score", **kwargs)


def message(error: type[Exception], thunk: Callable[[], object]) -> str:
    """The message of the ``error`` that ``thunk()`` raises (exactly that type)."""
    with pytest.raises(error) as info:
        thunk()
    assert type(info.value) is error
    return str(info.value)


# --- the decorator keeps the function intact ---------------------------------------


def test_signature_and_metadata_preserved() -> None:
    assert inspect.signature(STAND_IN) == inspect.signature(betweenstats)
    assert STAND_IN.__name__ == "betweenstats"
    assert STAND_IN.__doc__ == betweenstats.__doc__
    assert STAND_IN.__wrapped__ is betweenstats  # type: ignore[attr-defined]
    assert STAND_IN.__module__ == betweenstats.__module__


def test_static_type_is_the_wrapped_signature() -> None:
    # pyright checks these lines: the ParamSpec keeps parameters and return type.
    wrapped: Callable[..., dict[str, object]] = STAND_IN
    assert_type(STAND_IN("frame", "g", "y", conf_level=0.9), dict[str, object])
    assert wrapped is STAND_IN


def test_valid_calls_pass_every_argument_through() -> None:
    assert call() == {"data": "frame", "x": "group", "y": "score", **DEFAULTS}
    out = STAND_IN(data="frame", x="g", y="s", type="np", seed=3, ax=None)
    assert (out["data"], out["x"], out["y"], out["type"], out["seed"]) == (
        "frame",
        "g",
        "s",
        "np",
        3,
    )
    everything = {k: f"value-{k}" for k in KEYWORDS[3:]}
    assert call(**everything) == {
        "data": "frame",
        "x": "group",
        "y": "score",
        **everything,
    }


def test_python_errors_for_missing_and_duplicate_arguments_remain() -> None:
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'y'"):
        STAND_IN("frame", "group")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="multiple values for argument 'x'"):
        STAND_IN("frame", "group", "score", x="other")  # type: ignore[misc]


def test_decorator_requires_an_explicit_signature() -> None:
    def star_args(data: object, *args: object) -> None:
        del data, args

    def star_kwargs(data: object, **kwargs: object) -> None:
        del data, kwargs

    with pytest.raises(
        TypeError, match=r"explicit signature; star_args\(\) takes \*args"
    ):
        checked(star_args)
    with pytest.raises(TypeError, match=r"star_kwargs\(\) takes \*\*kwargs"):
        checked(star_kwargs)


def test_other_positional_counts() -> None:
    @checked
    def histostats(data: object, x: str, *, test_value: float = 0.0) -> float:
        del data, x
        return test_value

    with pytest.raises(TypeError) as info:
        histostats("frame", "x", 1.5)  # type: ignore[misc]
    assert str(info.value) == (
        "histostats() takes 2 positional arguments (data, x) but 3 were given. "
        "Everything after data, x is keyword-only; name each extra argument. "
        "Extra positional values: 1.5."
    )

    @checked
    def one(data: object) -> None:
        del data

    with pytest.raises(TypeError, match=r"takes 1 positional argument \(data\) but 2"):
        one("a", "b")  # type: ignore[call-arg]


# --- SPEC section 2: targeted messages ---------------------------------------------

# (spellings, exception, phrases the message must contain). Every row of SPEC.md
# section 2's table, in dotted and snake_case spelling.
TARGETED: list[tuple[tuple[str, ...], type[Exception], tuple[str, ...]]] = [
    (
        ("var.equal", "var_equal"),
        TypeError,
        ("removed in ggstatsplot 1.0.0", "Welch's test is always used"),
    ),
    (
        ("nboot",),
        TypeError,
        ("removed in ggstatsplot 1.0.0", "2,000 seeded bootstrap resamples", "seed="),
    ),
    (
        ("effsize.type", "effsize_type"),
        TypeError,
        (
            "removed in ggstatsplot 1.0.0",
            "Hedges' g and partial omega squared are always used",
        ),
    ),
    (("k",), TypeError, ("renamed digits in ggstatsplot 0.12.3", "digits=")),
    (("package",), TypeError, ("removed in ggstatsplot 1.0.0", "pass palette=")),
    (
        ("plot.type", "plot_type"),
        TypeError,
        ("removed in ggstatsplot 0.12.0", "use layers=", "('box', 'points')"),
    ),
    (
        ("pairwise.comparisons", "pairwise_comparisons"),
        TypeError,
        ("removed in ggstatsplot 0.12.0", "pairwise_display='none'"),
    ),
    (
        ("output",),
        TypeError,
        ("removed in ggstatsplot 0.10.0", "fig.result", "ip.analyze.betweenstats"),
    ),
    (("messages",), TypeError, ("inferplot never prints", "fig.result.warnings")),
    (
        ("sampling.plan", "sampling_plan"),
        TypeError,
        ("removed from ggbarstats in ggstatsplot 1.0.0",),
    ),
    (
        ("fixed.margin", "fixed_margin"),
        TypeError,
        ("removed from ggbarstats in ggstatsplot 1.0.0",),
    ),
    (
        ("prior.concentration", "prior_concentration"),
        TypeError,
        ("removed from ggbarstats in ggstatsplot 1.0.0",),
    ),
    (
        ("grouping.var", "grouping_var"),
        NotImplementedError,
        ("arrives in inferplot v0.2", "once per subset"),
    ),
    (
        ("hue",),
        TypeError,
        ("seaborn", "no second grouping", "grouping_var", "arrives in inferplot v0.2"),
    ),
    (
        ("ci", "errorbar"),
        TypeError,
        ("seaborn", "always conf_level confidence intervals", "conf_level="),
    ),
    # betweenstats has no subject_id: point to withinstats instead of subject_id=.
    (
        ("subject", "id", "subject.id", "subject_id"),
        TypeError,
        ("compares independent groups", "withinstats(data, x, y, subject_id=...)"),
    ),
]
TARGETED_CASES = [
    pytest.param(name, error, phrases, id=name)
    for names, error, phrases in TARGETED
    for name in names
]


@pytest.mark.parametrize(("name", "error", "phrases"), TARGETED_CASES)
def test_targeted_messages(
    name: str, error: type[Exception], phrases: tuple[str, ...]
) -> None:
    with pytest.raises(error) as info:
        call(**{name: 3})
    assert type(info.value) is error
    message = str(info.value)
    for phrase in phrases:
        assert phrase in message, (phrase, message)
    assert name in message  # names what was passed


def test_targeted_exact_wording() -> None:
    with pytest.raises(TypeError) as info:
        call(**{"var.equal": True})
    assert str(info.value) == (
        "var.equal= was removed in ggstatsplot 1.0.0; Welch's test is always used "
        "(it does not assume equal variances), so there is nothing to set. "
        "Remove var.equal=."
    )
    with pytest.raises(TypeError) as info:
        call(k=3)
    assert str(info.value) == (
        "k= was renamed digits in ggstatsplot 0.12.3; pass digits=3 instead."
    )
    with pytest.raises(NotImplementedError) as info:
        call(grouping_var="sex")
    assert str(info.value) == (
        "grouping_var= (ggstatsplot's grouped_* functions, one plot per value of a "
        "column) arrives in inferplot v0.2. For now call betweenstats() once per "
        "subset of the data."
    )


def test_subject_names_point_to_subject_id_where_it_exists() -> None:
    @checked
    def withinstats(
        data: object, x: str, y: str, *, subject_id: str | None = None
    ) -> str | None:
        del data, x, y
        return subject_id

    for name in ("subject", "id"):
        with pytest.raises(TypeError) as info:
            withinstats("frame", "x", "y", **{name: "person"})
        assert str(info.value) == (
            f"{name}= is not an argument of withinstats(); use subject_id='person', "
            "the column that identifies each subject."
        )
    with pytest.raises(TypeError, match=r"Write subject_id='person'"):
        withinstats("frame", "x", "y", **{"subject.id": "person"})
    assert withinstats("frame", "x", "y", subject_id="person") == "person"


def test_a_required_subject_id_is_never_left_to_python() -> None:
    @checked
    def withinstats(data: object, x: str, y: str, *, subject_id: str) -> str:
        del data, x, y
        return subject_id

    loose = cast("Callable[..., object]", withinstats)
    assert message(TypeError, lambda: loose("frame", "x", "y")) == str(
        v.subject_id_required("withinstats")
    )
    assert "never by row order" in str(v.subject_id_required("withinstats"))
    assert withinstats("frame", "x", "y", subject_id="s") == "s"
    # Keyword problems come first; another required keyword keeps Python's message.
    with pytest.raises(TypeError, match="use subject_id='s'"):
        loose("frame", "x", "y", subject="s")

    @checked
    def other(data: object, *, column: str) -> str:
        del data
        return column

    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        cast("Callable[..., object]", other)("frame")


def test_withinstats_names_do_not_change_betweenstats_messages() -> None:
    # paired= and the path-styling arguments get withinstats' messages only there.
    for name in ("point.path.args", "centrality_path_args"):
        text = message(TypeError, lambda name=name: call(**{name: 1}))
        assert text.startswith(
            f"betweenstats() got an unexpected keyword argument {name!r}."
        )
    # betweenstats' paired= points to withinstats instead (STATUS U3).
    text = message(TypeError, lambda: call(paired=True))
    assert text.startswith("betweenstats() compares independent groups")
    assert "not needed" not in text


def test_layout_messages_adapt_to_the_function() -> None:
    # A drawing function (it takes ax=) without layers= or pairwise comparisons.
    @checked
    def histostats(
        data: object, x: str, *, conf_level: float = 0.95, ax: object = None
    ) -> None:
        del data, x, conf_level, ax

    with pytest.raises(TypeError, match=r"histostats\(\) draws a fixed set of layers"):
        histostats("frame", "x", plot_type="box")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match=r"histostats\(\) has no pairwise comparisons"):
        histostats("frame", "x", pairwise_comparisons=False)  # type: ignore[call-arg]


def _analysis(
    data: object,
    x: str,
    y: str,
    *,
    type: str = "parametric",
    conf_level: float = 0.95,
    seed: int = 0,
) -> None:
    """Stand-in for ``ip.analyze.betweenstats``: analysis arguments only, no ax."""
    del data, x, y, type, conf_level, seed


ANALYSIS = checked(_analysis)


def analysis_call(**kwargs: Any) -> None:
    ANALYSIS("frame", "g", "y", **kwargs)


def test_drawing_options_on_an_analysis_function_point_to_the_drawing_functions() -> (
    None
):
    for name in ("plot.type", "plot_type"):
        bad = {name: "box"}
        text = message(TypeError, lambda bad=bad: analysis_call(**bad))
        assert text == (
            f"{name}= was removed in ggstatsplot 0.12.0; ip.analyze._analysis() only "
            "computes the statistics. What is drawn is chosen when drawing, with "
            "ip._analysis(...) or ip.render._analysis(result, ...). "
            f"Remove {name}=."
        )
    text = message(TypeError, lambda: analysis_call(package="x"))
    assert text == (
        "package= was removed in ggstatsplot 1.0.0; colors are a drawing option: "
        "pass palette= (a list of colors or a matplotlib colormap name) to "
        "ip._analysis(...) or ip.render._analysis(result, ...); "
        "ip.analyze._analysis() only computes the statistics. Remove package=."
    )


def test_real_analysis_function_points_to_the_drawing_functions() -> None:
    from inferplot import analyze

    frame = pd.DataFrame({"g": list("aabbcc") * 2, "y": range(12)})
    analysis = cast("Callable[..., object]", analyze.betweenstats)
    for name in ("plot.type", "plot_type"):
        bad = {name: "box"}
        text = message(TypeError, lambda bad=bad: analysis(frame, "g", "y", **bad))
        assert text == (
            f"{name}= was removed in ggstatsplot 0.12.0; ip.analyze.betweenstats() "
            "only computes the statistics; choose the layers when drawing with "
            "layers=, a tuple of 'violin', 'box' and 'points': "
            "ip.betweenstats(..., layers=('box', 'points')) or "
            "ip.render.betweenstats(result, layers=('box', 'points')). "
            f"Remove {name}=."
        )
    text = message(TypeError, lambda: analysis(frame, "g", "y", package="x"))
    assert "palette= (a list of colors" in text
    assert "ip.betweenstats(...) or ip.render.betweenstats(result, ...)" in text


def test_a_drawing_function_given_data_says_it_draws_a_result() -> None:
    from inferplot.result import StatsResult

    @checked
    def betweenstats(result: StatsResult, *, ax: object = None) -> StatsResult:
        del ax
        return result

    draw = cast("Callable[..., object]", betweenstats)
    frame = pd.DataFrame({"g": list("aabbcc") * 2, "y": range(12)})
    expected = (
        "render.betweenstats() draws the result of "
        "ip.analyze.betweenstats(data, x, y); got a pandas.DataFrame. Compute it "
        "first with result = ip.analyze.betweenstats(data, x, y, ...) and pass "
        "result, or use ip.betweenstats(data, x, y, ...) to analyze and draw in one "
        "call."
    )
    for thunk in (
        lambda: draw(frame, "g", "y"),
        lambda: draw(frame),
        lambda: draw(result=frame),
        lambda: draw(frame, ax=None),
    ):
        assert message(TypeError, thunk) == expected
    text = message(TypeError, lambda: draw({"a": 1}))
    assert "got {'a': 1}" in text
    # Python's own message for a missing result is left alone.
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        draw()


def test_real_render_function_given_data_says_it_draws_a_result() -> None:
    from inferplot import render

    draw = cast("Callable[..., object]", render.betweenstats)
    frame = pd.DataFrame({"g": list("aabbcc") * 2, "y": range(12)})
    text = message(TypeError, lambda: draw(frame, "g", "y"))
    assert text.startswith(
        "render.betweenstats() draws the result of ip.analyze.betweenstats(data, x, y)"
    )
    assert "ip.betweenstats(data, x, y, ...)" in text
    assert "keyword-only" not in text


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"g": list("aabbcc") * 2, "y": range(12)})


def test_real_render_function_given_data_by_keyword_says_it_draws_a_result() -> None:
    # Data-style keywords are reported as data given to a drawing function, never
    # as a misspelled drawing keyword (x= is not ax=).
    from inferplot import render

    draw = cast("Callable[..., object]", render.betweenstats)
    frame = _frame()
    expected = (
        "render.betweenstats() draws the result of "
        "ip.analyze.betweenstats(data, x, y); got a pandas.DataFrame. Compute it "
        "first with result = ip.analyze.betweenstats(data, x, y, ...) and pass "
        "result, or use ip.betweenstats(data, x, y, ...) to analyze and draw in one "
        "call."
    )
    for thunk in (
        lambda: draw(frame, x="g", y="y"),
        lambda: draw(data=frame, x="g", y="y"),
        lambda: draw(frame, y="y"),
        lambda: draw(frame, "g", "y", pairwise_display="all"),
        lambda: draw(frame, pointargs={}),
    ):
        assert message(TypeError, thunk) == expected
    text = message(TypeError, lambda: draw(x="g", y="y"))
    assert text.startswith(
        "render.betweenstats() draws the result of ip.analyze.betweenstats(data, x, "
        "y); got x='g', y='y' and no result."
    )


def test_analysis_keywords_given_to_a_drawing_function_point_to_the_analysis() -> None:
    from inferplot import analyze, render

    draw = cast("Callable[..., object]", render.betweenstats)
    result = analyze.betweenstats(_frame(), "g", "y")
    text = message(TypeError, lambda: draw(result, x="g"))
    assert text == (
        "x= is an argument of the analysis, ip.analyze.betweenstats(data, x, y, "
        "...); render.betweenstats() draws a result that already holds it. Pass x= "
        "to ip.analyze.betweenstats() or ip.betweenstats() and draw that result. "
        "Remove x=."
    )
    for name in ("pairwise_display", "conf_level", "seed", "data", "order"):
        bad = {name: 1}
        text = message(TypeError, lambda bad=bad: draw(result, **bad))
        assert text.startswith(f"{name}= is an argument of the analysis"), text
    # A real typo of a drawing keyword still gets its did-you-mean.
    text = message(TypeError, lambda: draw(result, pointargs={}))
    assert text.endswith("Did you mean point_args={}?")


def test_drawing_keywords_given_to_an_analysis_function_point_to_the_drawing() -> None:
    from inferplot import analyze

    analysis = cast("Callable[..., object]", analyze.betweenstats)
    frame = _frame()
    text = message(TypeError, lambda: analysis(frame, "g", "y", ax=None))
    assert text == (
        "ax= is a drawing option of ip.betweenstats(...) and "
        "ip.render.betweenstats(result, ...); ip.analyze.betweenstats() only "
        "computes the statistics. Pass ax= when drawing, or remove it."
    )
    for name in (
        "palette",
        "layers",
        "title",
        "subtitle",
        "caption",
        "xlab",
        "ylab",
        "point_args",
        "boxplot_args",
        "violin_args",
        "centrality_plotting",
    ):
        bad = {name: None}
        text = message(TypeError, lambda bad=bad: analysis(frame, "g", "y", **bad))
        assert text.startswith(f"{name}= is a drawing option of ip.betweenstats("), text


def test_drawing_and_analysis_keyword_sets_cover_the_real_signatures() -> None:
    import inferplot as ip

    def names(fn: Callable[..., object]) -> set[str]:
        return set(inspect.signature(fn).parameters)

    analysis = names(ip.analyze.betweenstats)
    drawing = names(ip.render.betweenstats) - {"result"}
    assert names(ip.betweenstats) == analysis | drawing
    assert drawing - analysis <= v.DRAWING_KEYWORDS["betweenstats"]
    assert analysis - drawing <= v.ANALYSIS_KEYWORDS


def test_did_you_mean_never_suggests_an_argument_the_call_fills() -> None:
    # x is given positionally, so ax= must not become "Did you mean x=None?".
    text = message(TypeError, lambda: analysis_call(ax=None))
    assert "x=None" not in text
    assert "drawing option" in text
    # Nor a keyword given in the same call.
    loose = cast("Callable[..., object]", ANALYSIS)
    text = message(TypeError, lambda: loose("frame", "g", "y", seeds=1, seed=2))
    assert "Did you mean" not in text
    # A snake_case spelling of an argument the call already gives says so.
    text = message(TypeError, lambda: loose("frame", "g", "y", X="g"))
    assert text == (
        "_analysis() got an unexpected keyword argument 'X', and x is already given. "
        "Remove X=."
    )


@pytest.mark.parametrize(
    ("name", "phrase"),
    [
        ("centrality.type", "always follows type"),
        ("centrality.point.args", "centrality point is not styled"),
        ("centrality.label.args", "centrality label is not styled"),
        ("ggsignif.args", "brackets are not styled"),
        ("ggtheme", "no ggplot2 themes"),
        ("ggplot.component", "no ggplot2 components"),
    ],
)
def test_other_ggstatsplot_arguments_get_no_misleading_suggestion(
    name: str, phrase: str
) -> None:
    for spelling in (name, name.replace(".", "_")):
        with pytest.raises(TypeError) as info:
            call(**{spelling: None})
        message = str(info.value)
        assert message.startswith(
            f"{spelling}= is a ggstatsplot argument that inferplot does not have: "
        )
        assert phrase in message
        assert "Did you mean" not in message


# --- R spellings, typos and unknown names ------------------------------------------

SNAKE = [k for k in KEYWORDS if "_" in k]


@pytest.mark.parametrize("name", SNAKE)
def test_dotted_spelling_of_a_valid_keyword(name: str) -> None:
    dotted = name.replace("_", ".")
    with pytest.raises(TypeError) as info:
        call(**{dotted: 0.9})
    assert str(info.value) == (
        f"betweenstats() has no argument {dotted!r}: inferplot spells every argument "
        f"in lowercase snake_case. Write {name}=0.9."
    )


@pytest.mark.parametrize(
    ("spelling", "fix"),
    [
        ("confLevel", "conf_level"),
        ("pAdjustMethod", "p_adjust_method"),
        ("Type", "type"),
        ("bf-prior", "bf_prior"),
        ("results.Subtitle", "results_subtitle"),
    ],
)
def test_other_spellings_of_a_valid_keyword(spelling: str, fix: str) -> None:
    with pytest.raises(TypeError, match=rf"Write {fix}='v'\.$"):
        call(**{spelling: "v"})


@pytest.mark.parametrize(
    ("typo", "fix"),
    [
        ("pairwise_dispay", "pairwise_display"),
        ("p_adjust", "p_adjust_method"),
        ("conf_lvl", "conf_level"),
        ("seeds", "seed"),
        ("layer", "layers"),
        ("boxplot_arg", "boxplot_args"),
        ("Pairwise_Displays", "pairwise_display"),
    ],
)
def test_did_you_mean(typo: str, fix: str) -> None:
    with pytest.raises(TypeError) as info:
        call(**{typo: "all"})
    assert str(info.value) == (
        f"betweenstats() got an unexpected keyword argument {typo!r}. "
        f"Did you mean {fix}='all'?"
    )


def test_did_you_mean_needs_a_likely_match() -> None:
    # difflib's ratio alone is not enough: 'ordqq' scores 0.6 against 'order' (the
    # old cutoff) but is two edits from a five-letter name, so nothing is
    # suggested; one edit is.
    assert difflib.SequenceMatcher(None, "ordqq", "order").ratio() == 0.6
    with pytest.raises(TypeError) as info:
        call(ordqq=1)
    assert "Did you mean" not in str(info.value)
    with pytest.raises(TypeError, match=r"Did you mean order=1\?"):
        call(ordr=1)


def test_unknown_keyword_without_a_match_lists_the_arguments() -> None:
    with pytest.raises(TypeError) as info:
        call(frobnicate=1)
    assert str(info.value) == (
        "betweenstats() got an unexpected keyword argument 'frobnicate'. Its "
        "arguments are: " + ", ".join(KEYWORDS) + "."
    )


def test_first_bad_keyword_in_call_order_is_reported() -> None:
    with pytest.raises(TypeError, match="'frobnicate'"):
        call(frobnicate=1, nboot=100)
    with pytest.raises(TypeError, match="nboot"):
        call(nboot=100, frobnicate=1)
    with pytest.raises(NotImplementedError):
        call(grouping_var="g", frobnicate=1)


def test_long_values_are_named_by_type() -> None:
    frame = pd.DataFrame({"a": range(100)})
    with pytest.raises(TypeError) as info:
        # (data is given positionally, so dataa= would not suggest it.)
        call(palete=frame)
    assert "palette=a pandas.DataFrame?" in str(info.value)
    assert "\n" not in str(info.value)


# --- positional arguments ----------------------------------------------------------


def test_too_many_positional_arguments() -> None:
    with pytest.raises(TypeError) as info:
        STAND_IN("frame", "group", "score", "np", "all")  # type: ignore[misc]
    assert str(info.value) == (
        "betweenstats() takes 3 positional arguments (data, x, y) but 5 were given. "
        "Everything after data, x, y is keyword-only; name each extra argument. "
        "Extra positional values: 'np', 'all'. For example "
        "betweenstats(..., type='np')."
    )


@pytest.mark.parametrize(
    ("extra", "hint"),
    [
        ("bayes", " For example betweenstats(..., type='bayes')."),
        ("ns", " For example betweenstats(..., pairwise_display='ns')."),
        (0.9, ""),
        ("holm", ""),
    ],
)
def test_positional_hint(extra: object, hint: str) -> None:
    with pytest.raises(TypeError) as info:
        STAND_IN("frame", "group", "score", extra)  # type: ignore[misc]
    assert str(info.value).endswith(f"Extra positional values: {extra!r}.{hint}")


def test_positional_data_frame_named_by_type() -> None:
    frame = pd.DataFrame({"a": range(100)})
    with pytest.raises(TypeError) as info:
        STAND_IN(frame, "group", "score", frame)  # type: ignore[misc]
    assert str(info.value).endswith("Extra positional values: a pandas.DataFrame.")


# --- type, pairwise_display, p_adjust_method, alternative --------------------------


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("parametric", "parametric"),
        ("p", "parametric"),
        ("nonparametric", "nonparametric"),
        ("np", "nonparametric"),
        ("robust", "robust"),
        ("r", "robust"),
        ("bayes", "bayes"),
        ("bf", "bayes"),
        (np.str_("np"), "nonparametric"),
    ],
)
def test_canonical_type(value: str, canonical: str) -> None:
    out = v.canonical_type(value)
    assert out == canonical
    assert type(out) is str


TYPE_ALLOWED = (
    "use one of 'parametric' (alias 'p'), 'nonparametric' (alias 'np'), "
    "'robust' (alias 'r'), 'bayes' (alias 'bf')."
)


@pytest.mark.parametrize(
    ("value", "hint"),
    [
        ("Parametric", " Did you mean type='parametric'?"),
        ("non-parametric", " Did you mean type='nonparametric'?"),
        ("bayesian", " Did you mean type='bayes'?"),
        ("NP", " Did you mean type='np'?"),
        ("frequentist", ""),
        ("", ""),
    ],
)
def test_type_rejected(value: str, hint: str) -> None:
    text = message(ValueError, lambda: v.canonical_type(value))
    assert text == f"type={value!r} is not allowed; {TYPE_ALLOWED}{hint}"


@pytest.mark.parametrize("value", [None, 1, True, ["p"]])
def test_type_not_a_string(value: object) -> None:
    with pytest.raises(TypeError, match=r"type must be a string, one of 'parametric'"):
        v.canonical_type(value)


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("significant", "significant"),
        ("s", "significant"),
        ("non-significant", "non-significant"),
        ("ns", "non-significant"),
        ("all", "all"),
        ("none", "none"),
    ],
)
def test_canonical_pairwise_display(value: str, canonical: str) -> None:
    assert v.canonical_pairwise_display(value) == canonical


def test_pairwise_display_rejected() -> None:
    allowed = "'significant' (alias 's'), 'non-significant' (alias 'ns'), 'all', 'none'"
    text = message(ValueError, lambda: v.canonical_pairwise_display("nonsignificant"))
    assert text == (
        f"pairwise_display='nonsignificant' is not allowed; use one of {allowed}. "
        "Did you mean pairwise_display='non-significant'?"
    )
    with pytest.raises(TypeError) as info:
        v.canonical_pairwise_display(None)
    assert str(info.value) == (
        f"pairwise_display must be a string, one of {allowed}; got None. To turn it "
        "off write pairwise_display='none'."
    )
    with pytest.raises(TypeError, match="got False"):
        v.canonical_pairwise_display(False)


@pytest.mark.parametrize("method", v.P_ADJUST_METHODS)
def test_p_adjust_methods(method: str) -> None:
    assert v.check_p_adjust_method(method) == method


def test_p_adjust_method_names_are_rs_exactly() -> None:
    assert v.P_ADJUST_METHODS == (
        "holm",
        "hochberg",
        "hommel",
        "bonferroni",
        "BH",
        "BY",
        "fdr",
        "none",
    )
    with pytest.raises(ValueError, match="Did you mean p_adjust_method='bonferroni'"):
        v.check_p_adjust_method("bonferoni")
    with pytest.raises(TypeError, match="got None"):
        v.check_p_adjust_method(None)


@pytest.mark.parametrize(
    ("wrong", "right"), [("bh", "BH"), ("Holm", "holm"), ("by", "BY"), ("bonf", None)]
)
def test_p_adjust_method_is_case_sensitive(wrong: str, right: str | None) -> None:
    text = message(ValueError, lambda: v.check_p_adjust_method(wrong))
    assert text.startswith(
        f"p_adjust_method={wrong!r} is not allowed; use one of 'holm', "
        "'hochberg', 'hommel', 'bonferroni', 'BH', 'BY', 'fdr', 'none'."
    )
    if right is not None:
        assert text.endswith(f"Did you mean p_adjust_method={right!r}?")
    else:
        assert "Did you mean" not in text


@pytest.mark.parametrize("value", ["two-sided", "less", "greater"])
def test_alternative(value: str) -> None:
    assert v.check_alternative(value) == value


def test_alternative_rejected() -> None:
    allowed = "'two-sided', 'less', 'greater'"
    text = message(ValueError, lambda: v.check_alternative("two.sided"))
    assert text == (
        f"alternative='two.sided' is not allowed; use one of {allowed}. R spells it "
        "'two.sided'; inferplot writes alternative='two-sided'."
    )
    with pytest.raises(ValueError, match="Did you mean alternative='two-sided'"):
        v.check_alternative("two_sided")
    with pytest.raises(ValueError, match="Did you mean alternative='greater'"):
        v.check_alternative("greater ")
    with pytest.raises(TypeError, match="alternative must be a string"):
        v.check_alternative(None)


# --- numbers -----------------------------------------------------------------------

NUMERIC_KINDS = [0.5, np.float64(0.5), np.float32(0.5), fractions.Fraction(1, 2)]


@pytest.mark.parametrize("value", NUMERIC_KINDS)
def test_real_numbers_become_plain_floats(value: object) -> None:
    for check in (
        v.check_conf_level,
        lambda x: v.check_probability(x, "pairwise_alpha"),
        lambda x: v.check_positive(x, "bf_prior"),
    ):
        out = check(value)
        assert out == 0.5
        assert type(out) is float


@pytest.mark.parametrize("kind", [np.float32, np.float16])
@pytest.mark.parametrize("value", [0.9, 0.1, 0.01, 0.2, 0.707, 0.95, 0.3])
def test_narrow_floats_are_read_as_they_print(kind: type, value: float) -> None:
    # np.float32(0.9) is 0.89999997615814208984375; it prints as 0.9, which is
    # what was meant, so it is read through its shortest repr (STATUS U12).
    narrow = kind(value)
    assert float(narrow) != value  # the binary value differs
    for check in (
        v.check_conf_level,
        lambda x: v.check_probability(x, "pairwise_alpha"),
        lambda x: v.check_positive(x, "bf_prior"),
        v.check_tr if value < 0.5 else v.check_conf_level,
    ):
        out = check(narrow)
        assert out == value
        assert type(out) is float


@pytest.mark.parametrize(
    "check",
    [
        v.check_conf_level,
        lambda x: v.check_probability(x, "pairwise_alpha"),
        lambda x: v.check_positive(x, "bf_prior"),
        v.check_tr,
    ],
    ids=["conf_level", "probability", "positive", "tr"],
)
@pytest.mark.parametrize(
    "value", [True, False, np.True_, "0.5", None, Decimal("0.5"), [0.5], 0.5j]
)
def test_numbers_reject_non_numbers(
    check: Callable[[object], float], value: object
) -> None:
    with pytest.raises(TypeError, match="must be a number, for example"):
        check(value)


@pytest.mark.parametrize(
    "value",
    [0.95, 0.5, 1e-300, 5e-324, math.nextafter(1.0, 0.0), np.float64(0.9)],
)
def test_conf_level_accepted(value: float) -> None:
    assert v.check_conf_level(value) == float(value)


@pytest.mark.parametrize(
    "value", [0, 1, 0.0, 1.0, -0.0, -0.5, 1.5, math.nan, math.inf, -math.inf]
)
def test_conf_level_bounds(value: float) -> None:
    text = message(ValueError, lambda: v.check_conf_level(value))
    assert text.startswith(
        "conf_level must be strictly between 0 and 1, for example conf_level=0.95; "
        f"got {value!r}."
    )


def test_conf_level_percentage_hint() -> None:
    text = message(ValueError, lambda: v.check_conf_level(95))
    assert text == (
        "conf_level must be strictly between 0 and 1, for example conf_level=0.95; "
        "got 95. It is a proportion, not a percentage: write conf_level=0.95."
    )
    with pytest.raises(ValueError, match=r"write conf_level=0\.9\."):
        v.check_conf_level(90.0)
    text = message(ValueError, lambda: v.check_conf_level(100))
    assert "percentage" not in text


@pytest.mark.parametrize("value", [0.05, 1e-10, math.nextafter(1.0, 0.0), 5e-324])
def test_probability_accepted(value: float) -> None:
    assert v.check_probability(value, "pairwise_alpha") == value


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, -0.01, 1.01, math.nan, math.inf])
def test_probability_bounds(value: float) -> None:
    text = message(ValueError, lambda: v.check_probability(value, "pairwise_alpha"))
    assert text == (
        "pairwise_alpha must be strictly between 0 and 1, for example "
        f"pairwise_alpha=0.05; got {value!r}."
    )


@pytest.mark.parametrize("value", [0.707, 1, 1e-300, 5e-324, 1e300, np.float64(2.0)])
def test_positive_accepted(value: float) -> None:
    assert v.check_positive(value, "bf_prior") == float(value)


@pytest.mark.parametrize("value", [0, 0.0, -0.0, -0.707, math.nan, math.inf, -math.inf])
def test_positive_bounds(value: float) -> None:
    text = message(ValueError, lambda: v.check_positive(value, "bf_prior"))
    assert text == (
        "bf_prior must be a finite number greater than 0, for example "
        f"bf_prior=0.707; got {value!r}."
    )


def test_positive_example_follows_the_argument() -> None:
    text = message(ValueError, lambda: v.check_positive(0, "dpi", example="300"))
    assert text == (
        "dpi must be a finite number greater than 0, for example dpi=300; got 0."
    )
    text = message(TypeError, lambda: v.check_positive("300", "dpi", example="300"))
    assert text == "dpi must be a number, for example dpi=300; got '300'."
    # The default example is bf_prior's.
    text = message(TypeError, lambda: v.check_positive("x", "bf_prior"))
    assert text == "bf_prior must be a number, for example bf_prior=0.707; got 'x'."


@pytest.mark.parametrize("value", [0, 0.0, 0.2, 0.1, math.nextafter(0.5, 0.0)])
def test_tr_accepted(value: float) -> None:
    out = v.check_tr(value)
    assert out == value
    assert type(out) is float


@pytest.mark.parametrize("value", [0.5, 0.6, 1, -1e-300, -0.1, math.nan, math.inf])
def test_tr_bounds(value: float) -> None:
    text = message(ValueError, lambda: v.check_tr(value))
    assert text == (
        "tr is the proportion trimmed from each end and must satisfy 0 <= tr < 0.5, "
        f"for example tr=0.2; got {value!r}."
    )


# --- integers ----------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 2, 15, np.int64(3), np.uint8(4)])
def test_digits_accepted(value: int) -> None:
    out = v.check_digits(value)
    assert out == int(value)
    assert type(out) is int


def test_digits_rejected() -> None:
    text = message(ValueError, lambda: v.check_digits(-1))
    assert text == (
        "digits is a number of decimal places and must be >= 0, for example "
        "digits=2; got -1."
    )
    for big in (16, 100, 10**6):
        text = message(ValueError, lambda big=big: v.check_digits(big))
        assert text == (
            "digits is a number of decimal places and must be at most 15, the most "
            "a double-precision number holds meaningfully, for example digits=2; "
            f"got {big}."
        )
    with pytest.raises(TypeError) as info:
        v.check_digits(2.0)
    assert str(info.value) == (
        "digits must be a whole number (an int), for example digits=2; got 2.0 "
        "(write digits=2)."
    )
    for bad in (True, False, np.True_, 2.5, math.nan, math.inf, "2", None, Decimal(2)):
        with pytest.raises(TypeError) as info:
            v.check_digits(bad)
        assert str(info.value).startswith("digits must be a whole number (an int)")
        assert "(write" not in str(info.value)


@pytest.mark.parametrize("value", [0, 1, 42, 2**70, np.int64(7), np.uint64(2**63)])
def test_seed_accepted(value: int) -> None:
    out = v.check_seed(value)
    assert out == int(value)
    assert type(out) is int


def test_seed_rejected() -> None:
    text = message(ValueError, lambda: v.check_seed(-1))
    assert text == ("seed must be a whole number >= 0, for example seed=0; got -1.")
    for bad in (True, False, np.False_, 1.5, math.nan, "0", None):
        with pytest.raises(TypeError, match=r"seed must be a whole number \(an int\)"):
            v.check_seed(bad)
    with pytest.raises(TypeError, match=r"\(write seed=3\)"):
        v.check_seed(np.float64(3.0))


# --- booleans and text -------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False, np.True_, np.False_])
def test_bool_accepted(value: bool) -> None:
    out = v.check_bool(value, "results_subtitle")
    assert out is bool(value)
    assert v.check_optional_bool(value, "bf_message") is bool(value)


@pytest.mark.parametrize("value", [0, 1, 1.0, "True", "false", [True]])
def test_bool_rejected(value: object) -> None:
    with pytest.raises(TypeError) as info:
        v.check_bool(value, "results_subtitle")
    assert str(info.value) == (
        f"results_subtitle must be True or False; got {value!r}."
    )
    with pytest.raises(TypeError) as info:
        v.check_optional_bool(value, "bf_message")
    assert str(info.value) == (
        f"bf_message must be None, True or False; got {value!r}."
    )


def test_bool_none() -> None:
    assert v.check_optional_bool(None, "bf_message") is None
    with pytest.raises(TypeError, match="must be True or False; got None"):
        v.check_bool(None, "results_subtitle")


def test_optional_text() -> None:
    assert v.check_optional_text(None, "title") is None
    assert v.check_optional_text("", "title") == ""
    out = v.check_optional_text(np.str_("Score"), "xlab")
    assert out == "Score"
    assert type(out) is str
    for bad in (1, 1.5, True, b"title", ["a"]):
        with pytest.raises(TypeError) as info:
            v.check_optional_text(bad, "title")
        assert str(info.value) == (
            f"title must be a string or None; got {bad!r}. Convert it with str() "
            "first if you mean its text."
        )


# --- layers and artist arguments ---------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        ("violin", "box", "points"),
        ["points"],
        ("box", "violin"),
        (np.str_("box"),),
    ],
)
def test_layers_accepted(value: Sequence[str]) -> None:
    out = v.check_layers(value)
    assert out == tuple(value)
    assert type(out) is tuple
    assert all(type(item) is str for item in out)


def test_layers_rejected() -> None:
    allowed = "'violin', 'box', 'points'"
    with pytest.raises(TypeError) as info:
        v.check_layers("box")
    assert str(info.value) == (
        f"layers must be a tuple of layer names from {allowed}, for example "
        "layers=('box',); got 'box'."
    )
    for bad in ({"box"}, None, 3, (x for x in ["box"])):
        with pytest.raises(TypeError, match=r"for example layers=\('box', 'points'\)"):
            v.check_layers(bad)
    text = message(ValueError, lambda: v.check_layers(()))
    assert text == (
        f"layers must name at least one of {allowed}; got an empty sequence."
    )
    text = message(ValueError, lambda: v.check_layers(("box", "box")))
    assert text == "layers names 'box' more than once."
    text = message(ValueError, lambda: v.check_layers(("violin", "boxplot")))
    assert text == (
        f"layers contains 'boxplot', which is not a layer; use names from {allowed}. "
        "Did you mean 'box'?"
    )
    with pytest.raises(ValueError, match="Did you mean 'points'"):
        v.check_layers(["point"])
    with pytest.raises(ValueError, match="contains 'jitter', which is not a layer"):
        v.check_layers(["jitter"])
    with pytest.raises(ValueError, match="contains 1, which is not a layer"):
        v.check_layers([1])


def test_artist_args() -> None:
    assert v.check_artist_args(None, "point_args") is None
    given = {"alpha": 0.5, "color": "k"}
    out = v.check_artist_args(given, "point_args")
    assert out == given
    assert out is not given
    assert type(out) is dict
    proxy = types.MappingProxyType({"linewidth": 2})
    assert v.check_artist_args(proxy, "boxplot_args") == {"linewidth": 2}
    assert v.check_artist_args({}, "violin_args") == {}


def test_artist_args_rejected() -> None:
    for bad in ([("alpha", 0.5)], "alpha=0.5", 0.5, True):
        with pytest.raises(TypeError) as info:
            v.check_artist_args(bad, "point_args")
        assert str(info.value) == (
            "point_args must be a dict of matplotlib properties, for example "
            f"point_args={{'alpha': 0.5}}; got {bad!r}."
        )
    with pytest.raises(TypeError) as info:
        v.check_artist_args({1: "a"}, "violin_args")
    assert str(info.value) == (
        "violin_args keys must be matplotlib property names (strings); got 1."
    )


# --- hints that must not lead to another valid analysis (STATUS U3) ----------------
#
# The user is an AI agent that follows a hint literally, so a did-you-mean is given
# only when the suggestion is very likely what was meant; otherwise the valid names
# are listed without one. Common names of things inferplot spells differently get
# a message of their own.


def _public_callables() -> dict[str, Callable[..., object]]:
    """Every public function that checks keywords, each called with valid
    positional arguments, so only the keyword under test is wrong."""
    import matplotlib

    matplotlib.use("Agg")
    import inferplot as ip

    frame = pd.DataFrame(
        {"g": list("ab") * 6, "y": np.arange(12.0) % 5, "s": np.repeat(range(6), 2)}
    )
    between = ip.analyze.betweenstats(frame, "g", "y")
    loose = cast("Any", ip)
    return {
        "ip.betweenstats": lambda **kw: loose.betweenstats(frame, "g", "y", **kw),
        "ip.analyze.betweenstats": lambda **kw: loose.analyze.betweenstats(
            frame, "g", "y", **kw
        ),
        "ip.render.betweenstats": lambda **kw: loose.render.betweenstats(between, **kw),
        "ip.withinstats": lambda **kw: loose.withinstats(
            frame, "g", "y", subject_id="s", **kw
        ),
        "ip.analyze.withinstats": lambda **kw: loose.analyze.withinstats(
            frame, "g", "y", subject_id="s", **kw
        ),
        "ip.render.withinstats": lambda **kw: loose.render.withinstats(
            _within_result(), **kw
        ),
    }


def _within_result() -> Any:
    """A within result built from made-up parts (no within statistics exist yet)."""
    from tests.test_withinstats_api import _fake_result  # pyright: ignore[reportPrivateUsage]

    frame = pd.DataFrame(
        {
            "subject": np.repeat(range(6), 2),
            "condition": list("ab") * 6,
            "desire": np.arange(12.0) % 5,
        }
    )
    return _fake_result(frame)


PUBLIC = _public_callables()

# (keyword, value, the suggestion it must not get)
WRONG_SUGGESTIONS = [
    ("tail", "greater", "title="),
    ("tails", 1, "tr="),
    ("alternative_hypothesis", "greater", "Did you mean"),
    ("one_sided", True, "Did you mean"),
    ("levels", ["b", "a"], "conf_level="),
    ("categories", ["b", "a"], "Did you mean"),
    ("x_order", ["b", "a"], "Did you mean"),
    ("legend", False, "seed="),
    ("bayes", True, "layers="),
    ("label", "p", "Did you mean ylab="),
    ("labels", "p", "Did you mean"),
    ("centrality", "median", "centrality_plotting="),
    ("alpha", 0.01, "Did you mean"),
    ("hue_order", ["a"], "order="),
]


@pytest.mark.parametrize("where", list(PUBLIC))
@pytest.mark.parametrize(("keyword", "value", "wrong"), WRONG_SUGGESTIONS)
def test_unlikely_keyword_suggestions_are_not_made(
    where: str, keyword: str, value: object, wrong: str
) -> None:
    with pytest.raises(TypeError) as info:
        PUBLIC[where](**{keyword: value})
    text = str(info.value)
    assert wrong not in text, text
    assert "Did you mean" not in text, text
    assert keyword in text


@pytest.mark.parametrize(
    "where", ["ip.betweenstats", "ip.analyze.betweenstats", "ip.render.betweenstats"]
)
@pytest.mark.parametrize("keyword", ["point_path", "centrality_path"])
def test_path_keywords_in_betweenstats_belong_to_withinstats(
    where: str, keyword: str
) -> None:
    text = message(TypeError, lambda: PUBLIC[where](**{keyword: False}))
    assert text == (
        f"{keyword}= is an argument of withinstats(), which joins each subject's "
        "points (point_path) or the centrality points (centrality_path) across "
        "conditions; betweenstats() compares independent groups and draws no "
        f"paths. Remove {keyword}=."
    )


def test_targeted_keyword_messages_exact() -> None:
    text = message(TypeError, lambda: call(tail="greater"))
    assert text == (
        "tail= is not an argument of betweenstats(); the direction of the test is "
        "set by alternative=: 'two-sided' (the default), 'greater' (the first "
        "level of x is greater than the second) or 'less' (it is less). The first "
        "level is the first in order= when given, else the first declared "
        "category, else the first in sorted order. Remove tail=."
    )
    for name in ("tails", "alternative_hypothesis", "one_sided", "one.sided"):
        text = message(TypeError, lambda name=name: call(**{name: True}))
        assert text.startswith(
            f"{name}= is not an argument of betweenstats(); the direction of the test "
            "is set by alternative=:"
        )
    text = message(TypeError, lambda: call(alpha=0.01))
    assert text == (
        "alpha= is not an argument of betweenstats(): pairwise_alpha= sets the "
        "threshold below which a pair's adjusted p-value counts as significant, "
        "and conf_level= sets the level of the confidence intervals (conf_level=0.95 "
        "for 95%). For transparency, use point_args={'alpha': 0.5}, boxplot_args or "
        "violin_args. Remove alpha= and pass the one you mean."
    )
    text = message(TypeError, lambda: call(paired=True))
    assert text == (
        "betweenstats() compares independent groups and takes no paired=. For "
        "repeated measures of the same subjects use "
        "withinstats(data, x, y, subject_id=...); otherwise remove paired=."
    )
    for name in ("levels", "categories", "x_order"):
        text = message(TypeError, lambda name=name: call(**{name: ["b", "a"]}))
        assert text == (
            f"{name}= is not an argument of betweenstats(); the order of the levels "
            "of x is set by order=, a list of every level present, for example "
            f"order=['b', 'a']. Remove {name}=."
        )
    text = message(TypeError, lambda: call(levels=3))
    assert "for example order=['control', 'treatment']" in text
    text = message(TypeError, lambda: call(legend=False))
    assert text == (
        "legend= is not an argument of betweenstats(): there is no legend; the "
        "colors follow the levels of x, which the x axis names. Remove legend=."
    )
    text = message(TypeError, lambda: call(bayes=True))
    assert text == (
        "bayes= is not an argument of betweenstats(); for Bayesian statistics pass "
        "type='bayes', and for the Bayes factor caption under a parametric "
        "subtitle pass bf_message=True or False. Remove bayes=."
    )
    for name in ("label", "labels"):
        text = message(TypeError, lambda name=name: call(**{name: "p"}))
        assert text == (
            f"{name}= is not an argument of betweenstats(); the labels are set by "
            "xlab= (the x axis), ylab= (the y axis) and title=. "
            f"Remove {name}=."
        )


def test_targeted_keyword_messages_point_to_the_right_function() -> None:
    text = message(TypeError, lambda: PUBLIC["ip.render.betweenstats"](tail="less"))
    assert text.startswith("tail= is not an argument of render.betweenstats(); ")
    assert (
        "alternative= is an argument of the analysis: pass it to "
        "ip.analyze.betweenstats() or ip.betweenstats() and draw that result. "
        "Remove tail=." in text
    )
    text = message(TypeError, lambda: PUBLIC["ip.analyze.betweenstats"](label="p"))
    assert text.startswith("label= is not an argument of betweenstats(); ")
    assert (
        "xlab= is a drawing option of ip.betweenstats(...) and "
        "ip.render.betweenstats(result, ...). Remove label=." in text
    )
    text = message(TypeError, lambda: PUBLIC["ip.render.betweenstats"](alpha=0.5))
    assert text == (
        "alpha= is not an argument of render.betweenstats(). For transparency, use "
        "point_args={'alpha': 0.5}, boxplot_args or violin_args. pairwise_alpha= "
        "and conf_level= are arguments of the analysis: pass them to "
        "ip.analyze.betweenstats() or ip.betweenstats() and draw that result. "
        "Remove alpha=."
    )
    # The drawing function names itself and points to both within functions;
    # paired= is no argument of any betweenstats or withinstats form.
    text = message(TypeError, lambda: PUBLIC["ip.render.betweenstats"](paired=True))
    assert text == (
        "render.betweenstats() draws independent groups and takes no paired=. For "
        "repeated measures of the same subjects use "
        "withinstats(data, x, y, subject_id=...), or ip.analyze.withinstats() and "
        "ip.render.withinstats(); otherwise remove paired=."
    )
    assert "argument of the analysis" not in text
    text = message(TypeError, lambda: PUBLIC["ip.render.withinstats"](paired=True))
    assert text == (
        "paired= is not needed: render.withinstats() draws a result whose "
        "measurements are already paired by subject_id. Remove paired=."
    )
    text = message(TypeError, lambda: PUBLIC["ip.analyze.withinstats"](alpha=0.5))
    assert "For transparency" not in text  # the analysis draws nothing
    assert "pairwise_alpha= sets the threshold" in text
    # withinstats keeps its own paired= message (SPEC.md section 2).
    text = message(TypeError, lambda: PUBLIC["ip.withinstats"](paired=True))
    assert text == (
        "paired= is not needed: withinstats() always pairs the measurements of each "
        "subject, matched by subject_id. Remove paired=."
    )


# Suggestions that are very likely what was meant stay.
LIKELY = [
    ("conf_levle", "conf_level"),
    ("pairwise_dispay", "pairwise_display"),
    ("p_adjust", "p_adjust_method"),
    ("conf_lvl", "conf_level"),
    ("seeds", "seed"),
    ("layer", "layers"),
    ("boxplot_arg", "boxplot_args"),
    ("Pairwise_Displays", "pairwise_display"),
    ("ndigits", "digits"),
    ("trim", "tr"),
    ("prior", "bf_prior"),
    ("ci_level", "conf_level"),
    ("level", "conf_level"),
    ("pointargs", "point_args"),
    ("ylabel", "ylab"),
    ("palete", "palette"),
]


@pytest.mark.parametrize(("typo", "fix"), LIKELY)
def test_likely_keyword_suggestions_stay(typo: str, fix: str) -> None:
    text = message(TypeError, lambda: call(**{typo: 1}))
    assert text == (
        f"betweenstats() got an unexpected keyword argument {typo!r}. "
        f"Did you mean {fix}=1?"
    )


@pytest.mark.parametrize(
    ("value", "right"),
    [
        ("benjamini-hochberg", "BH"),
        ("Benjamini-Hochberg", "BH"),
        ("fdr_bh", "BH"),
        ("fdr-bh", "BH"),
        ("b-h", "BH"),
        ("benjamini-yekutieli", "BY"),
        ("fdr_by", "BY"),
        ("holm-bonferroni", "holm"),
    ],
)
def test_adjustment_names_map_to_rs_spelling(value: str, right: str) -> None:
    text = message(ValueError, lambda: v.check_p_adjust_method(value))
    assert text.startswith(f"p_adjust_method={value!r} is not allowed; use one of ")
    assert text.endswith(f"write p_adjust_method={right!r}.")
    assert "Did you mean" not in text


def test_adjustment_names_exact_wording() -> None:
    allowed = "'holm', 'hochberg', 'hommel', 'bonferroni', 'BH', 'BY', 'fdr', 'none'"
    start = f"is not allowed; use one of {allowed}."
    assert message(ValueError, lambda: v.check_p_adjust_method("fdr_bh")) == (
        f"p_adjust_method='fdr_bh' {start} The Benjamini-Hochberg procedure (the "
        "false discovery rate) is R's 'BH': write p_adjust_method='BH'."
    )
    assert message(ValueError, lambda: v.check_p_adjust_method("fdr_by")) == (
        f"p_adjust_method='fdr_by' {start} The Benjamini-Yekutieli procedure is R's "
        "'BY': write p_adjust_method='BY'."
    )
    assert message(ValueError, lambda: v.check_p_adjust_method("holm-bonferroni")) == (
        f"p_adjust_method='holm-bonferroni' {start} The Holm-Bonferroni procedure is "
        "R's 'holm': write p_adjust_method='holm'."
    )
    for name, label in (
        ("sidak", "Sidak's correction"),
        ("holm-sidak", "The Holm-Sidak procedure"),
        ("tukey", "Tukey's method"),
    ):
        text = message(ValueError, lambda name=name: v.check_p_adjust_method(name))
        assert text == (
            f"p_adjust_method={name!r} {start} {label} is not one of R's p.adjust() "
            "methods, which inferplot follows, so it is not available; choose one "
            "of those listed."
        )


@pytest.mark.parametrize(
    "value",
    ["one-sided", "one.sided", "one_sided", "onesided", "One-Sided", "one-tailed"],
)
def test_one_sided_asks_for_the_direction(value: str) -> None:
    text = message(ValueError, lambda: v.check_alternative(value))
    assert text == (
        f"alternative={value!r} is not allowed; use one of 'two-sided', 'less', "
        "'greater'. A one-sided test needs its direction: alternative='greater' "
        "tests whether the first level of x is greater than the second, and "
        "alternative='less' whether it is less. The first level is the first in "
        "order= when given, else the first declared category, else the first in "
        "sorted order."
    )


def test_two_tailed_is_two_sided() -> None:
    text = message(ValueError, lambda: v.check_alternative("two-tailed"))
    assert text.endswith(
        "A two-tailed test is alternative='two-sided': write alternative='two-sided'."
    )


@pytest.mark.parametrize(
    ("value", "hint"),
    [
        (95, " It is a proportion, not a percentage: write conf_level=0.95."),
        (90, " It is a proportion, not a percentage: write conf_level=0.9."),
        (99.0, " It is a proportion, not a percentage: write conf_level=0.99."),
        (99.9, " It is a proportion, not a percentage: write conf_level=0.999."),
        (5, ""),
        (1.96, ""),
        (80, ""),
        (100, ""),
        (1, ""),
    ],
)
def test_conf_level_proposes_a_value_only_for_a_common_percentage(
    value: float, hint: str
) -> None:
    text = message(ValueError, lambda: v.check_conf_level(value))
    assert text == (
        "conf_level must be strictly between 0 and 1, for example conf_level=0.95; "
        f"got {value!r}.{hint}"
    )


@pytest.mark.parametrize(
    ("check", "value", "unlikely"),
    [
        (v.check_p_adjust_method, "benjamini", "hochberg"),
        (v.check_p_adjust_method, "fdr-tsbh", "fdr"),
        (v.check_alternative, "one-side", "two-sided"),
        (v.canonical_type, "anova", None),
        (v.canonical_pairwise_display, "some", None),
    ],
)
def test_unlikely_value_suggestions_are_not_made(
    check: Callable[[object], object], value: str, unlikely: str | None
) -> None:
    text = message(ValueError, lambda: check(value))
    assert "Did you mean" not in text, text
    if unlikely is not None:
        assert f"={unlikely!r}" not in text
