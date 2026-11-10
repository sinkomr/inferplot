"""The APA sentence and the alt text of within-subjects results (SPEC.md section 7).

The APA sentence names each within test and effect size and says what n counts:
"n = 10 pairs" with two conditions, "n = 88 subjects" with three or more. The alt
text says "conditions", that the measurements are paired ("measured on the same
<n> subjects"), and which paths the figure draws: the analysis describes the figure
``ip.withinstats`` draws by default, and a figure drawn with other path or layer
arguments carries an alt text that describes what it drew (``fig.result``). Both
keep the one-sided direction (D68) and describe only what is drawn (D70).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot import text as tx
from inferplot.result import StatsResult
from tests import within_pb_cases as wc
from tests.oracle_fixtures import read_csv

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_within: Any = ip.analyze.withinstats
_draw: Any = ip.render.withinstats


def _sentences(text: str) -> int:
    return len(_SENTENCE_END.findall(text))


def _sleep(**kwargs: Any) -> StatsResult:
    return _within(read_csv("sleep.csv"), "group", "extra", subject_id="ID", **kwargs)


def _bugs(**kwargs: Any) -> StatsResult:
    frame = read_csv("bugs_long.csv")
    return _within(frame, "condition", "desire", subject_id="subject", **kwargs)


# --------------------------------------------------------------------------- #
# APA sentence
# --------------------------------------------------------------------------- #

# sleep (10 subjects, two conditions) and bugs_long (88 complete subjects, four
# conditions): the numbers are ggstatsplot's (tests/test_withinstats_texts.py).
_APA = {
    ("sleep", "parametric"): (
        "Paired t-test: t(9) = -4.06, p = .003, Hedges' g = -1.17, 95% CI [-1.94, "
        "-0.38], n = 10 pairs."
    ),
    ("sleep", "nonparametric"): (
        "Wilcoxon signed-rank test: V = 0.00, p = .009, rank-biserial r = -1.00, 95% "
        "CI [-1.00, -1.00], n = 10 pairs."
    ),
    ("bugs", "parametric"): (
        "Repeated-measures ANOVA (Greenhouse-Geisser): F(2.63, 228.95) = 20.59, "
        "p < .001, partial omega squared = .08, 95% CI [.03, 1.00], n = 88 subjects."
    ),
    ("bugs", "nonparametric"): (
        "Friedman test: chi2(3) = 55.83, p < .001, Kendall's W = .21, 95% CI "
        "[.15, 1.00], n = 88 subjects."
    ),
}


@pytest.mark.parametrize(("data", "kind"), list(_APA))
def test_apa_sentence_names_the_within_test(data: str, kind: str) -> None:
    result = (_sleep if data == "sleep" else _bugs)(type=kind)
    apa = result.text.apa
    if kind == "nonparametric" and data == "bugs":
        # Kendall's W's lower bound is a seeded bootstrap: check the rest exactly.
        head, _, _ = _APA[(data, kind)].partition("95% CI")
        assert apa.startswith(head)
        assert apa.endswith(", 1.00], n = 88 subjects.")
    else:
        assert apa == _APA[(data, kind)]


def test_apa_sentence_robust_and_bayes() -> None:
    robust = _sleep(type="robust")
    assert robust.text.apa.startswith(
        "Yuen's t-test for dependent samples (20% trimmed means): t(5) = -2.73, "
        "p = .041, AKP robust standardized difference = -2.56, 95% CI ["
    )
    assert robust.text.apa.endswith("], n = 10 pairs.")
    many = _bugs(type="robust")
    assert many.text.apa.startswith(
        "Repeated-measures trimmed-means ANOVA (20% trimmed means): F(2.73, 144.71) "
        "= 20.98, p < .001, average AKP robust standardized difference = 0.32, 95% CI ["
    )
    assert many.text.apa.endswith("], n = 88 subjects.")
    bayes = _sleep(type="bayes")
    assert bayes.text.apa == (
        "Bayesian paired t-test (JZS prior, r = 0.707): BF10 = 17.26, "
        "log(BF01) = -2.85, n = 10 pairs."
    )
    pairs = _bugs(type="bayes", results_subtitle=False)
    assert pairs.text.apa.startswith(
        "Bayesian paired t-tests of each pair (JZS prior, r = 0.707): HDHF vs HDLF, "
        "BF10 = "
    )
    assert pairs.text.apa.endswith("; n = 88 subjects.")
    none = _sleep(type="bayes", results_subtitle=False)
    assert (
        none.text.apa == "No test was computed (results_subtitle=False), n = 10 pairs."
    )


def test_apa_sentence_between_is_unchanged() -> None:
    plants = read_csv("PlantGrowth.csv")
    result = ip.analyze.betweenstats(plants, "group", "weight")
    assert result.text.apa.endswith(", n = 30.")


def test_one_sided_within_names_the_direction() -> None:
    result = _sleep(alternative="less")
    assert result.text.apa.startswith("Paired t-test (one-sided: 1 less than 2): ")
    assert result.text.apa.endswith(", n = 10 pairs.")
    assert "(one-sided: 1 less than 2)" in result.text.alt_text
    signed = _sleep(type="nonparametric", alternative="greater")
    assert signed.text.apa.startswith(
        "Wilcoxon signed-rank test (one-sided: 1 greater than 2): "
    )


# --------------------------------------------------------------------------- #
# alt text
# --------------------------------------------------------------------------- #


def test_alt_text_two_conditions() -> None:
    result = _sleep()
    assert result.text.alt_text == (
        "Plot comparing extra across the 2 conditions of group, measured on the same "
        "10 subjects: 1 (mean 0.75), 2 (mean 2.33); lines join each subject's values "
        "and the means. " + result.text.apa
    )


def test_alt_text_three_or_more_conditions() -> None:
    result = _bugs()
    alt = result.text.alt_text
    # Four conditions with the result and the pair count: the means as a range.
    assert alt.startswith(
        "Plot comparing desire across the 4 conditions of condition, measured on the "
        "same 88 subjects (means from 5.66 (LDLF) to 7.82 (HDHF)); a line joins the "
        "means. "
    )
    assert result.text.apa in alt
    assert alt.endswith(
        " 4 of 6 pairwise comparisons are significant (Student's t, Holm adjusted, "
        "alpha = 0.05)."
    )
    assert len(alt) <= tx.ALT_TEXT_MAX_CHARS
    assert _sentences(alt) == 3


@pytest.mark.parametrize(
    ("kind", "word"),
    [
        ("nonparametric", "medians"),
        ("robust", "20% trimmed means"),
        ("bayes", "MAP estimates"),
    ],
)
def test_alt_text_names_the_centrality_path(kind: str, word: str) -> None:
    result = _sleep(type=kind)
    assert f"; lines join each subject's values and the {word}. " in (
        result.text.alt_text
    )


def test_alt_text_without_subtitle_and_brackets() -> None:
    quiet = _bugs(results_subtitle=False)
    assert "No test result is shown (results_subtitle=False), n = 88 subjects." in (
        quiet.text.alt_text
    )
    assert quiet.text.apa not in quiet.text.alt_text
    assert " pairwise comparisons " in quiet.text.alt_text
    bare = _bugs(pairwise_display="none")
    assert "pairwise" not in bare.text.alt_text
    assert bare.text.apa in bare.text.alt_text
    bayes = _bugs(type="bayes", results_subtitle=False)
    assert (
        "Bayesian paired t-tests of each of the 6 pairs (JZS prior, r = 0.707): "
        in (bayes.text.alt_text)
    )
    assert bayes.text.alt_text.endswith("brackets show log(BF01) for every pair.")


def test_alt_text_shortens_long_names_within_the_limit() -> None:
    levels = [f"{'condition with a long descriptive name ' * 2}{i}" for i in range(6)]
    rows = [
        {"s": s, "x": level, "y": float((s * 7 + i * 3) % 11) + i}
        for s in range(12)
        for i, level in enumerate(levels)
    ]
    result = _within(pd.DataFrame(rows), "x", "y", subject_id="s")
    alt = result.text.alt_text
    assert len(alt) <= tx.ALT_TEXT_MAX_CHARS
    assert "…" in alt
    assert "measured on the same 12 subjects" in alt
    assert "a line joins the means" in alt
    assert result.text.apa in alt


@pytest.mark.parametrize(
    ("kwargs", "phrase"),
    [
        ({}, "; lines join each subject's values and the means. "),
        ({"point_path": False}, "; a line joins the means. "),
        ({"centrality_path": False}, "; lines join each subject's values. "),
        (
            {"point_path": False, "centrality_path": False},
            ": 1 (mean 0.75), 2 (mean 2.33). ",
        ),
        ({"layers": ("violin", "box")}, "; a line joins the means. "),
        ({"centrality_plotting": False}, "; lines join each subject's values. "),
    ],
)
def test_figure_alt_text_describes_the_paths_drawn(
    kwargs: dict[str, Any], phrase: str
) -> None:
    result = _sleep()
    fig = _draw(result, **kwargs)
    alt = fig.result.text.alt_text
    assert phrase in alt
    assert "join" not in alt or phrase.startswith(";")
    assert len(alt) <= tx.ALT_TEXT_MAX_CHARS
    if not kwargs:
        assert alt == result.text.alt_text


def test_figure_alt_text_three_conditions_without_centrality_path() -> None:
    result = _bugs()
    fig = _draw(result, centrality_path=False)
    alt = fig.result.text.alt_text
    assert "join" not in alt
    # Without the path clause every condition's mean fits.
    assert alt == (
        "Plot comparing desire across the 4 conditions of condition, measured on the "
        "same 88 subjects: HDHF (mean 7.82), HDLF (mean 6.68), LDHF (mean 7.35), "
        f"LDLF (mean 5.66). {result.text.apa} 4 of 6 pairwise comparisons are "
        "significant (Student's t, Holm adjusted, alpha = 0.05)."
    )
    assert len(alt) <= tx.ALT_TEXT_MAX_CHARS


def test_figure_alt_text_names_a_subsample_of_subjects() -> None:
    n = 3000  # 6,000 values: the paths show a sample of whole subjects
    rows = [
        {"s": s, "x": level, "y": float((s * 13 + j * 5) % 17) + j}
        for s in range(n)
        for j, level in enumerate(("pre", "post"))
    ]
    result = _within(pd.DataFrame(rows), "x", "y", subject_id="s")
    fig = _draw(result)
    assert fig.subsample is not None
    m = fig.subsample.shown // 2
    assert (
        f"; lines join the values of a random sample of {m:,} of the {n:,} subjects "
        "and the means. "
    ) in fig.result.text.alt_text


def test_embedded_copy_carries_the_figure_alt_text(tmp_path: Any) -> None:
    from tests.render._helpers import svg_description

    fig = _draw(_sleep(), point_path=False)
    path = tmp_path / "f.svg"
    fig.save(path)
    saved = StatsResult.from_json(svg_description(path))
    assert saved.text.alt_text == fig.result.text.alt_text
    assert "; a line joins the means. " in saved.text.alt_text


def test_alt_text_on_every_within_fixture() -> None:
    """Every ggwithinstats fixture's result: at most 400 characters, two to four
    sentences, conditions and the subjects measured, the default paths, the test
    only when the subtitle is drawn and the pair count only with brackets."""
    for case in wc.ALL_CASES:
        result = wc.analyze(case.name, "pandas")
        alt = result.text.alt_text
        k, n = len(result.levels), tx.format_n(result.n_obs)
        assert len(alt) <= tx.ALT_TEXT_MAX_CHARS, case.name
        assert 2 <= _sentences(alt) <= 4, (case.name, alt)
        assert f"across the {k} conditions of {result.x}" in alt, case.name
        assert f"measured on the same {n} subjects" in alt, case.name
        words = {
            "mean": "means",
            "median": "medians",
            "trimmed_mean": f"{tx.r_number_text(100 * result.settings.tr)}% "
            "trimmed means",
            "map": "MAP estimates",
        }
        what = words[result.descriptives[0].centrality]
        paths = (
            f"; lines join each subject's values and the {what}"
            if k == 2
            else f"; a line joins the {what}"
        )
        # The paths are left out only when the text cannot hold them.
        assert paths + ". " in alt or len(alt) + len(paths) > tx.ALT_TEXT_MAX_CHARS, (
            case.name
        )
        unit = "pairs" if k == 2 else "subjects"
        bayes_pairs = bool(result.pairwise) and result.pairwise[0].bayes is not None
        if result.text.subtitle is not None:
            assert result.text.apa in alt, case.name
        if not bayes_pairs:
            assert f"n = {n} {unit}." in alt, case.name
        brackets = any(p.displayed for p in result.pairwise)
        frequentist = result.pairwise and result.pairwise[0].bayes is None
        if frequentist and result.settings.pairwise_display != "none":
            assert " pairwise comparison" in alt, case.name
        else:
            assert " pairwise comparison" not in alt, case.name
        if not brackets:
            assert "brackets show" not in alt, case.name
