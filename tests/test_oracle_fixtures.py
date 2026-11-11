"""The committed oracle fixtures are intact, complete, and read correctly.

These tests check the fixtures themselves, not inferplot: hashes against the manifest,
datasets and generator files, the ggstatsplot version, the case list, what
ggstatsplot hid, and that every supported dataframe backend sees the same groups and
counts that R analyzed, and, within subjects, the same complete subjects and values.
"""

import math
from collections.abc import Sequence
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pytest
from scipy import stats as sp_stats

from inferplot.stats import histogram_bins, marginal_bins, ols_band
from tests.oracle_args import translate
from tests.oracle_fixtures import (
    ACCURATE_BAYES_FACTORS,
    PARITY_VERSION,
    SIMULATED_LINCON_CRITICAL_VALUES,
    Backend,
    OracleCase,
    Row,
    cases_for,
    integrity_problems,
    load_cases,
    load_probes,
    number,
    oracle_params,
    probe,
    r_double,
    r_level_label,
    read_dataset,
    require_cases,
)
from tests.tolerances import assert_within_monte_carlo

TYPES = ("parametric", "nonparametric", "robust", "bayes")
BACKENDS: Sequence[Backend] = ("pandas", "polars", "pyarrow")

# The cases oracle/generate.R writes. The Phase 0 checklist asked for ggbetweenstats,
# all types, on ToothGrowth, PlantGrowth and mtcars; the rest settle the SPEC.md
# claims the Phase 0 review found untested (see the comments in generate.R).
ALL_TYPE_LABELS = (
    "ToothGrowth",
    "PlantGrowth",
    "mtcars",
    "mtcars-am",
    "edge_k2",
    "iris",
    "mtcars-disp",
    "edge_unbalanced",
    "edge_k5",
    "bugs_long",
)
# Phase 1 (betweenstats, parametric): edge data and every argument the parametric
# slice exposes (see generate.R).
P_ADJUST_METHODS = ("bonferroni", "hochberg", "hommel", "BH", "BY", "fdr", "none")
PARAMETRIC_LABELS = (
    "edge_missing",
    "edge_empty_level",
    "PlantGrowth-declared",
    "edge_n2",
    "edge_ties",
    "edge_negative",
    "edge_outliers",
    "InsectSprays",
    "edge_null",
    "edge_large_n",
    *(f"PlantGrowth-padj-{method}" for method in P_ADJUST_METHODS),
    "ToothGrowth-cl90",
    "PlantGrowth-cl90",
    "mtcars-am-digits3",
    "mtcars-digits3",
    "PlantGrowth-display-all",
    "PlantGrowth-display-ns",
    "PlantGrowth-display-none",
    "ToothGrowth-nosubtitle",
    "PlantGrowth-nosubtitle",
    "ToothGrowth-nobf",
    "ToothGrowth-prior08",
)
# Phase 2 (betweenstats nonparametric, robust, bayes): the Phase 1 edge data and every
# argument these types take, with the types each label runs (see generate.R).
PHASE2_TYPES = ("nonparametric", "robust", "bayes")
PHASE2_LABELS: dict[str, tuple[str, ...]] = {
    "edge_missing": PHASE2_TYPES,
    "edge_empty_level": PHASE2_TYPES,
    "PlantGrowth-declared": PHASE2_TYPES,
    "edge_n2": PHASE2_TYPES,
    "edge_ties": PHASE2_TYPES,
    "edge_ties_small": ("nonparametric",),
    "edge_negative": PHASE2_TYPES,
    "edge_outliers": PHASE2_TYPES,
    "InsectSprays": PHASE2_TYPES,
    "edge_null": PHASE2_TYPES,
    "edge_large_n": PHASE2_TYPES,
    "edge_k9": ("robust",),
    "iris-petal": PHASE2_TYPES,
    "iris_no_setosa-petal": PHASE2_TYPES,
    "iris_no_setosa-petalwidth": ("robust",),
    **{
        f"PlantGrowth-padj-{method}": ("nonparametric", "robust")
        for method in P_ADJUST_METHODS
    },
    "ToothGrowth-cl90": PHASE2_TYPES,
    "PlantGrowth-cl90": PHASE2_TYPES,
    "ToothGrowth-tr01": ("robust",),
    "PlantGrowth-tr01": ("robust",),
    "mtcars-am-digits3": PHASE2_TYPES,
    "mtcars-digits3": PHASE2_TYPES,
    "PlantGrowth-display-all": PHASE2_TYPES,
    "PlantGrowth-display-ns": PHASE2_TYPES,
    "PlantGrowth-display-none": PHASE2_TYPES,
    "ToothGrowth-nosubtitle": PHASE2_TYPES,
    "PlantGrowth-nosubtitle": PHASE2_TYPES,
    "edge_k2-greater": ("nonparametric",),
    "edge_k2-less": ("nonparametric",),
    "ToothGrowth-greater": ("robust", "bayes"),
    "ToothGrowth-prior08": ("bayes",),
    "PlantGrowth-prior08": ("bayes",),
}
# Phase 3: robust ggbetweenstats cases whose bootstrap resamples include infinite
# effect sizes (heavy ties in small groups).
PHASE3_BETWEEN_LABELS = ("edge_heavy_ties2", "edge_heavy_ties3")
# Phase 3 (withinstats): ggwithinstats cases, label -> types (see generate.R).
WITHIN_EDGE_LABELS = (
    "sleep",
    "edge_within2_small",
    "edge_within2_small-declared",
    "edge_within2_ties",
    "edge_within2_negative",
    "edge_within2_outliers",
    "edge_within2_shuffled",
    "edge_within2_null",
    "edge_within2_large_n",
    "edge_within2_missing",
    "bugs_long",
    "edge_within_missing",
    "edge_within_missing-declared",
    "edge_within_small",
    "edge_within_ties",
    "edge_within_negative",
    "edge_within_outliers",
    "edge_within_shuffled",
    "edge_within_null",
    "edge_within_large_n",
    "edge_within_k5",
    "edge_within_nonspherical",
)
WITHIN_LABELS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(WITHIN_EDGE_LABELS, TYPES),
    **{
        f"bugs_long-padj-{method}": ("parametric", "nonparametric", "robust")
        for method in P_ADJUST_METHODS
    },
    "sleep-cl90": TYPES,
    "bugs_long-cl90": TYPES,
    "sleep-tr01": ("robust",),
    "bugs_long-tr01": ("robust",),
    "bugs_long-display-all": TYPES,
    "bugs_long-display-ns": TYPES,
    "bugs_long-display-none": TYPES,
    "sleep-digits3": TYPES,
    "bugs_long-digits3": TYPES,
    "sleep-nosubtitle": TYPES,
    "bugs_long-nosubtitle": TYPES,
    "sleep-greater": TYPES,
    "sleep-less": ("parametric", "nonparametric"),
    "sleep-prior08": ("parametric", "bayes"),
    "bugs_long-prior08": ("bayes",),
    "sleep-nobf": ("parametric",),
    "sleep-nopaths": ("parametric",),
    "bugs_long-nopaths": ("parametric",),
}
# Within-subject datasets whose rows are another dataset's in a random order: the
# results must be the same (pairing is by subject, not by row order).
SHUFFLED_TWINS = {
    "edge_within2_shuffled": "edge_within2_outliers",
    "edge_within_shuffled": "edge_within_outliers",
}
# Review of Phases 0-3, T7: small cases that decide open rules (see generate.R).
T7_CASES = frozenset(
    {
        "ggbetweenstats-edge_akp_undefined2-robust",
        "ggbetweenstats-edge_tied3_small-nonparametric",
        "ggbetweenstats-edge_tr029-robust",
        "ggbetweenstats-edge_n2_vs50-parametric",
        "ggwithinstats-edge_rm_all_tied-nonparametric",
        "ggwithinstats-edge_rm_some_tied-nonparametric",
    }
)
# Cases generate.R runs into probe-heldout instead of a fixture (see there).
HELD_OUT_CASES = frozenset(
    {
        "ggbetweenstats-edge_akp_undefined-robust",
        "ggwithinstats-edge_rm_n3k4-parametric",
        # Review round 3: a value equal to the test value 0.3 as written but not in
        # binary (D105); ratings whose bootstrap-t resamples are 0/0 (D65); an exact
        # zero printed as rounding noise (C3); percent labels at one-digit ties (D108);
        *(
            f"gghistostats-edge_one_near_mu-{column}-nonparametric"
            for column in ("sum", "below", "both", "sum_ties")
        ),
        *(
            f"gghistostats-edge_one_ratings-{column}-robust"
            for column in ("whole", "tenths", "centred_whole", "centred_tenths")
        ),
        "ggbarstats-edge_bar_zero_v-parametric",
        "ggbarstats-edge_bar_ties-digitsperc1-parametric",
        # and Fisher pairs listed sorted as text where inferplot keeps level order.
        *(
            f"ggbarstats-edge_bar_reduced-declared-{side}-parametric"
            for side in ("two.sided", "greater", "less")
        ),
    }
)
# Phase 4 (histostats): gghistostats cases, label -> types (see generate.R).
HISTO_LABELS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        (
            "mtcars",
            "mtcars-tv20",
            "iris-sepal",
            "ToothGrowth",
            "ToothGrowth-tv100",
            "bugs_long",
            "edge_one_small",
            "edge_one_ties",
            "edge_one_negative",
            "edge_one_outlier",
            "edge_one_null",
            "edge_one_large_n",
            "edge_one_missing",
            "ToothGrowth-cl90",
            "ToothGrowth-digits3",
            "ToothGrowth-nosubtitle",
            "ToothGrowth-greater",
        ),
        TYPES,
    ),
    "ToothGrowth-tr01": ("robust",),
    "ToothGrowth-less": ("parametric", "nonparametric"),
    "ToothGrowth-prior08": ("parametric", "bayes"),
    "ToothGrowth-nobf": ("parametric",),
    "ToothGrowth-binwidth2": ("parametric",),
    "ToothGrowth-nocentrality": ("parametric",),
}
# Phase 4 (scatterstats): ggscatterstats cases, label -> types (see generate.R).
SCATTER_LABELS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        (
            "mtcars",
            "iris",
            "edge_corr_small",
            "edge_corr_ties",
            "edge_corr_perfect",
            "edge_corr_perfect-neg",
            "edge_corr_null",
            "edge_corr_outlier",
            "edge_corr_large_n",
            "edge_corr_missing",
            "edge_corr_constant_x",
            "mtcars-cl90",
            "mtcars-digits3",
            "mtcars-nosubtitle",
        ),
        TYPES,
    ),
    "mtcars-tr01": ("robust",),
    "mtcars-prior08": ("parametric", "bayes"),
    "mtcars-nobf": ("parametric",),
    "mtcars-nomarginal": ("parametric",),
    "mtcars-loess": ("parametric",),
    "mtcars-labels": ("parametric",),
    "mtcars-labelexpr": ("parametric",),
}
# The small samples the foundation package asked for, and constant data (lead's
# request): n = 2 and 3, ten values of 5 with the default bin width and with 1.
HISTO_LABELS.update(
    dict.fromkeys(
        (
            "edge_one_n2",
            "edge_one_n3",
            "edge_one_constant",
            "edge_one_constant-binwidth1",
        ),
        TYPES,
    )
)
SCATTER_LABELS.update(
    {
        "edge_corr_n3": TYPES,
        "edge_corr_n4": TYPES,
        "edge_corr_label_missing": ("parametric",),
    }
)
# Review round 3 (generate.R's ROUND3_CASES): the test value 0.3 written as a literal
# in the data (an exact zero of x - 0.3).
ROUND3_HISTO_LABELS: dict[str, tuple[str, ...]] = {
    "edge_one_near_mu-literal": ("nonparametric",),
}
HISTO_LABELS.update(ROUND3_HISTO_LABELS)
# Phase 4 (barstats): ggbarstats cases, label -> types (see generate.R).
P_ADJUST_LABELS = tuple(f"Titanic_full-class-padj-{m}" for m in P_ADJUST_METHODS)
BAR_LABELS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        (
            "mtcars-cyl-am",
            "mtcars-am-cyl",
            "mtcars-am-vs",
            "Titanic_full-class",
            "mtcars-cyl",
            "edge_bar_paired",
        ),
        TYPES,
    ),
    **dict.fromkeys(
        (
            "Titanic_full-survived",
            "Titanic_counts-class",
            *(
                f"mtcars-cyl-am-{arg}"
                for arg in (
                    "cl90",
                    "digits3",
                    "nosubtitle",
                    "nosubtitle-proptest",
                    "noproptest",
                    "nobf",
                    "greater",
                    "less",
                    "counts",
                    "both",
                    "digitsperc1",
                    "ratio",
                )
            ),
            *P_ADJUST_LABELS,
            *(
                f"mtcars-cyl-{arg}"
                for arg in (
                    "cl90",
                    "digits3",
                    "nosubtitle",
                    "nobf",
                    "greater",
                    "both",
                    "digitsperc1",
                    "ratio",
                    "ratio-sum4",
                    "ratio-short",
                )
            ),
            "edge_bar_paired-cl90",
            "edge_bar_paired-less",
            "edge_bar_zero_cell",
            "edge_bar_zero_cell-declared",
            "edge_bar_zero_cell-oneway-declared",
            "edge_bar_small_expected",
            "edge_bar_2x2_one",
            "edge_bar_large",
            "edge_bar_missing",
            "edge_bar_missing-counts",
            "edge_bar_missing-oneway",
            "edge_bar_perfect",
            "edge_bar_independent",
            "edge_bar_degenerate-oneway-one",
            "edge_bar_degenerate-one-row",
            "edge_bar_degenerate-one-column",
            "edge_bar_degenerate-one-cell",
            "edge_bar_degenerate-single-x",
            "edge_bar_rounding",
            "edge_bar_rounding-digitsperc1",
            "edge_bar_rounding-digitsperc2",
            "edge_bar_rounding-both",
            "edge_bar_rounding-counts",
            "edge_bar_counts",
            "edge_bar_counts-frac",
            "edge_bar_paired3",
            "edge_bar_paired_agree",
            "edge_bar_paired_small",
            "edge_bar_paired_unequal",
            "edge_bar_paired_unequal-declared",
        ),
        ("parametric",),
    ),
}
# Review round 3 (generate.R's ROUND3_CASES): wide tables, exact zeros, percent labels
# at rounding ties, one-sided Fisher pairs on reduced pair tables.
ROUND3_BAR_LABELS: dict[str, tuple[str, ...]] = dict.fromkeys(
    (
        "edge_bar_wide20",
        "edge_bar_wide40",
        "edge_bar_wide61",
        "edge_bar_wide61r",
        "edge_bar_exact_fit",
        "edge_bar_ties-digitsperc0",
        "edge_bar_ties-digitsperc2",
        *(f"edge_bar_reduced-{side}" for side in ("two.sided", "greater", "less")),
    ),
    ("parametric",),
)
BAR_LABELS.update(ROUND3_BAR_LABELS)
# ggpiestats: the one-way design, to show it computes what ggbarstats computes.
PIE_LABELS: dict[str, tuple[str, ...]] = {
    "mtcars-cyl": TYPES,
    "mtcars-cyl-ratio": ("parametric",),
}
BAR_CASES = frozenset(
    f"ggbarstats-{label}-{t}" for label, types in BAR_LABELS.items() for t in types
) | frozenset(
    f"ggpiestats-{label}-{t}" for label, types in PIE_LABELS.items() for t in types
)
HISTO_CASES = frozenset(
    f"gghistostats-{label}-{t}" for label, types in HISTO_LABELS.items() for t in types
)
SCATTER_CASES = frozenset(
    f"ggscatterstats-{label}-{t}"
    for label, types in SCATTER_LABELS.items()
    for t in types
)
EXPECTED_CASES = frozenset(
    {f"ggbetweenstats-{label}-{t}" for label in ALL_TYPE_LABELS for t in TYPES}
    | {"ggbetweenstats-mtcars-am-drat-parametric"}
    | {
        f"ggbetweenstats-ToothGrowth-{side}-{t}"
        for side in ("greater", "less")
        for t in ("parametric", "nonparametric")
    }
    | {f"ggbetweenstats-{label}-parametric" for label in PARAMETRIC_LABELS}
    | {
        f"ggbetweenstats-{label}-{t}"
        for label, types in PHASE2_LABELS.items()
        for t in types
    }
    | {f"ggbetweenstats-{label}-robust" for label in PHASE3_BETWEEN_LABELS}
    | {
        f"ggwithinstats-{label}-{t}"
        for label, types in WITHIN_LABELS.items()
        for t in types
    }
    | T7_CASES
    | HISTO_CASES
    | SCATTER_CASES
    | BAR_CASES
)
# Cases where ggstatsplot 1.1.4 itself fails, as generate.R declares them.
EXPECTED_FAILURES = frozenset(
    {
        "ggbetweenstats-edge_n2-parametric",
        "ggbetweenstats-edge_n2-robust",
        "ggbetweenstats-edge_n2-bayes",
        # Every subject constant across conditions: the subtitle fails silently and
        # the Durbin-Conover table has no rows (T7).
        "ggwithinstats-edge_rm_all_tied-nonparametric",
        # A weight of 2.5: tidyr::uncount() refuses it while ggbarstats prepares the
        # data, before any hidden call.
        "ggbarstats-edge_bar_counts-frac-parametric",
    }
)
EXPECTED_PROBES = frozenset(
    {
        "probe-format",
        "probe-padjust",
        "probe-ptukey",
        "probe-omega",
        "probe-smm",
        "probe-wrs2p",
        "probe-akp",
        "probe-map",
        "probe-ranks",
        "probe-signrank",
        "probe-pairedbf",
        "probe-friedman",
        "probe-rmanova",
        "probe-wrs2within",
        "probe-bootrules",
        "probe-corr",
        "probe-onesamplebf",
        "probe-heldout",
        "probe-contingency",
        # Review round 3 (tests/test_probe_round3.py).
        "probe-nearzero",
        "probe-trimcibt",
        "probe-bartables",
        "probe-round",
    }
)


def test_fixtures_match_manifest_datasets_and_generator() -> None:
    assert integrity_problems() == []


def test_case_list_is_what_the_generator_writes() -> None:
    names = {case.name for case in load_cases()}
    assert names == EXPECTED_CASES, (
        f"missing {sorted(EXPECTED_CASES - names)}, unexpected "
        f"{sorted(names - EXPECTED_CASES)}"
    )
    assert len(PARAMETRIC_LABELS) == 28
    assert sum(len(types) for types in PHASE2_LABELS.values()) == 88
    assert sum(len(types) for types in WITHIN_LABELS.values()) == 159
    assert len(T7_CASES) == 6
    assert len(HISTO_CASES) == 76 + 16 + 1
    assert len(SCATTER_CASES) == 64 + 9
    assert len(BAR_CASES) == 90 + 10


ALL_CASES = [
    *oracle_params("ggbetweenstats"),
    *oracle_params("ggbetweenstats", failed=True),
]
FUNCTIONS = ("ggbetweenstats", "ggwithinstats")


def _completed_params() -> list[Any]:
    """Every completed case of the functions whose fixtures share one layout."""
    return [param for function in FUNCTIONS for param in oracle_params(function)]


def _failed_params() -> list[Any]:
    """Every case where ggstatsplot failed as declared, for either function."""
    cases = require_cases(
        [c for function in FUNCTIONS for c in cases_for(function, failed=True)],
        "failed ggbetweenstats or ggwithinstats cases",
    )
    return [pytest.param(case, id=case.name) for case in cases]


COMPLETED = _completed_params()
# gghistostats and ggscatterstats: designs without groups (analyzed_data).
NO_LEVEL = ("gghistostats", "ggscatterstats")


def _no_level_params() -> list[Any]:
    """Every completed gghistostats and ggscatterstats case."""
    return [param for function in NO_LEVEL for param in oracle_params(function)]


def _no_level_all() -> list[Any]:
    """Every gghistostats and ggscatterstats case, failed ones included."""
    return [
        pytest.param(case, id=case.name)
        for case in load_cases()
        if case.function in NO_LEVEL
    ]


WITHIN_ALL = [
    pytest.param(case, id=case.name)
    for case in load_cases()
    if case.function == "ggwithinstats"
]


def test_failures_are_the_declared_ones() -> None:
    failed = {case.name for case in load_cases() if case.error is not None}
    assert failed == EXPECTED_FAILURES
    for case in load_cases():
        assert (case.expected_error is not None) == (case.name in EXPECTED_FAILURES)
        assert (case.raw["partial"] is not None) == (case.name in EXPECTED_FAILURES)


@pytest.mark.parametrize("case", _failed_params())
def test_failed_case_fails_where_declared(case: OracleCase) -> None:
    # ggstatsplot stopped inside the declared hidden call, with the declared message,
    # after every hidden call before it ran; what those calls returned is kept.
    expected = case.expected_error
    assert expected is not None
    assert case.error is not None
    assert expected["message"] in case.error
    assert expected["reason"]
    *before, last = case.suppressed
    assert last["component"] == expected["component"]
    assert last["error"] is not None
    # A call before the failure failed only where the case declares that
    # ggstatsplot swallows its error (edge_rm_all_tied: the subtitle); such a call
    # returned nothing.
    declared = set(case.expected_hidden_errors)
    assert {e["component"] for e in before if e["error"] is not None} == declared
    assert case.raw["subtitle"] is None
    assert case.raw["stats"] is None
    assert case.raw["plot_layers"] is None
    for entry in before:
        value = case.partial(entry["component"])
        if entry["component"] in declared:
            assert value is None, entry["component"]
        else:
            assert value, f"{entry['component']} returned nothing"
    # The failing call is recorded too, with no value: it returned nothing.
    components = [entry["component"] for entry in case.raw["partial"]]
    assert components.count(expected["component"]) == 1
    assert components[-1] == expected["component"]
    assert case.partial(expected["component"]) is None
    # The centrality labels R computed carry the counts generate.R recorded.
    labels = case.partial("centrality_description")
    x = case.args["x"]
    assert {str(r[x]): int(r["n.obs"]) for r in labels} == case.n_per_level


def test_probe_list_is_what_the_generator_writes() -> None:
    assert set(load_probes()) == EXPECTED_PROBES
    with pytest.raises(KeyError, match="probe-formats"):
        probe("probe-format" + "s")
    with pytest.raises(KeyError, match="value"):
        probe("probe-format").section("value")
    with pytest.raises(KeyError, match="name"):
        probe("probe-format").section("name")  # envelope fields are not sections


def test_r_double_reads_r_hex_exactly() -> None:
    assert r_double("0x1.999999999999ap-4") == 0.1
    assert r_double("0x0p+0") == 0.0
    negative_zero = r_double("-0x0p+0")
    assert negative_zero is not None
    assert negative_zero == 0.0
    assert math.copysign(1.0, negative_zero) < 0
    assert r_double("NA") is None
    assert math.isnan(r_double("NaN") or 0.0)
    assert r_double("-Inf") == -math.inf
    with pytest.raises(ValueError, match="hex"):
        r_double("0.1")


def _hex_or_na(text: str) -> float | None:
    value = r_double(text)
    assert value is None or isinstance(value, float)
    return value


def test_probe_format_covers_its_grid() -> None:
    fmt = probe("probe-format")
    rows = fmt.rows("values")
    # The generator loops over the grid, then over digits 0 to 4: blocks of five
    # rows with the same value.
    assert len(rows) % 5 == 0
    for start in range(0, len(rows), 5):
        block = rows[start : start + 5]
        assert [row["digits"] for row in block] == [0, 1, 2, 3, 4]
        assert len({(row["hex"], row["negative_zero"]) for row in block}) == 1
    values = [_hex_or_na(row["hex"]) for row in rows[::5]]
    assert None in values  # NA
    finite = [v for v in values if v is not None and math.isfinite(v)]
    # The grid asks for 0 and -0 (sprintf("%a") prints both as 0x0p+0), but R
    # received -0 as +0: neither zero row is negative (1 / x > 0, "%.0f" gives "0").
    zeros = [row for row in rows if row["hex"] == "0x0p+0"]
    assert len(zeros) == 10
    assert not any(row["negative_zero"] for row in rows)
    assert any(v is not None and math.isnan(v) for v in values)
    assert {math.inf, -math.inf} <= {v for v in values if v is not None}
    assert max(abs(v) for v in finite) > 1e300
    for row in rows:
        assert not str(row["to_char"]).startswith("<error"), row
    assert len(fmt.rows("counts")) == 34
    assert [r["label"] for r in fmt.rows("conf_levels")][:5] == [
        "50%",
        "80%",
        "89%",
        "90%",
        "95%",
    ]
    assert [c["type"] for c in fmt.rows("centrality")] == [
        "parametric",
        "nonparametric",
        "robust",
    ]


def test_probe_padjust_covers_every_method() -> None:
    padjust = probe("probe-padjust")
    methods = padjust.section("methods")
    assert set(methods) == {
        "holm",
        "hochberg",
        "hommel",
        "bonferroni",
        "BH",
        "BY",
        "fdr",
        "none",
    }
    cases = padjust.rows("cases")
    assert len(cases) == 20
    for case in cases:
        n = len(case["p_hex"])
        assert set(case["adjusted"]) == set(methods)
        for method in methods:
            assert len(case["adjusted"][method]["hex"]) == n, (case["name"], method)


def test_probe_ptukey_covers_its_grid_and_every_games_howell_case() -> None:
    ptukey = probe("probe-ptukey")
    tails = ptukey.rows("tails")
    assert len(tails) == 15 * 7 * 13
    assert {(r["q"], r["nmeans"], r["df_hex"]) for r in tails}.__len__() == len(tails)
    expected = {
        case.name
        for case in load_cases()
        if case.function == "ggbetweenstats"
        and case.type == "parametric"
        and len(case.analysis_levels) > 2
    }
    entries = ptukey.rows("games_howell")
    assert {entry["case"] for entry in entries} == expected
    by_name = {case.name: case for case in load_cases()}
    for entry in entries:
        case = by_name[entry["case"]]
        k = len(case.analysis_levels)
        assert tuple(entry["levels"]) == case.analysis_levels
        assert {str(k): int(v) for k, v in entry["n_per_level"].items()} == (
            case.n_per_level
        )
        assert len(entry["pairs"]) == k * (k - 1) // 2
        for pair in entry["pairs"]:
            p_value = _hex_or_na(pair["p_value_hex"])
            welch_df = _hex_or_na(pair["welch_df_hex"])
            assert p_value is not None
            assert welch_df is not None
            if welch_df < 2:
                # R's ptukey is undefined below 2 df and returns NaN (edge_n2).
                assert math.isnan(p_value), pair
            else:
                assert 0.0 <= p_value <= 1.0


def test_probe_omega_covers_its_grid() -> None:
    rows = probe("probe-omega").rows("rows")
    assert len(rows) == 8 * 4 * 6 * 3 * 3 + 3
    greater = [r for r in rows if r["alternative"] == "greater"]
    assert all(r["accurate_low"] is not None for r in greater)
    assert all(r["accurate_low"] is None for r in rows if r["alternative"] != "greater")


def test_probe_smm_covers_its_grid() -> None:
    smm = probe("probe-smm")
    alphas = {"0.05", "1 - 0.95", "0.01", "1 - 0.99", "0.1", "1 - 0.9"}
    lincon = smm.rows("lincon")
    assert {(e["groups"], e["alpha"]) for e in lincon} == {
        (k, a) for k in range(2, 11) for a in alphas
    }
    # lincon's table functions, found by recording what lincon calls: one per alpha
    # the table supports.
    grids = {g["name"]: g for g in smm.rows("grids")}
    assert set(grids) == {"smmcrit", "smmcrit01"}
    for grid in grids.values():
        assert (grid["df_arg"], grid["c_arg"]) == ("nuhat", "C")
        assert len(grid["rows"]) == len(smm.section("df_grid_hex"))
        assert all(len(row["values_hex"]) == 50 for row in grid["rows"])
    calls = smm.rows("calls")
    assert len(calls) == 3 * len(alphas)
    for entry in calls:
        found = [c["name"] for c in entry["critical_value_calls"]]
        tabled = entry["alpha"] in {"0.05", "0.01"} and entry["comparisons"] <= 28
        assert found == ([] if not tabled else [grids_for(entry["alpha"])])
        assert ("smmvalv2" in entry["call_counts"]) == (not tabled)


def grids_for(alpha: str) -> str:
    return {"0.05": "smmcrit", "0.01": "smmcrit01"}[alpha]


def test_probe_wrs2p_covers_its_grid() -> None:
    wrs2p = probe("probe-wrs2p")
    shifts = 28
    assert len(wrs2p.rows("yuen")) == 3 * shifts
    assert len(wrs2p.rows("t1way")) == 3 * shifts
    lincon = wrs2p.rows("lincon")
    assert len(lincon) == 3 * shifts
    assert all(len(row["pairs"]) == 3 for row in lincon)
    # The grid reaches p = 0 (1 - cdf exactly 0) and p = 1.
    p_values = [r_double(row["p_value_hex"]) for row in wrs2p.rows("yuen")]
    assert 0.0 in p_values
    assert 1.0 in p_values


def test_probe_akp_covers_its_grid() -> None:
    akp = probe("probe-akp")
    rows = akp.rows("rows")
    assert len(rows) == len(akp.section("data")) * len(akp.section("tr_grid"))
    assert all([e["EQVAR"] for e in row["akp"]] == [False, True] for row in rows)
    assert len(akp.rows("constants")) == 49


def test_probe_map_pins_the_grid() -> None:
    # ggstatsplot's MAP label for type "bayes" (centrality_description) is
    # bayestestR::map_estimate with its defaults, which is the grid point of highest
    # density of stats::density(x, bw = "SJ", n = 1024, from = min(x), to = max(x)):
    # on every vector where bw.SJ works. Where it fails (constant data, zero IQR),
    # map_estimate still returns a value; the probe records what.
    rows = probe("probe-map").rows("rows")
    assert len(rows) >= 50
    assert len({row["name"] for row in rows}) == len(rows)
    sizes = {row["n"] for row in rows}
    assert {2, 501, 1000, 100000} <= sizes
    for row in rows:
        what = row["name"]
        density = row["density_data_range"]
        if row["bw_sj"]["error"] is not None:
            assert density["error"] is not None, what
            continue
        assert row["centrality_error"] is None, what
        assert row["map_estimate_error"] is None, what
        assert row["centrality_hex"] == row["map_estimate_hex"], what
        assert density["n"] == 1024, what
        assert row["map_estimate_hex"] == density["x_at_max_hex"], what
        assert density["bw_hex"] == row["bw_sj"]["hex"], what
        # Without from/to, density's grid is wider and the maximum moves.
        assert row["density_default_range"]["n"] == 1024, what
    failures = [row["name"] for row in rows if row["bw_sj"]["error"] is not None]
    assert {"constant", "zero_iqr"} <= set(failures)


def test_probe_ranks_covers_its_grid() -> None:
    ranks = probe("probe-ranks")
    two = ranks.rows("two_sample")
    assert len(two) == 9
    assert all(len(entry["wilcox"]) == 6 for entry in two)
    assert all(len(entry["rank_biserial"]) == 3 for entry in two)
    k = ranks.rows("k_sample")
    assert [entry["name"] for entry in k] == ["edge_ties", "InsectSprays", "mtcars_cyl"]


@pytest.mark.parametrize("case", COMPLETED)
def test_fixture_contents(case: OracleCase) -> None:
    versions = case.raw["versions"]
    assert versions["packages"]["ggstatsplot"] == PARITY_VERSION
    assert versions["R"].startswith("4.5.")
    assert case.error is None, case.error

    # What ggstatsplot hid (SPEC.md section 9), one entry per hidden call in call
    # order. With results.subtitle (default TRUE) ggstatsplot computes the subtitle,
    # then the Bayes caption when its bf.condition ("parametric") and bf.message
    # (default TRUE) both hold; without results.subtitle it computes neither. It
    # computes the centrality labels whether or not it draws them.
    with_subtitle = case.args.get("results.subtitle", True) is not False
    with_caption = (
        with_subtitle
        and case.type == "parametric"
        and case.args.get("bf.message", True) is not False
    )
    expected_calls: list[str] = []
    if with_subtitle:
        expected_calls.append("subtitle")
    if with_caption:
        expected_calls.append("caption")
    expected_calls.append("centrality_description")
    if (
        len(case.analysis_levels) > 2
        and case.args.get("pairwise.display", "significant") != "none"
    ):
        expected_calls.append("pairwise_comparisons")
    assert [entry["component"] for entry in case.suppressed] == expected_calls
    # No call swallowed an error unless the case declares it; a declared component
    # is missing from the output, and the checks that need it do not apply.
    swallowed = {
        entry["component"]: entry["error"]
        for entry in case.suppressed
        if entry["error"] is not None
    }
    declared = set(case.expected_hidden_errors)
    assert set(swallowed) == declared, swallowed
    if with_caption and "caption" not in declared:
        assert case.caption, "no Bayes caption"
        assert case.caption_data is not None, "no caption_data"
    else:
        assert case.caption is None
        assert case.caption_data is None
    if "subtitle" in declared or not with_subtitle:
        assert case.subtitle is None
        assert case.subtitle_rows == []
        return

    assert case.subtitle, "ggstatsplot drew no subtitle"
    rows = case.subtitle_rows
    assert rows, "no subtitle_data"
    if case.type == "bayes":
        # Bayes ANOVA repeats the omnibus values on one row per posterior parameter.
        assert len({number(r["bf10"]) for r in rows}) == 1
        assert number(rows[0]["bf10"]) > 0.0
    else:
        row = case.subtitle_data
        for column in ("statistic", "p.value", "estimate", "conf.low", "conf.high"):
            assert column in row, (
                f"subtitle_data has no {column!r} column: {sorted(row)}"
            )
        assert 0.0 <= number(row["p.value"]) <= 1.0
    call = translate(case.function, case.args)
    assert call.function == case.function.removeprefix("gg")
    assert call.args == (case.args["x"], case.args["y"])
    assert call.kwargs["type"] == case.type
    if case.function == "ggwithinstats":
        assert call.kwargs["subject_id"] == case.subject_id
        if case.type != "bayes" or len(case.analysis_levels) == 2:
            # n in the subtitle counts subjects (pairs), not observations.
            n_obs = {int(number(r["n.obs"])) for r in rows}
            assert n_obs == {len(case.within_data["subjects"])}, n_obs


# R warnings and messages a fixture may record, each read and found harmless for the
# numbers inferplot compares: (kind, hidden call, start of the text) -> why.
DECLARED_R_NOISE: dict[tuple[str, str, str], str] = {
    ("message", "caption", "Multiple `BFBayesFactor` models detected"): (
        "the Bayesian repeated-measures ANOVA of the parametric caption (anovaBF "
        "with a random subject effect): insight says which model the posterior "
        "comes from; a v0.2 output, compared by no v0.1 test"
    ),
    ("message", "subtitle", "Multiple `BFBayesFactor` models detected"): (
        "the same message for the type bayes subtitle with three or more conditions"
    ),
    ("message", "subtitle", "Registered S3 method overwritten by 'lme4'"): (
        "afex loads lme4 once per R process; the message lands in the first "
        "parametric repeated-measures case of each generator shard"
    ),
    ("warning", "subtitle", "HF eps > 1 treated as 1"): (
        "afex computes the Huynh-Feldt epsilon alongside; ggstatsplot reports the "
        "Greenhouse-Geisser correction, which the warning does not concern"
    ),
    # ggscatterstats on constant x (edge_corr_constant_x), where no correlation is
    # defined: R's cor() warns and returns NA, and ggstatsplot prints NA for r, the
    # statistic, p and the CI (parametric, nonparametric, robust); the Bayesian
    # correlation then stops ("No valid values found"), which ggstatsplot swallows
    # (the parametric caption, the bayes subtitle). A finding, not noise: inferplot
    # must refuse such data with a plain error instead (CLAUDE.md rule 3).
    ("warning", "subtitle", "the standard deviation is zero"): (
        "constant x: R's correlation is NA and ggstatsplot prints NA "
        "(edge_corr_constant_x)"
    ),
    ("warning", "caption", "the standard deviation is zero"): (
        "constant x: the Bayesian caption's correlation, before it fails"
    ),
    ("warning", "correlation record", "the standard deviation is zero"): (
        "constant x: the correlation record replays the subtitle's call"
    ),
    ("message", "subtitle", "`ci` is too small or x does not contain enough data"): (
        "constant x: bayestestR finds no posterior draws for the interval"
    ),
    ("message", "caption", "`ci` is too small or x does not contain enough data"): (
        "constant x: as for the bayes subtitle, in the parametric caption"
    ),
    ("message", "correlation record", "`ci` is too small or x does not contain"): (
        "constant x: the correlation record replays the bayes subtitle's call"
    ),
    # ggbarstats: chisq.test warns when an expected count is below 5. ggstatsplot
    # mutes it (subtitle) or suppressWarnings() it (each bar's test); it changes no
    # number (statsExpressions never asks for a simulated p).
    ("warning", "subtitle", "Chi-squared approximation may be incorrect"): (
        "chisq.test with an expected count below 5 (independence or goodness of fit)"
    ),
    ("warning", "proportion_test", "Chi-squared approximation may be incorrect"): (
        "the same warning for a bar's goodness-of-fit test"
    ),
    ("warning", "proportion_test", "Using an external vector in selections was"): (
        "tidyselect's once-per-session deprecation note on contingency_table()'s "
        "select(); it lands in the first per-bar test of each generator shard"
    ),
}


def test_r_warnings_and_messages_are_declared() -> None:
    """Every R warning or message a ggstatsplot call, a hidden call or a bootstrap
    reference run recorded is one that has been read and declared above. A warning
    is R saying a number may be off, and no parity test reads these fields, so an
    undeclared one fails here; so does a declared one that no fixture records."""
    undeclared: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    def read(case: OracleCase, kind: str, where: str, texts: Any) -> None:
        for text in texts or []:
            keys = [
                key
                for key in DECLARED_R_NOISE
                if key[:2] == (kind, where) and str(text).startswith(key[2])
            ]
            if keys:
                seen.update(keys)
            else:
                undeclared.append(f"{case.name} {where} {kind}: {text}")

    for case in load_cases():
        raw = case.raw
        read(case, "warning", "call", raw["r_warnings"])
        read(case, "message", "call", raw["r_messages"])
        for entry in raw["suppressed"] or []:
            read(case, "warning", str(entry["component"]), entry["warnings"])
            read(case, "message", str(entry["component"]), entry["messages"])
        reference = raw["bootstrap_reference"]
        if reference is not None:
            read(case, "warning", "bootstrap reference", reference["r_warnings"])
        if case.function == "ggscatterstats":  # its replay of the subtitle's call
            record = raw["correlation"]
            read(case, "warning", "correlation record", record["warnings"])
            read(case, "message", "correlation record", record["messages"])
    assert undeclared == []
    assert seen == set(DECLARED_R_NOISE)
    # Only the repeated-measures cases are noisy; ggbetweenstats ran silently.
    assert all(
        not (c.raw["r_warnings"] or c.raw["r_messages"])
        and not any(e["warnings"] or e["messages"] for e in c.raw["suppressed"] or [])
        for c in load_cases()
        if c.function == "ggbetweenstats"
    )


def test_pinned_exceptions_name_pairs_that_exist() -> None:
    # An exception keyed by a case or pair that no longer exists would never be
    # looked up, and so could never fail.
    pairs = {
        (case.name, str(row["group1"]), str(row["group2"]))
        for case in load_cases()
        if case.error is None
        for row in case.pairwise
    }
    assert set(ACCURATE_BAYES_FACTORS) <= pairs
    assert set(SIMULATED_LINCON_CRITICAL_VALUES) <= pairs


def test_lookups_of_unknown_names_fail_loudly() -> None:
    # A misspelled table or geom must not pass as an empty one (SPEC.md section 9).
    case = next(c for c in load_cases() if c.name == "ggbetweenstats-mtcars-parametric")
    assert len(case.pairwise) == 3
    assert case.descriptive == []  # R gave NULL: empty, not an error
    with pytest.raises(KeyError, match="pairwise_comparison_data"):
        case.table("pairwise_comparison_data")
    with pytest.raises(KeyError, match="GeomLabelRepell"):
        case.layer_data("GeomLabelRepel" + "l")
    with pytest.raises(KeyError, match="recorded only for"):
        case.layer_params("GeomPoint")
    assert len(case.layer_params("GeomSignif")) == 1
    assert case.layer_data("GeomViolin") == []  # draws the plot's own data
    # Two point layers: the raw points (plot data) and the centrality points.
    assert len(case.layer_data("GeomPoint")) == 1


def _levels_and_counts(frame: Any, x: str, y: str) -> tuple[list[str], dict[str, int]]:
    df = nw.from_native(frame, eager_only=True)
    used = df.select(x, y).drop_nulls()
    counts = used.group_by(x).agg(nw.len().alias("n"))
    keys = counts[x].to_list()
    ns = counts["n"].to_list()
    dtype = df.schema[x]
    if isinstance(dtype, nw.Enum | nw.Categorical):
        order = [str(v) for v in df[x].cat.get_categories().to_list()]
        keys_sorted = [k for k in order if k in set(keys)]
    else:
        keys_sorted = sorted(keys)
    by_key = dict(zip(keys, ns, strict=True))
    labels = [r_level_label(k) for k in keys_sorted]
    return labels, {r_level_label(k): int(by_key[k]) for k in keys_sorted}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ALL_CASES)
def test_backends_see_the_groups_r_analyzed(case: OracleCase, backend: Backend) -> None:
    frame = read_dataset(case, backend)
    levels, counts = _levels_and_counts(frame, case.args["x"], case.args["y"])
    assert tuple(levels) == case.analysis_levels
    assert counts == case.n_per_level


@pytest.mark.parametrize("case", ALL_CASES)
def test_declared_level_order_is_kept(case: OracleCase) -> None:
    # A declared order (factor_levels) survives, minus levels without data; text
    # columns sort in the C locale, as Python's sorted() does.
    declared = case.factor_levels.get(case.args["x"])
    if declared is None:
        assert list(case.analysis_levels) == sorted(case.analysis_levels)
        return
    assert case.analysis_levels == tuple(
        level for level in declared if level in case.n_per_level
    )
    assert set(declared) - set(case.analysis_levels) == {
        level for level in declared if level not in case.n_per_level
    }


@pytest.mark.parametrize("case", COMPLETED)
def test_centrality_labels_are_recorded(case: OracleCase) -> None:
    # extract_stats() in ggstatsplot 1.1.4 leaves descriptive_data empty, so the
    # centrality labels come from the plot layer that draws them: one row per group.
    labelled = case.layer_data("GeomLabelRepel")
    assert labelled, (
        f"no label layer; geoms are {[g['geom'] for g in case.plot_layers]}"
    )
    rows = labelled[0]
    assert [str(r[case.args["x"]]) for r in rows] == list(case.analysis_levels)
    # ggstatsplot's own n per group agrees with the counts generate.R recorded.
    assert {str(r[case.args["x"]]): int(r["n.obs"]) for r in rows} == case.n_per_level


@pytest.mark.parametrize("case", COMPLETED)
def test_brackets_are_the_displayed_pairs(case: OracleCase) -> None:
    # SPEC.md section 4: brackets exist only for three or more groups and never with
    # pairwise.display "none"; they mark the pairs whose adjusted p is strictly below
    # pairwise.alpha (0.05) by default, those at or above it for "non-significant",
    # every pair for "all", and every pair for type "bayes". Each label is the pair's
    # expression from the pairwise table.
    brackets = case.layer_params("GeomSignif")
    display = case.args.get("pairwise.display", "significant")
    if len(case.analysis_levels) == 2 or display == "none":
        assert brackets == []
        assert case.pairwise == []
        return
    rows = case.pairwise
    alpha = float(case.args.get("pairwise.alpha", 0.05))
    if case.type == "bayes" or display == "all":
        shown = rows
    elif display == "significant":
        shown = [r for r in rows if number(r["p.value"]) < alpha]
    else:
        assert display == "non-significant", display
        shown = [r for r in rows if number(r["p.value"]) >= alpha]
    expected = {
        (str(r["group1"]), str(r["group2"])): str(r["expression"]) for r in shown
    }
    if not expected:
        assert brackets == []
        return
    assert len(brackets) == 1
    stat = brackets[0]["stat"]
    annotations = stat["annotations"]
    if isinstance(annotations, str):  # R writes a single label as a scalar
        annotations = [annotations]
    drawn = dict(
        zip(
            (tuple(map(str, pair)) for pair in stat["comparisons"]),
            annotations,
            strict=True,
        )
    )
    assert drawn == expected


@pytest.mark.parametrize("case", [*COMPLETED, *_no_level_params()])
def test_bootstrap_reference_is_consistent(case: OracleCase) -> None:
    ref = case.bootstrap_reference
    k = len(case.analysis_levels)
    # Bootstrapped designs (SPEC.md section 9), and only when ggstatsplot computed the
    # effect size: without a subtitle (results.subtitle = FALSE) there is no CI.
    # gghistostats bootstraps only its robust test (trimcibt); ggscatterstats nothing.
    if case.function == "ggscatterstats":
        bootstrapped = False
    elif case.function == "gghistostats":
        bootstrapped = case.type == "robust"
    else:
        bootstrapped = (case.type == "robust") or (
            case.type == "nonparametric" and k > 2
        )
    # A subtitle ggstatsplot swallowed an error for has no effect size either
    # (edge_within_small, nonparametric).
    with_subtitle = (
        case.args.get("results.subtitle", True) is not False
        and "subtitle" not in case.expected_hidden_errors
    )
    if not (bootstrapped and with_subtitle):
        assert ref is None
        return
    assert ref is not None
    assert ref["resamples"] >= 20000
    # The reference runs at the case's own confidence level and trimming. Within
    # subjects, the robust effect sizes (dep.effect, wmcpAKP) take no confidence
    # level: statsExpressions passes none, and ggstatsplot reports 95% whatever the
    # case asks (requested_conf_level).
    requested = case.args.get("conf.level", 0.95)
    if case.function == "ggwithinstats" and case.type == "robust":
        assert ref["requested_conf_level"] == requested
        assert ref["conf_level"] == 0.95
        assert number(case.subtitle_data["conf.level"]) == 0.95
    else:
        assert ref["conf_level"] == requested
    tr = case.args.get("tr", 0.2)
    if case.type == "robust":
        assert f"tr = {tr:g}," in ref["call"], ref["call"]
    else:
        assert f"ci = {ref['conf_level']:g}," in ref["call"], ref["call"]
    reps = ref["replicates"]
    assert reps["resamples"] == 2000
    # The oracle replayed ggstatsplot's own bootstrap (set.seed, then the same call
    # with the same 100 resamples on the same rows) and reproduced its estimate and
    # both CI bounds exactly. That proves the reference call is the method, variant,
    # sidedness and alpha ggstatsplot used, which a tolerance check could not.
    assert ref["replay"]["matches"], (
        f"no candidate call reproduces ggstatsplot {ref['ggstatsplot']}: "
        f"{ref['replay']['candidates']}"
    )
    keys: tuple[str, ...] = ("estimate", "conf_low", "conf_high")
    if case.function == "gghistostats":
        # trimcibt: statsExpressions passes no nboot, so ggstatsplot uses WRS2's
        # documented default of 200; the replay with nboot left out and the one with
        # nboot = 200 both reproduce it, at alpha = 1 - conf.level as computed. Its
        # p-value is a bootstrap quantity too.
        keys = (*keys, "p_value")
        assert ref["ggstatsplot"]["resamples"] == 200
        candidates = ref["replay"]["candidates"]
        by_mode = {(c["replay_resamples"], c["call"]): c for c in candidates}
        computed = candidates[0]["call"]
        assert "alpha = 1 - " in computed, computed
        assert by_mode[("default", computed)]["matches"]
        assert by_mode[(200, computed)]["matches"]
        assert (
            by_mode[("default", computed)]["values"]
            == by_mode[(200, computed)]["values"]
        )
        assert f"nv = {case.args.get('test.value', 0):g}," in ref["call"], ref["call"]
    else:
        assert ref["ggstatsplot"]["resamples"] == 100
    estimate = number(ref["estimate"])
    low, high = number(ref["conf_low"]), number(ref["conf_high"])
    if math.isnan(low) or math.isnan(high):
        # Constant data (edge_one_constant): every resample's winsorized SD is 0,
        # trimcibt's bootstrap t is 0/0 and its interval NA, in ggstatsplot's run,
        # the reference and every replicate alike; the case declares the NA.
        assert case.function == "gghistostats"
        assert "subtitle" in case.raw["expected_na_values"]
        assert ref["ggstatsplot"]["conf_low"] is None
        keys = tuple(k for k in keys if not math.isnan(number(ref[k])))
    else:
        assert low <= estimate <= high
    # The 20,000-resample reference lies within the spread of 2,000-resample runs.
    for key in keys:
        assert_within_monte_carlo(
            number(ref[key]),
            number(reps[f"{key}_mean"]),
            number(reps[f"{key}_sd"]),
            what=f"{case.name} reference {key}",
        )


# Within subjects (ggwithinstats) ----------------------------------------------------


def _r_order_key(value: object) -> tuple[int, float, str]:
    """R's sort order of subject ids: numbers numerically, text in the C locale."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return (0, float(value), "")
    return (1, 0.0, str(value))


def _complete_subjects(case: OracleCase, frame: Any) -> dict[str, Any]:
    """statsExpressions::complete_subjects() as SPEC.md section 3 describes it, on any
    backend: rows without x or subject are dropped; the levels are those left
    (declared order, else sorted); a subject is kept only with a non-missing y in
    every level; subjects sort as R sorts them."""
    df = nw.from_native(frame, eager_only=True)
    x, y, subject = case.args["x"], case.args["y"], case.subject_id
    usable = df.filter(~nw.col(x).is_null() & ~nw.col(subject).is_null())
    raw_levels = set(usable[x].to_list())
    present = {r_level_label(v) for v in raw_levels}
    dtype = df.schema[x]
    if isinstance(dtype, nw.Enum | nw.Categorical):
        order = [str(v) for v in df[x].cat.get_categories().to_list()]
        levels = [v for v in order if v in present]
    else:
        levels = [r_level_label(v) for v in sorted(raw_levels, key=_r_order_key)]
    cells: dict[Any, dict[str, float | None]] = {}
    for sid, level, value in zip(
        usable[subject].to_list(), usable[x].to_list(), usable[y].to_list(), strict=True
    ):
        row = cells.setdefault(sid, {})
        key = r_level_label(level)
        assert key not in row, f"subject {sid!r} repeated in level {key!r}"
        if value is None or (isinstance(value, float) and math.isnan(value)):
            row[key] = None
        else:
            row[key] = float(value)
    complete = [
        sid
        for sid, row in cells.items()
        if all(row.get(lv) is not None for lv in levels)
    ]
    kept = sorted(complete, key=_r_order_key)
    dropped = sorted(set(cells) - set(complete), key=_r_order_key)
    return {
        "levels": levels,
        "subjects": [r_level_label(s) for s in kept],
        "values": [[cells[s][level] for level in levels] for s in kept],
        "dropped_subjects": [r_level_label(s) for s in dropped],
        "rows_without_x_or_subject": len(df) - len(usable),
    }


def _within_values(case: OracleCase) -> list[list[float]]:
    return [[number(v) for v in row] for row in case.within_data["values"]]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", WITHIN_ALL)
def test_backends_see_the_subjects_r_analyzed(
    case: OracleCase, backend: Backend
) -> None:
    # The subjects-by-conditions table ggstatsplot analyzed (within_data), rebuilt
    # from the CSV by subject id on every backend: the same levels, the subjects in
    # R's order, the same values and the same dropped subjects.
    data = case.within_data
    assert data["subject_id"] == case.subject_id
    mine = _complete_subjects(case, read_dataset(case, backend))
    assert mine["levels"] == list(case.analysis_levels) == list(data["levels"])
    assert mine["subjects"] == list(data["subjects"])
    assert mine["values"] == _within_values(case)
    assert mine["dropped_subjects"] == list(data["dropped_subjects"])
    assert mine["rows_without_x_or_subject"] == data["rows_without_x_or_subject"]
    # Every level counts each complete subject once.
    assert case.n_per_level == dict.fromkeys(
        case.analysis_levels, len(mine["subjects"])
    )


@pytest.mark.parametrize("case", WITHIN_ALL)
def test_declared_within_level_order_is_kept(case: OracleCase) -> None:
    declared = case.factor_levels.get(case.args["x"])
    if declared is None:
        assert list(case.analysis_levels) == sorted(case.analysis_levels)
    else:
        assert case.analysis_levels == tuple(
            level for level in declared if level in case.n_per_level
        )


# A message R prints once per process, when afex first loads lme4: the first
# parametric case with three or more conditions in each oracle shard carries it in
# `suppressed`, whichever dataset that case reads.
PACKAGE_LOAD_MESSAGE = "Registered S3 method overwritten by 'lme4'"


def _comparable(case: OracleCase) -> dict[str, Any]:
    """A fixture's outputs, without the fields that name its dataset, and without
    the one-time package-load message in what ggstatsplot muted."""
    keep = (
        "analysis_levels",
        "n_per_level",
        "within_data",
        "subtitle",
        "caption",
        "stats",
        "plot_layers",
        "bootstrap_reference",
    )
    out = {key: case.raw[key] for key in keep}
    out["suppressed"] = [
        {
            **entry,
            "messages": [
                m for m in entry["messages"] if not m.startswith(PACKAGE_LOAD_MESSAGE)
            ],
        }
        for entry in case.suppressed
    ]
    return out


@pytest.mark.parametrize(
    ("shuffled", "twin"),
    [
        pytest.param(f"{a}-{t}", f"{b}-{t}", id=f"{a}-{t}")
        for a, b in SHUFFLED_TWINS.items()
        for t in TYPES
    ],
)
def test_shuffled_rows_give_the_same_results(shuffled: str, twin: str) -> None:
    # ggwithinstats pairs by subject.id and sorts by it, so a random row order changes
    # nothing: every output, bootstrap included, equals the sorted twin's.
    by_name = {case.name: case for case in load_cases()}
    a = by_name[f"ggwithinstats-{shuffled}"]
    b = by_name[f"ggwithinstats-{twin}"]
    assert _comparable(a) == _comparable(b)


def test_probe_map_pins_the_anchor_threshold() -> None:
    # The base vector shifted by c: 1e5 and 1e8 bin widths, then just below and just
    # above 2^31 bin widths (STATUS, "MAP binning").
    rows = {row["name"]: row for row in probe("probe-map").rows("rows")}
    base = [0, 1, 2, 2, 3, 3, 3, 4, 5, 7, 9, 12]
    for shift in (1200, 1211988, 26027440, 26027507):
        row = rows[f"anchor_{shift}"]
        assert [r_double(v) for v in row["x_hex"]] == [v + shift for v in base]
        assert row["bw_sj"]["error"] is None
        assert row["map_estimate_hex"] == row["density_data_range"]["x_at_max_hex"]


def test_probe_signrank_covers_its_grid() -> None:
    pairs = probe("probe-signrank").rows("pairs")
    assert len(pairs) == 12
    for entry in pairs:
        assert len(entry["a_hex"]) == len(entry["b_hex"])
        # exact NULL, TRUE, FALSE x correct TRUE, FALSE x three alternatives.
        assert len(entry["wilcox"]) == 18
        assert len(entry["rank_biserial"]) == 6


def test_probe_pairedbf_covers_its_grid() -> None:
    rows = probe("probe-pairedbf").rows("rows")
    assert len(rows) == 4 * 19 * 2
    assert {row["n"] for row in rows} == {5, 10, 30, 200}
    assert all(row["call_error"] is None for row in rows)


def test_probe_friedman_covers_its_designs() -> None:
    designs = probe("probe-friedman").rows("designs")
    assert [d["name"] for d in designs][-2:] == ["tied_rows", "untied"]
    for design in designs:
        k = len(design["levels"])
        assert design["friedman_error"] is None, design["name"]
        assert design["durbin_error"] is None, design["name"]
        assert design["kendalls_w_error"] is None, design["name"]
        assert len(design["durbin_pairs"]) == k * (k - 1) // 2


def test_probe_rmanova_covers_its_designs() -> None:
    designs = probe("probe-rmanova").rows("designs")
    assert len(designs) == 9
    for design in designs:
        assert [t["correction"] for t in design["anova"]] == ["none", "GG", "HF"]
        assert all(t["error"] is None for t in design["anova"]), design["name"]
        assert design["tidy_model_parameters"], design["name"]


def test_probe_wrs2within_covers_its_grid() -> None:
    wrs2 = probe("probe-wrs2within")
    shifts = 28
    assert len(wrs2.rows("yuend")) == 3 * shifts
    rm = wrs2.rows("rmanova")
    assert len(rm) == 3 * shifts
    assert all(len(row["pairs"]) == 3 and len(row["rmmcp"]) == 3 for row in rm)
    assert len(wrs2.rows("dep_effect")) == 4 * 2
    assert len(wrs2.rows("wmcp_akp")) == 4 * 2


def test_probe_bootrules_covers_its_grid() -> None:
    # The rules themselves are tested in tests/stats/test_probe_bootrules.py.
    bootrules = probe("probe-bootrules")
    rules = {rule["name"]: rule for rule in bootrules.rows("rules")}
    assert list(rules) == [
        "akp.effect",
        "t1way",
        "rank_epsilon_squared",
        "dep.effect",
        "wmcpAKP",
        "kendalls_w",
        "trimcibt",
    ]
    for name, rule in rules.items():
        levels = 1 if name in ("dep.effect", "wmcpAKP") else 2
        assert len(rule["rows"]) == 6 * levels, name
        for row in rule["rows"]:
            assert row["error"] is None, (name, row["B"])
            assert row["unbalanced_traces"] == 0
            assert row["traced"] or row["boots"], (name, row["B"])
    undefined = [entry["name"] for entry in bootrules.rows("undefined")]
    assert undefined == [
        "edge_akp_undefined",
        "edge_akp_undefined2",
        "edge_tied3_small",
        "edge_rm_some_tied",
    ]


def test_probe_corr_covers_its_designs() -> None:
    # The rules are tested in tests/test_reference_vs_oracle.py.
    designs = probe("probe-corr").rows("designs")
    assert [d["name"] for d in designs] == [
        "mtcars",
        "iris",
        "small",
        "ties",
        "null",
        "outlier",
        "missing",
        "perfect",
        "perfect_neg",
    ]
    for design in designs:
        assert len(design["x_hex"]) == len(design["y_hex"]) == design["n"]
        assert len(design["pearson"]) == 2
        assert [s["exact"] for s in design["spearman"]] == ["NULL", True, False]
        assert len(design["correlation"]) == 4
        assert [w["tr"] for w in design["winsorized"]] == [0.2, 0.1]
        assert [b["prior"] for b in design["bayes"]][:3] == [0.707, 0.8, "medium"]


def test_probe_heldout_holds_the_held_out_cases() -> None:
    # Cases R completes but inferplot computes differently by a rule the lead has
    # yet to decide: each is a whole fixture, kept out of the parity tests.
    entries = {entry["name"]: entry for entry in probe("probe-heldout").rows("cases")}
    assert set(entries) == HELD_OUT_CASES
    for name, entry in entries.items():
        fixture = entry["fixture"]
        assert fixture["name"] == name
        assert fixture["error"] is None
        assert fixture["versions"]["packages"]["ggstatsplot"] == PARITY_VERSION
    # edge_rm_n3k4: with 3 subjects and 4 conditions afex computes no sphericity
    # correction and warns; ggstatsplot prints the uncorrected F(3, 6).
    n3k4 = entries["ggwithinstats-edge_rm_n3k4-parametric"]["fixture"]
    (row,) = n3k4["stats"]["subtitle_data"]
    assert (number(row["df"]), number(row["df.error"])) == (3.0, 6.0)
    # edge_akp_undefined: about 17% of the AKP resamples are 0/0, and WRS2's upper
    # bound is NA (positions counted over all B after the 0/0 values are dropped);
    # ggstatsplot prints CI [0.22, NA], at 100 resamples and in the 20,000-resample
    # reference alike, which the exact replay reproduces.
    akp = entries["ggbetweenstats-edge_akp_undefined-robust"]["fixture"]
    (row,) = akp["stats"]["subtitle_data"]
    assert row["conf.high"] is None
    assert '"NA" * "]"' in akp["subtitle"]
    reference = akp["bootstrap_reference"]
    assert reference["replay"]["matches"]
    assert reference["conf_high"] is None
    assert number(reference["conf_low"]) == number(row["conf.low"])
    (subtitle,) = [e for e in n3k4["suppressed"] if e["component"] == "subtitle"]
    assert any(
        "non-sphericity tests and corrections not available" in w
        for w in subtitle["warnings"]
    )


# One-sample and correlation designs (gghistostats, ggscatterstats) ------------------

# R arguments of these cases that inferplot has no keyword for (SPEC.md section 2):
# ggscatterstats' label.expression (which points get labels) and smooth.line.args
# (any geom_smooth method and its arguments; inferplot draws the OLS line).
NO_INFERPLOT_KEYWORD = frozenset({"label.expression", "smooth.line.args"})
# The numbers of a subtitle or caption row that generate.R checks for NA.
NUMBER_COLUMNS = ("statistic", "p.value", "estimate", "conf.low", "conf.high", "bf10")


def _hidden_calls(case: OracleCase) -> list[str]:
    """The hidden calls generate.R expects, in order, for these functions."""
    calls: list[str] = []
    if case.args.get("results.subtitle", True) is not False:
        calls.append("subtitle")
        if case.type == "parametric" and case.args.get("bf.message", True) is not False:
            calls.append("caption")
    if case.function == "ggscatterstats":
        if case.args.get("marginal", True) is not False:
            calls += ["xsidehistogram", "ysidehistogram"]
    elif case.args.get("centrality.plotting", True) is not False:
        calls.append("centrality_description")
    return calls


@pytest.mark.parametrize("case", _no_level_params())
def test_no_level_fixture_contents(case: OracleCase) -> None:
    versions = case.raw["versions"]
    assert versions["packages"]["ggstatsplot"] == PARITY_VERSION
    assert case.error is None, case.error
    # No levels: an empty level list and count map; the values are in analyzed_data.
    assert case.analysis_levels == ()
    assert case.n_per_level == {}
    assert [entry["component"] for entry in case.suppressed] == _hidden_calls(case)
    swallowed = {e["component"] for e in case.suppressed if e["error"] is not None}
    declared = set(case.expected_hidden_errors)
    assert swallowed == declared
    with_subtitle = case.args.get("results.subtitle", True) is not False
    if "caption" in _hidden_calls(case) and "caption" not in declared:
        assert case.caption, "no Bayes caption"
        assert case.caption_data is not None
    else:
        assert case.caption is None
        assert case.caption_data is None
    n = case.analyzed_data["n"]
    assert n == len(case.analyzed_values("x")) == len(case.analyzed_data["rows"])
    if not with_subtitle or "subtitle" in declared:
        assert case.subtitle is None
        assert case.subtitle_rows == []
    else:
        assert case.subtitle
        row = case.subtitle_data
        assert int(number(row["n.obs"])) == n
    # NA numbers (printed as "NA" or "") only where the case declares them: constant
    # x, and the Bayes factor at r = +-1.
    declared_na = set(case.raw["expected_na_values"])
    for table in ("subtitle", "caption"):
        rows = case.table(f"{table}_data")
        has_na = any(
            math.isnan(number(row[column]))
            for row in rows
            for column in NUMBER_COLUMNS
            if column in row
        )
        assert has_na == (table in declared_na), table
    args = {k: v for k, v in case.args.items() if k not in NO_INFERPLOT_KEYWORD}
    call = translate(case.function, args)
    assert call.function == case.function.removeprefix("gg")
    positional = ("x",) if case.function == "gghistostats" else ("x", "y")
    assert call.args == tuple(case.args[name] for name in positional)
    assert call.kwargs["type"] == case.type


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _present_rows(
    frame: Any, columns: Sequence[str]
) -> tuple[list[int], dict[str, list[float]]]:
    """1-based rows where every column is present, and those columns' values."""
    df = nw.from_native(frame, eager_only=True)
    lists = {column: df[column].to_list() for column in columns}
    keep = [
        i
        for i in range(len(df))
        if not any(_is_missing(lists[column][i]) for column in columns)
    ]
    values = {column: [float(lists[column][i]) for i in keep] for column in columns}
    return [i + 1 for i in keep], values


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", _no_level_all())
def test_backends_see_the_values_r_analyzed(case: OracleCase, backend: Backend) -> None:
    # gghistostats drops rows without x; ggscatterstats keeps the complete pairs. Each
    # backend reads the same rows, in file order, with the same values.
    data = case.analyzed_data
    columns = [case.args["x"]]
    if case.function == "ggscatterstats":
        columns.append(case.args["y"])
    rows, values = _present_rows(read_dataset(case, backend), columns)
    assert rows == list(data["rows"])
    assert values[case.args["x"]] == case.analyzed_values("x")
    if case.function == "ggscatterstats":
        assert values[case.args["y"]] == case.analyzed_values("y")
    assert data["rows_dropped"] == case.raw["dataset"]["rows"] - len(rows)


@pytest.mark.parametrize("case", oracle_params("gghistostats"))
def test_histogram_bins_and_centrality_line(case: OracleCase) -> None:
    # gghistostats' bin width: `binwidth`, else (max - min) / sqrt(n) of the analyzed
    # values (ggstatsplot's .binwidth()); stat_bin lays the bins edge to edge on the
    # grid (k + 1/2) w, and every value is counted once.
    x = case.analyzed_values("x")
    n = len(x)
    (params,) = case.layer_params("GeomBar")
    width = float(params["stat"]["binwidth"])
    given = case.args.get("binwidth")
    expected = float(given) if given is not None else (max(x) - min(x)) / math.sqrt(n)
    if given is None and max(x) == min(x):
        # No range (edge_one_constant, ten values of 5): ggstatsplot's .binwidth()
        # gives 0.1, one bin [4.95, 5.05] (not |x| / 1000 = 0.005).
        expected = 0.1
    assert math.isclose(width, expected, rel_tol=1e-13), (width, expected)
    (bins,) = case.layer_built("GeomBar")
    assert sum(int(b["count"]) for b in bins) == n
    for b in bins:
        assert math.isclose(number(b["xmax"]) - number(b["xmin"]), width, rel_tol=1e-9)
        k = number(b["xmin"]) / width - 0.5
        assert abs(k - round(k)) < 1e-6, (b["xmin"], width)
    assert number(bins[0]["xmin"]) <= min(x)
    assert max(x) <= number(bins[-1]["xmax"])
    # inferplot's own bins (tests/stats/test_plot_geometry.py checks every edge).
    assert list(histogram_bins(x, binwidth=given).counts) == [
        int(b["count"]) for b in bins
    ]
    # The centrality line sits at the value centrality_description() labels: the
    # mean, median, trimmed mean (at the case's tr) or MAP.
    centrality = case.centrality
    lines = case.layer_built("GeomVline")
    if case.args.get("centrality.plotting", True) is False:
        assert centrality is None
        assert lines == []
        return
    assert centrality is not None
    (row,) = centrality
    assert int(row["n.obs"]) == n
    ((line,),) = lines
    assert number(line["xintercept"]) == number(row["var"])
    values = sorted(x)
    if case.type == "parametric":
        assert math.isclose(number(row["var"]), math.fsum(x) / n, rel_tol=1e-14)
    elif case.type == "nonparametric":
        middle = (values[(n - 1) // 2] + values[n // 2]) / 2
        assert math.isclose(number(row["var"]), middle, rel_tol=1e-14)
    elif case.type == "robust":
        g = math.floor(float(case.args.get("tr", 0.2)) * n)
        trimmed = values[g : n - g]
        assert math.isclose(
            number(row["var"]), math.fsum(trimmed) / len(trimmed), rel_tol=1e-14
        )
    kind = {
        "parametric": "mean",
        "nonparametric": "median",
        "robust": "trimmed",
        "bayes": "MAP",
    }[case.type]
    expression = str(row["expression"])
    assert expression.startswith(f"list(widehat(mu)[{kind}] == "), expression


@pytest.mark.parametrize("case", oracle_params("ggscatterstats"))
def test_scatter_smooth_band_and_marginals(case: OracleCase) -> None:
    x = np.array(case.analyzed_values("x"))
    y = np.array(case.analyzed_values("y"))
    n = x.size
    empty = set(case.raw["expected_empty_layers"])
    (params,) = case.layer_params("GeomSmooth")
    level = float(case.args.get("conf.level", 0.95))
    assert float(params["stat"]["level"]) == level
    smooth = case.args.get("smooth.line.args")
    method = "lm" if smooth is None else smooth["method"]
    assert params["stat"]["method"] == method
    (built,) = case.layer_built("GeomSmooth")
    if "GeomSmooth" in empty:
        assert built == []
    else:
        assert len(built) == 80
        assert number(built[0]["x"]) == x.min()
        assert number(built[-1]["x"]) == x.max()
        if method == "lm":
            _check_ols_band(built, x, y, level)
            band = ols_band(x, y, conf_level=level)
            for row, fit in zip(built, band.fit, strict=True):
                assert math.isclose(number(row["y"]), fit, rel_tol=1e-9, abs_tol=1e-9)
    # Marginal histograms: 30 bins of each variable (one of width 0.1 for a constant
    # one, edge_corr_constant_x), every value counted once.
    for geom, values in {"GeomXsidebar": x, "GeomYsidebar": y}.items():
        layers = [layer["built"] for layer in case.plot_layers if layer["geom"] == geom]
        if case.args.get("marginal", True) is False:
            assert layers == []
            continue
        (side,) = layers
        assert len(side) == (1 if values.min() == values.max() else 30)
        assert sum(int(b["count"]) for b in side) == values.size
        assert list(marginal_bins(values).counts) == [int(b["count"]) for b in side]
    # Point labels: every analyzed point, or those label.expression selects.
    labelled = [
        layer for layer in case.plot_layers if layer["geom"] == "GeomLabelRepel"
    ]
    label = case.args.get("label.var")
    if label is None:
        assert labelled == []
        return
    (layer,) = labelled
    rows = layer["data"]
    assert all(label in row for row in rows)
    if "label.expression" not in case.args:
        assert len(rows) == n


def _check_ols_band(built: list[Row], x: Any, y: Any, level: float) -> None:
    """The OLS line and its pointwise confidence band at ``level``: fit -/+
    t(1 - (1 - level) / 2, n - 2) x the fit's standard error."""
    n = x.size
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    s2 = float(residual @ residual) / (n - 2)
    centre = float(x.mean())
    sxx = float(((x - centre) ** 2).sum())
    q = float(sp_stats.t.ppf(1 - (1 - level) / 2, n - 2))
    for row in built:
        at = number(row["x"])
        fit = intercept + slope * at
        se = math.sqrt(s2 * (1 / n + (at - centre) ** 2 / sxx))
        assert math.isclose(number(row["y"]), fit, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(number(row["se"]), se, rel_tol=1e-8, abs_tol=1e-12)
        assert math.isclose(
            number(row["ymin"]), fit - q * se, rel_tol=1e-8, abs_tol=1e-9
        )
        assert math.isclose(
            number(row["ymax"]), fit + q * se, rel_tol=1e-8, abs_tol=1e-9
        )


@pytest.mark.parametrize("case", oracle_params("ggscatterstats"))
def test_correlation_row_replays_the_subtitle(case: OracleCase) -> None:
    # correlation::correlation() as corr_test() calls it, after set.seed(seed): its
    # row is ggstatsplot's subtitle row (the Bayes type's posterior too).
    record = case.correlation
    method = "spearman" if case.type == "nonparametric" else "pearson"
    assert f'method = "{method}"' in record["call"]
    assert f"bayesian = {'TRUE' if case.type == 'bayes' else 'FALSE'}" in record["call"]
    tr = case.args.get("tr", 0.2)
    winsorize = f"winsorize = {tr:g}" if case.type == "robust" else "winsorize = FALSE"
    assert winsorize in record["call"]
    if case.subtitle_rows:
        assert record["matches_subtitle"] is True
    if record["error"] is None:
        (row,) = record["value"]
        assert int(row["n_Obs"]) == case.analyzed_data["n"]


# Contingency tables (ggbarstats, ggpiestats) ----------------------------------------

BAR = ("ggbarstats", "ggpiestats")


def _bar_all() -> list[Any]:
    """Every ggbarstats and ggpiestats case, failed ones included."""
    cases = require_cases(
        [c for c in load_cases() if c.function in BAR], "ggbarstats or ggpiestats"
    )
    return [pytest.param(case, id=case.name) for case in cases]


def _bar_completed() -> list[Any]:
    return [param for function in BAR for param in oracle_params(function)]


def _bar_design(case: OracleCase) -> tuple[int, int, int]:
    """x's levels, y's levels (0 without y) and the levels of y that hold data, as
    ggstatsplot's .pie_bar_data_prep() leaves them."""
    data = case.contingency_data
    table = case.counts_table()
    y_levels = len(data["y_levels"])
    held = sum(1 for j in range(y_levels) if sum(row[j] for row in table) > 0)
    return len(data["x_levels"]), y_levels, held


def _bar_hidden_calls(case: OracleCase) -> list[str]:
    """The hidden calls of ggbarstats and ggpiestats 1.1.4, in order: with
    results.subtitle the subtitle; the Bayes caption when bf.message holds, the type
    is not bayes, paired is FALSE and y does not have exactly one level (so also
    without y); the pairwise Fisher tests when x has more than two levels, y more
    than one and paired is FALSE; then, with y, one proportion test per level of y
    that holds data, whatever results.subtitle and proportion.test say."""
    x_levels, y_levels, held = _bar_design(case)
    paired = case.args.get("paired", False) is True
    calls: list[str] = []
    if case.args.get("results.subtitle", True) is not False:
        calls.append("subtitle")
        if (
            case.args.get("bf.message", True) is not False
            and case.type != "bayes"
            and not paired
            and y_levels != 1
        ):
            calls.append("caption")
        if x_levels > 2 and y_levels > 1 and not paired:
            calls.append("pairwise_comparisons")
    return calls + ["proportion_test"] * held


def _bar_proportion_test(case: OracleCase) -> bool:
    """Whether the per-bar proportion tests are drawn (ggbarstats: never for type
    bayes or with fewer than two levels of x or of y; ggpiestats: never for type
    bayes or one level of x with several pies)."""
    x_levels, y_levels, _ = _bar_design(case)
    wanted = case.args.get(
        "proportion.test", case.args.get("results.subtitle", True) is not False
    )
    if case.function == "ggbarstats":
        off = case.type == "bayes" or x_levels < 2 or y_levels < 2
    else:
        off = case.type == "bayes" or (x_levels == 1 and y_levels > 1)
    return bool(wanted) and not off


@pytest.mark.parametrize("case", _bar_completed())
def test_bar_fixture_contents(case: OracleCase) -> None:
    versions = case.raw["versions"]
    assert versions["packages"]["ggstatsplot"] == PARITY_VERSION
    data = case.contingency_data
    assert data["error"] is None
    calls = _bar_hidden_calls(case)
    assert [entry["component"] for entry in case.suppressed] == calls
    # The per-bar tests run in the order of y's levels that hold data.
    table = case.counts_table()
    held = [
        str(level)
        for j, level in enumerate(data["y_levels"])
        if sum(row[j] for row in table) > 0
    ]
    tests = [e for e in case.suppressed if e["component"] == "proportion_test"]
    assert [str(e["level"]) for e in tests] == held
    swallowed = {e["component"] for e in case.suppressed if e["error"] is not None}
    declared = set(case.expected_hidden_errors)
    assert swallowed == declared
    nothing = set(case.raw["expected_null_outputs"])
    if "subtitle" in calls and not {"subtitle"} & (declared | nothing):
        assert case.subtitle
        for row in case.subtitle_rows:
            # The one-way Bayes factor's row has no n.
            if "n.obs" in row:
                assert int(number(row["n.obs"])) == data["n"]
    else:
        assert case.subtitle is None
        assert case.subtitle_rows == []
    if "caption" in calls and not {"caption"} & (declared | nothing):
        assert case.caption
        assert case.caption_data is not None
    else:
        assert case.caption is None
    if "pairwise_comparisons" in calls and "pairwise_comparisons" not in declared:
        assert case.pairwise
    else:
        assert case.pairwise == []
    # The counts ggstatsplot labels are the recorded table, cell by cell.
    x, y = data["x"], data["y"]
    x_levels = [str(v) for v in data["x_levels"]]
    y_levels = [str(v) for v in data["y_levels"]]
    for row in case.table("descriptive_data"):
        i = x_levels.index(str(row[x]))
        j = 0 if y is None else y_levels.index(str(row[y]))
        assert int(row["counts"]) == table[i][j]
    assert sum(map(sum, table)) == data["n"]
    # NA numbers only where the case declares them.
    declared_na = set(case.raw["expected_na_values"])
    tables = {
        "subtitle": "subtitle_data",
        "caption": "caption_data",
        "pairwise": "pairwise_comparisons_data",
        "one_sample": "one_sample_data",
    }
    for kind, name in tables.items():
        rows = case.table(name)
        has_na = any(
            math.isnan(number(row[column]))
            for row in rows
            for column in (*NUMBER_COLUMNS, "p.value.adj")
            if column in row
        )
        assert has_na == (kind in declared_na), kind
    # The layers the arguments imply.
    geoms = [layer["geom"] for layer in case.plot_layers]
    drawn = _bar_proportion_test(case)
    if case.function == "ggbarstats":
        expected = ["GeomCol", "GeomLabel"]
        expected += (
            (["GeomText"] if drawn else []) + ["GeomText"] if y else ["GeomText"]
        )
    else:
        expected = ["GeomCol", "GeomLabel"] + (
            ["GeomText"] if len(y_levels) > 1 and drawn else []
        )
    assert geoms == expected
    # The texts: the fill legend names x; ggbarstats' x axis names y (no title
    # without y); the y axis has none.
    texts = case.plot_texts
    assert texts["legend_title"] == x
    labels = texts["labels"]
    assert labels["fill"] == x
    assert labels["y"] is None
    assert labels["x"] == (y if case.function == "ggbarstats" and y else None)
    call = translate(case.function, case.args)
    assert call.function == "barstats"
    assert call.args == tuple(case.args[k] for k in ("x", "y") if k in case.args)
    assert call.kwargs["type"] == case.type


def _r_sort_key(value: object) -> tuple[int, float, str]:
    """R's order of factor levels made by as.factor(): numbers numerically, text in
    the C locale."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return (0, float(value), "")
    return (1, 0.0, str(value))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", _bar_all())
def test_backends_see_the_table_r_analyzed(case: OracleCase, backend: Backend) -> None:
    # Rows with x, y or the counts column missing are dropped, each row counts its
    # weight (1 without counts), and the levels are the declared ones (empty ones
    # included) or the sorted values that remain: the same table on every backend.
    data = case.contingency_data
    columns = [data[k] for k in ("x", "y", "counts") if data[k] is not None]
    frame = nw.from_native(read_dataset(case, backend), eager_only=True)
    lists = {column: frame[column].to_list() for column in columns}
    keep = [
        i
        for i in range(len(frame))
        if not any(_is_missing(lists[column][i]) for column in columns)
    ]
    assert [i + 1 for i in keep] == list(data["rows"])
    assert data["rows_dropped"] == len(frame) - len(keep)
    weights = [1.0] * len(keep)
    if data["counts"] is not None:
        weights = [float(lists[data["counts"]][i]) for i in keep]
    if data["error"] is not None:
        # tidyr::uncount() refuses a weight that is not whole.
        assert any(not w.is_integer() for w in weights)
        assert "loss of precision" in data["error"]
        return

    def levels(column: str) -> list[str]:
        declared = case.factor_levels.get(column)
        if declared is not None:
            return list(declared)
        present = {
            lists[column][i] for i, w in zip(keep, weights, strict=True) if w > 0
        }
        return [r_level_label(v) for v in sorted(present, key=_r_sort_key)]

    x_levels = levels(data["x"])
    y_levels = levels(data["y"]) if data["y"] is not None else []
    assert x_levels == [str(v) for v in data["x_levels"]]
    assert y_levels == [str(v) for v in data["y_levels"]]
    table = [[0] * max(1, len(y_levels)) for _ in x_levels]
    for i, w in zip(keep, weights, strict=True):
        if w == 0:  # uncount() drops the row
            continue
        row = x_levels.index(r_level_label(lists[data["x"]][i]))
        column = (
            0
            if data["y"] is None
            else y_levels.index(r_level_label(lists[data["y"]][i]))
        )
        table[row][column] += int(w)
    assert table == case.counts_table()
    assert sum(map(sum, table)) == data["n"]
    assert case.n_per_level == dict(zip(x_levels, map(sum, table), strict=True))


def _r_round_text(value: float, digits: int) -> str:
    """paste0(round(value, digits)): R's round() (probe-contingency's 927 labels:
    Python's round() on the same double, ties to even on the exact binary value)
    printed with up to 15 significant digits, no trailing zeros."""
    return format(round(value, digits), ".15g")


@pytest.mark.parametrize("case", _bar_completed())
def test_bar_labels(case: OracleCase) -> None:
    # The label inside each cell: the percentage of its bar ((counts / bar total) x
    # 100, rounded to digits.perc as R rounds, no trailing zeros: "12.5%", "33%",
    # "6.2%"), the count (with a thousands comma) or both ("3\n(16%)"); a zero cell
    # has no label (NA). Percentages of a bar need not sum to 100 (three thirds:
    # 33% each).
    data = case.contingency_data
    x, y = data["x"], data["y"]
    table = case.counts_table()
    x_levels = [str(v) for v in data["x_levels"]]
    y_levels = [str(v) for v in data["y_levels"]]
    digits = int(case.args.get("digits.perc", 0))
    kind = case.args.get("label", "percentage")
    labels: list[str] = []
    for row in case.table("descriptive_data"):
        j = 0 if y is None else y_levels.index(str(row[y]))
        count = table[x_levels.index(str(row[x]))][j]
        total = sum(r[j] for r in table)
        perc = (count / total) * 100
        assert math.isclose(number(row["perc"]), perc, rel_tol=1e-14, abs_tol=0)
        if count == 0:
            assert row[".label"] is None
            continue
        percent = f"{_r_round_text(perc, digits)}%"
        expected = {
            "percentage": percent,
            "counts": f"{count:,}",
            "both": f"{count:,}\n({percent})",
        }[kind]
        assert row[".label"] == expected
        labels.append(expected)
    (built,) = case.layer_built("GeomLabel")
    assert sorted(str(b["label"]) for b in built if b["label"] is not None) == sorted(
        labels
    )
    # Under each bar its n; ggbarstats without y: under the single bar.
    texts = [
        layer["built"] for layer in case.plot_layers if layer["geom"] == "GeomText"
    ]
    if case.function == "ggbarstats":
        if y is None:
            assert [str(b["label"]) for b in texts[-1]] == [f"(n = {data['n']:,})"]
        else:
            totals = {
                level: sum(r[j] for r in table) for j, level in enumerate(y_levels)
            }
            expected_n = {f"(n = {n:,})" for n in totals.values() if n > 0}
            assert {str(b["label"]) for b in texts[-1]} == expected_n
    pies = case.function == "ggpiestats" and len(y_levels) > 1
    if _bar_proportion_test(case) and (case.function == "ggbarstats" or pies):
        # The proportion tests' labels: p alone above each bar (ggbarstats), the
        # whole goodness-of-fit result beside each pie (ggpiestats); no stars.
        column = ".p.label" if case.function == "ggbarstats" else ".label"
        drawn = {str(b["label"]) for b in texts[0]}
        assert drawn == {str(row[column]) for row in case.one_sample}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=c.name)
        for c in load_cases()
        if c.function in BAR and c.error is not None
    ],
)
def test_bar_failed_case(case: OracleCase) -> None:
    # ggstatsplot failed while preparing the data: no hidden call ran, and the
    # generator's own preparation (ggstatsplot's function) failed the same way.
    expected = case.expected_error
    assert expected is not None
    assert expected["component"] == "data"
    assert case.error is not None
    assert expected["message"] in case.error
    assert case.suppressed == []
    assert case.raw["partial"] == []
    assert case.contingency_data["error"] == case.error
    assert case.raw["stats"] is None


def test_probe_contingency_covers_its_grid() -> None:
    # The rules are tested in tests/test_reference_vs_oracle.py; the percent labels
    # here: paste0(round(perc, d), "%") is Python's round() on the same double
    # (ties to even on the exact binary value: 12.5 -> 12, 6.25 -> 6.2, 31.25 ->
    # 31.2), printed with up to 15 significant digits.
    contingency = probe("probe-contingency")
    assert len(contingency.rows("chisq")) == 22
    assert len(contingency.rows("goodness_of_fit")) == 13
    assert len(contingency.rows("cramers_v")) == 22
    assert all(len(entry["rows"]) == 12 for entry in contingency.rows("cramers_v"))
    assert len(contingency.rows("paired")) == 12
    assert len(contingency.rows("fisher")) == 19
    labels = contingency.rows("percent_labels")
    assert len(labels) == 927
    for row in labels:
        perc = r_double(str(row["perc_hex"]))
        assert perc is not None
        assert perc == (row["k"] / row["n"]) * 100
        rounded = round(perc, row["digits"])
        assert r_double(str(row["rounded_hex"])) == rounded
        assert row["label"] == f"{_r_round_text(perc, row['digits'])}%"
    ties = {row["label"] for row in labels if row["n"] in (8, 16)}
    assert {"12%", "38%", "6.2%", "31.2%", "6.25%"} <= ties


def test_pie_and_bar_compute_the_same() -> None:
    # ggpiestats' one-way cases next to ggbarstats': the same tables, statistics and
    # labels (the one-way Bayes factor is a Monte Carlo estimate from the same seed).
    by_name = {case.name: case for case in load_cases()}
    pies = [name for name in by_name if name.startswith("ggpiestats-")]
    assert len(pies) == 5
    for name in pies:
        pie, bar = by_name[name], by_name[name.replace("ggpiestats-", "ggbarstats-", 1)]
        assert pie.contingency_data == bar.contingency_data
        for table in ("subtitle_data", "caption_data", "descriptive_data"):
            assert pie.table(table) == bar.table(table), (name, table)
