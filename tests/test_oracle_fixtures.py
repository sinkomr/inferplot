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
import pytest

from tests.oracle_args import translate
from tests.oracle_fixtures import (
    PARITY_VERSION,
    Backend,
    OracleCase,
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
)
# Cases where ggstatsplot 1.1.4 itself fails, as generate.R declares them.
EXPECTED_FAILURES = frozenset(
    {
        "ggbetweenstats-edge_n2-parametric",
        "ggbetweenstats-edge_n2-robust",
        "ggbetweenstats-edge_n2-bayes",
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
    assert all(entry["error"] is None for entry in before)
    assert case.raw["subtitle"] is None
    assert case.raw["stats"] is None
    assert case.raw["plot_layers"] is None
    for entry in before:
        value = case.partial(entry["component"])
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


@pytest.mark.parametrize("case", COMPLETED)
def test_bootstrap_reference_is_consistent(case: OracleCase) -> None:
    ref = case.bootstrap_reference
    k = len(case.analysis_levels)
    # Bootstrapped designs (SPEC.md section 9), and only when ggstatsplot computed the
    # effect size: without a subtitle (results.subtitle = FALSE) there is no CI.
    bootstrapped = (case.type == "robust") or (case.type == "nonparametric" and k > 2)
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
    assert ref["ggstatsplot"]["resamples"] == 100
    estimate = number(ref["estimate"])
    low, high = number(ref["conf_low"]), number(ref["conf_high"])
    assert low <= estimate <= high
    # The 20,000-resample reference lies within the spread of 2,000-resample runs.
    for key in ("estimate", "conf_low", "conf_high"):
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
