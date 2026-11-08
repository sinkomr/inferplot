"""The committed oracle fixtures are intact, complete, and read correctly.

These tests check the fixtures themselves, not inferplot: hashes against the manifest,
datasets and generator files, the ggstatsplot version, the case list, what
ggstatsplot hid, and that every supported dataframe backend sees the same groups and
counts that R analyzed.
"""

from collections.abc import Sequence
from typing import Any

import narwhals.stable.v2 as nw
import pytest

from tests.oracle_args import translate
from tests.oracle_fixtures import (
    PARITY_VERSION,
    Backend,
    OracleCase,
    integrity_problems,
    load_cases,
    number,
    oracle_params,
    r_level_label,
    read_dataset,
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
EXPECTED_CASES = frozenset(
    {f"ggbetweenstats-{label}-{t}" for label in ALL_TYPE_LABELS for t in TYPES}
    | {"ggbetweenstats-mtcars-am-drat-parametric"}
    | {
        f"ggbetweenstats-ToothGrowth-{side}-{t}"
        for side in ("greater", "less")
        for t in ("parametric", "nonparametric")
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


@pytest.mark.parametrize("case", oracle_params("ggbetweenstats"))
def test_between_fixture_contents(case: OracleCase) -> None:
    versions = case.raw["versions"]
    assert versions["packages"]["ggstatsplot"] == PARITY_VERSION
    assert versions["R"].startswith("4.5.")
    assert case.error is None, case.error
    assert case.subtitle, "ggstatsplot drew no subtitle"

    # What ggstatsplot hid (SPEC.md section 9), one entry per hidden call in call
    # order; no call swallowed an error unless the case says so. ggstatsplot's
    # bf.condition draws the Bayes caption only for type "parametric".
    with_caption = case.type == "parametric" and case.args.get("bf.message", True)
    expected_calls = ["subtitle"]
    if with_caption:
        expected_calls.append("caption")
    if case.args.get("centrality.plotting", True):
        expected_calls.append("centrality_description")
    if (
        len(case.analysis_levels) > 2
        and case.args.get("pairwise.display", "significant") != "none"
    ):
        expected_calls.append("pairwise_comparisons")
    assert [entry["component"] for entry in case.suppressed] == expected_calls
    swallowed = {
        entry["component"]: entry["error"]
        for entry in case.suppressed
        if entry["error"] is not None
    }
    assert set(swallowed) == set(case.expected_hidden_errors), swallowed
    if with_caption:
        assert case.caption, "no Bayes caption"
        assert case.caption_data is not None, "no caption_data"
    else:
        assert case.caption is None
        assert case.caption_data is None

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
    assert call.function == "betweenstats"
    assert call.args == (case.args["x"], case.args["y"])
    assert call.kwargs["type"] == case.type


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
@pytest.mark.parametrize("case", oracle_params("ggbetweenstats"))
def test_backends_see_the_groups_r_analyzed(case: OracleCase, backend: Backend) -> None:
    frame = read_dataset(case, backend)
    levels, counts = _levels_and_counts(frame, case.args["x"], case.args["y"])
    assert tuple(levels) == case.analysis_levels
    assert counts == case.n_per_level


@pytest.mark.parametrize("case", oracle_params("ggbetweenstats"))
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


@pytest.mark.parametrize("case", oracle_params("ggbetweenstats"))
def test_brackets_are_the_displayed_pairs(case: OracleCase) -> None:
    # SPEC.md section 4: brackets exist only for three or more groups; by default
    # they mark the pairs whose adjusted p is strictly below 0.05, and every pair
    # for type "bayes". Each label is the pair's expression from the pairwise table.
    brackets = case.layer_params("GeomSignif")
    if len(case.analysis_levels) == 2:
        assert brackets == []
        return
    rows = case.pairwise
    shown = (
        rows
        if case.type == "bayes"
        else [r for r in rows if number(r["p.value"]) < 0.05]
    )
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


@pytest.mark.parametrize("case", oracle_params("ggbetweenstats"))
def test_bootstrap_reference_is_consistent(case: OracleCase) -> None:
    ref = case.bootstrap_reference
    k = len(case.analysis_levels)
    expects_reference = (case.type == "robust") or (
        case.type == "nonparametric" and k > 2
    )
    if not expects_reference:
        assert ref is None
        return
    assert ref is not None
    assert ref["resamples"] >= 20000
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
