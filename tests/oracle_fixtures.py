"""Load and verify the R oracle fixtures in ``tests/fixtures/oracle/``.

``oracle/generate.R`` writes the fixtures in GitHub Actions; they are never edited by
hand. Loading checks that every fixture's SHA-256 matches ``manifest.json``, that the
CSV each fixture recorded still has the same SHA-256, and that the generator files
are the ones that wrote the fixtures, so a fixture can never drift silently from the
data the Python tests read or from the committed generator. The schema is described
in ``docs/design/SPEC.md`` section 9.

Probe files (``probe-*.json``) record what R functions behind ggstatsplot return on
fixed grids (number formatting, ``p.adjust``, ``ptukey``, the omega squared CI). They
are hashed in the manifest and checked the same way, and are read through
``load_probes()`` and ``probe()``, never as cases.

Typical use in a parity test::

    @pytest.mark.parametrize("case", oracle_params("ggbetweenstats"))
    def test_parity(case: OracleCase) -> None:
        frame = read_dataset(case)
        ...
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "oracle"
DATASET_DIR = ROOT / "oracle" / "datasets"
MANIFEST = "manifest.json"
PROBE_PREFIX = "probe-"
SCHEMA_VERSION = 2
PARITY_VERSION = "1.1.4"
# Files that decide what the fixtures contain (repository-relative). The manifest
# records their SHA-256 when the oracle runs.
GENERATOR_FILES = (
    "oracle/generate.R",
    "oracle/export_datasets.R",
    ".github/workflows/oracle.yml",
)

Backend = Literal["pandas", "polars", "pyarrow"]
Row = Mapping[str, Any]
# R functions whose design has no groups: a one-sample test and a correlation. Their
# fixtures record the analyzed values (`analyzed_data`) instead of levels.
NO_LEVEL_FUNCTIONS = frozenset({"gghistostats", "ggscatterstats"})


class FixtureIntegrityError(AssertionError):
    """The fixture directory does not match its manifest or its datasets."""


def number(value: object) -> float:
    """Convert a fixture number to float.

    R's NA is stored as JSON null and becomes NaN here; NaN, Inf and -Inf are
    stored as the strings "NaN", "Inf" and "-Inf".
    """
    if value is None:
        return math.nan
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got the boolean {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value in {"NaN", "Inf", "-Inf"}:
        return float(value)
    raise TypeError(f"expected a number, got {value!r}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class OracleCase:
    """One fixture: a single ggstatsplot call and everything it reported."""

    name: str
    path: Path
    raw: Mapping[str, Any]

    def __repr__(self) -> str:  # short ids in pytest output
        return f"OracleCase({self.name!r})"

    @property
    def function(self) -> str:
        return str(self.raw["function"])

    @property
    def args(self) -> Mapping[str, Any]:
        return self.raw["args"]

    @property
    def type(self) -> str:
        return str(self.args["type"])

    @property
    def dataset_file(self) -> str:
        return str(self.raw["dataset"]["file"])

    @property
    def factor_levels(self) -> Mapping[str, Sequence[str]]:
        return self.raw["dataset"]["factor_levels"] or {}

    @property
    def analysis_levels(self) -> tuple[str, ...]:
        return tuple(self.raw["analysis_levels"])

    @property
    def n_per_level(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.raw["n_per_level"].items()}

    @property
    def subject_id(self) -> str:
        """The subject column of a ggwithinstats case; other functions raise."""
        if self.function != "ggwithinstats":
            raise ValueError(f"{self.name}: {self.function} has no subject column")
        return str(self.args["subject.id"])

    @property
    def within_data(self) -> Mapping[str, Any]:
        """The subjects-by-conditions data a ggwithinstats case analyzed.

        As ``statsExpressions::complete_subjects()`` left it for ggwithinstats:
        ``subject_id`` (the subject column), ``levels``, ``subjects`` (as R prints
        them, in the order ggstatsplot holds them: numbers numerically, text in the
        C locale), ``values`` (one row per subject, one column per level),
        ``dropped_subjects`` (subjects with a row, x and subject present, left out
        for a missing level or value) and ``rows_without_x_or_subject``.
        """
        if self.function != "ggwithinstats":
            raise ValueError(f"{self.name}: only ggwithinstats records within_data")
        return self.raw["within_data"]

    @property
    def analyzed_data(self) -> Mapping[str, Any]:
        """The values a gghistostats or ggscatterstats case analyzed.

        These designs have no levels (``analysis_levels`` is empty and
        ``n_per_level`` is ``{}``). gghistostats: ``x`` (the column), ``rows`` (the
        1-based CSV rows kept, in file order: rows with x present), ``x_values``,
        ``n`` and ``rows_dropped``. ggscatterstats: the same with ``y`` and
        ``y_values``, for the complete pairs.
        """
        if self.function not in NO_LEVEL_FUNCTIONS:
            raise ValueError(
                f"{self.name}: only {sorted(NO_LEVEL_FUNCTIONS)} record analyzed_data"
            )
        return self.raw["analyzed_data"]

    def analyzed_values(self, column: Literal["x", "y"] = "x") -> list[float]:
        """The analyzed values of x (or, for a correlation, y) as floats."""
        return [number(v) for v in self.analyzed_data[f"{column}_values"]]

    @property
    def centrality(self) -> list[Row] | None:
        """gghistostats: what ``centrality_description()`` returned (the line's
        value, its label ``expression`` and ``n.obs``); None without the line."""
        if self.function != "gghistostats":
            raise ValueError(f"{self.name}: only gghistostats records centrality")
        rows = self.raw["centrality"]
        return None if rows is None else list(rows)

    @property
    def correlation(self) -> Mapping[str, Any]:
        """ggscatterstats: ``correlation::correlation()`` as ``corr_test()`` calls
        it, replayed after ``set.seed(seed)``: ``call``, ``value`` (its row),
        ``error``, ``warnings``, ``messages`` and ``matches_subtitle``."""
        if self.function != "ggscatterstats":
            raise ValueError(f"{self.name}: only ggscatterstats records correlation")
        return self.raw["correlation"]

    def within_columns(self) -> dict[str, list[float]]:
        """Each level's values in subject order, as floats (``within_data``)."""
        data = self.within_data
        levels = [str(level) for level in data["levels"]]
        return {
            level: [number(row[j]) for row in data["values"]]
            for j, level in enumerate(levels)
        }

    @property
    def error(self) -> str | None:
        return self.raw["error"]

    @property
    def expected_error(self) -> Mapping[str, str] | None:
        """The failure the case declares: ``component``, ``message``, ``reason``.

        Null unless ggstatsplot itself fails on the case; the generator stops on any
        failure a case does not declare.
        """
        return self.raw["expected_error"]

    @property
    def partial_components(self) -> tuple[str, ...]:
        """The hidden calls that ran before ggstatsplot failed, in call order.

        A Bayes caption is computed for type parametric only, so the other types'
        failed cases have no ``caption`` entry.
        """
        entries = self.raw["partial"]
        if entries is None:
            raise ValueError(f"{self.name}: ggstatsplot did not fail; read its outputs")
        return tuple(str(entry["component"]) for entry in entries)

    def partial(self, component: str) -> Any:
        """What a hidden call returned before ggstatsplot failed (failed cases only).

        ``component`` is ``subtitle``, ``caption`` or ``centrality_description``
        for the hidden calls that completed; the value is in the fixture's JSON form
        (a data frame as row objects). The failing call (``expected_error``'s
        component, ``pairwise_comparisons`` in ``edge_n2``) appears last, with value
        None.
        """
        entries = self.raw["partial"]
        if entries is None:
            raise ValueError(f"{self.name}: ggstatsplot did not fail; read its outputs")
        for entry in entries:
            if entry["component"] == component:
                return entry["value"]
        known = [entry["component"] for entry in entries]
        raise KeyError(f"{self.name}: no partial output {component!r}; have {known}")

    @property
    def subtitle(self) -> str | None:
        return self.raw["subtitle"]

    @property
    def caption(self) -> str | None:
        return self.raw["caption"]

    @property
    def bootstrap_reference(self) -> Mapping[str, Any] | None:
        return self.raw["bootstrap_reference"]

    @property
    def suppressed(self) -> list[Row]:
        """What ggstatsplot hid, one entry per hidden call, in call order.

        Each entry has ``component`` (``subtitle``, ``caption``,
        ``centrality_description`` or ``pairwise_comparisons``), ``error`` (null
        unless ggstatsplot swallowed one) and the muted ``warnings`` and ``messages``.
        """
        return list(self.raw["suppressed"])

    @property
    def expected_hidden_errors(self) -> tuple[str, ...]:
        """Components whose error the case declares ggstatsplot swallows."""
        return tuple(self.raw["expected_hidden_errors"])

    def table(self, name: str) -> list[Row]:
        """One ``extract_stats()`` table as row objects; empty when R gave NULL.

        A name ``extract_stats()`` never returns raises ``KeyError``, so a misspelled
        table cannot pass as an empty one.
        """
        stats = self.raw["stats"]
        if stats is None:
            raise ValueError(f"{self.name}: ggstatsplot failed ({self.error})")
        if name not in stats:
            raise KeyError(
                f"{self.name}: extract_stats() has no table {name!r}; "
                f"tables are {sorted(stats)}"
            )
        rows = stats[name]
        return [] if rows is None else list(rows)

    @property
    def subtitle_rows(self) -> list[Row]:
        """All ``subtitle_data`` rows. Bayes ANOVA has one per posterior parameter."""
        return self.table("subtitle_data")

    @property
    def subtitle_data(self) -> Row:
        """The single ``subtitle_data`` row of a frequentist or two-group test."""
        rows = self.subtitle_rows
        if len(rows) != 1:
            raise ValueError(
                f"{self.name}: expected one subtitle_data row, got {len(rows)}"
            )
        return rows[0]

    @property
    def caption_data(self) -> Row | None:
        rows = self.table("caption_data")
        return rows[0] if rows else None

    @property
    def pairwise(self) -> list[Row]:
        return self.table("pairwise_comparisons_data")

    @property
    def descriptive(self) -> list[Row]:
        return self.table("descriptive_data")

    @property
    def plot_layers(self) -> list[Row]:
        """Each ggplot layer's geom and its own data (null when it uses the plot's)."""
        layers = self.raw["plot_layers"]
        if layers is None:
            raise ValueError(f"{self.name}: ggstatsplot failed ({self.error})")
        return list(layers)

    def layer_params(self, geom: str) -> list[Mapping[str, Any]]:
        """Parameters (``stat``, ``geom``, ``aes``) of the layers drawn with ``geom``.

        For ggbetweenstats and ggwithinstats the generator records parameters only
        for the significance brackets (``GeomSignif``); any other geom raises
        ``KeyError``. Empty when this plot draws no brackets. gghistostats and
        ggscatterstats record them for every layer but the raw points.
        """
        if self.function in NO_LEVEL_FUNCTIONS:
            _check_geom(geom)
            return [
                layer["params"]
                for layer in self.plot_layers
                if layer["geom"] == geom and layer["params"] is not None
            ]
        if geom not in PARAMETER_GEOMS:
            raise KeyError(
                f"layer parameters are recorded only for {sorted(PARAMETER_GEOMS)}, "
                f"not {geom!r}"
            )
        return [layer["params"] for layer in self.plot_layers if layer["geom"] == geom]

    def layer_built(self, geom: str) -> list[list[Row]]:
        """The rows ggplot2 computed for drawing each layer drawn with ``geom`` (the
        histogram's bins, the smooth line and its band, the marginal histograms).
        Recorded for the brackets, and for every gghistostats and ggscatterstats
        layer but the raw points; a geom no fixture draws raises ``KeyError``."""
        _check_geom(geom)
        return [
            list(layer["built"])
            for layer in self.plot_layers
            if layer["geom"] == geom and layer["built"] is not None
        ]

    def layer_data(self, geom: str) -> list[list[Row]]:
        """Own data of the layers drawn with ``geom``, e.g. ``"GeomLabelRepel"``.

        Layers that draw the plot's main data (points, boxes, violins) have no data
        of their own and are left out, so the result is empty for them and when this
        plot has no such layer. A geom no fixture draws raises ``KeyError``.
        """
        _check_geom(geom)
        return [
            list(layer["data"])
            for layer in self.plot_layers
            if layer["geom"] == geom and layer["data"]
        ]


def fixture_paths() -> list[Path]:
    """Case fixtures: every JSON file except the manifest and the probes."""
    return sorted(
        p
        for p in FIXTURE_DIR.glob("*.json")
        if p.name != MANIFEST and not p.name.startswith(PROBE_PREFIX)
    )


def probe_paths() -> list[Path]:
    return sorted(FIXTURE_DIR.glob(f"{PROBE_PREFIX}*.json"))


@functools.cache
def load_manifest() -> Mapping[str, Any]:
    path = FIXTURE_DIR / MANIFEST
    if not path.exists():
        raise FixtureIntegrityError(
            f"{path} is missing; download fixtures from a successful oracle run"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def generator_problems(manifest: Mapping[str, Any]) -> list[str]:
    """Generator files that differ from the ones that wrote the fixtures."""
    recorded = manifest.get("generator_files")
    if not isinstance(recorded, Mapping) or set(recorded) != set(GENERATOR_FILES):
        return [
            f"manifest generator_files {recorded!r}; expected the SHA-256 of "
            f"{list(GENERATOR_FILES)}"
        ]
    problems: list[str] = []
    for relative in GENERATOR_FILES:
        path = ROOT / relative
        if not path.exists():
            problems.append(f"{relative} is missing")
        elif sha256_file(path) != recorded[relative]:
            problems.append(
                f"{relative} differs from the generator that wrote these fixtures; "
                "run the oracle on this branch and commit its fixtures (CLAUDE.md)"
            )
    return problems


def integrity_problems() -> list[str]:
    """Every way the fixtures disagree with their manifest, datasets or generator."""
    problems: list[str] = []
    manifest = load_manifest()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"manifest schema_version {manifest.get('schema_version')!r}")
    if manifest.get("parity_target") != f"ggstatsplot {PARITY_VERSION}":
        problems.append(f"manifest parity_target {manifest.get('parity_target')!r}")
    problems.extend(generator_problems(manifest))
    problems.extend(
        _file_problems(manifest, "fixtures", fixture_paths(), _fixture_datasets)
    )
    problems.extend(_file_problems(manifest, "probes", probe_paths(), _probe_datasets))
    return problems


def _fixture_datasets(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [raw["dataset"]]


def _probe_datasets(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(raw["datasets"])


def _file_problems(
    manifest: Mapping[str, Any],
    key: str,
    paths: Sequence[Path],
    datasets: Callable[[Mapping[str, Any]], list[Mapping[str, Any]]],
) -> list[str]:
    """Problems of the files the manifest lists under ``key`` (fixtures or probes)."""
    listed = manifest.get(key)
    if not isinstance(listed, Mapping):
        return [f"manifest has no {key!r} map"]
    problems: list[str] = []
    present = {p.name for p in paths}
    for name in sorted(present - listed.keys()):
        problems.append(f"{name} is not in the manifest's {key}")
    for name in sorted(listed.keys() - present):
        problems.append(f"{name} is in the manifest's {key} but missing")
    for name in sorted(present & listed.keys()):
        path = FIXTURE_DIR / name
        if sha256_file(path) != listed[name]:
            problems.append(
                f"{name} does not match its manifest hash (edited by hand?)"
            )
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            problems.append(f"{name}: schema_version {raw.get('schema_version')!r}")
        if raw.get("name") != path.stem:
            problems.append(f"{name}: names itself {raw.get('name')!r}")
        version = raw.get("versions", {}).get("packages", {}).get("ggstatsplot")
        if version != PARITY_VERSION:
            problems.append(f"{name}: generated by ggstatsplot {version!r}")
        for entry in datasets(raw):
            dataset = DATASET_DIR / entry["file"]
            if not dataset.exists():
                problems.append(f"{name}: dataset {dataset.name} is missing")
            elif sha256_file(dataset) != entry["sha256"]:
                problems.append(f"{name}: {dataset.name} differs from the CSV R read")
    return problems


@functools.cache
def load_cases() -> tuple[OracleCase, ...]:
    """All fixtures, after verifying integrity; raises if anything is off."""
    problems = integrity_problems()
    if problems:
        raise FixtureIntegrityError(
            "oracle fixtures failed verification:\n" + "\n".join(problems)
        )
    cases: list[OracleCase] = []
    for path in fixture_paths():
        raw = json.loads(path.read_text(encoding="utf-8"))
        cases.append(OracleCase(name=str(raw["name"]), path=path, raw=raw))
    return tuple(cases)


@dataclass(frozen=True)
class OracleProbe:
    """One probe file: what R functions returned on a fixed grid of inputs.

    Doubles are stored as R's C99 hex text (``sprintf("%a")``, read exactly with
    ``r_double``), most of them also as 15-digit JSON numbers.
    """

    name: str
    path: Path
    raw: Mapping[str, Any]

    def __repr__(self) -> str:
        return f"OracleProbe({self.name!r})"

    @property
    def description(self) -> str:
        return str(self.raw["description"])

    @property
    def versions(self) -> Mapping[str, Any]:
        return self.raw["versions"]

    def section(self, key: str) -> Any:
        """One top-level entry, e.g. ``"values"``; an unknown key raises ``KeyError``.

        A misspelled section must not pass as an empty one.
        """
        if key in _PROBE_ENVELOPE or key not in self.raw:
            known = sorted(set(self.raw) - _PROBE_ENVELOPE)
            raise KeyError(f"{self.name} has no section {key!r}; sections are {known}")
        return self.raw[key]

    def rows(self, key: str) -> list[Row]:
        """A section that is a list of row objects."""
        value = self.section(key)
        if not isinstance(value, list):
            raise TypeError(f"{self.name}: section {key!r} is not a list of rows")
        return list(value)


# Fields every probe file has; the rest are its sections.
_PROBE_ENVELOPE = frozenset(
    {"schema_version", "name", "kind", "description", "datasets", "versions"}
)


@functools.cache
def load_probes() -> Mapping[str, OracleProbe]:
    """All probe files by name, after the same verification as ``load_cases()``."""
    problems = integrity_problems()
    if problems:
        raise FixtureIntegrityError(
            "oracle fixtures failed verification:\n" + "\n".join(problems)
        )
    probes: dict[str, OracleProbe] = {}
    for path in probe_paths():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("kind") != "probe":
            raise FixtureIntegrityError(f"{path.name}: kind {raw.get('kind')!r}")
        name = str(raw["name"])
        probes[name] = OracleProbe(name=name, path=path, raw=raw)
    return MappingProxyType(probes)


def probe(name: str) -> OracleProbe:
    """One probe, e.g. ``probe("probe-padjust")``; an unknown name raises."""
    probes = load_probes()
    if name not in probes:
        raise KeyError(f"no probe {name!r}; probes are {sorted(probes)}")
    return probes[name]


def r_double(text: str) -> float | None:
    """A double R wrote with ``sprintf("%a")``, exactly; R's NA is ``None``.

    NaN, Inf and -Inf come as R prints them (``"NaN"``, ``"Inf"``, ``"-Inf"``),
    negative zero as ``"-0x0p+0"``.
    """
    if text == "NA":
        return None
    if text in {"NaN", "Inf", "-Inf"}:
        return float(text)
    if not text.removeprefix("-").startswith("0x"):
        raise ValueError(f"not a hex double: {text!r}")
    return float.fromhex(text)


# Bayes factors where BayesFactor::ttestBF (ggstatsplot's pairwise Bayes factors)
# misses the accurate value by more than `ttest_bayes_factor` (rtol 1e-8). Following
# DECISIONS.md D31, inferplot reports the accurate value and is compared with it.
# (case, group1, group2) -> (accurate BF10, ggstatsplot's relative miss). iris
# Petal.Length, 50 + 50 values: at Student t -39.49 and -49.99 BayesFactor's quadrature
# is off by +1.16e-3 and -6.1e-5. The accurate values are 50-digit mpmath quadratures
# of Rouder et al. (2009), eq. 1, which tools/reference matches to 1.6e-14; every
# other one of the 77 distinct two-sample Bayes factors in the fixtures is within
# 2.3e-9 of it.
ACCURATE_BAYES_FACTORS: Mapping[tuple[str, str, str], tuple[float, float]] = (
    MappingProxyType(
        {
            ("ggbetweenstats-iris-petal-bayes", "setosa", "versicolor"): (
                1.1443263236860359e58,
                1.16e-3,
            ),
            ("ggbetweenstats-iris-petal-bayes", "setosa", "virginica"): (
                3.183438242038274e67,
                -6.1e-5,
            ),
        }
    )
)


def expected_bf10(case_name: str, group1: str, group2: str, oracle: float) -> float:
    """The BF10 to compare with: ggstatsplot's, or the accurate one where it misses.

    For a registered miss, also checks that ggstatsplot's value still misses by the
    recorded amount, so a stale entry fails instead of hiding a difference.
    """
    entry = ACCURATE_BAYES_FACTORS.get((case_name, group1, group2))
    if entry is None:
        return oracle
    accurate, miss = entry
    measured = oracle / accurate - 1
    assert abs(measured - miss) <= 0.01 * abs(miss), (
        case_name,
        group1,
        group2,
        measured,
    )
    return accurate


@functools.cache
def wrs2_akp_constant(tr: float) -> float:
    """WRS2::akp.effect's scaling constant at trimming ``tr``, as probe-akp measures
    it: AKP / ((trimmed mean 1 - trimmed mean 2) / winsorized SD of group 1), the
    same to 4 ulps on every dataset at a given tr, for tr = 0.01 to 0.49 by 0.01.

    WRS2 evaluates the winsorized standard normal SD numerically, off the exact value
    inferplot uses by an amount that varies irregularly with tr (1.2e-9 relative at
    0.2, 4.6e-9 at 0.1, up to 4.2e-7 at 0.49; tolerance ``akp_constant``), so AKP
    estimates are compared after scaling inferplot's by this constant over the exact
    one, under ``closed_form``.
    """
    akp = probe("probe-akp")
    # The constants were run on R's seq(0.01, 0.49, 0.01), whose 0.1 is one ulp
    # below 0.1; match on the printed tr and confirm with the per-dataset rows,
    # which use tr exactly, where they have it.
    found = [
        r_double(str(row["ratio_to_group1_hex"]))
        for row in akp.rows("constants")
        if float(row["tr"]) == tr and row["error"] is None
    ]
    if len(found) != 1 or found[0] is None:
        raise KeyError(f"probe-akp has no AKP constant at tr {tr!r}")
    value = found[0]
    for row in akp.rows("rows"):
        if r_double(str(row["tr_hex"])) == tr:
            ratio = r_double(str(row["akp"][0]["ratio_to_group1_hex"]))
            if ratio is not None and math.isfinite(ratio):
                assert abs(ratio - value) <= 4 * math.ulp(value), (tr, ratio, value)
    return value


# lincon's critical values above 28 comparisons, where WRS2 replaces its table by a
# simulation seeded by itself and inferplot computes the exact Studentized maximum
# modulus quantile at the pair's df (DECISIONS.md D52). WRS2's simulated values are
# not that quantile, so the intervals differ by more than any rounding: (case,
# group1, group2) -> (R's critical value, implied by its interval as
# (conf.high - conf.low) / (2 x Yuen SE), and inferplot's), 6 decimals. edge_k9
# (9 groups of 6, Yuen df 4.69 to 5.98): R's value is 0.35% below to 4.7% above the
# exact quantile, the most where the df is lowest (a-i); on probe-smm's lincon runs
# (36 and 45 comparisons, df 9.7 to 19) it is 3.4% to 6.6% below.
SIMULATED_LINCON_CRITICAL_VALUES: Mapping[tuple[str, str, str], tuple[float, float]] = (
    MappingProxyType(
        {
            ("ggbetweenstats-edge_k9-robust", g1, g2): values
            for (g1, g2), values in {
                ("a", "b"): (4.847880, 4.849568),
                ("a", "c"): (4.923635, 4.896593),
                ("a", "d"): (4.988970, 4.959015),
                ("a", "e"): (5.099571, 5.039438),
                ("a", "f"): (5.243255, 5.122068),
                ("a", "g"): (5.361642, 5.215088),
                ("a", "h"): (5.492718, 5.304154),
                ("a", "i"): (5.649805, 5.398081),
                ("b", "c"): (4.844058, 4.849251),
                ("b", "d"): (4.910205, 4.886927),
                ("b", "e"): (4.979111, 4.945127),
                ("b", "f"): (5.053385, 5.010888),
                ("b", "g"): (5.171081, 5.089548),
                ("b", "h"): (5.300584, 5.168226),
                ("b", "i"): (5.427776, 5.254010),
                ("c", "d"): (4.834097, 4.844806),
                ("c", "e"): (4.902295, 4.878290),
                ("c", "f"): (4.948830, 4.924376),
                ("c", "g"): (5.020942, 4.985385),
                ("c", "h"): (5.109359, 5.050485),
                ("c", "i"): (5.245551, 5.124785),
                ("d", "e"): (4.827567, 4.844669),
                ("d", "f"): (4.890908, 4.872359),
                ("d", "g"): (4.947312, 4.916205),
                ("d", "h"): (4.974801, 4.967526),
                ("d", "i"): (5.074250, 5.029598),
                ("e", "f"): (4.839591, 4.841890),
                ("e", "g"): (4.879637, 4.867211),
                ("e", "h"): (4.932912, 4.903161),
                ("e", "i"): (4.971316, 4.951107),
                ("f", "g"): (4.834818, 4.841961),
                ("f", "h"): (4.875191, 4.863588),
                ("f", "i"): (4.926057, 4.897994),
                ("g", "h"): (4.831618, 4.840112),
                ("g", "i"): (4.875203, 4.860048),
                ("h", "i"): (4.831624, 4.840116),
            }.items()
        }
    )
)


def check_simulated_lincon_pair(
    case_name: str,
    group1: str,
    group2: str,
    *,
    r_critical: float,
    critical: float,
) -> None:
    """Assert a pinned pair (``SIMULATED_LINCON_CRITICAL_VALUES``) still reads as
    pinned: R's implied critical value and inferplot's, each to 6 decimals."""
    r_pinned, ours_pinned = SIMULATED_LINCON_CRITICAL_VALUES[
        (case_name, group1, group2)
    ]
    what = f"{case_name} {group1}-{group2}"
    assert abs(r_critical - r_pinned) <= 5e-7, (what, r_critical, r_pinned)
    assert abs(critical - ours_pinned) <= 5e-7, (what, critical, ours_pinned)


# Geoms whose layer parameters generate.R records (the drawn pairs and labels).
PARAMETER_GEOMS = frozenset({"GeomSignif"})


@functools.cache
def known_geoms() -> frozenset[str]:
    """Every geom drawn by some fixture's plot (cases where R failed have none)."""
    return frozenset(
        str(layer["geom"])
        for case in load_cases()
        for layer in case.raw["plot_layers"] or []
    )


def _check_geom(geom: str) -> None:
    if geom not in known_geoms():
        raise KeyError(
            f"no fixture draws a layer with {geom!r}; geoms are {sorted(known_geoms())}"
        )


def cases_for(function: str, *, failed: bool = False) -> tuple[OracleCase, ...]:
    """Fixtures for one R function, e.g. ``"ggbetweenstats"``.

    By default the cases ggstatsplot completed; with ``failed=True`` the cases where
    it failed, as the case declares (``expected_error``), which have no outputs to
    compare against, only ``partial`` ones.
    """
    return tuple(
        c
        for c in load_cases()
        if c.function == function and (c.error is not None) == failed
    )


def require_cases(cases: Sequence[OracleCase], what: str) -> Sequence[OracleCase]:
    """``cases``, or ``LookupError`` if there are none.

    An empty parametrization would make pytest skip the test silently (CLAUDE.md
    rule 2), so every fixture selection used to parametrize a test goes through here.
    """
    if not cases:
        raise LookupError(f"no oracle fixtures for {what}; add cases to generate.R")
    return cases


def oracle_params(function: str, *, failed: bool = False) -> list[Any]:
    """``pytest.param`` per fixture of ``function``, with the case name as id.

    ``failed`` selects as in ``cases_for``.
    """
    what = f"{function!r}{' (failed in R)' if failed else ''}"
    cases = require_cases(cases_for(function, failed=failed), what)
    return [pytest.param(case, id=case.name) for case in cases]


def read_dataset(case: OracleCase, backend: Backend = "pandas") -> Any:
    """The CSV a fixture was generated from, read the way R read it.

    Only empty fields are missing values (R: ``na.strings = ""``). Declared factor
    levels become a pandas Categorical, a polars Enum or a pyarrow dictionary
    column marked ordered with that level order, including empty levels.
    """
    return read_csv(case.dataset_file, backend, case.factor_levels)


def read_csv(
    file: str,
    backend: Backend = "pandas",
    factor_levels: Mapping[str, Sequence[str]] | None = None,
) -> Any:
    """A CSV of ``oracle/datasets`` read the way R reads it (see ``read_dataset``)."""
    import pandas as pd

    factor_levels = factor_levels or {}
    frame = pd.read_csv(
        DATASET_DIR / file,
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8",
    )
    for column, levels in factor_levels.items():
        frame[column] = pd.Categorical(frame[column], categories=list(levels))
    if backend == "pandas":
        return frame
    if backend == "polars":
        import polars as pl

        # polars 1.0 (the floor, D22) cannot convert a pandas object column holding
        # NaN for missing text (bugs_long's gender, region, education); pandas'
        # string dtype carries those as nulls.
        text = {c: "string" for c in frame.columns if frame[c].dtype == object}
        out = pl.from_pandas(frame.astype(text))
        for column, levels in factor_levels.items():
            out = out.with_columns(
                pl.col(column).cast(pl.String).cast(pl.Enum(list(levels)))
            )
        return out
    if backend == "pyarrow":
        import pyarrow as pa

        # Arrow declares a dictionary's order only when its type is marked ordered
        # (an unordered dictionary is read as text); R's factor declared it.
        for column in factor_levels:
            frame[column] = frame[column].cat.as_ordered()
        return pa.Table.from_pandas(frame, preserve_index=False)
    raise ValueError(f"unknown backend {backend!r}")


def r_level_label(value: object) -> str:
    """Label R gives a value when it makes a factor from a CSV column.

    Exact for strings, integers and short decimals such as 0.5; R prints doubles
    with 15 significant digits, so values like 0.1 + 0.2 would need more care.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
