"""Load and verify the R oracle fixtures in ``tests/fixtures/oracle/``.

``oracle/generate.R`` writes the fixtures in GitHub Actions; they are never edited by
hand. Loading checks that every fixture's SHA-256 matches ``manifest.json`` and that
the CSV each fixture recorded still has the same SHA-256, so a fixture can never
drift silently from the data the Python tests read. The schema is described in
``docs/design/SPEC.md`` section 9.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "oracle"
DATASET_DIR = ROOT / "oracle" / "datasets"
MANIFEST = "manifest.json"
SCHEMA_VERSION = 1
PARITY_VERSION = "1.1.4"

Backend = Literal["pandas", "polars", "pyarrow"]
Row = Mapping[str, Any]


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
    def error(self) -> str | None:
        return self.raw["error"]

    @property
    def subtitle(self) -> str | None:
        return self.raw["subtitle"]

    @property
    def caption(self) -> str | None:
        return self.raw["caption"]

    @property
    def bootstrap_reference(self) -> Mapping[str, Any] | None:
        return self.raw["bootstrap_reference"]

    def table(self, name: str) -> list[Row]:
        """One ``extract_stats()`` table as row objects; empty when R gave NULL."""
        stats = self.raw["stats"]
        if stats is None:
            raise ValueError(f"{self.name}: ggstatsplot failed ({self.error})")
        rows = stats.get(name)
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
        layers = self.raw.get("plot_layers")
        return [] if layers is None else list(layers)

    def layer_params(self, geom: str) -> list[Mapping[str, Any]]:
        """Parameters (``stat``, ``geom``, ``aes``) of layers drawn with ``geom``."""
        return [
            layer["params"]
            for layer in self.plot_layers
            if layer["geom"] == geom and layer.get("params")
        ]

    def layer_data(self, geom: str) -> list[list[Row]]:
        """Own data of every layer drawn with ``geom``, e.g. ``"GeomLabelRepel"``."""
        return [
            list(layer["data"])
            for layer in self.plot_layers
            if layer["geom"] == geom and layer["data"]
        ]


def fixture_paths() -> list[Path]:
    return sorted(p for p in FIXTURE_DIR.glob("*.json") if p.name != MANIFEST)


@functools.cache
def load_manifest() -> Mapping[str, Any]:
    path = FIXTURE_DIR / MANIFEST
    if not path.exists():
        raise FixtureIntegrityError(
            f"{path} is missing; download fixtures from a successful oracle run"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def integrity_problems() -> list[str]:
    """Every way the fixture directory disagrees with its manifest or datasets."""
    problems: list[str] = []
    manifest = load_manifest()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"manifest schema_version {manifest.get('schema_version')!r}")
    if manifest.get("parity_target") != f"ggstatsplot {PARITY_VERSION}":
        problems.append(f"manifest parity_target {manifest.get('parity_target')!r}")
    listed: Mapping[str, str] = manifest.get("fixtures", {})
    present = {p.name for p in fixture_paths()}
    for name in sorted(present - listed.keys()):
        problems.append(f"{name} is not in the manifest")
    for name in sorted(listed.keys() - present):
        problems.append(f"{name} is in the manifest but missing")
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
        version = raw.get("versions", {}).get("packages", {}).get("ggstatsplot")
        if version != PARITY_VERSION:
            problems.append(f"{name}: generated by ggstatsplot {version!r}")
        dataset = DATASET_DIR / raw["dataset"]["file"]
        if not dataset.exists():
            problems.append(f"{name}: dataset {dataset.name} is missing")
        elif sha256_file(dataset) != raw["dataset"]["sha256"]:
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


def cases_for(function: str) -> tuple[OracleCase, ...]:
    """Fixtures for one R function, e.g. ``"ggbetweenstats"``."""
    return tuple(c for c in load_cases() if c.function == function)


def oracle_params(function: str) -> list[Any]:
    """``pytest.param`` per fixture of ``function``, with the case name as id."""
    return [pytest.param(case, id=case.name) for case in cases_for(function)]


def read_dataset(case: OracleCase, backend: Backend = "pandas") -> Any:
    """The CSV a fixture was generated from, read the way R read it.

    Only empty fields are missing values (R: ``na.strings = ""``). Declared factor
    levels become a pandas Categorical, a polars Enum or a pyarrow dictionary
    column with that level order, including empty levels.
    """
    import pandas as pd

    frame = pd.read_csv(
        DATASET_DIR / case.dataset_file,
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8",
    )
    for column, levels in case.factor_levels.items():
        frame[column] = pd.Categorical(frame[column], categories=list(levels))
    if backend == "pandas":
        return frame
    if backend == "polars":
        import polars as pl

        out = pl.from_pandas(frame)
        for column, levels in case.factor_levels.items():
            out = out.with_columns(
                pl.col(column).cast(pl.String).cast(pl.Enum(list(levels)))
            )
        return out
    if backend == "pyarrow":
        import pyarrow as pa

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
