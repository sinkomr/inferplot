"""inferplot.stats' correlations against every ggscatterstats fixture and probe-corr.

Each method runs on the complete pairs the fixture analyzed (``analyzed_data``, in
file order) with the fixture's arguments, and every number the fixture holds for
it is compared: r or rho, t or S, df, p, the interval and its level, n, and the
Bayes factor with its prior (subtitle of the bayes type, caption of the parametric
type). probe-corr adds what correlation::correlation runs underneath (cor.test,
Spearman's exact = FALSE, the winsorized values and r, correlationBF) on its nine
designs, at 95% and 90% and tr 0.2 and 0.1, from R's own doubles.

Where R prints NA or a number R itself got by accident, inferplot differs on
purpose, and those cases are listed by name below (``R_DIFFERS``, ``REFUSED``).
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
import pytest
from numpy.typing import NDArray

from inferplot.stats import (
    Correlation,
    correlation_bf10,
    correlation_log_bf10,
    pearson,
    spearman,
    winsorized_correlation,
)
from inferplot.stats.correlation import (
    _winsorize,  # pyright: ignore[reportPrivateUsage]
)
from inferplot.stats.robust import trim_count
from tests.oracle_fixtures import (
    OracleCase,
    cases_for,
    number,
    probe,
    r_double,
    require_cases,
)
from tests.stats.fixture_data import bf_prior, conf_level, tr, value
from tests.tolerances import assert_close

FloatArray = NDArray[np.float64]
CASES = require_cases(cases_for("ggscatterstats"), "ggscatterstats fixtures")
DESIGNS = probe("probe-corr").rows("designs")

# Constant x: R reports NA for every number (and the Bayesian correlation stops);
# inferplot's functions raise a plain ValueError instead.
REFUSED: Final = frozenset(
    f"ggscatterstats-edge_corr_constant_x-{t}"
    for t in ("parametric", "nonparametric", "robust", "bayes")
)

# Three pairs (barstats oracle run): the Fisher z interval needs 4 (D85), so
# inferplot's coefficients raise; R prints r, t (or S) and p with the interval
# [NA, NA], or [-1, 1] for Spearman's rho. The Bayes factor (type bayes and the
# parametric caption) is answered and compared.
THREE_PAIRS: Final = frozenset(
    f"ggscatterstats-edge_corr_n3-{t}"
    for t in ("parametric", "nonparametric", "robust")
)

# Exactly collinear data (y = 2x + 1 and y = 27 - 3x on x = 1..10): inferplot's
# exact sums give r = +/-1, so t = +/-inf, S = 0 and p = 0, and the Bayes factor is
# its limit, +inf (n = 10 >= 5 at prior 0.707). R's r is 1 - 2^-53 or so on the
# positive data (t = 2^27, p 1.06e-62; S = 3.66e-14), -1 exactly on the negative
# data (t = -Inf, p 0, which statsExpressions prints as 2.23e-308), and
# correlationBF returns NaN (an empty log BF01). case -> column -> (R, inferplot).
R_DIFFERS: Final[dict[str, dict[str, tuple[float | None, float]]]] = {
    "ggscatterstats-edge_corr_perfect-parametric": {
        "statistic": (134217728.0, math.inf),
        "p.value": (1.0635035875251e-62, 0.0),
    },
    "ggscatterstats-edge_corr_perfect-robust": {
        "statistic": (134217728.0, math.inf),
        "p.value": (1.0635035875251e-62, 0.0),
    },
    "ggscatterstats-edge_corr_perfect-nonparametric": {
        "statistic": (3.66373598126302e-14, 0.0),
        "p.value": (1.0635035875251e-62, 0.0),
    },
    "ggscatterstats-edge_corr_perfect-neg-parametric": {
        "p.value": (2.2250738585072e-308, 0.0),
    },
    "ggscatterstats-edge_corr_perfect-neg-nonparametric": {
        "p.value": (1.70160574004017e-61, 0.0),
    },
    "ggscatterstats-edge_corr_perfect-neg-robust": {
        "statistic": (-189812531.248503, -math.inf),
        "p.value": (6.64689742203201e-64, 0.0),
    },
    "ggscatterstats-edge_corr_perfect-bayes": {"bf10": (None, math.inf)},
    "ggscatterstats-edge_corr_perfect-parametric-caption": {"bf10": (None, math.inf)},
    "ggscatterstats-edge_corr_perfect-neg-parametric-caption": {
        "bf10": (None, math.inf)
    },
    "ggscatterstats-edge_corr_perfect-neg-bayes": {"bf10": (None, math.inf)},
}


def _pairs(case: OracleCase) -> tuple[FloatArray, FloatArray]:
    return (
        np.array(case.analyzed_values("x"), dtype=np.float64),
        np.array(case.analyzed_values("y"), dtype=np.float64),
    )


def _params(type_: str, *, refused: bool = False) -> list[Any]:
    chosen = [
        c
        for c in CASES
        if c.type == type_
        and c.subtitle_rows
        and ((c.name in REFUSED) == refused)
        and c.name not in THREE_PAIRS
    ]
    return [pytest.param(c, id=c.name) for c in require_cases(chosen, type_)]


def _compare(case: OracleCase, column: str, mine: float, row: Any) -> None:
    differs = R_DIFFERS.get(case.name, {})
    if column in differs:
        theirs, expected = differs[column]
        oracle = row[column]
        assert (oracle is None) if theirs is None else number(oracle) == theirs
        assert mine == expected, (case.name, column, mine)
        return
    assert_close("closed_form", mine, value(row, column), what=f"{case.name} {column}")


def _run(case: OracleCase) -> Correlation:
    x, y = _pairs(case)
    level = conf_level(case)
    if case.type == "parametric":
        return pearson(x, y, conf_level=level)
    if case.type == "nonparametric":
        return spearman(x, y, conf_level=level)
    return winsorized_correlation(x, y, tr=tr(case), conf_level=level)


def test_case_counts() -> None:
    # 64, and 9 from the barstats oracle run: edge_corr_n3 and edge_corr_n4 in four
    # types each, edge_corr_label_missing (parametric).
    assert len(CASES) == 73
    counts = {
        t: sum(c.type == t for c in CASES)
        for t in ("parametric", "nonparametric", "robust", "bayes")
    }
    assert counts == {"parametric": 23, "nonparametric": 16, "robust": 17, "bayes": 17}
    names = {c.name for c in CASES}
    assert names >= REFUSED | THREE_PAIRS
    assert names >= {k.removesuffix("-caption") for k in R_DIFFERS}


@pytest.mark.parametrize(
    "case",
    _params("parametric") + _params("nonparametric") + _params("robust"),
)
def test_correlation_subtitle(case: OracleCase) -> None:
    row = case.subtitle_data
    result = _run(case)
    x, _ = _pairs(case)
    assert result.n == x.size == int(value(row, "n.obs"))
    assert result.conf_level == value(row, "conf.level")
    method = {
        "parametric": "Pearson correlation",
        "nonparametric": "Spearman correlation",
        "robust": "Winsorized Pearson correlation",
    }[case.type]
    assert row["method"] == method
    for column, mine in (
        ("estimate", result.estimate),
        ("statistic", result.statistic),
        ("p.value", result.p_value),
        ("conf.low", result.conf_low),
        ("conf.high", result.conf_high),
    ):
        _compare(case, column, mine, row)
    if case.type != "nonparametric":
        assert result.df == value(row, "df.error") == x.size - 2
        assert result.t == result.statistic
    # The correlation::correlation row the subtitle came from says the same.
    (corr,) = case.correlation["value"]
    estimate = number(corr["rho" if case.type == "nonparametric" else "r"])
    assert_close("closed_form", result.estimate, estimate, what="correlation row")


def _check_bf(case: OracleCase, row: Any, *, caption: bool = False) -> None:
    x, y = _pairs(case)
    prior = bf_prior(case)
    bf10, log_bf01 = correlation_bf10(x, y, prior_scale=prior)
    differs = R_DIFFERS.get(case.name + ("-caption" if caption else ""), {})
    if "bf10" in differs:
        assert row["bf10"] is None
        assert bf10 == differs["bf10"][1]
        assert log_bf01 == -math.inf
        return
    assert math.isclose(log_bf01, -math.log(bf10), rel_tol=1e-15, abs_tol=1e-15)
    assert_close(
        "correlation_bayes_factor", bf10, value(row, "bf10"), what=f"{case.name} BF10"
    )


@pytest.mark.parametrize("case", _params("bayes"))
def test_correlation_bayes_factor(case: OracleCase) -> None:
    row = case.subtitle_data
    assert row["method"] == "Bayesian Pearson correlation"
    # ggstatsplot prints the beta prior's parameter, 1 / bf.prior.
    assert math.isclose(value(row, "prior.scale"), 1.0 / bf_prior(case), rel_tol=1e-14)
    x, _ = _pairs(case)
    assert int(value(row, "n.obs")) == x.size
    _check_bf(case, row)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=c.name)
        for c in CASES
        if c.type == "parametric" and c.caption_data is not None
    ],
)
def test_correlation_bayes_factor_caption(case: OracleCase) -> None:
    caption = case.caption_data
    assert caption is not None
    _check_bf(case, caption, caption=True)


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in CASES if c.name in REFUSED],
)
def test_constant_x_is_refused(case: OracleCase) -> None:
    # R: every number NA ("the standard deviation is zero"); the Bayes type's
    # correlation stops. inferplot raises a plain error naming the variable.
    x, y = _pairs(case)
    assert float(x.min()) == float(x.max())
    if case.type == "bayes":
        with pytest.raises(ValueError, match="every value of x is the same"):
            correlation_bf10(x, y, prior_scale=bf_prior(case))
    else:
        with pytest.raises(ValueError, match="every value of x is the same"):
            _run(case)
    rows = case.subtitle_rows
    if rows:
        assert rows[0]["estimate"] is None


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c.name) for c in CASES if c.name in THREE_PAIRS]
)
def test_three_pairs_are_refused(case: OracleCase) -> None:
    x, _ = _pairs(case)
    assert x.size == 3
    with pytest.raises(ValueError, match="needs at least 4 pairs"):
        _run(case)
    # R's side: r (or rho) = 0.5, p = 2/3 and an interval of NA (Pearson,
    # winsorized: nothing is winsorized at floor(0.2 x 3) = 0) or [-1, 1]
    # (Spearman: the Fisher z standard error is infinite at n = 3).
    row = case.subtitle_data
    assert int(value(row, "n.obs")) == 3
    assert_close("closed_form", value(row, "estimate"), 0.5, what="r")
    assert_close("closed_form", value(row, "p.value"), 2 / 3, what="p")
    if case.type == "nonparametric":
        assert value(row, "statistic") == 2.0
        assert (value(row, "conf.low"), value(row, "conf.high")) == (-1.0, 1.0)
    else:
        # t = r sqrt(n - 2) / sqrt(1 - r^2) = 1 / sqrt(3) on 1 df.
        assert_close("closed_form", value(row, "statistic"), 3**-0.5, what="t")
        assert value(row, "df.error") == 1
        assert (row["conf.low"], row["conf.high"]) == (None, None)


# ------------------------------------------------------------------ probe-corr


def _hex(text: object) -> float:
    found = r_double(str(text))
    assert found is not None, text
    return found


def _design_pairs(design: Any) -> tuple[FloatArray, FloatArray]:
    x = np.array([float.fromhex(v) for v in design["x_hex"]])
    y = np.array([float.fromhex(v) for v in design["y_hex"]])
    return x, y


def _close(mine: float, theirs: float, what: str, design: Any) -> None:
    if design["name"] in ("perfect", "perfect_neg") and what in ("t", "p", "S"):
        # R's accidents on exactly collinear data (R_DIFFERS): inferplot's t is
        # infinite, p 0 and S 0 (perfect) or (n^3 - n) / 3 = 330 (perfect_neg).
        assert math.isinf(mine) or mine == 0.0 or (what == "S" and mine == 330.0)
        return
    assert_close("closed_form", mine, theirs, what=f"{design['name']} {what}")


@pytest.mark.parametrize(
    "design", [pytest.param(d, id=str(d["name"])) for d in DESIGNS]
)
def test_probe_corr(design: Any) -> None:
    x, y = _design_pairs(design)
    assert x.size == design["n"]
    for entry in design["pearson"]:
        mine = pearson(x, y, conf_level=entry["conf"])
        _close(mine.estimate, _hex(entry["estimate_hex"]), "r", design)
        _close(mine.t, _hex(entry["statistic_hex"]), "t", design)
        _close(mine.p_value, _hex(entry["p_value_hex"]), "p", design)
        low, high = (_hex(v) for v in entry["conf_int_hex"])
        _close(mine.conf_low, low, "low", design)
        _close(mine.conf_high, high, "high", design)
    (no_exact,) = [s for s in design["spearman"] if s["exact"] is False]
    rho = spearman(x, y)
    _close(rho.estimate, _hex(no_exact["estimate_hex"]), "rho", design)
    _close(rho.statistic, _hex(no_exact["statistic_hex"]), "S", design)
    _close(rho.p_value, _hex(no_exact["p_value_hex"]), "p", design)
    for entry in design["correlation"]:
        (row,) = entry["row"]
        method = entry["method"]
        result = (pearson if method == "pearson" else spearman)(
            x, y, conf_level=entry["ci"]
        )
        key = "r_hex" if method == "pearson" else "rho_hex"
        _close(result.estimate, _hex(row[key]), "estimate", design)
        _close(result.conf_low, _hex(row["CI_low_hex"]), "CI low", design)
        _close(result.conf_high, _hex(row["CI_high_hex"]), "CI high", design)
        _close(result.p_value, _hex(row["p_hex"]), "p", design)
        if method == "spearman":
            _close(result.statistic, _hex(row["S_hex"]), "S", design)
        else:
            _close(result.t, _hex(row["t_hex"]), "t", design)
    for entry in design["winsorized"]:
        trim = float(entry["tr"])
        g = trim_count(x.size, trim)
        assert _winsorize(x, g).tolist() == [
            float.fromhex(v) for v in entry["winsorized_x_hex"]
        ]
        assert _winsorize(y, g).tolist() == [
            float.fromhex(v) for v in entry["winsorized_y_hex"]
        ]
        (row,) = entry["correlation"]
        result = winsorized_correlation(x, y, tr=trim)
        _close(result.estimate, _hex(row["r_hex"]), "r_w", design)
        _close(result.estimate, _hex(entry["wincor"]["cor"][0]), "wincor r", design)
        _close(result.t, _hex(row["t_hex"]), "t", design)
        _close(result.p_value, _hex(row["p_hex"]), "p", design)
        _close(result.conf_low, _hex(row["CI_low_hex"]), "CI low", design)
        _close(result.conf_high, _hex(row["CI_high_hex"]), "CI high", design)


@pytest.mark.parametrize(
    "design", [pytest.param(d, id=str(d["name"])) for d in DESIGNS]
)
def test_probe_corr_bayes_factor(design: Any) -> None:
    # correlationBF at bf.prior 0.707 and 0.8 and BayesFactor's "medium" (1/3),
    # on the data and at R's own r (which isolates BayesFactor's arithmetic).
    x, y = _design_pairs(design)
    (cor_test,) = [p for p in design["pearson"] if p["conf"] == 0.95]
    r_r = _hex(cor_test["estimate_hex"])
    for entry in design["bayes"]:
        prior = 1.0 / 3.0 if entry["prior"] == "medium" else float(entry["prior"])
        theirs = r_double(str(entry["bf10_hex"]))
        bf10, _ = correlation_bf10(x, y, prior_scale=prior)
        if design["name"] in ("perfect", "perfect_neg"):
            assert theirs is None or math.isnan(theirs)
            assert bf10 == math.inf
            continue
        assert theirs is not None
        what = f"{design['name']} prior {entry['prior']}"
        assert_close("correlation_bayes_factor", bf10, theirs, what=what)
        at_r = math.exp(correlation_log_bf10(r_r, design["n"], prior_scale=prior))
        assert_close("correlation_bayes_factor", at_r, theirs, what=f"{what} at R's r")
