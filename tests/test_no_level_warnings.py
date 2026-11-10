"""Arguments without effect in histostats and scatterstats (DECISIONS.md D39), as a
table.

Every analysis argument, for every type: a value that changes nothing reported or
drawn adds one ``warnings`` entry naming the argument and the value as given; a
value that would change what the reported numbers mean raises; a value with an
effect, a default and a default's alias add nothing. The entries are compared with
the warnings of the same call without the argument, so each row isolates one
argument.

histostats: ``seed`` acts only for type robust (its bootstrap; the histogram is
drawn from every value). scatterstats: no correlation draws random numbers, but
the drawing uses ``seed`` to choose the points it shows of more than 5,000
(SPEC.md section 3), so it never warns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot.result import StatsResult

ONE = pd.DataFrame({"v": [1.0, 2.0, 4.0, 3.5, 6.0, 2.2, 5.1, 3.3, 7.4, 4.8]})
PAIR = pd.DataFrame(
    {
        "a": [1.0, 2.0, 4.0, 3.5, 6.0, 2.2, 5.1, 3.3, 7.4, 4.8],
        "b": [2.0, 2.5, 5.5, 3.0, 8.0, 3.9, 5.0, 4.4, 6.1, 4.0],
    }
)
TYPES = ("parametric", "nonparametric", "robust", "bayes")
_NO_CAPTION = (
    "bf_message=False has no effect: the Bayes factor caption is drawn only for "
    "type='parametric' with the results subtitle (as in ggstatsplot), and "
)
_NO_BF = "bf_prior=0.8 has no effect: no Bayes factor is computed ("

# (type, extra keywords of the baseline call, the argument under test, its value,
# what it gives): None (no new entry), a substring of the one new warnings entry,
# or an exception type. The same rows hold for both functions unless marked.
COMMON: list[tuple[str, dict[str, Any], str, Any, Any]] = [
    # defaults and their aliases: nothing
    *[
        (t, {}, name, value, None)
        for t in TYPES
        for name, value in (
            ("bf_prior", 0.707),
            ("conf_level", 0.95),
            ("tr", 0.2),
            ("bf_message", None),
            ("seed", 0),
            ("digits", 2),
            ("results_subtitle", True),
        )
    ],
    # digits always acts (every text)
    *[(t, {}, "digits", 3, None) for t in TYPES],
    # conf_level: every type with an interval; type bayes shows none
    *[(t, {}, "conf_level", 0.9, None) for t in TYPES[:3]],
    (
        "bayes",
        {},
        "conf_level",
        0.9,
        "conf_level=0.9 has no effect: type='bayes' shows no interval",
    ),
    (
        "bayes",
        {"results_subtitle": False},
        "conf_level",
        0.9,
        "conf_level=0.9 has no effect: type='bayes' shows no interval",
    ),
    # bf_prior: only where a Bayes factor is computed with it
    ("parametric", {}, "bf_prior", 0.8, None),
    ("bayes", {}, "bf_prior", 0.8, None),
    (
        "parametric",
        {"bf_message": False},
        "bf_prior",
        0.8,
        _NO_BF + "bf_message=False leaves the Bayes factor out).",
    ),
    (
        "parametric",
        {"results_subtitle": False},
        "bf_prior",
        0.8,
        _NO_BF + "results_subtitle=False leaves the Bayes factor out).",
    ),
    (
        "bayes",
        {"results_subtitle": False},
        "bf_prior",
        0.8,
        _NO_BF + "results_subtitle=False leaves the Bayes factor out).",
    ),
    ("nonparametric", {}, "bf_prior", 0.8, _NO_BF + "type='nonparametric' has no"),
    ("robust", {}, "bf_prior", 0.8, _NO_BF + "type='robust' has no Bayes factor)."),
    # bf_message: the parametric caption only
    ("parametric", {}, "bf_message", False, None),
    ("parametric", {}, "bf_message", True, None),
    (
        "parametric",
        {"results_subtitle": False},
        "bf_message",
        False,
        _NO_CAPTION + "results_subtitle=False leaves the Bayes factor caption out.",
    ),
    ("parametric", {"results_subtitle": False}, "bf_message", True, ValueError),
    ("nonparametric", {}, "bf_message", False, _NO_CAPTION + "type='nonparametric'"),
    ("robust", {}, "bf_message", False, _NO_CAPTION + "type='robust' draws none."),
    (
        "bayes",
        {},
        "bf_message",
        False,
        _NO_CAPTION + "type='bayes' draws none; type='bayes' reports its Bayes "
        "factor in the subtitle.",
    ),
    *[(t, {}, "bf_message", True, ValueError) for t in TYPES[1:]],
    # results_subtitle=False acts for every type (no subtitle)
    *[(t, {}, "results_subtitle", False, None) for t in TYPES],
    # tr: only the robust type trims
    *[(t, {}, "tr", 0.1, ValueError) for t in ("parametric", "nonparametric", "bayes")],
    ("robust", {}, "tr", 0.1, None),
]
HISTO_ONLY: list[tuple[str, dict[str, Any], str, Any, Any]] = [
    # test_value always acts
    *[(t, {}, "test_value", 3.0, None) for t in TYPES],
    # alternative: one-sided only for the t-test and the signed-rank test
    ("parametric", {}, "alternative", "less", None),
    ("nonparametric", {}, "alternative", "greater", None),
    ("robust", {}, "alternative", "less", ValueError),
    ("bayes", {}, "alternative", "greater", ValueError),
    *[(t, {}, "alternative", "two-sided", None) for t in TYPES],
    # seed: only the robust type resamples
    ("robust", {}, "seed", 3, None),
    *[
        (
            t,
            {},
            "seed",
            3,
            f"seed=3 has no effect: type={t!r} draws no random numbers (only "
            "type='robust' resamples",
        )
        for t in ("parametric", "nonparametric", "bayes")
    ],
]
SCATTER_ONLY: list[tuple[str, dict[str, Any], str, Any, Any]] = [
    # seed: the drawing subsamples with it (SPEC section 3), so it never warns
    *[(t, {}, "seed", 3, None) for t in TYPES],
    # label_var always acts (the point labels)
    *[(t, {}, "label_var", "a", None) for t in TYPES],
]


def _ids(rows: list[tuple[str, dict[str, Any], str, Any, Any]]) -> list[str]:
    return [f"{t}-{name}={value!r}-{extra}" for t, extra, name, value, _ in rows]


def _check(call: Any, name: str, value: Any, gives: Any) -> None:
    if isinstance(gives, type) and issubclass(gives, Exception):
        with pytest.raises(gives):
            call(**{name: value})
        return
    base: tuple[str, ...] = call().warnings
    result: StatsResult = call(**{name: value})
    added = [w for w in result.warnings if w not in base and "has no effect" in w]
    if gives is None:
        assert added == []
    else:
        assert isinstance(gives, str)
        assert len(added) == 1, added
        assert gives in added[0], added[0]
        assert added[0].startswith(f"{name}={value!r} has no effect: ")


@pytest.mark.parametrize(
    ("type_", "extra", "name", "value", "gives"),
    COMMON + HISTO_ONLY,
    ids=_ids(COMMON + HISTO_ONLY),
)
def test_histostats_argument_without_effect(
    type_: str, extra: dict[str, Any], name: str, value: Any, gives: Any
) -> None:
    def call(**kwargs: Any) -> StatsResult:
        return ip.analyze.histostats(ONE, "v", type=type_, **extra, **kwargs)

    _check(call, name, value, gives)


@pytest.mark.parametrize(
    ("type_", "extra", "name", "value", "gives"),
    COMMON + SCATTER_ONLY,
    ids=_ids(COMMON + SCATTER_ONLY),
)
def test_scatterstats_argument_without_effect(
    type_: str, extra: dict[str, Any], name: str, value: Any, gives: Any
) -> None:
    def call(**kwargs: Any) -> StatsResult:
        return ip.analyze.scatterstats(PAIR, "a", "b", type=type_, **extra, **kwargs)

    _check(call, name, value, gives)


def test_the_warnings_name_the_value_as_given() -> None:
    import numpy as np

    # (NumPy 2 prints np.float32(0.8), NumPy 1 prints 0.8: the repr as given.)
    prior = np.float32(0.8)
    result = ip.analyze.histostats(ONE, "v", type="np", bf_prior=prior)
    assert result.warnings[-1].startswith(f"bf_prior={prior!r} has no effect")
    seed = np.int64(5)
    result = ip.analyze.histostats(ONE, "v", seed=seed)
    assert result.warnings[-1].startswith(f"seed={seed!r} has no effect")
