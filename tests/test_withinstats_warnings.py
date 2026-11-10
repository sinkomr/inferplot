"""Arguments without effect in withinstats (DECISIONS.md D39), as a table.

Every withinstats argument, for every type and for two and for three conditions:
a value that changes nothing reported or drawn adds one ``warnings`` entry naming
the argument and the value as given; a value that would change what the reported
numbers mean raises; a value with an effect, a default and a default's alias add
nothing. The entries are compared with the warnings of the same call without the
argument, so each row isolates one argument.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot.result import StatsResult


def _long(columns: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {"subject": f"s{i}", "condition": level, "score": value}
        for level, values in columns.items()
        for i, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


TWO = _long(
    {
        "pre": [1.0, 2.0, 4.0, 3.5, 6.0, 2.2, 5.1, 3.3],
        "post": [2.0, 2.5, 5.5, 3.0, 8.0, 3.9, 5.0, 4.4],
    }
)
THREE = _long(
    {
        "a": [1.0, 2.0, 4.0, 3.5, 6.0, 2.2, 5.1, 3.3],
        "b": [2.0, 2.5, 5.5, 3.0, 8.0, 3.9, 5.0, 4.4],
        "c": [0.5, 2.5, 3.0, 4.5, 6.5, 2.9, 4.0, 3.1],
    }
)
_analyze: Any = ip.analyze.withinstats
_render: Any = ip.render.withinstats

_PAIRS2 = "x='condition' has 2 conditions, so there are no pairwise comparisons to"
_NO_CAPTION = "the Bayes factor caption is drawn only for type='parametric'"

# (type, conditions, extra keywords of the baseline call, the argument under
# test, its value, what it gives): None (no new entry), a substring of the one
# new warnings entry, or an exception type.
ROWS: list[tuple[str, int, dict[str, Any], str, Any, Any]] = [
    # pairwise arguments with two conditions, every type
    *[
        (t, 2, {}, name, value, f"{name}={value!r} has no effect: {_PAIRS2} {what}.")
        for t in ("parametric", "nonparametric", "robust", "bayes")
        for name, value, what in (
            ("p_adjust_method", "BH", "adjust"),
            ("pairwise_display", "ns", "display"),
            ("pairwise_display", "all", "display"),
            ("pairwise_display", "none", "display"),
            ("pairwise_alpha", 0.1, "call significant"),
        )
    ],
    # ... their defaults and aliases
    *[
        (t, k, {}, name, value, None)
        for t in ("parametric", "nonparametric", "robust")
        for k in (2, 3)
        for name, value in (
            ("p_adjust_method", "holm"),
            ("pairwise_display", "s"),
            ("pairwise_display", "significant"),
            ("pairwise_alpha", 0.05),
            ("bf_prior", 0.707),
            ("conf_level", 0.95),
            ("tr", 0.2),
            ("bf_message", None),
            ("seed", 0),
        )
    ],
    # ... and with three conditions, where they act (frequentist types)
    *[
        (t, 3, {}, name, value, None)
        for t in ("parametric", "nonparametric", "robust")
        for name, value in (
            ("p_adjust_method", "BH"),
            ("pairwise_display", "ns"),
            ("pairwise_display", "all"),
            ("pairwise_display", "none"),
            ("pairwise_alpha", 0.1),
        )
    ],
    # bayes with three conditions: Bayes factors are neither adjusted nor tested
    (
        "bayes",
        3,
        {"results_subtitle": False},
        "p_adjust_method",
        "BH",
        "p_adjust_method='BH' has no effect: Bayes factors are not adjusted",
    ),
    (
        "bayes",
        3,
        {"results_subtitle": False},
        "pairwise_alpha",
        0.1,
        "pairwise_alpha=0.1 has no effect: Bayes factors are not compared",
    ),
    (
        "bayes",
        3,
        {"results_subtitle": False},
        "pairwise_display",
        "ns",
        "pairwise_display='ns' has no effect: Bayes factors have no significance",
    ),
    ("bayes", 3, {"results_subtitle": False}, "pairwise_display", "all", None),
    ("bayes", 3, {"results_subtitle": False}, "pairwise_display", "none", None),
    # conf_level: no interval of type bayes, robust intervals always 95%
    ("parametric", 2, {}, "conf_level", 0.9, None),
    ("parametric", 3, {}, "conf_level", 0.9, None),
    ("nonparametric", 2, {}, "conf_level", 0.9, None),
    ("nonparametric", 3, {}, "conf_level", 0.9, None),
    (
        "robust",
        2,
        {},
        "conf_level",
        0.9,
        "conf_level=0.9 has no effect: the AKP effect size's bootstrap interval is",
    ),
    (
        "robust",
        3,
        {},
        "conf_level",
        0.9,
        "conf_level=0.9 has no effect: the average AKP effect size's bootstrap",
    ),
    (
        "bayes",
        2,
        {},
        "conf_level",
        0.9,
        "conf_level=0.9 has no effect: type='bayes' shows no interval",
    ),
    (
        "bayes",
        3,
        {"results_subtitle": False},
        "conf_level",
        0.9,
        "conf_level=0.9 has no effect: type='bayes' shows no interval",
    ),
    # bf_prior: only where a Bayes factor is computed with it
    ("parametric", 2, {}, "bf_prior", 0.8, None),
    (
        "parametric",
        2,
        {"bf_message": False},
        "bf_prior",
        0.8,
        "bf_prior=0.8 has no effect: no Bayes factor is computed (bf_message=False",
    ),
    (
        "parametric",
        2,
        {"results_subtitle": False},
        "bf_prior",
        0.8,
        "bf_prior=0.8 has no effect: no Bayes factor is computed (results_subtitle",
    ),
    (
        "parametric",
        3,
        {},
        "bf_prior",
        0.8,
        "bf_prior=0.8 has no effect: no Bayes factor is computed (x='condition' has "
        "3 conditions",
    ),
    (
        "nonparametric",
        2,
        {},
        "bf_prior",
        0.8,
        "(type='nonparametric' has no Bayes factor).",
    ),
    (
        "nonparametric",
        3,
        {},
        "bf_prior",
        0.8,
        "(type='nonparametric' has no Bayes factor).",
    ),
    ("robust", 2, {}, "bf_prior", 0.8, "(type='robust' has no Bayes factor)."),
    ("robust", 3, {}, "bf_prior", 0.8, "(type='robust' has no Bayes factor)."),
    ("bayes", 2, {}, "bf_prior", 0.8, None),
    (
        "bayes",
        2,
        {"results_subtitle": False},
        "bf_prior",
        0.8,
        "(results_subtitle=False leaves the Bayes factor out).",
    ),
    (
        "bayes",
        3,
        {"results_subtitle": False},
        "bf_prior",
        0.8,
        "bf_prior=0.8 has no effect: the pairwise Bayes factors always use the prior "
        "scale 0.707",
    ),
    # bf_message: the parametric caption only
    ("parametric", 2, {}, "bf_message", False, None),
    ("parametric", 2, {}, "bf_message", True, None),
    ("parametric", 3, {}, "bf_message", False, None),  # silences the v0.2 note
    ("parametric", 3, {}, "bf_message", True, NotImplementedError),
    (
        "parametric",
        2,
        {"results_subtitle": False},
        "bf_message",
        False,
        "bf_message=False has no effect: results_subtitle=False leaves the Bayes "
        "factor caption out.",
    ),
    ("parametric", 2, {"results_subtitle": False}, "bf_message", True, ValueError),
    *[
        (
            t,
            k,
            {"results_subtitle": False} if (t, k) == ("bayes", 3) else {},
            "bf_message",
            False,
            f"bf_message=False has no effect: {_NO_CAPTION}",
        )
        for t in ("nonparametric", "robust", "bayes")
        for k in (2, 3)
    ],
    *[
        (t, 2, {}, "bf_message", True, ValueError)
        for t in ("nonparametric", "robust", "bayes")
    ],
    # tr: only the robust type trims
    *[(t, 2, {}, "tr", 0.1, ValueError) for t in ("parametric", "nonparametric")],
    ("bayes", 2, {}, "tr", 0.1, ValueError),
    ("robust", 2, {}, "tr", 0.1, None),
    ("robust", 3, {}, "tr", 0.1, None),
    # alternative: one-sided only for two conditions, parametric or nonparametric
    ("parametric", 2, {}, "alternative", "less", None),
    ("nonparametric", 2, {}, "alternative", "greater", None),
    ("robust", 2, {}, "alternative", "less", ValueError),
    ("bayes", 2, {}, "alternative", "less", ValueError),
    ("parametric", 3, {}, "alternative", "less", ValueError),
    ("nonparametric", 3, {}, "alternative", "less", ValueError),
    # seed, digits and order always act (the jitter is seeded; every text; the
    # level order)
    *[
        (
            t,
            k,
            {"results_subtitle": False} if (t, k) == ("bayes", 3) else {},
            name,
            value,
            None,
        )
        for t in ("parametric", "nonparametric", "robust", "bayes")
        for k in (2, 3)
        for name, value in (("seed", 3), ("digits", 3))
    ],
]


def _ids() -> list[str]:
    return [
        f"{t}-k{k}-{name}={value!r}-{extra}" for t, k, extra, name, value, _ in ROWS
    ]


@pytest.mark.parametrize(
    ("type_", "k", "extra", "name", "value", "gives"), ROWS, ids=_ids()
)
def test_argument_without_effect(
    type_: str, k: int, extra: dict[str, Any], name: str, value: Any, gives: Any
) -> None:
    frame = TWO if k == 2 else THREE

    def call(**kwargs: Any) -> StatsResult:
        return _analyze(
            frame,
            "condition",
            "score",
            subject_id="subject",
            type=type_,
            **extra,
            **kwargs,
        )

    if isinstance(gives, type) and issubclass(gives, Exception):
        with pytest.raises(gives):
            call(**{name: value})
        return
    base = call().warnings
    warnings = call(**{name: value}).warnings
    # Other entries (a one-sided caption note, infinite bootstrap values) are
    # about the data or the method, not about an argument without effect.
    added = [w for w in warnings if w not in base and "has no effect" in w]
    if gives is None:
        assert added == []
    else:
        assert isinstance(gives, str)
        assert len(added) == 1, added
        assert gives in added[0]
        assert added[0].startswith(f"{name}={value!r} has no effect: ")


# --------------------------------------------------------------------------- #
# drawing arguments
# --------------------------------------------------------------------------- #

# (conditions, drawing keywords, the one new entry or None)
DRAWING_ROWS: list[tuple[int, dict[str, Any], str | None]] = [
    (2, {"point_path": True}, None),
    (2, {"point_path": False}, None),
    (
        2,
        {"layers": ("violin", "box"), "point_path": False},
        "point_path=False has no effect: subject paths join the points",
    ),
    (
        2,
        {"layers": ("violin", "box"), "point_path": True},
        "point_path=True has no effect: subject paths join the points",
    ),
    (2, {"layers": ("violin", "box")}, None),
    (
        3,
        {"point_path": False},
        "point_path=False has no effect: subject paths are drawn only with two",
    ),
    (
        3,
        {"point_path": True},
        "point_path=True has no effect: subject paths are drawn only with two",
    ),
    (3, {}, None),
    (2, {"centrality_path": True}, None),
    (2, {"centrality_path": False}, None),
    (2, {"centrality_plotting": False}, None),
    (
        2,
        {"centrality_plotting": False, "centrality_path": False},
        "centrality_path=False has no effect: the centrality path joins the "
        "centrality points",
    ),
    (
        2,
        {"centrality_plotting": False, "centrality_path": True},
        "centrality_path=True has no effect: the centrality path joins the "
        "centrality points",
    ),
    (
        3,
        {"centrality_plotting": False, "centrality_path": True},
        "centrality_path=True has no effect",
    ),
]


@pytest.mark.parametrize(("k", "kwargs", "gives"), DRAWING_ROWS)
def test_drawing_argument_without_effect(
    k: int, kwargs: dict[str, Any], gives: str | None
) -> None:
    result = _analyze(
        TWO if k == 2 else THREE, "condition", "score", subject_id="subject"
    )
    added = list(_render(result, **kwargs).result.warnings[len(result.warnings) :])
    if gives is None:
        assert added == []
    else:
        assert len(added) == 1, added
        assert added[0].startswith(gives)
