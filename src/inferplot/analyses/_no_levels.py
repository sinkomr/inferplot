"""Argument rules shared by ``histostats`` and ``scatterstats``, the analyses
without levels of x (SPEC.md section 2).

The rules are those of betweenstats and withinstats where they apply: ``tr`` only
with type robust, ``bf_message=True`` only where ggstatsplot draws a Bayes factor
caption (type parametric with the results subtitle), and one ``warnings`` entry
for each argument changed from its default that has no effect (D39). Nothing
here imports matplotlib.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from inferplot import stats

__all__ = [
    "DEFAULT_BF_PRIOR",
    "DEFAULT_CONF_LEVEL",
    "DEFAULT_TR",
    "canonical_alternative",
    "check_bayes_caption",
    "check_type_arguments",
    "no_effect_warnings",
]

# ggstatsplot's defaults of the arguments that can have no effect.
DEFAULT_TR: Final = 0.2
DEFAULT_BF_PRIOR: Final = 0.707
DEFAULT_CONF_LEVEL: Final = 0.95


def canonical_alternative(value: str) -> stats.Alternative:
    """A validated ``alternative`` as the statistics' literal type."""
    for allowed in stats.ALTERNATIVES:
        if value == allowed:
            return allowed
    raise AssertionError(f"unvalidated alternative {value!r}")  # pragma: no cover


def check_type_arguments(
    type_name: str, *, tr: object, trim: float, show_bf: bool | None
) -> None:
    """Arguments that cannot apply to ``type_name`` whatever the data: they would
    change what the reported numbers mean, so they raise instead of being ignored
    as ggstatsplot ignores them (betweenstats' rule and wording)."""
    if trim != DEFAULT_TR and type_name != "robust":
        raise ValueError(
            f"tr={tr!r} sets the trimming of the robust methods (type='robust'), but "
            f"type={type_name!r} trims nothing, so it would have no effect. Remove "
            "tr=, or pass type='robust' to use the trimmed methods."
        )
    if show_bf is True and type_name != "parametric":
        raise ValueError(
            f"bf_message=True asks for a Bayes factor caption, which ggstatsplot "
            f"draws only for type='parametric'; type={type_name!r} has none. Remove "
            "bf_message=True"
            + (
                "; type='bayes' reports the Bayes factor in the subtitle."
                if type_name == "bayes"
                else "."
            )
        )


def check_bayes_caption(*, show_bf: bool | None, subtitle_on: bool) -> None:
    """``bf_message=True`` with ``results_subtitle=False``: ggstatsplot draws the
    caption only with the subtitle."""
    if show_bf is True and not subtitle_on:
        raise ValueError(
            "bf_message=True asks for the Bayes factor caption, which is drawn only "
            "together with the results subtitle; results_subtitle=False turns both "
            "off. Pass results_subtitle=True, or drop bf_message=True."
        )


def no_effect_warnings(
    *,
    type_name: str,
    bayes_computed: bool,
    subtitle_on: bool,
    show_bf: bool | None,
    prior_scale: float,
    conf_level: float,
    passed: Mapping[str, object],
) -> list[str]:
    """One plain-English entry per argument changed from its default that has no
    effect here (D39), naming the value as the caller gave it (``passed``):
    ``bf_prior`` where no Bayes factor is computed, ``bf_message=False`` where no
    caption could be drawn, and ``conf_level`` with type bayes, which shows no
    interval in this version."""
    out: list[str] = []
    if show_bf is False and (type_name != "parametric" or not subtitle_on):
        where = (
            f"type={type_name!r} draws none"
            + (
                "; type='bayes' reports its Bayes factor in the subtitle"
                if type_name == "bayes"
                else ""
            )
            if type_name != "parametric"
            else "results_subtitle=False leaves the Bayes factor caption out"
        )
        out.append(
            "bf_message=False has no effect: the Bayes factor caption is drawn only "
            f"for type='parametric' with the results subtitle (as in ggstatsplot), "
            f"and {where}."
        )
    if not bayes_computed and prior_scale != DEFAULT_BF_PRIOR:
        if type_name in ("nonparametric", "robust"):
            why = f"type={type_name!r} has no Bayes factor"
        elif not subtitle_on:
            why = "results_subtitle=False leaves the Bayes factor out"
        else:
            why = f"bf_message={show_bf!r} leaves the Bayes factor out"
        out.append(
            f"bf_prior={passed.get('bf_prior', prior_scale)!r} has no effect: no "
            f"Bayes factor is computed ({why})."
        )
    if type_name == "bayes" and conf_level != DEFAULT_CONF_LEVEL:
        out.append(
            f"conf_level={passed.get('conf_level', conf_level)!r} has no effect: "
            "type='bayes' shows no interval in this version (the posterior "
            "credible interval arrives in inferplot 0.2)."
        )
    return out
