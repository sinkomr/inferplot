"""Statistical methods on clean float64 NumPy arrays (no data frames, no plotting).

Every function takes arrays or plain floats and returns plain Python floats or
frozen dataclasses of them; each docstring cites its method and names the R
function it must match (SPEC.md section 4).
"""

from __future__ import annotations

from inferplot.stats.bayes import jzs_bf10_two_sample, jzs_log_bf10
from inferplot.stats.descriptive import (
    common_scale,
    mean,
    scaled_variance,
    variance,
)
from inferplot.stats.effect_sizes import (
    EffectSizeCI,
    Sidedness,
    hedges_g,
    hedges_j,
    partial_omega_squared,
)
from inferplot.stats.noncentral import f_ncp_bound, t_ncp_interval
from inferplot.stats.padjust import P_ADJUST_METHODS, PAdjustMethod, p_adjust
from inferplot.stats.pairwise import PairResult, games_howell
from inferplot.stats.parametric import (
    ALTERNATIVES,
    Alternative,
    StudentT,
    WelchAnova,
    WelchT,
    student_t,
    welch_anova,
    welch_t,
)
from inferplot.stats.srange import srange_logsf, srange_sf

__all__ = [
    "ALTERNATIVES",
    "P_ADJUST_METHODS",
    "Alternative",
    "EffectSizeCI",
    "PAdjustMethod",
    "PairResult",
    "Sidedness",
    "StudentT",
    "WelchAnova",
    "WelchT",
    "common_scale",
    "f_ncp_bound",
    "games_howell",
    "hedges_g",
    "hedges_j",
    "jzs_bf10_two_sample",
    "jzs_log_bf10",
    "mean",
    "p_adjust",
    "partial_omega_squared",
    "scaled_variance",
    "srange_logsf",
    "srange_sf",
    "student_t",
    "t_ncp_interval",
    "variance",
    "welch_anova",
    "welch_t",
]
