"""Statistical methods on clean float64 NumPy arrays (no data frames, no plotting).

Every function takes arrays or plain floats and returns plain Python floats or
frozen dataclasses of them; each docstring cites its method and names the R
function it must match (SPEC.md section 4).
"""

from __future__ import annotations

from inferplot.stats.bayes import jzs_bf10_two_sample, jzs_log_bf10
from inferplot.stats.density import MapEstimate, bw_sj, map_estimate
from inferplot.stats.descriptive import (
    common_scale,
    mean,
    median,
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
from inferplot.stats.nonparametric import (
    BOOTSTRAP_RESAMPLES,
    BootstrapEffectSizeCI,
    DunnPair,
    KruskalWallis,
    MannWhitney,
    dunn,
    kruskal_wallis,
    mann_whitney,
    rank_biserial,
    rank_epsilon_squared,
)
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
from inferplot.stats.robust import (
    LinconPair,
    TrimmedMeansAnova,
    YuenT,
    akp_effect,
    explanatory_effect,
    lincon,
    smm_critical_value,
    trimmed_mean,
    trimmed_means_anova,
    winsorized_normal_variance,
    winsorized_variance,
    yuen_t,
)
from inferplot.stats.smm import smm_cdf, smm_quantile
from inferplot.stats.srange import srange_logsf, srange_sf

__all__ = [
    "ALTERNATIVES",
    "BOOTSTRAP_RESAMPLES",
    "P_ADJUST_METHODS",
    "Alternative",
    "BootstrapEffectSizeCI",
    "DunnPair",
    "EffectSizeCI",
    "KruskalWallis",
    "LinconPair",
    "MannWhitney",
    "MapEstimate",
    "PAdjustMethod",
    "PairResult",
    "Sidedness",
    "StudentT",
    "TrimmedMeansAnova",
    "WelchAnova",
    "WelchT",
    "YuenT",
    "akp_effect",
    "bw_sj",
    "common_scale",
    "dunn",
    "explanatory_effect",
    "f_ncp_bound",
    "games_howell",
    "hedges_g",
    "hedges_j",
    "jzs_bf10_two_sample",
    "jzs_log_bf10",
    "kruskal_wallis",
    "lincon",
    "mann_whitney",
    "map_estimate",
    "mean",
    "median",
    "p_adjust",
    "partial_omega_squared",
    "rank_biserial",
    "rank_epsilon_squared",
    "scaled_variance",
    "smm_cdf",
    "smm_critical_value",
    "smm_quantile",
    "srange_logsf",
    "srange_sf",
    "student_t",
    "t_ncp_interval",
    "trimmed_mean",
    "trimmed_means_anova",
    "variance",
    "welch_anova",
    "welch_t",
    "winsorized_normal_variance",
    "winsorized_variance",
    "yuen_t",
]
