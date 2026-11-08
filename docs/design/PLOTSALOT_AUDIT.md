# plotsalot audit (2026-11-07)

plotsalot 1.0.0 (https://github.com/joshuamyers22/plotsalot, MIT) is the closest prior
Python adaptation of ggstatsplot. Verdict: correct for what it implements, not a
ggstatsplot equivalent, not worth forking. Method: source and document review,
independent SciPy recomputation of its retained outputs, and comparison with the
ggstatsplot 1.1.1 output frozen in its oracle. Its test suite was not run (PyPI was
blocked in the audit sandbox).

## Findings

| Check | Result | Evidence |
| --- | --- | --- |
| Correctness of implemented methods | Pass | 37 retained outputs matched independent SciPy within 7.6e-14 relative |
| R oracle reproducible in Python | Pass | ggstatsplot 1.1.1 Welch F, omega squared and Games-Howell q matched exactly; Games-Howell + Holm p within 2.6e-7 |
| Nonparametric coverage | Fail | no `type="nonparametric"` anywhere |
| Effect-size confidence intervals | Fail | point estimates only, except Cramér's V and Cohen's w |
| pandas input | Fail | raises `TypeError: data must be a polars.DataFrame` |
| Scatter plot | Fail | no regression line, no marginal distributions |
| Multi-panel output | Fail | `combine_plots` rasterizes panels |
| Oracle pinning | Stale | ggstatsplot 1.1.1; 1.1.3 and 1.1.4 later fixed result bugs |
| Agent-facing errors | Mixed | strict signatures with no `**kwargs`; but Bayesian calls require a raw-unit prior scale, and warnings contain milestone jargon ("not approved for M6A") |
| Provenance | Concern | 34 commits by one author over 47 hours; 21.7k lines of code; 104 markdown documents; 0 stars |

## Divergences from ggstatsplot on identical data

| Quantity | ggstatsplot | plotsalot |
| --- | --- | --- |
| Pairwise after 3-group Welch ANOVA (oracle `m3-between`) | Games-Howell + Holm: p = 0.963, 0.113, 0.113; none significant | Welch t + Holm: p = 0.803, 0.0499, 0.0499; two significant |
| One-sample effect size, x = 1..5 vs 0 | Hedges' g ≈ 1.51 | Cohen's d = 1.90 |
| Bayes factor, same data | JZS, r = 0.707: BF10 = 5.76 | normal-inverse-gamma prior, scale 2.0 outcome units: 13.3 |

## What we keep from plotsalot

Analyze/render split; frozen typed results; keyword-only allowlisted signatures;
explicit subject IDs with duplicate detection; seeded randomness with recorded seeds;
atomic failure instead of partial results; an R oracle with pinned versions and hashed
fixtures. Its `oracle/generate.R` (MIT) is a usable starting point.

## Reproducing

`tools/reference/stats_reference.py` contains the independent implementations used
here and runs known-answer checks against the frozen ggstatsplot 1.1.1 values.
