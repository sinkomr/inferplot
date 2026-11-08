# Status

Read this at the start of every session; update it at the end of every slice.
A slice is done only when its definition of done in `CLAUDE.md` is met.

Current phase: **1 — betweenstats, parametric, end to end** (Phase 0 done 2026-11-08)

## Phase 0 — Scaffold

- [x] `pyproject.toml` (hatchling, src layout, `requires-python >= 3.11`), `uv.lock`, MIT `LICENSE` ("inferplot contributors"), `NOTICE` (ggstatsplot MIT notice; plotsalot credit)
- [x] ruff, pyright (strict on `src/`), pytest, hypothesis, pytest-mpl configured
- [x] `.github/workflows/ci.yml`: lint, types, tests on Ubuntu, macOS, Windows × Python 3.11 and the newest Python with numpy, scipy and matplotlib wheels; one job with `uv sync --resolution lowest-direct`
- [x] `oracle/` with `generate.R` and datasets; `.github/workflows/oracle.yml`; first fixtures (ggbetweenstats, all types, three datasets) committed under `tests/fixtures/oracle/`
- [x] `tests/tolerances.py` and a fixture loader; `tools/reference/stats_reference.py` checks run in CI

Phase 0 notes: fixtures come from oracle run 42462799705 (R 4.5.3, ggstatsplot
1.1.4, statsExpressions 2.2.1, Posit snapshot 2026-10-23): 45 ggbetweenstats cases,
namely all four types on ten dataset and variable pairs from eight CSV files (40),
one-sided alternatives on ToothGrowth (4) and mtcars `am` x `drat` (parametric). In
every fixture `tests/test_reference_vs_oracle.py` independently reproduces each test
statistic, df and p-value, omnibus and pairwise; Hedges' g and the rank-biserial r with
their CIs; the partial omega squared estimate; and every JZS Bayes factor (two-group
subtitles, parametric captions, pairs). Games-Howell p-values match only to R's ptukey
precision (open below). Not reproduced: the partial omega squared CI (Phase 1 parity),
the Bayes ANOVA (the caption of a parametric test with three or more groups and the
subtitle of the bayes type) and posterior estimates (v0.2), and the bootstrapped effect
sizes, whose method is proved instead by replay: for all 17 bootstrapped effect-size
CIs (3 AKP, 7 t1way, 7 rank epsilon squared) the oracle reproduces ggstatsplot's own
100-resample values exactly, so the 20,000-resample references use the same method.
The centrality labels' bootstrap CIs, which ggstatsplot computes but never shows, are
not replayed. Both end-of-phase reviews' findings are fixed or listed below, and a
third, fresh-context review confirmed the fixes; oracle findings that changed SPEC.md
are logged in DECISIONS.md.

## Phase 1 — betweenstats, parametric, end to end

- [x] `_data.py` boundary (pandas, polars, pyarrow) and `_validate.py` (aliases, did-you-mean, removed-argument messages)
- [x] Welch t and ANOVA, Hedges' g and partial omega squared with CIs, Games-Howell then Holm (accurate studentized range tail, D31): `inferplot.stats` reproduces every completed parametric fixture's statistics (`tests/stats/test_parity.py`); end-to-end parity is the last item
- [ ] Two-group JZS Bayes factor caption (D32)
- [ ] `StatsResult`, `text.py` (subtitle parity, APA, caption, alt text), `check()`
- [ ] Renderer (violin, box, points, centrality label, brackets, subtitle, caption) and `save()`
- [ ] Oracle parity tests pass for every betweenstats parametric fixture

Phase 1 notes: fixtures come from oracle run 42522555302: the 45 Phase 0 cases,
unchanged, plus 28 parametric ggbetweenstats cases (edge data and every argument the
parametric slice exposes) and four probe files (number formatting, `p.adjust`,
`ptukey`, the omega squared CI). In `edge_n2` (two values per group) ggstatsplot 1.1.4
itself fails, because R's ptukey returns NaN below 2 df; the fixture keeps the
subtitle, caption and labels computed before the failure, and inferplot computes the
accurate p-values there (D37, flagged for the owner). The probes pinned every number
formatting rule except negative zero, showed that ggstatsplot's adjusted Games-Howell p
is exactly `p.adjust` of gamesHowellTest's raw p, measured ptukey's error by df, and
confirmed the omega squared CI construction (D33, flagged for the owner).

## Phase 2 — betweenstats nonparametric, robust, bayes

- [ ] nonparametric (Mann-Whitney / Kruskal-Wallis, Dunn)
- [ ] robust (Yuen / t1way, lincon, AKP and explanatory effect sizes with bootstrap CIs)
- [ ] bayes (two-group JZS Bayes factor; MAP centrality)

## Phase 3 — withinstats, all types

- [ ] parametric, nonparametric, robust, bayes (two conditions); `subject_id` handling

## Phase 4 — histostats, scatterstats, barstats

- [ ] histostats, all types
- [ ] scatterstats, all types (line, band, marginals)
- [ ] barstats (independence, McNemar, goodness of fit, proportion tests)

## Phase 5 — documentation, evaluation, release 0.1.0

- [ ] README, `docs/api.md` (size test), `llms.txt`, `skills/inferplot/SKILL.md`, examples
- [ ] simulation tests (`slow` marker)
- [ ] agent evaluation harness (ask the human before running it)
- [ ] release workflow; human publishes to TestPyPI, then PyPI

## v0.2 backlog

Bayes ANOVA and contingency, posterior estimates, dotplotstats, piestats, corrmat,
coefstats, `grouping_var`, replicate-aware SuperPlots.

## Known gaps and open oracle discrepancies

- Decided (D31, D37): Games-Howell p is computed from an accurate studentized range
  tail (`inferplot.stats.srange_sf`), not R's ptukey, whose error grows as the df falls
  (probe-ptukey: up to 1.35e-3 absolute at df 2, 1.8e-7 at df 10, about 4.7e-10 in the
  far tail; NaN below 2 df). `games_howell_p` (atol 3e-9) holds on every fixture, but
  InsectSprays B-E already uses 2.8e-9 of it (ptukey's error times the Holm multiplier).
  A fixture with more groups or smaller df will exceed it; its entry must then be
  rebuilt from measurement with the reason it has (R's raw error, which the raw-p
  check against probe-ptukey bounds, times the adjustment's multiplier, as `wrs2_p`
  scales), never widened to fit.
- D33 and D37 follow the owner's D31 precedent (accurate value over R's numerical
  accident) and await the owner's confirmation.
- R's `pt` with a noncentrality parameter is documented as less accurate for large ncp
  (`?TDist`); no fixture has a two-group \|t\| above 3.7, so Hedges' g CI parity at
  large \|t\| is untested.
- The generator handles ggbetweenstats only: `between_analysis`,
  `statsexpressions_rows` (`paired = FALSE`) and `bootstrap_candidates` stop on any
  other function. Each one needs its own bookkeeping (complete subjects for
  withinstats via `statsExpressions::complete_subjects`, one sample for histostats)
  and bootstrap replay (withinstats: `kendalls_w`, `dep.effect`, `wmcpAKP`; histostats
  robust: `trimcibt`) before its cases are added, and so do `name_hidden`, the caption
  rule in `outcome_problems` and `label_counts`, which assume ggbetweenstats (barstats
  computes its pairwise tests through the same hidden helper, and `.chisq_test_safe`
  swallows errors into NA rows the recorders cannot see). Unused edge CSV:
  `edge_within_missing` (Phase 3).
- Phase 2: compute WRS2's p-values as R does, 1 - cdf (`wrs2_p`; the statistics rule
  has an exception for it). Every fixture's p is large enough that `sf` passes too, so
  add a robust case whose p underflows (for example iris Petal.Length) to pin how
  ggstatsplot prints a p of 0 or one rounded on the double grid; and a two-group robust
  case with tiny p to confirm Yuen's 1 - pt. lincon's critical values need the
  published Studentized maximum modulus table (C = 3 at df 5 is 3.39, not the rounded
  3.40); no fixture has a Yuen df above 200 or alpha other than 0.05.
- Phases 3 and 4: `ttest_bayes_factor` was measured on two-sample Bayes factors only;
  measure paired and one-sample ones against an exact integral when their fixtures
  arrive, and log any change.
- AKP: WRS2's scaling constant differs from the exact winsorized-normal value by
  1.2e-9 relative (review measurement), which is above the 1e-10 closed-form tolerance.
  Phase 2 must match WRS2's constant or register a measured tolerance.
- MAP centrality: the oracle records the values; the grid behind SPEC §10's "one grid
  step" tolerance is bayestestR's 2^10-point density (help page), to confirm in Phase 2.
- ggstatsplot's own bootstrap CIs use 100 resamples, so they differ from the 20,000-
  resample references by Monte Carlo noise; parity tests compare against the
  references (SPEC §10, D29), not against ggstatsplot's 100-resample values.
