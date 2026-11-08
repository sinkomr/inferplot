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

- [ ] `_data.py` boundary (pandas, polars, pyarrow) and `_validate.py` (aliases, did-you-mean, removed-argument messages)
- [ ] Welch t and ANOVA, Hedges' g and partial omega squared with CIs, Games-Howell then Holm (accurate studentized range tail, D31)
- [ ] Two-group JZS Bayes factor caption (D32)
- [ ] `StatsResult`, `text.py` (subtitle parity, APA, caption, alt text), `check()`
- [ ] Renderer (violin, box, points, centrality label, brackets, subtitle, caption) and `save()`
- [ ] Oracle parity tests pass for every betweenstats parametric fixture

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

- Decided (D31): Games-Howell p is computed from an accurate studentized range tail,
  not R's ptukey, which is off by up to about 5e-10 in the far tail (iris
  setosa-versicolor: ggstatsplot 1.41e-09, true about 2.2e-14). SciPy's
  `studentized_range.sf` is no ground truth there either (1.9e-14, 14% low; it can
  return 0), so Phase 1 needs its own stable tail computation.
- The generator handles ggbetweenstats only: `between_analysis`,
  `statsexpressions_rows` (`paired = FALSE`) and `bootstrap_candidates` stop on any
  other function. Each one needs its own bookkeeping (complete subjects for
  withinstats via `statsExpressions::complete_subjects`, one sample for histostats)
  and bootstrap replay (withinstats: `kendalls_w`, `dep.effect`, `wmcpAKP`; histostats
  robust: `trimcibt`) before its cases are added, and so do `name_hidden`, the caption
  rule in `outcome_problems` and `label_counts`, which assume ggbetweenstats (barstats
  computes its pairwise tests through the same hidden helper, and `.chisq_test_safe`
  swallows errors into NA rows the recorders cannot see). A case that declares a
  swallowed subtitle (`expect_hidden_errors`) would still get a bootstrap reference;
  skip it in `run_case` with the first such case. Unused edge CSVs: `edge_n2`,
  `edge_ties`, `edge_negative`, `edge_empty_level`, `edge_missing`, `edge_outliers`,
  `edge_within_missing`.
- Phase 1: Hedges' g CI bounds agree with a tight SciPy inversion to 7.5e-7 relative
  and up to 1.9e-8 absolute; set `noncentral_ci` from the implementation's own
  measurement and add an absolute part before a fixture has a bound near 0.
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
