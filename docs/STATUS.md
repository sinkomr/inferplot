# Status

Read this at the start of every session; update it at the end of every slice.
A slice is done only when its definition of done in `CLAUDE.md` is met.

Current phase: **2 — betweenstats nonparametric, robust, bayes** (Phase 1 merged to main
2026-11-08; its close-out findings are open below)

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
- [x] Two-group JZS Bayes factor caption (D32): every two-sample ttestBF value within 1.5e-9; the caption is ggstatsplot's first and last items, AST-equal on every two-group parametric fixture
- [x] `StatsResult`, `text.py` (subtitle parity, APA, caption, alt text), `check()`: every fixture expression AST-equal when rebuilt from its numbers; number formatting reproduces all 5,010 probe-format rows
- [x] Renderer (violin, box, points, centrality label, brackets, subtitle, caption) and `save()`: all 40 completed parametric fixtures render with no clipped or overlapping text; SVG, PDF and PNG round trips; identical bytes on re-save
- [x] Oracle parity tests pass for every betweenstats parametric fixture: `ip.analyze.betweenstats` matches all 40 completed parametric fixtures on pandas, polars and pyarrow (`tests/test_betweenstats_parity.py`); three bracket labels differ by design (D31, SPEC §5 item 6); `edge_n2`, where ggstatsplot fails, matches its partial outputs (D37)

Slices: two groups (ToothGrowth) and three or more groups (iris) render to SVG with
`fig.check()` empty (`tests/test_betweenstats_end_to_end.py`).

End-of-phase review: done. A fresh-context review of the Phase 1 diff against SPEC.md
confirmed 21 findings, all fixed. The four major ones: the omega squared interval took
minutes, and crashed near F = 1e17, for very well separated groups (now at most 0.04 s
at any F); a bracket label drawn across another bracket's line went unreported with
`ax=`; artist dicts for layers that are not drawn were ignored, and a bad one left a
half-drawn Axes; statsExpressions' templates were adapted without its notice (NOTICE
added; the D5 question is below). The others: statistics that depended on the units
of y (now bit-identical under any power of two), -0.00 for Hedges' g at df 1,
arguments with no effect accepted silently (now a warning each, D39), Games-Howell
references for every fixture pair (109), coverage simulated at nonzero effects,
`order=` with NumPy values, user text as mathtext (D40), clearer messages, and stale
counts. The lead's review of the fixes found that very wide ranges between groups gave
silent errors or NaN; such data now compute exactly or raise a plain error (SPEC §4,
"Scale and range").

Phase 1 notes: fixtures come from oracle run 42522555302: the 45 Phase 0 cases,
unchanged, plus 28 parametric ggbetweenstats cases (edge data and every argument the
parametric slice exposes) and four probe files (number formatting, `p.adjust`,
`ptukey`, the omega squared CI). In `edge_n2` (two values per group) ggstatsplot 1.1.4
itself fails, because R's ptukey returns NaN below 2 df; the fixture keeps the
subtitle, caption and labels computed before the failure, and inferplot computes the
accurate p-values there (D37, confirmed by the owner). The probes pinned every number
formatting rule except negative zero, showed that ggstatsplot's adjusted Games-Howell p
is exactly `p.adjust` of gamesHowellTest's raw p, measured ptukey's error by df, and
confirmed the omega squared CI construction (D33, confirmed by the owner). The owner
extended D5's exception to statsExpressions, whose templates the text layer adapts.

Phase 1 close-out. A final fresh-context verification of the review fixes at 0469bdc
was cut off by a restart; its rendering and documentation parts finished and
confirmed 9 problems. The four documentation ones are fixed on main (a broken SPEC §2
table, the Bayes factor count 35 to 36 re-measured, the D31 row's 2 of 31 to 3 of
106, a stale large-|t| note). Open:

- [x] `point_args` values matplotlib checks only when drawing (`{'linewidth': 'bad'}`)
  pass the pre-draw check: with `ax=` the call draws into the user's Axes and a later
  `savefig` fails; without it the user gets matplotlib's raw message (SPEC §2; major).
  Fixed: each artist dict styles a scratch artist that is drawn and written as SVG and
  PDF first; all 603 bad values tried are refused as `ValueError`
- [x] a known property with a wrongly typed value (`alpha='x'`) is reported as an
  unknown property, `TypeError` instead of `ValueError` (SPEC §2)
- [x] `ip.render.betweenstats(df, x='g', y='y')` suggests `ax='g'` instead of saying
  it draws the result of `ip.analyze.betweenstats`
- [x] `ip.analyze.betweenstats(df, 'g', 'y', ax=None)` suggests `x=None`, which the
  call already gives
- [x] `fig.check()` reports sub-pixel contact (under 0.5 px) between a bracket leg and
  a label as a crossing on large figures. Fixed with the bracket layout (D38): levels are
  spaced in points, so three levels take 4.8 in instead of 7.3 in and InsectSprays with
  every pair 7.15 in instead of 9.8 in
- [ ] the statistics part did not finish. Its unjudged leads: `variance()` on subnormal
  data; the noncentral F lower tail near 1e-300 (log P off by about 3); the noncentral
  F next to the Poisson-mean switch with df2 = 0.5 (1.6e-10 relative in log P) and at
  df2 = 1e8 (4.6e-11); the refusal of a group whose values span more than the largest
  double. Re-run from a fresh context.

## Phase 2 — betweenstats nonparametric, robust, bayes

- [ ] nonparametric (Mann-Whitney / Kruskal-Wallis, Dunn). Statistics layer done
  (`inferplot.stats.nonparametric`): every nonparametric fixture reproduced, closed-form
  values within 1.9e-14 relative, the rank epsilon squared lower bound within 1.21
  Monte Carlo SDs of the 20,000-resample reference; analysis, text and rendering next
- [ ] robust (Yuen / t1way, lincon, AKP and explanatory effect sizes with bootstrap CIs)
- [ ] bayes (two-group JZS Bayes factor; MAP centrality). MAP centrality done
  (`inferplot.stats.map_estimate`): all 30 MAP labels of the 10 bayes fixtures on
  ggstatsplot's grid point; v0.1 behaviour decided (D41); analysis, text and rendering
  next

## Phase 3 — withinstats, all types

- [ ] parametric, nonparametric, robust, bayes (two conditions); `subject_id` handling

## Phase 4 — histostats, scatterstats, barstats

- [ ] histostats, all types
- [ ] scatterstats, all types (line, band, marginals)
- [ ] barstats (independence, McNemar, goodness of fit, proportion tests)

## Phase 5 — documentation, evaluation, release 0.1.0

- [ ] README, `docs/api.md` (size test), `llms.txt`, `skills/inferplot/SKILL.md`, examples
- [ ] simulation tests (`slow` marker); started in Phase 1: Welch t and ANOVA Type I error under the null (0.0475, 0.0515) and Hedges' g CI coverage at nonzero standardized differences (0.9445, 0.9480, 0.9515), with unequal variances and n (`tests/test_simulation.py`)
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
- Renderer: pytest-mpl image baselines are not generated yet (Linux CI only; candidates:
  ToothGrowth, PlantGrowth-display-all, InsectSprays, a subsampled panel, an `ax=`
  figure); in progress after the Phase 1 close-out. With `ax=`,
  inferplot sets its font only on the axis texts and the texts it draws; the user's own
  texts keep theirs.
- Negative zero in number formatting is not pinned (R received the probe's -0 as +0);
  inferplot prints "-0.00" as C `sprintf` does.
- Wide dynamic range: one constant group with a value of at least about 1e154 times the
  other group's standard deviation raises (SPEC §5 item 8), although its t fits in a
  double; a second scale per group could compute it. Not worth it for such data unless
  a user needs it.
- R's `pt` with a noncentrality parameter is documented as less accurate for large ncp
  (`?TDist`); the largest two-group \|t\| in the fixtures is 5.65 (mtcars `am` x `drat`),
  whose Hedges' g CI matches within `noncentral_ci`; parity above that is untested.
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
- MAP centrality: reproduced at the same grid point on all 30 fixture labels
  (`MAP_CENTRALITY`, 0 steps), by matching the fixtures. Not yet pinned (every fixture
  value is positive, n <= 93): bins of negative values (truncation toward zero
  assumed), R's binning above n = 500, the tolerance after the lower search end is
  enlarged, integer bins once |x| exceeds 2^31 bin widths, and what bayestestR does
  when bw.SJ has no answer (n = 2, ties, zero IQR; inferplot raises `ValueError`).
  probe-map settles these.
- Rank epsilon squared: the resampling scheme (within groups) and percentile rule are
  settled by Monte Carlo agreement and the boot.ci help page, not by exact replay. A
  resample with every value tied raises; R's handling of it is unverified until a
  heavy-ties fixture arrives.
- ggstatsplot's own bootstrap CIs use 100 resamples, so they differ from the 20,000-
  resample references by Monte Carlo noise; parity tests compare against the
  references (SPEC §10, D29), not against ggstatsplot's 100-resample values.
