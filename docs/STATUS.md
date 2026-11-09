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
- [x] the statistics part did not finish; re-run from a fresh context. All 13 earlier
  statistics findings and the wide-range follow-up are confirmed closed; of the five
  unjudged leads, three were false alarms (subnormal variance, df2 = 0.5, the one-sided
  bound at 50%) and two are real but minor (below)
- [ ] new, major: t, F, Games-Howell q and Hedges' g subtract means that were each
  rounded first, so groups that differ only in the last digits get silently wrong
  results (0.3 against 0.1 + 0.2 data: p = 0.04 where the exact p on the same doubles
  is 0.54; 1.7e9 timestamps lose six digits). R has the same flaw, so the oracle cannot
  catch it. Being fixed (exact differences, rounded once), also checked in the robust
  layer
- [ ] minor, being fixed: the noncentral F distribution function is inaccurate or turns
  a NaN into P = 1 in deep tails and at extreme df2 that no shipped number reaches;
  three analysis messages state wrong reasons (the span error, an overflow message in
  stats words, the Hedges' g J = 0 warning)

## Phase 2 — betweenstats nonparametric, robust, bayes

- [x] nonparametric (Mann-Whitney / Kruskal-Wallis, Dunn): `ip.betweenstats` matches all
  12 nonparametric fixtures on pandas, polars and pyarrow (statistics within 2.3e-15
  relative, the rank epsilon squared bound within 1.21 Monte Carlo SDs); every subtitle,
  centrality and bracket label AST-equal; `heavy_ties` check (D47)
- [x] robust (Yuen / t1way, lincon, AKP and explanatory effect sizes): all 10 robust
  fixtures on three backends (p within `wrs2_p`, D43; bootstrap bounds within 2.72 MC
  SDs); texts AST-equal; every bootstrap method proved by exact replay; three or more
  groups at a `conf_level` other than 0.95 raise `NotImplementedError` (lincon's table
  is at 95% only)
- [x] bayes (D41): the two-group subtitle (3 fixtures, BF within 3.5e-11); three or more
  groups with `results_subtitle=False` (7 fixtures; 31 pair BFs within 1.5e-9, 31
  bracket labels and all 30 MAP labels on ggstatsplot's grid point)
- [ ] Phase 2 oracle cases and probes (oracle/phase2-cases: edge data, conf_level 0.90,
  tr 0.1, every p_adjust method, probe-smm, probe-map, probe-wrs2p, probe-akp) pass
- [ ] end-of-phase fresh-context review of the Phase 2 diff against SPEC.md

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
- Renderer: five pytest-mpl image baselines are in (`tests/render/test_images.py`,
  generated by CI run 42565747622, compared at RMS 0.1; a bracket line width change and
  a centrality colour change both fail CI, runs 42566517893 and 42567213432). A
  failing `--mpl` job does not yet upload its difference images. With `ax=`, inferplot
  sets its font only on the axis texts and the texts it draws; the user's own texts
  keep theirs.
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
- Robust: Yuen's p is 2 (1 - pt) in WRS2 (confirmed: every robust omnibus p lies on the
  1 - cdf grid); inferplot reports the accurate tail (D43, confirmed by the owner). Unconfirmed until probe-smm and the Phase 2 oracle cases land:
  lincon's critical value at df >= 200 (the infinite-df row is used), below df 2 or at
  `conf_level` other than 0.95 (both raise `NotImplementedError`), WRS2's constants at tr
  other than 0.2, and 879 of 900 SMM table entries. A bootstrap resample whose effect
  size is 0/0 raises `ValueError`. Bootstrap time grows with n: about 5 s for xi at
  10,000 values per group; a counting-based sort could make it about 5 times faster.
- Phases 3 and 4: `ttest_bayes_factor` was measured on two-sample Bayes factors only;
  measure paired and one-sample ones against an exact integral when their fixtures
  arrive, and log any change.
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
