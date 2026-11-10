# Status

Read this at the start of every session; update it at the end of every slice.
A slice is done only when its definition of done in `CLAUDE.md` is met.

Current phase: **4 — histostats, scatterstats, barstats** (Phase 3 closed 2026-11-10)

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
- [x] new, major: t, F, Games-Howell q and Hedges' g subtracted means that were each
  rounded first, so groups that differ only in the last digits got silently wrong
  results (0.3 against 0.1 + 0.2 data: p = 0.04 where the exact p on the same doubles
  is 0.54; 1.7e9 timestamps lost six digits). R has the same flaw, so the oracle
  cannot catch it. Fixed (D49): every mean and trimmed-mean difference is formed
  exactly and rounded once, in the parametric and robust layers; checked against
  exact fraction arithmetic (`tests/stats/test_exact_differences.py`, at most 9.2 ulp);
  the fixtures move by at most 1.5e-13
- [x] minor: the noncentral F distribution function was inaccurate or turned a NaN
  into P = 1 in deep tails and at extreme df2 that no shipped number reaches; three
  analysis messages stated wrong reasons (the span error, an overflow message in
  stats words, the Hedges' g J = 0 warning). Fixed: beta factors in log space, deep
  tails by continued fraction, a NaN raises (5.0e-14 up to ncp 2e12, 3.1e-13 at df2
  1e8, log P to 2.3e-15 down to -2000; the worst omega squared interval now takes
  0.23 s instead of 37 ms); messages reworded; robust equivariance tested exactly on
  dyadic data

## Phase 2 — betweenstats nonparametric, robust, bayes

- [x] nonparametric (Mann-Whitney / Kruskal-Wallis, Dunn): `ip.betweenstats` matches all
  12 nonparametric fixtures on pandas, polars and pyarrow (statistics within 2.3e-15
  relative, the rank epsilon squared bound within 1.21 Monte Carlo SDs); every subtitle,
  centrality and bracket label AST-equal; `heavy_ties` check (D47)
- [x] robust (Yuen / t1way, lincon, AKP and explanatory effect sizes): all 10 robust
  fixtures on three backends (p within `wrs2_p`, D43; bootstrap bounds within 2.72 MC
  SDs); texts AST-equal; every bootstrap method proved by exact replay; three or more
  groups keep 95% pairwise intervals at any `conf_level`, as ggstatsplot does, with a
  warning (D53)
- [x] bayes (D41): the two-group subtitle (3 fixtures, BF within 3.5e-11); three or more
  groups with `results_subtitle=False` (7 fixtures; 31 pair BFs within 1.5e-9, 31
  bracket labels and all 30 MAP labels on ggstatsplot's grid point)
- [x] Phase 2 oracle cases and probes pass: fixtures from oracle run 42579842568
  (generator at 9c4994c), the 73 earlier cases plus 88 nonparametric, robust and bayes
  ggbetweenstats cases (edge data, every argument these types take, iris Petal.Length,
  iris_no_setosa, edge_ties_small, edge_k9) and five probes (smm, wrs2p, akp, map,
  ranks); all 72 bootstrap references reproduce ggstatsplot's 100-resample values
  exactly, also at conf.level 0.90 and tr 0.1. 64 of the 73 earlier fixtures are
  byte-identical; in the other nine only BayesFactor's Bayes ANOVA values moved (v0.2
  outputs; OpenBLAS kernels, now pinned). 11,192 tests pass. Deliberate differences
  pinned: 3 subtitles and 3 bracket labels (D43), 1 label (D50), 36 edge_k9 critical
  values (D52). New: WRS2's published lincon table (D51), the exact SMM quantile above
  28 pairs (D52), tied resamples left out (D54), MAP without a density (D55)
- [x] end-of-phase fresh-context review of the Phase 2 diff against SPEC.md: two
  reviewers (statistics; API, text and rendering) found no wrong formula, tail, df or
  seeding error and 13 smaller findings, all fixed (DECISIONS log 2026-11-09): BF10
  beyond a double printed from log(BF01) (4.19e+330, not Inf), Bayes alt text, robust
  pairs below 2 df at any pair count, Bayes pair errors, infinite AKP and xi
  resamples counted and reported (721 of 2,000 in the review's example), MAP bins
  anchored at min(x) far from zero, NumPy integer seeds, probe-ranks and probe-wrs2p
  run through inferplot's own functions. CI green on all nine jobs, including the
  lowest-dependency job

## Phase 3 — withinstats, all types

- [x] foundation (D56): `read_within` (complete subjects, paired by `subject_id`; the
  same matrix on pandas, polars and pyarrow and under shuffled rows; bugs_long 88 of 93
  subjects, sleep 10, edge_within_missing 11), `ip.withinstats`,
  `ip.analyze.withinstats` and `ip.render.withinstats` with every argument check,
  `StatsResult.subjects`; every type raises NotImplementedError until its statistics
  land. 11,475 tests pass
- [x] oracle: fixtures from run 42780911147 (generator at 55f7900, ten shards and a
  merge job): the 161 earlier cases, 159 ggwithinstats cases (sleep, bugs_long and 20
  within-subject edge datasets exported by `oracle/export_datasets.R`, all four types;
  every other p.adjust method, conf.level 0.90, tr 0.1, display all/ns/none, digits 3,
  no subtitle, one-sided, bf.prior 0.8, no Bayes caption, no paths), two heavy-tie
  robust ggbetweenstats cases, and five probes (signrank, pairedbf, friedman, rmanova,
  wrs2within); probe-map gains four anchor vectors. All 136 bootstrap references
  reproduce ggstatsplot's 100-resample values exactly (62 within: 25 wmcpAKP, 23
  kendalls_w, 14 dep.effect). Earlier fixtures byte-identical except
  edge_large_n-bayes's posterior R-squared (v0.2 output, at most 8.1e-14). Each fixture
  records the complete subjects it analyzed (`within_data`); `read_within` gives the
  same subjects and values on all 159 cases and three backends. 14,105 tests pass
- [x] parametric: paired t, Hedges' g of the differences (noncentral-t CI), paired JZS
  caption; repeated-measures ANOVA with Greenhouse-Geisser, partial omega squared (CI
  at the uncorrected df), paired t pairs. 44 of 44 fixtures on three backends and
  probe-rmanova's 9 designs; worst disagreement with R 8e-12 relative (F on
  edge_within_null), CI bounds 7.5e-8 absolute (D62)
- [x] nonparametric: Wilcoxon signed-rank with the matched-pairs rank-biserial r
  (analytic CI); Friedman with Kendall's W (seeded bootstrap CI) and Durbin-Conover
  pairs. 40 of 40 fixtures on three backends, probe-signrank and probe-friedman; worst
  7.8e-15 relative; W = 1 reported with a warning where ggstatsplot drops the subtitle
  (D64)
- [x] robust: Yuen for dependent samples with dep.effect's AKP; rmanova with the
  average AKP (wmcpAKP) and rmmcp pairs. 41 of 41 fixtures on three backends and
  probe-wrs2within's 252 rows; worst 2.5e-14 relative, bootstrap bounds at most 2.62
  of 4 allowed Monte Carlo SDs (D66)
- [x] bayes: paired JZS Bayes factor (two conditions); three or more conditions only
  with `results_subtitle=False` (MAP estimates and a pair Bayes factor at prior 0.707;
  the Bayesian RM ANOVA is v0.2). 34 of 34 fixtures (19 through the replay), worst
  9.3e-10 relative; accurate where BayesFactor misses large |t| (D61, for the owner)
- [x] renderer (D57): subject paths (`point_path`, two conditions, one LineCollection,
  one offset per subject) and the centrality path; labels kept off the paths;
  whole-subject subsampling; `path_hidden` and `paths_crowded` checks. 40 random within
  data sets clean in own figure and `ax=` at 72/100/200 dpi; betweenstats output
  byte-identical (49 figures); 11,514 tests pass. Drawn from hand-built results until
  the statistics land
- [x] integration: every text of the 159 ggwithinstats fixtures compared with
  ggstatsplot's as plotmath (subtitle with n_pairs and the corrected df, caption,
  centrality, tick and bracket labels, drawn pairs); one pinned difference,
  edge_within_small nonparametric, whose subtitle ggstatsplot drops (D64). APA sentence
  and alt text for within designs (D73); no-effect warnings complete (169-row table;
  D72); `heavy_ties` counts what the paired tests rank (D71). Every fixture drawn,
  checked and saved as SVG, PDF and PNG; three within image baselines. 20,864 tests
  pass on the newest and on the oldest supported versions
- [x] end-of-phase fresh-context review of the Phase 3 diff against SPEC.md: two
  fresh reviewers, both halves' correctness findings fixed; W3, W4, W6, W8 and W9 stay
  open below as documentation items or the owner's call.
  Statistics half done (2026-11-10; 28 of 29 planted bugs caught; test statistics, df,
  p-values, closed-form effect sizes and the paired Bayes factor right to about 1e-15
  on hostile data; scripts in `reviewW/`). Findings:
  - [x] W1 (high): the robust bootstraps misclassified resamples whose winsorized
    values are all equal (0/0 or infinite came out as arbitrary numbers): a bound wrong
    on 31 of 83 rating-scale and 38 of 110 one-decimal data sets within subjects, and
    between groups on decimal data with a nearly constant first group. Fixed (D74);
    now 0 of 282 against an exact decimal replay.
  - [x] W2 (high): every subject changing by the same decimal amount gave
    t = 1,007,035,491,024,313 (parametric), BF10 = 1e58 and a robust t of 6e14 without a
    warning. Fixed (D74): refused like exactly constant differences.
  - [x] W5: robust with two subjects and three or more conditions silently used a
    sphericity correction of 1 where the formula is 0/0. Fixed: refused.
  - [x] W7: wrong remedy in the message for a conf_level too close to 1 (Kendall's W,
    rank epsilon squared); `estimate_outside_interval` fired when estimate and bound
    print the same. Fixed.
  - [x] W3 (as in R): Kendall's W's interval can exclude W under no effect (3 to 8 of
    60 data sets); `estimate_outside_interval` reports it. Measured by simulation (4%
    of data sets) and documented in `docs/intervals.md`.
  - [ ] W4 (as in R): the average AKP's interval (D66) covers 99 to 100% where 95% is
    nominal and is 1.3 to 2 times wider than the percentile interval of the averaged
    resamples; say so in the docs. "One shared resample for all pairs" is pinned by no
    test (planted bug P15 survived) and D66's evidence for it is weak.
  - [ ] W6: swapping two conditions negates the AKP estimate exactly but mirrors its
    interval only within Monte Carlo error; Kendall's W's interval depends on the
    condition order when seven or more conditions tie on rank sum and ranks. Document.
  - [x] W8: the percentile rules of the within bootstraps were pinned only
    incidentally. Now pinned on R's own replicates (probe-bootrules): dep.effect 6 of
    6, wmcpAKP 6 of 6; the within helpers use R's rounded rule (D66).
  - [ ] W9 (owner's call): the signed-rank test ranks R's rounded differences (D64), so
    differences equal as written but different in binary (0.1 stored two ways) get
    different ranks, as in R. D74's rule would tie them and differ from R.
  - Leads: what WRS2 does with 0/0 resamples within subjects and with two subjects;
    seeded intervals depend on NumPy's `Generator.integers` stream across versions.
  Second half (data, arguments, texts, checks, drawing; 2026-11-10; scripts and images
  in `reviewX/`): counts, centrality values, paths, brackets, one-sided direction and
  saved files matched hand computation on 30 hand-built frames and 18 drawn designs;
  the three forms agree byte for byte (48 cases); no argument value is silently
  ignored (about 700 calls). Findings, all fixed (merge bf5b3b1; D75 to D80; the lead
  re-ran the reviewer's scripts, swept 196 more figures and changed two things, below):
  - [x] X1 (figure): with four or more conditions a centrality label was moved beside
    a neighbouring condition's point and read as that condition's (58 of 228 default
    figures, 47 unreported). Fixed (D75): now 0 of 241 on the reviewer's script and 0
    of 196 on the lead's sweep of 3 to 8 conditions in both functions; inferplot's own
    figure widens when no place fits; with `ax=` the new check
    `centrality_label_ambiguous` reports it. `path_hidden`/`label_overlap` on own
    default figures: 15 of 222 before, 0 after.
  - [x] X2: numbered text levels ("week 10" before "week 2") stay in text order, as in
    R, and a `warnings` entry now says so and gives `order=` (D76). Lead's change: no
    entry where the numbers read two ways (the first version advised
    order=['0.5', '0.25', '1'] for decimal names).
  - [x] X3: shortened names stay distinguishable in the alt text and APA sentence (D77).
  - [x] X4: wide-format data gets a message that says so and shows the reshape.
  - [x] X5, X6: `pairwise_conf_level` wording for the within robust type; the average
    AKP's resample counts are the largest of one pair (D78; "72 of 2,000" now 24).
  - [x] X7: the alt text names only what the figure draws.
  - [x] X8, X9: words that fit a paired design; hints that lead to a call that works
    (one more dead end found and fixed: the Durbin-Conover tied-pair message).
  - [x] X10: control characters in level names are drawn replaced, with a `warnings`
    entry, and the SVG is well-formed (D79). Lead's change: column names too.
  - [x] X11: `rows_dropped` is a warning above half (D80).
  - [ ] Small, open: on a `transparent=True` save the link line's casing (D75) stays
    in the Axes background colour, a short light stroke on a dark page.

## Phase 4 — histostats, scatterstats, barstats

- [x] oracle: fixtures from run 42829005373 (generator at 85d6c2f): the 322 earlier
  cases, 6 review cases (T7), 76 gghistostats and 64 ggscatterstats cases, and 4 new
  probes (probe-bootrules, probe-corr, probe-onesamplebf, probe-heldout); 468 cases
  and 18 probes in all. All 17 trimcibt references replay ggstatsplot's 200-resample
  values exactly. Earlier fixtures are byte-identical except 7 where an lme4 start-up
  message moved to another case and 2 v0.2 Bayes ANOVA outputs no test reads (8.1e-14
  and 8.3e-11). SPEC §4 holds each method as read from the fixtures. R is measurably
  off at very large |t| (ToothGrowth against 100, t = -82: Hedges' g interval -0.30%
  and -0.55%, one-sample Bayes factor +6.1e-4); inferplot will report the accurate
  values (D81, owner to confirm)
- [x] foundation: `read_one` and `read_pair` reproduce the analyzed values of all 140
  fixtures on pandas, polars and pyarrow (bugs_long 367 of 372 rows; the point labels
  of both label fixtures, 32 of 32 and 9 of 32); `ip.histostats` and `ip.scatterstats`
  exist in three forms with every argument check and targeted messages for
  ggstatsplot's, seaborn's and matplotlib's argument names; result schema 3 (designs
  one-sample and correlation, `point_labels`, `label_var`, `settings.test_value`,
  shapes checked on construction; D82 to D84); every type raises NotImplementedError
  until its statistics land, and the renderers until they are built. Oracle cases
  asked for: n = 2 and 3 (histostats), n = 3 and 4 (scatterstats), a missing label
- [x] statistics for both functions (`stats/one_sample.py`, `stats/correlation.py`,
  `stats/plot_geometry.py`): all 76 gghistostats and 64 ggscatterstats fixtures and
  probe-corr match R at statistics level. Worst disagreement: 1.3e-12 for closed-form
  numbers (allowance 1e-10), 2.6 Monte Carlo SDs for bootstrap bounds and p (allowance
  4), 3.07e-12 for correlation Bayes factors (allowance 5e-12, new kind). Histogram
  bins, 126 marginal histograms and 59 OLS bands match every built layer. Accurate
  where R is not: t = -82 (D81); exactly collinear data (D85, D86; owner to
  confirm). 40-digit references for the noncentral bounds and both Bayes factors.
  Lead's independent check: about 270 random data sets against SciPy and
  from-the-definition integrals, 1e-13 or better. Refused where R answers: a
  constant variable (R prints NA), a correlation interval at n = 3, the default bin
  width of a constant column (D89)
- [x] analyses: `ip.analyze.histostats` matches all 76 fixtures and
  `ip.analyze.scatterstats` the 60 answerable ones through the public API on pandas,
  polars and pyarrow (identical JSON on the three). Refused by design: one-sided
  robust and bayes histostats (2; the two-sided replay equals R), a constant column
  (4; R prints NA). Reported exactly where R prints rounding noise: 8 exactly
  collinear cases, with a warning (D85). Worst disagreement: statistics 5.1e-13
  relative, Hedges' g bounds 1.1e-7 absolute, Bayes factors 3.3e-9 and 3.07e-12,
  bootstrap-t bounds and p 2.3 and 2.6 of 4 Monte Carlo SDs. Lead's check: 60 random
  frames on three backends against SciPy, 1e-14. To do: the figures (in progress),
  then the final texts (the new test names need text templates), image baselines
- [ ] oracle for barstats (ggbarstats, ggpiestats); in progress
- [x] texts: histostats and scatterstats print ggstatsplot's subtitle, Bayes caption and
  histostats' centrality label: 120 of 128 subtitles, 35 of 37 captions and 75 of 75
  labels equal ggstatsplot's character for character (the 17 robust subtitles under
  §10's bootstrap rule); the 10 differences are pinned (ToothGrowth-tv100, D81; 9
  exactly collinear texts, D85, D86). All 140 fixtures' expressions are rebuilt from
  their own numbers; 330 expressions render on matplotlib 3.8.0 and current. APA
  sentence and alt text for both designs (D91, D92); `figure_alt_text` takes the new
  figures' keywords. A bootstrap p of 0 is explained in `warnings` (lead's review)
- [x] figures: `ip.histostats` and `ip.scatterstats` draw end to end on shared machinery
  (`render/base.py`; 327 existing between and within figures byte-identical). Histogram:
  ggstatsplot's bins on all 76 fixtures (counts exact, edges within 1e-13), percent
  axis, centrality line and label, test value mark (D94). Scatter plot: OLS line and
  band within 1e-8 of R's on every lm fixture, marginal histograms, point labels
  placed clear of each other (D97), a seeded subsample above 5,000 pairs. `fig.check()`
  reports nothing about the figure on any fixture; 120 random data sets clean in the
  own figure and in `ax=` at three dpi; passes on matplotlib 3.8.0. Six new image
  baselines. Open, small: a link line can cross another label in a crowded figure
- [x] histostats, all types: reference data, statistics, analysis, texts, figure
- [x] scatterstats, all types (line, band, marginals): the same
- [ ] barstats (independence, McNemar, goodness of fit, proportion tests)

## Phase 5 — documentation, evaluation, release 0.1.0

- [ ] README, `docs/api.md` (size test), `llms.txt`, `skills/inferplot/SKILL.md`, examples
- [x] simulation tests (`slow`): 57 settings and 89 rates over every test, pairwise
  family and interval of the four implemented functions (kind-2 rates established at
  10,000 to 20,000 replications by `tools/reference/simulation_rates.py`). No inferplot
  bug found: every far-off rate was reproduced by an independent implementation and is
  a property of R's method; the lead's own spot check of three rates agrees
  (rank-biserial 90.4% against 90.9%, winsorized r 89.4% against 89.6%, Games-Howell
  with Holm 1.0% against 1.55%). Hold: Welch, paired, one-sample and Pearson tests,
  Hedges' g, Fisher's interval, the rank tests, the trimmed-means tests, the
  bootstrap-t, the between AKP. Conservative: Holm families (Games-Howell with Holm
  1.55% where 5% is allowed), within omega squared (99.8 to 100%), average AKP (99.6%,
  W4). Do not hold: between omega squared at an effect (91.6%; its estimate
  converges to 0.20 for a true 1/7), rank-biserial intervals (88 to 94%), rank
  epsilon squared, Kendall's W and xi (never cover 0; 85%, 68% and 92% at an effect),
  winsorized r at rho 0.8 (89.6%), Yuen with a small, more variable group (6%). User
  documentation: `docs/intervals.md`. barstats' simulations follow with barstats
- [ ] agent evaluation harness (ask the human before running it)
- [ ] release workflow; human publishes to TestPyPI, then PyPI

## v0.2 backlog

Bayes ANOVA and contingency, posterior estimates, dotplotstats, piestats, corrmat,
coefstats, `grouping_var`, replicate-aware SuperPlots.

## Known gaps and open oracle discrepancies

### Fable review of Phases 0-3 (2026-11-10, owner's request), open findings

Statistics review (no wrong formula, tail, df, correction, adjustment or resampling
scheme found; scripts in the session scratch folder `reviewS/`). To fix:

- [x] S1 (wrong numbers, silent): an integer `y` above 2^53 (nanosecond timestamps,
  large counts) was rounded by the float64 cast; 12 values 1.7e18 + 10..300 kept 2
  distinct values (Welch t -3.16 for the true -4.32; Mann-Whitney p 0.02 for 0.008).
  Fixed: such a column raises `ValueError` naming the remedy (subtract an offset);
  large integers a double holds exactly are read, also on pyarrow, whose cast refused
  them. Decimal columns with more than 17 significant digits are still rounded
  (a decimal is rarely exact in a double, so there is no exactness test to apply).
- [x] S2: Hedges' g with a group of n = 2 collapses with Welch's df near 1 (n 2 against
  50: d 1.16, g 0.026); the formula is the documented one (J at Welch's df), but the
  warning fired only at df exactly 1. Now warns below 2 df. Still to do: pin
  effectsize's value with a two-group n = 2 case in the next oracle run (T7).
- [x] S3: `type="robust"`: one 0/0 bootstrap resample of 2,000 raises, so small rating-
  scale data are refused (5-point scale, n 8 per group: 33 of 40 data sets) and the
  outcome depends on the seed (fixture edge_ties raises at seed 3 only). Decide with
  the Phase 3 probe of WRS2's behaviour (D54 leaves undefined resamples out).
  Fixed (D65): such a resample is left out and counted, with a warning; the call
  fails only when too few remain. Which rule WRS2 itself uses awaits two oracle cases
  (T7); the fixtures cannot tell the candidates apart.
- [x] S4: AKP and xi intervals stopped widening above conf_level about 0.999 (the order
  statistics run out) without saying so. Now refused with a plain error, as rank
  epsilon squared does (SPEC §5 item 14).
- [x] S5: the explanatory effect size and its interval depend on the order of the
  groups when sizes differ (xi 0.5036 against 0.5125 with `order=` reversed), within
  Monte Carlo error: the groups draw from the random stream in level order. Fix:
  resample the groups in an order fixed by their contents, as rank epsilon squared
  does (tried: 5 lines). It changes every seed's stream, so do it together with S3,
  which decides what happens to the undefined resamples the pinned tests count.
  Fixed (D65): the groups are resampled in an order fixed by their contents.
- [x] S6: `smm_quantile` was off by up to 3e-12 relative between df 20 and 30 (4-term
  Stirling series from df/2 = 10) against a claimed 1.1e-14. Fixed with a seven-term
  series (within 3e-17 of 40-digit values from df 20); 36 reference rows added at df
  20, 21.3 and 25.7, all of which failed before. No printed digit changed.
- [x] S7: the robust pre-check in betweenstats worked in raw units: y x 1e-165 raised
  with a false reason (winsorized variance 0), y x 1e156 an overflow error. Now asked
  of the values, not their squares; results are the same from 1e-300 to 1e300.
- [x] S8 (wording): WRS2's critical values above 28 pairs match the quantile of the
  largest of C signed t variates (one-sided; within -0.4% to +0.9% at alpha 0.05 and
  0.1 on the probe), not a noisy modulus quantile; D52, SPEC §5 and the docstring now
  say so. The decision stands; the owner was told the description changed.
- [x] S9 (was a lead): the between-groups rank-biserial r of exactly ±1 got the
  interval [±1, ±1] without a warning. Fixed: a `warnings` entry is added; it fires on
  no fixture (the most extreme two-group r is -0.9644).
- Leads, not verified: the exact-replay emulator behind the robust percentile rules is
  not committed (the committed tests compare within 4 Monte Carlo SDs, which cannot
  resolve a one-order-statistic difference, about 0.14 SD); rank-biserial r of ±1 gets
  the interval [±1, ±1] without a warning; a p below the smallest double is stored as
  0.0.

Tests and oracle review (46 planted bugs: the parity tests caught 29, other tests 12,
5 survived all 11,514 tests; every documented exception from R held up against the
reviewer's own high-precision values; the oracle generator, manifest and references
held up; scripts in `reviewT/`). To fix:

- [x] T1: the unadjusted pairwise p (Dunn, lincon) and lincon's t and df in the result
  were compared with nothing (3 planted bugs survived). Now compared on every fixture
  with the normal tail of R's z and with SciPy's Yuen test (worst 2.8e-13 relative).
- [x] T2: D54 (tied resamples left out, not counted as 0) was pinned by no test. Now
  pinned on data where more than 5% of resamples are tied, so counting them as 0
  gives a lower bound of exactly 0.
- [x] T3: the trim count floor(tr n) had no case with tr n just below an integer
  (0.29 x 100 = 28.999999999999996). Now checked against SciPy's separate count.
- [x] T4: the test of the cases ggstatsplot fails on accepted any error mentioning
  "x=", "conf_level" or "type=". Now each case names its refusal; R's partial subtitle
  of edge_n2-robust is compared (F, df, p, xi and its interval), and edge_n2-bayes
  without the subtitle is compared with R's estimates and the independent Bayes
  factor reference.
- [x] T5: the slow tests (simulations, random layouts: 161) ran in no workflow; CI job
  `slow` runs them. R warnings recorded in fixtures and stale keys in the exception
  lists now fail a test.
- [x] T6: the 4-SD bootstrap rule could not tell the documented percentile rule from
  its neighbours. Done (probe-bootrules, run 42829005373): R's own sorted replicates
  are recorded at 39, 40, 99, 100, 199 and 200 resamples and 95%/90%, and inferplot's
  helpers return R's bounds from them exactly (akp.effect's floor rule 12 of 12, bar
  the 39-resample 95% case inferplot refuses, S4; t1way's xi rounded rule 12 of 12;
  `boot.ci`'s percentile rule for rank epsilon squared and Kendall's W within 3e-15;
  trimcibt 12 of 12). New: dep.effect and wmcpAKP use the rounded rule at alpha 0.05,
  not the floor rule D66 documents (the two coincide at inferplot's 2,000 resamples, so
  no number changes), and wmcpAKP takes its two bounds from separate bootstrap runs
  per pair (D66 said one shared resample). Done in Phase 4: the within helpers use
  the rounded rule, pinned on the probe (W8).
- [x] T7: the review's oracle cases (run 42829005373). Results:
  - 0/0 AKP resamples (D65): WRS2 leaves them out of the sort but counts the bound
    positions over all resamples, so its upper bound moves up and is NA once more than
    2.5% are undefined: `edge_akp_undefined` (17% undefined) prints CI [0.22, NA],
    where inferplot gives [0.22, 0.88] with a warning; `edge_akp_undefined2` (2%) has
    R's upper bound at 1.23 against inferplot's 0.94. An accident of R's NaN handling,
    not a method: inferplot keeps D65 (owner to confirm). `edge_akp_undefined` is
    held out of the parity tests in probe-heldout; a test now reproduces R's NA from
    inferplot's own replicates under R's rule, and on `edge_akp_undefined2` R's rule
    lands 1.4 SDs above R's reference and D65 2.5 SDs below it.
  - D54 confirmed in R itself: on `edge_tied3_small` 7 to 9% of resamples are fully
    tied and R's lower bound is 0.1458, not 0; inferplot matches.
  - Trim count: R trims 28 from each end at tr 0.29, n 100; inferplot matches.
  - Hedges' g at Welch df 1.0023 (`edge_n2_vs50`): inferplot matches R and warns.
  - Fewer subjects than conditions (`edge_rm_n3k4`, 3 subjects, 4 conditions; D62):
    afex cannot compute the sphericity correction and ggstatsplot prints the
    uncorrected F(3, 6) = 7.70, p = 0.018 without a word; inferplot applies the
    Greenhouse-Geisser correction it documents, F(1.05, 2.10), p = 0.104. inferplot
    keeps the correction (owner to confirm), names the difference in `warnings`, and
    a test pins both sides. Held out in probe-heldout.
  - Friedman: every subject tied, R fails (declared), inferplot raises plainly (D64);
    some subjects tied, R completes and inferplot matches (52 of 2,000 resamples left
    out, with a warning).
- [ ] T8 (owner): `fixtures.yml` is the only guard against a fixture and the manifest
  edited together, and main is not protected, so it is advisory. Making it a required
  check is a repository setting.

Data, arguments, result and text review (no wrong number: 60 random between and 60
within frames on up to seven column types matched plain-Python grouping, 40 random
results matched SciPy field by field, 108 results round-trip through JSON; scripts in
`reviewU/`). Reproduced by the lead; to fix:

- [x] U1: a pyarrow dictionary column (`dictionary_encode()`, `auto_dict_encode`) and
  a pandas categorical made by polars `.to_pandas()` hold their categories in order of
  first appearance, which inferplot takes as the declared order: shuffling the rows
  swaps the levels (t 4.32, p .0001 against t -4.32, p .9999 with
  `alternative="greater"`) and, in withinstats, the subject order. SPEC §3's "does
  not depend on the input row order" is false for these columns.
  Fixed (D58): an unordered pyarrow dictionary is read as its values and a warning
  names both orders; the same levels, t and p for every row order on six input kinds.
  A pandas categorical built in row order (polars `.to_pandas()`) still follows it,
  which U2 makes visible.
- [x] U2: nothing says which way a one-sided test points (`alternative="greater"` is
  first level greater than second, and "control" sorts before "treatment").
  Fixed (D68): the APA sentence and alt text name the direction with the level names.
- [x] U3: hints that lead to another valid analysis: "benjamini-hochberg" suggests
  'hochberg', "holm-bonferroni" suggests 'bonferroni', "one-sided" suggests
  'two-sided', `tail=` suggests `title=`, `conf_level=5` suggests 0.05, `levels=`
  suggests `conf_level=`; `paired=` and `alpha=` in betweenstats get no pointer.
  Fixed (D59): suggestions only for likely names; targeted messages for these and
  other common names, on all six public entry points.
- [x] U4: a text level whose y is all missing disappears without a warning (Welch's
  ANOVA becomes a t-test); the same data as a categorical does warn.
  Fixed (D60): the level is named in `warnings` for every column type.
- [x] U5: the caption says "significant only" without the threshold when
  `pairwise_alpha` is not 0.05.
  Fixed: the caption states the threshold when it is not 0.05.
- [x] U6: an effect size outside its own bootstrap interval is printed without a
  check (12 noise groups: rank epsilon squared 0.074, interval [0.107, 1]).
  Fixed (D69): new check `estimate_outside_interval`.
- [x] U7: result contract: `effect_size.undefined_resamples` is in SPEC §6 but not
  in the result; the JSON has no schema version and `from_json` refuses a result
  saved before `subjects` existed; `to_frame()` has no column saying an interval is
  one-sided; the rows of dropped subjects are not in a within result, so its counts
  do not add up to the input rows.
  Fixed (D67): `undefined_resamples`, `dropped_subject_rows`, `ci_sidedness`,
  and a `schema_version` that `from_json` checks.
- [x] U8: sign and df conventions differ between tests in the same columns (Welch t
  signed, Yuen t absolute, Games-Howell q as group2 minus group1, Dunn z unsigned)
  and are documented only in SPEC §4.
  Fixed: stated in the result docstrings and SPEC §6.
- [x] U9: messages: integer column labels have no way forward; a datetime or binned x
  is told to convert to a categorical, which fails too; a positional subject column
  gets a `pairwise_display` example; a Series passed as x is printed whole; swapped x
  and y; duplicate column names raise narwhals' own error.
  Fixed: each message now gives a fix that works.
- [x] U10: `type="bayes"` with levels named like "group 7" and a pair that cannot be
  computed raised a bare IndexError (the inner message was renamed twice). Fixed: a
  pair's message is renamed once, in one pass.
- [x] U11: `pairs_hidden` fires for an explicit `pairwise_display="none"`; the copy
  embedded in a saved figure loses the `heavy_ties` check.
  Fixed (D69); `heavy_ties` for paired designs counts what the paired tests rank (D71).
- [x] U12: fixed (D60, D70): whitespace look-alikes warn, `digits` is at most 15,
  float32 arguments are read as printed, the alt text describes only what is drawn,
  and the docs no longer claim an ASCII-only APA sentence.

Drawing, saving and figure checks review (784 brackets over 72 figures: every one
spans its pair with that pair's p; points, boxes and paths are the result's values;
saved files embed the right copy and no raw value, byte-identical on re-save; no
global matplotlib state is touched, also under 80 hostile rcParams; 70 random own
figures clean; scripts and images in `reviewV/`). The oldest matplotlib (3.8.0) was
not drawn on by the reviewer; CI's lowest-dependency job is the evidence there. For a
figure inferplot makes itself an empty `fig.check()` held up except for V2, V3 and
V7; with `ax=` it is not sufficient (V1, V4, V5). To fix:

- [x] V1 (fixed, D63): `ax=` on shared axes: with `sharex` a panel shows the other panel's tick
  labels, n included (left panel n = 12, 30, 25 labelled 50, 8, 44; image checked by
  the lead); with `sharey` the second call resets the limits and cuts the first
  panel's data (72 of 80 values outside the view). `fig.check()` is empty.
- [x] V2 (fixed, D63): a centrality label covers the next group's centrality point whenever
  neighbouring centres are close (66 of 120 random data sets; InsectSprays B under
  A's label, image checked by the lead). No check compares labels with points.
- [x] V3 (fixed, D63): data that nothing draws: (a) without a points layer, values beyond the
  whiskers have no mark; (b) a group of more than 2,048 values with one extreme value
  loses its violin (0/0 in the binned density) while the warning says the violins
  show all values; (c) the 2,000-point subsample is uniform over all values, so a
  group of 8 next to 60,000 gets no point.
- [x] V4 (fixed, D63): `ax=` under a dark style: subtitle, brackets, labels and caption are drawn
  in near-black on black.
- [x] V5 (fixed, D63): overlaps between panels (one panel's x label through the next panel's
  title) are not checked; only the Axes' own texts are compared.
- [x] V6 (fixed, D63): a sequential colormap as `palette` gives the first group a near-white
  colour ("Blues", "Greys"), invisible on white.
- [x] V7 (fixed, D63): Chinese, Japanese, Korean and Thai names are drawn as empty boxes, and the
  theme overrides any font the user sets.
- [x] V8 (documented in SPEC §6; a generic `sans-serif` fallback added; mathtext's
  per-character text is matplotlib's and stays): the SVG names DejaVu Sans and STIX fonts without a fallback (on macOS and
  Windows the hats land beside their letters), and mathtext is one positioned
  character per `<tspan>`, so it is not editable or searchable as text. PDF embeds
  its fonts and is faithful.
- [x] V9 (fixed, D63): the artist dicts accept properties that replace or hide the data
  (`offsets`, `visible`, `alpha=0`, `xy`, `path`, `transform`, a white palette)
  without a warning; palette entries beyond the number of levels are not validated.
- [x] V10 (fixed, D63): inverting an axis on a returned figure strikes each bracket label through
  with its own line (y) or puts labels on their own points (x); `fig.check()` is
  empty.
- [ ] V11 (docs, Phase 5; the `save()` docstring and SPEC §6 say it): `plt.savefig()` after `ip.betweenstats(...)` writes a blank
  image (the figure is not pyplot's); `fig.figure.savefig()` writes text as paths
  and no embedded result.
- [x] V12 (fixed, D63): with `ax=` under constrained layout the wide subtitle collapses the Axes;
  it is reported, but the advice does not name the cause.
- [x] V13 (fixed, D63): smaller: one long level name widens every level (60 characters: a 30 in
  figure); y x 1e200 raises a raw ZeroDivisionError in the brackets and y x 1e-200
  "figure size must be positive finite"; a 3D Axes fails after artists were added; a
  check message quotes mathtext source; `dpi=1e6` raises MemoryError.

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
- The generator handles ggbetweenstats, ggwithinstats, gghistostats and
  ggscatterstats; barstats still stops (it computes its pairwise tests through a
  hidden helper, and `.chisq_test_safe` swallows errors into NA rows the recorders
  cannot see). An NA among the subtitle or caption numbers and an empty plot layer
  stop the run unless the case declares them. `ORACLE_SUBSET` limits a debugging run (no
  manifest, never attested).
- betweenstats does not warn for `bf_message=False` where no Bayes caption can be
  drawn (withinstats does since D72); align it when betweenstats' warnings next change.
- Deliberate differences confirmed by the owner on 2026-11-09 (accuracy over R's
  numbers, like D31): D50, two pairwise Bayes factors BayesFactor misses (iris
  Petal.Length, +1.16e-3 and -6.1e-5; one label -133.68 for ggstatsplot's -133.69);
  D52, the exact SMM quantile for nine or more robust groups, where WRS2's seeded
  simulation is off by -6.6% to +4.7% (only the pairwise intervals differ, not the
  figure).
- Robust: WRS2's yuen, t1way and lincon p are 1 - cdf (probe-wrs2p, 420 of 420);
  inferplot reports the accurate tail (D43, confirmed). lincon below df 2 raises
  `NotImplementedError` (WRS2 stops too). An AKP or xi resample whose effect size is
  0/0 is left out of the interval and counted (D65). Bootstrap time grows with n: about 5 s for xi at 10,000
  values per group; a counting-based sort could make it about 5 times faster.
- MAP binning: the anchor threshold is pinned at 2^31 bin widths to within 2e-6
  (probe-map anchor_26027440 and anchor_26027507); `_ANCHOR_AT_MIN = 2**31` matches all
  four anchor vectors.
- WRS2 returns ±Inf for an infinite AKP resample and ggstatsplot prints it
  (edge_heavy_ties2, edge_within2_small, edge_within_small robust); 20,000 resamples
  can make the bound finite again (edge_within_small: -10.18). inferplot reproduces
  edge_heavy_ties2's -Inf. A 0/0 resample is left out and counted (D65); WRS2 counts the bound
  positions over all resamples instead, which gives an NA upper bound on
  edge_akp_undefined (T7).
- Paired Bayes factors: BayesFactor misses large |t| by far more than its own error
  estimate (probe-pairedbf against 25-digit mpmath: +68% at n 5, t 1,000; +5.3% at
  n 10, t 1,000; -0.78% at n 30, t 50; within 2.6e-6 at |t| <= 12). Every paired
  fixture value is within 9.3e-10 of tools/reference, so `ttest_bayes_factor` (1e-8)
  holds on the fixtures. inferplot computes the accurate value (the paired version of
  D50, which the owner confirmed for the two-sample case on 2026-11-09); flag it for
  the owner with the Phase 3 report. tools/reference overflows above BF 1e308.
- The APA sentence prints a huge statistic in fixed notation (an F of 1e250 gives 251
  digits), the only way the alt text can exceed 400 characters.
- Unverified lead (statistics review): SciPy's betainc is used where both beta
  parameters are at least 1e4 within a factor of 10 (large Welch df2 and ncp in the
  omega squared interval); `noncentral_f_df1_two` covers df1 = 2 only.
- Phase 4: `ttest_bayes_factor` is measured on two-sample and paired Bayes factors;
  measure one-sample ones (histostats) against an exact integral when their fixtures
  arrive, and log any change.
- MAP centrality: every fixture label and 49 of probe-map's 52 vectors on R's grid
  point (`MAP_CENTRALITY`, 0 steps); constant data and zero IQR behave as D55 says;
  normal_n1e5 is pinned one step away (R's two densities differ by 6.2e-15). For values
  near 1e8 and above R's bw.SJ anchors its bins at min(x), inferplot at zero: the
  bandwidth differs by up to 9.6e-4 relative (shifted_large, timestamps), the MAP
  does not. Copying R there would copy a numerical accident; not done.
- Rank epsilon squared: the resampling scheme (within groups) and percentile rule are
  settled by Monte Carlo agreement and the boot.ci help page, not by exact replay.
  Fully tied resamples are left out (D54); the data cannot tell that from sorting them
  last (0.46 standard errors apart on edge_ties_small).
- ggstatsplot's own bootstrap CIs use 100 resamples, so they differ from the 20,000-
  resample references by Monte Carlo noise; parity tests compare against the
  references (SPEC §10, D29), not against ggstatsplot's 100-resample values.
