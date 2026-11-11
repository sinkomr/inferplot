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
- [x] oracle for barstats: fixtures from attested run 42878070705 (generator at
  347c23a): 90 barstats cases (85 ggbarstats, 5 ggpiestats one-way cross-checks), a
  new probe (probe-contingency), and 25 edge cases for the existing functions (16
  gghistostats: n = 2, n = 3, constant data; 9 ggscatterstats: n = 3, n = 4, a missing
  label); 583 cases and 19 probes. Earlier fixtures byte-identical except a moved
  lme4 message and one v0.2 posterior (8.1e-14). Every chi-square, McNemar-Bowker,
  goodness-of-fit and Fisher p, Cramér's V, Pearson's C and Cohen's g is reproduced
  independently, except 9 interval bounds where R's root search stops short (pinned;
  D99, owner to confirm). SPEC §4 barstats rewritten (D98 to D101). Merged (30e177a):
  of the 25 edge cases 16 match on three backends (n = 2 and 3 one-sample, constant
  data nonparametric, 3 pairs bayes, 4 pairs, a missing label) and 9 are refused with
  R's side pinned (constant data parametric, bayes and robust; 3 pairs parametric,
  nonparametric, robust; D102, D103). Constant-data histograms draw one 0.1-wide bin
  (D89 amended). Open: two n = 2 subtitle ties pinned (t 1.13 and 0.68 in R, 1.12
  and 0.67 in inferplot; the robust one is being fixed)
- [x] barstats foundation: `read_table` reproduces R's prepared count table (levels,
  cells, n, dropped rows) on all 90 fixtures and three backends; arguments with
  targeted messages; result schema 4 (design "contingency": `counts`, `cells`, `bars`,
  `y_levels`); new check `small_expected` (Cochran's rule); analysis skeleton with 68
  fixtures answerable and 21 refused by design (D101, D104). The statistics join must
  raise for no discordant pair and for a Fisher pair with one column; the texts
  package must confirm the percent formula on edge_bar_rounding
- [x] barstats statistics (`stats/contingency.py`; a noncentral chi-square in
  `noncentral.py`): all 89 completed fixtures and probe-contingency match (273 parity
  tests; worst 6.3e-11, R's own error in a Fisher p); all 22 pinned interval bounds
  (D99) exercised; exact Fisher test for 2 x c tables with cost limits that raise
  rather than approximate (D107). Lead's check: 300 random tables against SciPy and
  221 brute-force enumerations, 1e-11 or better
- [x] barstats texts: ggbarstats' subtitle for the three designs, the bar p, n and
  cell labels, pair labels, APA sentence and alt text (D108). Rebuilt from the
  fixtures' numbers they equal ggstatsplot's on all 89 fixtures with output (71
  subtitles, 158 bar p labels, 175 n labels, 485 cell labels, 152 pair labels); on
  results, 69 of 70 subtitles (one pinned: D99 at three digits). To do when the
  analysis lands: run the text tests on real results
- [x] barstats analyses: `ip.analyze.barstats` returns complete results for the
  two-way, paired and one-way designs. 75 fixture calls (67 answered, 8 replays of
  refused calls) on pandas, polars and pyarrow match R on every number (worst
  closed-form gap 5.4e-11, R's own Fisher error; interval bounds within 2.6e-7), with
  identical JSON across backends; 22 fixtures refused by design, each tested; a pair
  without a Fisher test stays in the result with its reason (D113). Lead's check: 80
  random frames in three designs against pandas and SciPy, 1e-14
- [x] barstats figure: `ip.barstats` works end to end. Stacked bars on the shared
  machinery; all 67 answerable fixtures drawn with segment heights equal to the shares
  (1e-12) and every cell, p and n label equal to ggbarstats' built layers; small-cell
  labels are moved beside their bar, never dropped in the own figure (D114); 120
  random tables clean in the own figure and in `ax=` at three dpi; the four existing
  figures byte-identical (492 of 492); four image baselines (18 in all)
- [ ] end-of-phase review of barstats (two fresh reviewers). Visible half done
  (2026-11-11; scripts and images in `reviewBB/`): 43 hand-built frames on three
  backends and the same table written 14 ways (672 of 672 comparisons identical);
  texts of 159 results re-derived; about 190 figures measured; saved files and the
  three forms sound. Findings (2 high, 10 medium, 5 low), being fixed:
  - [ ] B1 (high; reproduced by the lead): two long level names that differ only at
    the end are shortened to the same legend entry and tick label; nothing reports
    it (shared tick wrapping: betweenstats too).
  - [ ] B2 (high; reproduced): `bar_args` properties silently wreck the chart
    (`color` paints every level alike, as the error message for `color=` itself
    suggests; `zorder` hides every cell label; `angle`; wide edges); identical
    palette colours are not reported.
  - [ ] B3 (reproduced: a 3 x 8 table of 377 observations takes 19 s and 3.2 GB; 8 x 8
    of 682, 28 s; 10 x 10 does not finish in minutes): the pairwise Fisher tests,
    which are never drawn. The limit is per pair, spent before the refusal, and the
    search's arrays are not in the memory bound.
  - [ ] B4: the APA sentence counts pairs that were never tested ("0 of 3 pairs").
  - [ ] B5: Cramér's V printed outside its own interval on wide tables (2 x 20: 10 of
    60), as the SPEC formula gives; the check's message speaks of a bootstrap; no
    fixture covers large df (oracle probe wanted).
  - [ ] B6 to B9: hints that do not lead to a working call (paired data in long
    format; the "declare both columns" recipe empties an integer column; `tail=` and
    `one_sided=`; `x_label=`).
  - [ ] B10: an aggregated table given without `counts=` is analysed as one
    observation per row, silently.
  - [ ] B11: the figure grows without bound with the number of bars (100 bars: 81
    in) and `figure_oversized` allows it.
  - [ ] B12: no test ties a level's colour to its legend entry.
  - [ ] B13 to B17 (low): moved labels carry no colour key; pandas-only hints on
    polars and pyarrow frames; "p = 2.23e-308" for an underflowed p (D108; owner's
    call); fifteen-digit `ratio` in texts; small items.
  Statistics half done (2026-11-11; scripts in `reviewBA/`): no wrong statistic, p,
  effect size or interval: 406 hostile two-way tables and 554 paired and one-way calls
  agree with exact rationals and mpmath to 3e-14; Fisher's test on 18,259 small tables
  against exact enumeration (10,084 with exact ties) with 0 misclassified; 1,954
  noncentral chi-square roots to 1.2e-13; all 22 pinned bounds of D99 confirmed; 2,837
  comparisons of the same table in different writings bit-identical; 39 of 41 planted
  bugs caught. Findings (2 high, 4 medium, 6 low), being fixed:
  - [ ] C1 (high): an unused declared category (a filtered pandas categorical) changes
    the p-values drawn above the bars (59/41 and 44/56 splits labelled p = 2.27e-05
    and 8.57e-06; 0.26 and 0.46 without the unused category), as in ggstatsplot, and
    the user is not told what the bar's test counts.
  - [ ] C2 (high): the Fisher cost again, measured independently (5 x 10 with 1,000
    rows: 27.8 s, 4.6 GB, 0 of 10 pairs tested); the 2 x 2 path has no limit either.
  - [ ] C3: rounding noise printed where the exact value is 0 ("V = 3.95e-09",
    "chi2 = 1.13e-31, C = 6.71e-17" for an exact fit): 32 of 206 small 2 x 2 tables
    whose corrected V is exactly 0.
  - [ ] C4: the refusal for a paired table with an unused category advises merging
    levels instead of removing the unused one.
  - [ ] C5: two planted bugs no test notices (a declared level empty in both paired
    measurements dropped; a wrong df from 7 categories up).
  - [ ] C6: a pandas object column of booleans is refused.
  - [ ] C7 (low): check wording for these intervals; raw errors for absurd `ratio`
    shares; D107's accuracy figure optimistic (1.3e-10 at 1e9 per cell, not 1.2e-11);
    small wording; unreachable code.
  - Oracle probes wanted: R's `round()` on percent values such as (1/2000) x 100 at
    one digit (Python's rule and R 4's may differ for 98 values of k/2000); one-sided
    Fisher pairs on reduced pair tables; what R prints for the exact-zero cases;
    Cramér's V interval at df 20 to 60; `wilcox.test` with a difference of 5.6e-17.
