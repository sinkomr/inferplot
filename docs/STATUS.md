# Status

Read this at the start of every session; update it at the end of every slice.
A slice is done only when its definition of done in `CLAUDE.md` is met.

Current phase: **1 — betweenstats, parametric, end to end** (Phase 0 done 2026-11-07)

## Phase 0 — Scaffold

- [x] `pyproject.toml` (hatchling, src layout, `requires-python >= 3.11`), `uv.lock`, MIT `LICENSE` ("inferplot contributors"), `NOTICE` (ggstatsplot MIT notice; plotsalot credit)
- [x] ruff, pyright (strict on `src/`), pytest, hypothesis, pytest-mpl configured
- [x] `.github/workflows/ci.yml`: lint, types, tests on Ubuntu, macOS, Windows × Python 3.11 and the newest Python with numpy, scipy and matplotlib wheels; one job with `uv sync --resolution lowest-direct`
- [x] `oracle/` with `generate.R` and datasets; `.github/workflows/oracle.yml`; first fixtures (ggbetweenstats, all types, three datasets) committed under `tests/fixtures/oracle/`
- [x] `tests/tolerances.py` and a fixture loader; `tools/reference/stats_reference.py` checks run in CI

Phase 0 notes: fixtures come from oracle run 42452307942 (R 4.5.3, ggstatsplot
1.1.4, statsExpressions 2.2.1, Posit snapshot 2026-10-23). `tests/test_reference_vs_oracle.py`
reproduces every omnibus and pairwise test statistic, df and p-value in them with
`tools/reference`, SciPy or an inline implementation (Dunn, lincon), plus partial omega
squared, except the Bayes ANOVA, which is v0.2. For every bootstrap CI the oracle
replays ggstatsplot's own bootstrap exactly, which proves the 20,000-resample reference
uses the same method. The end-of-phase review found no high-severity problem; its
findings are fixed or listed below. Oracle findings that changed SPEC.md are logged in
DECISIONS.md.

## Phase 1 — betweenstats, parametric, end to end

- [ ] `_data.py` boundary (pandas, polars, pyarrow) and `_validate.py` (aliases, did-you-mean, removed-argument messages)
- [ ] Welch t and ANOVA, Hedges' g and partial omega squared with CIs, Games-Howell then Holm
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

- No open discrepancies.
- Fixtures cover ggbetweenstats on ToothGrowth, PlantGrowth and mtcars only. Add
  edge-case and other-function cases to `oracle/generate.R` in the phase that needs
  them; the edge-case CSVs are already committed.
- Phase 1: the default caption of a two-group parametric test is the JZS Bayes factor
  (SPEC §7, D28), which the checklist lists under Phase 2. Implement it in Phase 1 or
  check caption parity in Phase 2.
- lincon's CI critical value: the 3-group fixtures cannot tell whether its C counts
  pairs or groups. Add a fixture with 4+ groups (`edge_k5.csv`) before Phase 2's robust
  slice.
- AKP: WRS2's scaling constant differs from the exact winsorized-normal value by
  1.2e-9 relative (review measurement), which is above the 1e-10 closed-form tolerance.
  Phase 2 must match WRS2's constant or register a measured tolerance.
- Mann-Whitney is never exact in ggbetweenstats (`exact = FALSE` in statsExpressions'
  source); a tie-free fixture would confirm it from the oracle itself.
- MAP centrality: the oracle records the values; the grid behind SPEC §10's "one grid
  step" tolerance is bayestestR's 2^10-point density (help page), to confirm in Phase 2.
- ggstatsplot's own bootstrap CIs use 100 resamples, so they differ from the 20,000-
  resample references by Monte Carlo noise; parity tests compare against the
  references (SPEC §10, D29), not against ggstatsplot's 100-resample values.
