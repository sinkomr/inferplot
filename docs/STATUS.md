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

Phase 0 notes: fixtures come from oracle run 42446767391 (R 4.5.3, ggstatsplot 1.1.4,
statsExpressions 2.2.1, Posit snapshot 2026-10-23). `tests/test_reference_vs_oracle.py`
reproduces every ggbetweenstats test statistic, df and p-value in them with
`tools/reference` or SciPy (Bayes ANOVA excepted, v0.2). Oracle findings that changed
SPEC.md are logged in DECISIONS.md (AKP variant, signed Games-Howell q, Mann-Whitney
correction, bootstrap details, centrality labels from plot layers).

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
- Mann-Whitney's exact branch (no ties, both n < 50) is not yet covered by a fixture.
- MAP centrality (Bayes) is recorded, but the density grid behind the "one grid step"
  tolerance is not; settle it in Phase 2.
- ggstatsplot's own bootstrap CIs use 100 resamples, so they differ from the 20,000-
  resample references by Monte Carlo noise; parity tests compare against the
  references (SPEC.md section 10), not against ggstatsplot's 100-resample values.
