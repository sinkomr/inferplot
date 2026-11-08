# Decisions

Decisions marked **fixed** change only with the human owner's agreement. Others may be
revised by the build agent when evidence warrants: add a dated entry to the log at the
bottom saying what changed, why, and the evidence, then update `SPEC.md`.

| # | Decision | Choice | Why |
| --- | --- | --- | --- |
| D1 | Build or fork **(fixed)** | New package; reuse ideas and MIT oracle scaffolding from plotsalot with credit | plotsalot lacks nonparametric methods, effect-size CIs and pandas input, and diverges from ggstatsplot by design (`PLOTSALOT_AUDIT.md`) |
| D2 | License **(fixed)** | MIT | Owner's choice |
| D3 | Name **(fixed)** | `inferplot` | PyPI page did not exist on 2026-11-07; `statsplot` is taken |
| D4 | Parity target **(fixed)** | ggstatsplot 1.1.4 statistics and subtitle content | Same answers as R on the same data; R becomes a complete oracle |
| D5 | Clean-room statistics **(fixed)** | Implement from published methods; never translate R package source except ggstatsplot (MIT, notice kept) | Keeps the MIT license clean |
| D6 | Runtime dependencies **(fixed)** | numpy, scipy, matplotlib, narwhals; pandas, polars, pyarrow optional; no GPL packages | License and install weight; fewer API-churn failures |
| D7 | Function names | ggstatsplot's without `gg`: `betweenstats`, `withinstats`, `histostats`, `scatterstats`, `barstats`, later `dotplotstats`, `piestats`, `corrmat`, `coefstats` | Agents know these names from R; `gg` wrongly implies ggplot2 |
| D8 | Arguments | `data, x, y` positional; ggstatsplot names in snake_case; `type` aliases accepted; no `**kwargs` | Familiar to agents trained on R or seaborn; strictness catches hallucinated arguments |
| D9 | Errors | Unknown keywords suggest the closest name; removed ggstatsplot names and seaborn habits get targeted messages | Agents trained on old code fail loudly with the fix |
| D10 | Statistical defaults | Exactly ggstatsplot 1.1.4, including Holm on top of Games-Howell and `bf_prior=0.707` | Parity |
| D11 | Deliberate differences | Required `subject_id`; 2,000 seeded bootstrap resamples; caption discloses hidden pairs; nothing silently dropped or aliased | Each fixes a documented failure; tests and point estimates still match R |
| D12 | Input | pandas, polars, pyarrow via narwhals; long format | Agents write pandas |
| D13 | Rendering | matplotlib object-oriented API only; headless; vector output with editable text; composed figures stay vector | Lowest LLM error rate among Python plotting libraries; journals need vectors |
| D14 | Return value | Figure plus typed `StatsResult` with JSON, APA sentence, caption, alt text, `check()` | Agents can report and self-check without reading pixels |
| D15 | Python | 3.11 and newer; CI on oldest and newest | AI sandboxes lag current Python; wider than SPEC 0 on purpose |
| D16 | v0.1 scope | Five functions, three classical types each, Bayes for t-tests and correlations | Core of ggstatsplot; Bayes ANOVA and contingency are hardest to match |
| D17 | Out of scope | Curve fits with uncertainty bands, residual plots | Different job |
| D18 | Oracle | GitHub Actions, CRAN snapshot pinned, hard version check, JSON fixtures committed | Nobody installs R locally |
| D19 | Visual style | Our own; Okabe-Ito palette | Parity is statistical and semantic, not pixels |
| D20 | Agent docs | `docs/api.md` under 32 KB, `llms.txt`, Claude Code skill in `skills/`; no MCP server in 1.0 | Agents learn a new package from context |
| D21 | Process documents | Only `docs/STATUS.md` and this file | plotsalot wrote 104 markdown documents in 47 hours, mostly milestone plans and sign-offs; they do not help users |
| D22 | Dependency floors | numpy >= 1.26, scipy >= 1.11, matplotlib >= 3.8, narwhals >= 2.0; extras pandas >= 2.1, polars >= 1.0, pyarrow >= 14; every dev tool has a floor too. A CI job installs the floors with `--resolution lowest-direct` on Python 3.11 | Sandboxes lag (D15). NumPy 1.26 is the last 1.x; matplotlib 3.8 is the first with inline type hints; narwhals 2.0 has the `stable.v2` API. Without floors, lowest-direct resolves to unusable ancient releases |
| D23 | Newest CI Python | 3.15 | 3.15.2 is released and numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, pandas 3.1.1 and pyarrow 26.0.1 publish cp315 wheels (PyPI, 2026-11-07) |
| D24 | Oracle fixture format | One JSON file per case with 15 significant digits, R's NA as null and non-finite numbers as strings; SHA-256 manifest; each fixture records the SHA-256 of its CSV; R collates strings in the C locale | Diffs stay reviewable; a hand edit or a changed dataset fails the tests; factor levels from text columns sort as Python's `sorted()` |
| D25 | Oracle datasets | R's and ggstatsplot's datasets exported by `oracle/export_datasets.R` and re-checked byte for byte on every oracle run; edge cases written by hand as `edge_*.csv` | Both sides read identical bytes; hand-written edge cases do not depend on R's random number generator |
| D26 | Bootstrap references | The underlying R function with 20,000 resamples, plus 50 replicate runs of 2,000 resamples giving the Monte Carlo SD of the estimate and of each bound. Among candidate calls, the one that reproduces ggstatsplot's point estimate is used | Gives SPEC section 10's bootstrap tolerance a measured SD; on the first run it caught that ggstatsplot's AKP is not `akp.effect`'s default |
| D27 | CI action pins | `astral-sh/setup-uv` pinned by commit SHA (v10.3.3); GitHub's own actions and r-lib by major tag | setup-uv publishes no floating major tags, so `@v10` does not resolve |

## Evidence

- ggstatsplot changelog (1.1.4 current; 1.0.0 removed `var.equal`, `nboot`,
  `effsize.type`; MIT since 0.13.1): https://www.indrapatil.com/ggstatsplot/news/
- plotsalot audit and its sources: `PLOTSALOT_AUDIT.md`
- JZS reference value used by `tools/reference/stats_reference.py`:
  https://pingouin-stats.org/build/html/generated/pingouin.bayesfactor_ttest.html

## Log

Format: `YYYY-MM-DD — D# — change — reason — evidence (test, fixture or link)`.

- 2026-11-07 — D1–D21 — initial decisions — design audit — `PLOTSALOT_AUDIT.md`
- 2026-11-07 — D22–D27 — added with the Phase 0 scaffold and oracle — tooling and first oracle runs — `pyproject.toml`, `.github/workflows/`, oracle runs 42445468125 and 42446300544
- 2026-11-07 — SPEC §4 — robust two-group AKP is `akp.effect(EQVAR = FALSE)` (group 1's winsorized SD), not the pooled default — ggstatsplot's estimate 0.682722711362298 is reproduced only by `EQVAR = FALSE`; `EQVAR = TRUE` gives 0.639261658700388 — `ggbetweenstats-ToothGrowth-robust.json`
- 2026-11-07 — SPEC §4 — Games-Howell q is signed as mean(group2) - mean(group1) in 1.1.4 (1.1.1 reported abs(q)) — reference cross-check against the fixtures — `ggbetweenstats-PlantGrowth-parametric.json`, `ggbetweenstats-mtcars-parametric.json`
- 2026-11-07 — SPEC §4 — Mann-Whitney follows R's defaults; with ties it is the normal approximation with continuity correction, although the method string reads only "Wilcoxon rank sum test" — SciPy reproduces p = 0.0644906721338357 only with the correction — `ggbetweenstats-ToothGrowth-nonparametric.json`
- 2026-11-07 — SPEC §4 — rank epsilon squared CI is one-sided (upper bound 1) percentile bootstrap with 100 iterations in ggstatsplot; t1way's explanatory effect size is random for unequal group sizes — `conf.iterations = 100`; replicate SD 0.0048 on mtcars — nonparametric and robust fixtures
- 2026-11-07 — SPEC §9 — `extract_stats()` returns no `descriptive_data` in 1.1.4; centrality values and labels are recorded from the plot layers (`plot_layers`); fixture schema version 1 documented — oracle run 42446300544
