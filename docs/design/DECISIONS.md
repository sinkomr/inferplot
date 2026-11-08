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

## Evidence

- ggstatsplot changelog (1.1.4 current; 1.0.0 removed `var.equal`, `nboot`,
  `effsize.type`; MIT since 0.13.1): https://www.indrapatil.com/ggstatsplot/news/
- plotsalot audit and its sources: `PLOTSALOT_AUDIT.md`
- JZS reference value used by `tools/reference/stats_reference.py`:
  https://pingouin-stats.org/build/html/generated/pingouin.bayesfactor_ttest.html

## Log

Format: `YYYY-MM-DD — D# — change — reason — evidence (test, fixture or link)`.

- 2026-11-07 — D1–D21 — initial decisions — design audit — `PLOTSALOT_AUDIT.md`
