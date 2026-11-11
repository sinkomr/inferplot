# Agent evaluation

This folder measures whether inferplot helps an AI coding agent produce a correct
statistical analysis (SPEC.md section 13). The same 32 requests, written as a
researcher would write them, go to an agent twice: once in a Python environment
without inferplot, once with inferplot and its skill. The answers are compared
with the numbers R's ggstatsplot 1.1.4 produced for the same data.

**A real run spends the owner's Claude usage.** It is started only by the owner,
with an explicit flag, after the project's rule in CLAUDE.md: ask the owner before
every run. The default mode, the dry run, makes no model call, needs no network
and costs nothing; CI runs it on every push through `tests/test_evals.py`.

## What it measures

For each task and each arm (one `claude -p` session in a fresh directory):

| Score | Meaning |
| --- | --- |
| ran | The session finished within its time limit without an error and left a readable `answer.json`. |
| correct test | The reported test is one the task accepts for its design (below). |
| numbers | The test's statistic, degrees of freedom, p-value and n are within tolerance of the accepted analysis the agent chose. |
| effect | The effect size and its interval are within tolerance (scored separately: some tools report another effect size). |
| trap | The reported numbers are those of a known mistake for this task (for example an unpaired test on paired data). |
| check clean | inferplot arm only: a figure was saved; no drawing problem from `fig.check()` is left; every problem the data cause (small groups, heavy ties, missing rows) is reported; the result inferplot embeds in the saved figure has no unexpected issue. |
| turns, tokens, cost, time | From the session's own summary. |

The two arms:

- **baseline**: a fresh virtual environment with pandas, numpy, scipy, matplotlib,
  seaborn and statsmodels. The run directory holds `data.csv` only.
- **inferplot**: the same packages plus inferplot (installed from this checkout), and
  the skill in the run directory's `.claude/skills/inferplot/`. inferplot and the
  skill are the only difference.

Each prompt is the task's request followed by one fixed paragraph, the same in both
arms, asking the agent to write `answer.json` (test, statistic, df, p, effect size
and interval, n, the tests it ran, the figure file, the problems its tools
reported, notes). The paragraph names no test. The task files, which hold the
expected numbers, are never copied into a run directory, and run directories must
be outside the repository.

## Running it

Dry run (the default; validates the tasks and prints what a real run would do):

```sh
uv run python evals/run.py
uv run python evals/run.py --tasks sleep-two-drugs,diets-three-groups
```

It checks every task (schema; the dataset present, readable and equal to its stated
source after the stated renames; every expected number equal to the oracle fixture
it cites; every number reproduced by `ip.analyze.<name>` or by its SciPy recipe
within the task's tolerance; the expected `check()` codes), builds both arms' run
directories in a temporary folder, prints the commands, and estimates the model
calls: 32 tasks x 2 arms = 64 sessions, each at most `--max-turns` (40) model
requests, so at most 2,560 requests.

Real run (the owner only, after deciding to spend the usage):

```sh
uv run python evals/run.py --i-understand-this-spends-usage --workdir ~/inferplot-eval-1
```

Options: `--tasks` (a subset), `--arms baseline` or `inferplot`, `--repeats N`
(each task N times per arm; a single run per task is noisy), `--model`,
`--max-turns`, `--time-limit` (seconds per session, default 900), `--claude` (the
executable). Without the flag, `--real` is refused; in CI (`CI` or
`GITHUB_ACTIONS` set) the real mode is refused.

What a real run does: creates the two environments with `uv` (network), then for
every task, arm and repeat runs, in that run's directory and with that arm's
environment first on `PATH`:

```sh
claude -p "<request + answer paragraph>" --output-format stream-json --verbose \
  --max-turns 40 --permission-mode acceptEdits \
  --allowedTools Bash,Read,Write,Edit,Glob,Grep \
  --setting-sources project --strict-mcp-config
```

It saves the transcript (`transcript.jsonl`), every file the agent produced and a
`score.json` per run under `<workdir>/results/<arm>/<task>/`, all scores in
`results/scores.json`, and prints a summary table per arm.

Before the first real run, check these assumptions against `claude --help` for the
installed version, since they were written without running `claude`:

- `--setting-sources project` keeps the owner's user settings out, so the
  inferplot arm sees only its own skill. Check that a user-level `CLAUDE.md` or
  memory is not loaded; if it is, run under a separate user account or point
  `CLAUDE_CONFIG_DIR` at an empty folder (with API-key authentication).
- Project skills in `.claude/skills/` load in `-p` mode.
- `--strict-mcp-config` without `--mcp-config` starts no MCP server.

Safety: with `Bash` allowed, each agent runs commands of its choosing in its run
directory, unattended. Run the evaluation in a container, a virtual machine or a
throwaway user account, never in a shell that holds credentials you care about.

## Tasks

Each task is a folder in `tasks/` with `task.toml` and `data.csv`:

- `request`: the question as a researcher would ask it, never naming the test.
- `[data]`: the source dataset in `oracle/datasets/` and the column and level renames
  that turn it into `data.csv` (the dry run rebuilds the file from the source and
  compares).
- `[reference]`: the inferplot call for the design, and the `check()` codes its result
  must give (or the error it must raise).
- `[[accepted]]`: each analysis that counts as the correct test, with its numbers:
  from an oracle fixture (`fixture` and the `key` of each number in it; `kwargs`, the
  inferplot call that reproduces them), from a SciPy `recipe` for variants
  ggstatsplot does not run (Yates' correction, the exact McNemar test, Student's
  t-test, a regression), or none (accepted without numbers).
- `[[traps]]`: analyses that a trap leads to, with their numbers, so the scorer can
  name the mistake.
- `rejected`: test families that are wrong for this design.
- `correct_rule`: the rule in words.

A reported test name is matched to a family (`FAMILIES` in `run.py`) by words, so
"Welch Two Sample t-test", "Welch's t-test" and `welch_t` are the same family, and
"t-test" alone is none: Welch's and Student's t-test must be named. Where an
accepted family has several variants (McNemar with or without correction, exact),
the numbers are scored against the variant they match best.

| Task | Function | Trap | Accepted analyses | Fixtures |
| --- | --- | --- | --- | --- |
| `clinic-ratings-heavy-ties` | betweenstats | ordinal 1-to-5 ratings with heavy ties | Kruskal-Wallis | `ggbetweenstats-edge_ties-nonparametric` |
| `diets-three-groups` | betweenstats | none (baseline) | Welch's ANOVA, Kruskal-Wallis, one-way ANOVA | `ggbetweenstats-PlantGrowth-parametric`, `-nonparametric` |
| `insect-sprays-six-groups` | betweenstats | six groups, very unequal variances | Welch's ANOVA, Kruskal-Wallis | `ggbetweenstats-InsectSprays-parametric`, `-nonparametric` |
| `reaction-times-real-outliers` | betweenstats | two real extreme values | Kruskal-Wallis, trimmed-means ANOVA | `ggbetweenstats-edge_outliers-nonparametric`, `-robust` |
| `strains-unbalanced-small-group` | betweenstats | a group of 4 next to 12 and 25 | Welch's ANOVA, Kruskal-Wallis, one-way ANOVA | `ggbetweenstats-edge_unbalanced-parametric`, `-nonparametric` |
| `supplement-no-significance-hunting` | betweenstats | asked to find significance | Welch's, Student's t, Mann-Whitney (two-sided) | `ggbetweenstats-ToothGrowth-parametric`, `-nonparametric` |
| `supplement-registered-direction` | betweenstats | one-sided hypothesis registered in advance | Welch's t, Mann-Whitney (one-sided) | `ggbetweenstats-ToothGrowth-greater-parametric`, `-nonparametric` |
| `transmission-unequal-variances` | betweenstats | unequal variances and sizes | Welch's t, Mann-Whitney | `ggbetweenstats-mtcars-am-parametric`, `-nonparametric` |
| `yield-missing-values` | betweenstats | missing values, a row without a group | Welch's ANOVA, Kruskal-Wallis, one-way ANOVA | `ggbetweenstats-edge_missing-parametric`, `-nonparametric` |
| `anxiety-ratings-before-after` | withinstats | paired ordinal ratings with ties | Wilcoxon signed-rank | `ggwithinstats-edge_within2_ties-nonparametric` |
| `bugs-four-situations` | withinstats | four conditions, incomplete subjects, sphericity | RM ANOVA (Greenhouse-Geisser), Friedman | `ggwithinstats-bugs_long-parametric`, `-nonparametric` |
| `caffeine-pilot-five-people` | withinstats | five participants | paired t, Wilcoxon signed-rank | `ggwithinstats-edge_within2_small-parametric`, `-nonparametric` |
| `grip-strength-missed-visits` | withinstats | missed visits | paired t, Wilcoxon signed-rank | `ggwithinstats-edge_within2_missing-parametric`, `-nonparametric` |
| `noise-four-conditions-sphericity` | withinstats | sphericity badly violated | RM ANOVA (Greenhouse-Geisser), Friedman | `ggwithinstats-edge_within_nonspherical-parametric`, `-nonparametric` |
| `sleep-two-drugs` | withinstats | long format, two rows per patient | paired t, Wilcoxon signed-rank | `ggwithinstats-sleep-parametric`, `-nonparametric` |
| `training-before-after-shuffled` | withinstats | rows not in subject order | paired t, Wilcoxon signed-rank | `ggwithinstats-edge_within2_shuffled-parametric`, `-nonparametric` |
| `bag-weights-missing` | histostats | missing values | one-sample t, Wilcoxon signed-rank | `gghistostats-edge_one_missing-parametric`, `-nonparametric` |
| `fuel-economy-against-20` | histostats | a stated test value (20), not 0 | one-sample t, Wilcoxon signed-rank | `gghistostats-mtcars-tv20-parametric`, `-nonparametric` |
| `recipe-tasters-five` | histostats | five values | one-sample t, Wilcoxon signed-rank | `gghistostats-edge_one_small-parametric`, `-nonparametric` |
| `sensor-calibration-outlier` | histostats | one real extreme reading | Wilcoxon signed-rank, bootstrap-t of the trimmed mean | `gghistostats-edge_one_outlier-nonparametric`, `-robust` |
| `height-leaf-area-missing-pairs` | scatterstats | missing values in either column | Pearson, Spearman | `ggscatterstats-edge_corr_missing-parametric`, `-nonparametric` |
| `reading-vocabulary-outlier` | scatterstats | one real point reverses Pearson's r | Spearman, winsorized r | `ggscatterstats-edge_corr_outlier-nonparametric`, `-robust` |
| `satisfaction-engagement-ordinal` | scatterstats | ordinal ratings with ties | Spearman | `ggscatterstats-edge_corr_ties-nonparametric` |
| `weight-fuel-correlation` | scatterstats | none (baseline) | Pearson, Spearman | `ggscatterstats-mtcars-parametric`, `-nonparametric` |
| `cylinders-planned-shares` | barstats | stated shares, not equal shares | chi-square goodness of fit | `ggbarstats-mtcars-cyl-ratio-parametric` |
| `engine-transmission-two-by-two` | barstats | 2 x 2: R and SciPy differ on the continuity correction | chi-square (R), chi-square (Yates), Fisher | `ggbarstats-mtcars-am-vs-parametric` and recipes |
| `headaches-before-after-paired` | barstats | paired yes/no answers | McNemar (R, corrected, exact) | `ggbarstats-edge_bar_paired-parametric` and recipes |
| `survey-regions-missing` | barstats | missing categories, small expected counts | chi-square, Fisher (no numbers) | `ggbarstats-edge_bar_missing-parametric` |
| `titanic-table-of-counts` | barstats | a table of counts (Freq column) | chi-square of independence | `ggbarstats-Titanic_counts-class-parametric` |
| `bayes-factor-three-groups` | betweenstats | refuse: Bayes factor for 3 groups (v0.1 raises) | no test with the reason, or a named Bayes factor | none |
| `dose-by-supplement-interaction` | betweenstats | redirect: two-factor design | two-way ANOVA, linear model with interaction | recipe |
| `weight-fuel-with-covariates` | scatterstats | redirect: regression with covariates | multiple regression | recipe |

## Tolerances

A reported number is right when |reported - expected| <= max(abs, rel x |expected|):

| Number | abs | rel | Why |
| --- | --- | --- | --- |
| statistic | 0.01 | 0.5% | agents round to 2 or 3 decimals; t, z, r and rho may have either sign (level order) |
| df, df_error | 0.02 | | Welch and Greenhouse-Geisser df are printed to 2 decimals |
| p-value | 0.001 | 5% | a p below 0.001 may be reported as "< .001" |
| n | 0 | | exact |
| effect size | 0.01 | | the effect size of the accepted analysis (Hedges' g, not Cohen's d, which differs by up to 10% at n = 10); another effect size fails this score only, which is why it is separate |
| interval bounds | 0.02 | | noncentral and normal intervals from other tools differ slightly; a mirrored interval counts when the sign is reversed |

The rank statistics may be reported for either group or direction (`alt` values:
U or n1 n2 - U, V or its complement, S or rho, t or r). The expected numbers follow
R's rules (exact or normal approximation, continuity corrections), which can differ
from a library's defaults; such an answer is still scored as the correct test, with
its numbers marked wrong, unless the task lists the variant.

## Reading the scores

`results/scores.json` has one row per run; the printed table has, per arm, the share
of runs that ran, chose a correct test, got the numbers and the effect right, were
clean (inferplot arm), and fell into a trap, with the median turns and the total
output tokens. Read them together: a correct test with wrong numbers usually means
a variant (a continuity correction, an exact p) or a mistake in the data (pairs,
missing values); a trap names the mistake. `contaminated` marks a baseline run that
installed or imported inferplot. With one repeat, a difference of one or two tasks
between arms is noise; use `--repeats 3` or more before drawing conclusions.

Limits: `answer.json` is what the agent says it found; the numbers are not
recomputed from its code. Drawing problems in the inferplot arm are those the agent
reports from `fig.check()` (a saved figure keeps the result but not the data, so the
figure cannot be checked again here). Pairwise comparisons are not scored.
