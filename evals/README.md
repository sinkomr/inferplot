# Agent evaluation

This folder measures whether inferplot helps an AI coding agent produce a correct
statistical analysis (SPEC.md section 13). The same 32 requests, written as a
researcher would write them, go to an agent twice: once in a Python environment
without inferplot, once with inferplot and its skill. The answers are compared
with the numbers R's ggstatsplot 1.1.4 produced for the same data, and with the
numbers of every other correct analysis a careful analyst could have run (below).

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
| numbers | The statistic, degrees of freedom, p-value and n are within tolerance of one accepted variant as a whole (below). |
| effect* | The effect size is the oracle's (estimate and interval) or one of the standard alternatives listed for that test (estimate only). |
| trap avoided | The run did not fall into the task's trap. Counted only for tasks whose trap can be read from `answer.json`; the others are listed for reading (below). |
| check clean | inferplot arm only: a figure was saved; no drawing problem from `fig.check()` is left; every problem the data cause (small groups, heavy ties, missing rows) is reported; the result inferplot embeds in the saved figure has no unexpected issue. |
| used inferplot | The run imported, installed or ran inferplot (in a shell command, a file it wrote or a script it left). In the baseline arm such a run is marked `contaminated`. |
| turns, tokens, cost, time | From the session's own summary. |

\* **effect is not a quality score.** ggstatsplot reports particular effect sizes
(Hedges' g with a non-pooled SD, partial omega squared, rank epsilon squared,
Pearson's C, ...). An answer with a standard alternative that is one formula away
(Cohen's d or d_z for Hedges' g, d with the average variance, eta or omega squared
from the sums of squares, eta squared H, the probability of superiority, phi or
Cramér's V without the bias correction, Cohen's w, partial or generalized eta
squared, an odds ratio) counts, each computed from the data by a recipe. Any other
choice fails this column only, so the column says more about which tool was used
than about the analysis.

The two arms:

- **baseline**: a fresh virtual environment with pandas, numpy, scipy, matplotlib,
  seaborn and statsmodels. The run directory holds `data.csv` only.
- **inferplot**: the same packages plus inferplot (a wheel built from this checkout
  into the arm's own directory, installed, then deleted), and the skill in the run
  directory's `.claude/skills/inferplot/`. inferplot and the skill are the only
  difference.

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
it cites; every number reproduced by `ip.analyze.<name>` or by its recipe within
the task's tolerance; every count for n recomputed; the expected `check()` codes),
builds both arms' run directories in a temporary folder, prints the commands, and
estimates the model calls: 32 tasks x 2 arms = 64 sessions, each at most
`--max-turns` (40) model requests, so at most 2,560 requests.

Real run (the owner only, after deciding to spend the usage):

```sh
uv run python evals/run.py --i-understand-this-spends-usage \
  --workdir ~/agent-eval-1 --results ~/eval-results-1
```

Options: `--tasks` (a subset), `--arms baseline` or `inferplot`, `--repeats N`
(each task N times per arm; a single run per task is noisy), `--model`,
`--max-turns`, `--time-limit` (seconds per session, default 900), `--claude` (the
executable), `--workdir` (the agents' side, default a new temporary folder named
`agent-eval-...`) and `--results` (the harness's side, default a new temporary
folder named `eval-results-...`; neither may contain the other, and neither may be
in the repository). Without the flag, `--real` is refused; in CI (`CI` or
`GITHUB_ACTIONS` set) and on Windows (no process groups) the real mode is refused.

What a real run does: builds inferplot's wheel and creates the two environments
with `uv` (network), then for every task, arm and repeat runs, in that run's
directory and with that arm's environment first on `PATH`:

```sh
claude -p "<request + answer paragraph>" --output-format stream-json --verbose \
  --max-turns 40 --permission-mode acceptEdits \
  --allowedTools Bash,Read,Write,Edit,Glob,Grep \
  --setting-sources project --strict-mcp-config
```

The layout keeps what an agent can see neutral and the two arms apart:

```text
<workdir>/arm-a/env/        one arm's environment   (which arm is a or b is drawn
<workdir>/arm-a/run-01/     its run directories      for each run)
<workdir>/arm-b/...         the other arm
<results>/runs.json         which arm and task each name stands for
<results>/prompts/          the prompts
<results>/<arm>/<task>/     transcript.jsonl, the agent's files, score.json
<results>/scores.json       all scores
```

Nothing in an agent's paths names the package, the arm or the task (choose a
`--workdir` whose own path does not either; the harness warns if it does). The agent's
environment is the harness's without any variable or `PATH` entry that names the
repository or the results root (`VIRTUAL_ENV`, `PYTHONPATH` and the repository's
`.venv` are dropped). Each session starts its own process group, which is ended
when the session exits or its time is up, so nothing it started in the background
outlives it. As soon as a session ends, its directory is moved to the results root,
so no later session finds an earlier one's files next to its own.

Before the first real run, check these assumptions against `claude --help` for the
installed version, since they were written without running `claude`:

- `--setting-sources project` keeps the owner's user settings out, so the
  inferplot arm sees only its own skill. Check that a user-level `CLAUDE.md` or
  memory is not loaded; if it is, run under a separate user account or point
  `CLAUDE_CONFIG_DIR` at an empty folder (with API-key authentication).
- Project skills in `.claude/skills/` load in `-p` mode.
- `--strict-mcp-config` without `--mcp-config` starts no MCP server.

Safety and the limit of the isolation: with `Bash` allowed, each agent runs
commands of its choosing, unattended, with the owner's permissions. Nothing stops
it from listing `<workdir>` (and finding the other arm's environment), searching
the temporary folder for the results root, or reading `~/.claude` (where Claude
Code keeps earlier sessions' transcripts). The layout removes the adjacent and
recorded paths, not the possibility. Run the evaluation in a container, a virtual
machine or a throwaway user account, never in a shell that holds credentials you
care about, and read `used inferplot` for the baseline arm.

## Tasks

Each task is a folder in `tasks/` with `task.toml` and `data.csv`:

- `request`: the question as a researcher would ask it, never naming the test.
- `trap` and `trap_scoring`: the trap in words, and how it is scored: `traps` (the
  numbers of a `[[traps]]` analysis, or a rejected test), `test` (the choice of
  test), `n` (the rows with missing values must be left out), `none` (a baseline
  task) or `manual` (not scored automatically; `read_for` says what to read).
- `[data]`: the source dataset in `oracle/datasets/` and the column and level renames
  that turn it into `data.csv` (the dry run rebuilds the file from the source and
  compares).
- `[reference]`: the inferplot call for the design, and the `check()` codes its result
  must give (or the error it must raise).
- `[counts]`: every count that is right for n, named by what it counts:
  `observations` (the values the test used), `subjects` or `pairs` (subjects with
  every condition), `nonzero_differences` (the signed-rank and sign tests drop the
  zeros), `responses` and `discordant_pairs` (paired yes/no answers). The dry run
  recomputes each from `data.csv`. A block's `n_also` lists the counts that are
  also right for its test (pairs or observations for a paired t-test; also the
  nonzero differences for a signed-rank test).
- `[[accepted]]`: each analysis that counts as the correct test, with its numbers:
  from an oracle fixture (`fixture` and the `key` of each number in it; `kwargs`, the
  inferplot call that reproduces them), from a `recipe` (SciPy and NumPy, run by the
  dry run), or none (accepted without numbers). `qualifier` lists words one of
  which the reported name must contain, unless the numbers are the variant's own
  (Welch's ANOVA of the log reaction times). `[[accepted.effects]]` lists the
  standard alternative effect sizes, each with its recipe.
- `[[traps]]`: analyses that a trap leads to, with their numbers, so the scorer can
  name the mistake.
- `rejected`: test families that are wrong for this design.
- `correct_rule`: the rule in words, including why a plausible analysis is not
  accepted.

### Variants

Every standard variant of an accepted test that a careful analyst gets from SciPy's
or statsmodels' defaults, or from R's, is accepted with its own numbers: exact or
normal-approximation p, with or without a continuity correction, Yates' correction
or not, pooled or Welch's variance where the task allows both, the
Greenhouse-Geisser or Huynh-Feldt sphericity correction. In particular SciPy's
Wilcoxon signed-rank test (exact up to 50 differences, also with ties), its
Mann-Whitney test (normal approximation with continuity correction unless the
samples are small and untied), its Spearman p (t distribution), Kendall's tau-b
(exact without ties), and statsmodels' McNemar test (exact binomial, the statistic
being the smaller discordant count). The sign test is accepted wherever the
signed-rank test is, and Kendall's tau wherever Spearman's rho is. An answer is
scored against every accepted variant of the test it names and keeps the best: right
as a whole first, then the p-value, then the statistic. Zero handling other than
dropping the zeros (SciPy's `zero_method="pratt"` or `"zsplit"`) is a deliberate
non-default choice and is not listed.

### Test names

A reported name is matched to a family (`FAMILIES` in `run.py`) by words, after
folding case, Unicode dashes, Greek letters (`χ²` is chi2, `ρ` rho, `τ` tau),
possessives and punctuation, so "Welch Two Sample t-test", "t-test (unequal
variances)", "Welch's t–test" and `welch_t` are the same family, and "t-test"
alone is none: Welch's and Student's t-test must be named. inferplot's own
identifiers (`result.test.name`, `.label` and `.method`) are recognized for every
accepted analysis. `tests/test_evals.py` holds a table of more than 100 spellings,
right and wrong.

| Task | Function | Trap (scoring) | Accepted analyses | Fixtures |
| --- | --- | --- | --- | --- |
| `clinic-ratings-heavy-ties` | betweenstats | ordinal 1-to-5 ratings with heavy ties (test) | Kruskal-Wallis | `ggbetweenstats-edge_ties-nonparametric` |
| `diets-three-groups` | betweenstats | none (baseline) | Welch's ANOVA, Kruskal-Wallis, one-way ANOVA | `ggbetweenstats-PlantGrowth-parametric`, `-nonparametric` |
| `insect-sprays-six-groups` | betweenstats | six groups, very unequal variances (traps) | Welch's ANOVA, Kruskal-Wallis | `ggbetweenstats-InsectSprays-parametric`, `-nonparametric` |
| `reaction-times-real-outliers` | betweenstats | two real extreme values (test) | Kruskal-Wallis, trimmed-means ANOVA, Welch's ANOVA of the logs | `ggbetweenstats-edge_outliers-nonparametric`, `-robust` |
| `strains-unbalanced-small-group` | betweenstats | a group of 4 next to 12 and 25 (manual) | Welch's ANOVA, Kruskal-Wallis, one-way ANOVA | `ggbetweenstats-edge_unbalanced-parametric`, `-nonparametric` |
| `supplement-no-significance-hunting` | betweenstats | asked to find significance (traps) | Welch's, Student's t, Mann-Whitney (two-sided) | `ggbetweenstats-ToothGrowth-parametric`, `-nonparametric` |
| `supplement-registered-direction` | betweenstats | one-sided hypothesis registered in advance (traps) | Welch's, Student's t, Mann-Whitney (one-sided) | `ggbetweenstats-ToothGrowth-greater-parametric`, `-nonparametric` |
| `transmission-unequal-variances` | betweenstats | unequal variances and sizes (traps) | Welch's t, Mann-Whitney | `ggbetweenstats-mtcars-am-parametric`, `-nonparametric` |
| `yield-missing-values` | betweenstats | missing values, a row without a group (n) | Welch's ANOVA, Kruskal-Wallis, one-way ANOVA | `ggbetweenstats-edge_missing-parametric`, `-nonparametric` |
| `anxiety-ratings-before-after` | withinstats | paired ordinal ratings with ties (test) | Wilcoxon signed-rank, sign test | `ggwithinstats-edge_within2_ties-nonparametric` |
| `bugs-four-situations` | withinstats | four conditions, incomplete subjects, sphericity (traps) | RM ANOVA (Greenhouse-Geisser, Huynh-Feldt), Friedman | `ggwithinstats-bugs_long-parametric`, `-nonparametric` |
| `caffeine-pilot-five-people` | withinstats | five participants (manual) | paired t, Wilcoxon signed-rank, sign test | `ggwithinstats-edge_within2_small-parametric`, `-nonparametric` |
| `grip-strength-missed-visits` | withinstats | missed visits (traps) | paired t, Wilcoxon signed-rank, sign test | `ggwithinstats-edge_within2_missing-parametric`, `-nonparametric` |
| `noise-four-conditions-sphericity` | withinstats | sphericity badly violated (traps) | RM ANOVA (Greenhouse-Geisser, Huynh-Feldt), Friedman | `ggwithinstats-edge_within_nonspherical-parametric`, `-nonparametric` |
| `sleep-two-drugs` | withinstats | long format, two rows per patient (traps) | paired t, Wilcoxon signed-rank, sign test | `ggwithinstats-sleep-parametric`, `-nonparametric` |
| `training-before-after-shuffled` | withinstats | rows not in subject order (traps) | paired t, Wilcoxon signed-rank, sign test | `ggwithinstats-edge_within2_shuffled-parametric`, `-nonparametric` |
| `bag-weights-missing` | histostats | missing values (n) | one-sample t, Wilcoxon signed-rank, sign test | `gghistostats-edge_one_missing-parametric`, `-nonparametric` |
| `fuel-economy-against-20` | histostats | a stated test value (20), not 0 (traps) | one-sample t, Wilcoxon signed-rank, sign test | `gghistostats-mtcars-tv20-parametric`, `-nonparametric` |
| `recipe-tasters-five` | histostats | five values (manual) | one-sample t, Wilcoxon signed-rank, sign test | `gghistostats-edge_one_small-parametric`, `-nonparametric` |
| `sensor-calibration-outlier` | histostats | one real extreme reading (traps) | Wilcoxon signed-rank, sign test, bootstrap-t of the trimmed mean | `gghistostats-edge_one_outlier-nonparametric`, `-robust` |
| `height-leaf-area-missing-pairs` | scatterstats | missing values in either column (n) | Pearson, Spearman, Kendall | `ggscatterstats-edge_corr_missing-parametric`, `-nonparametric` |
| `reading-vocabulary-outlier` | scatterstats | one real point reverses Pearson's r (traps) | Spearman, Kendall, winsorized r | `ggscatterstats-edge_corr_outlier-nonparametric`, `-robust` |
| `satisfaction-engagement-ordinal` | scatterstats | ordinal ratings with ties (test) | Spearman, Kendall | `ggscatterstats-edge_corr_ties-nonparametric` |
| `weight-fuel-correlation` | scatterstats | none (baseline) | Pearson, Spearman, Kendall | `ggscatterstats-mtcars-parametric`, `-nonparametric` |
| `cylinders-planned-shares` | barstats | stated shares, not equal shares (traps) | chi-square goodness of fit | `ggbarstats-mtcars-cyl-ratio-parametric` |
| `engine-transmission-two-by-two` | barstats | none: R and SciPy differ on the continuity correction | chi-square (R), chi-square (Yates), Fisher | `ggbarstats-mtcars-am-vs-parametric` and recipes |
| `headaches-before-after-paired` | barstats | paired yes/no answers (traps) | McNemar (R, corrected, exact) | `ggbarstats-edge_bar_paired-parametric` and recipes |
| `survey-regions-missing` | barstats | missing categories, small expected counts (n) | chi-square, Fisher (Freeman-Halton) | `ggbarstats-edge_bar_missing-parametric` and a recipe |
| `titanic-table-of-counts` | barstats | a table of counts (Freq column) (traps) | chi-square of independence | `ggbarstats-Titanic_counts-class-parametric` |
| `bayes-factor-three-groups` | betweenstats | refuse: Bayes factor for 3 groups, v0.1 raises (test) | no test with the reason, or a named Bayes factor | none |
| `dose-by-supplement-interaction` | betweenstats | redirect: two-factor design (test) | two-way ANOVA, linear model with interaction (dose as a factor) | recipe |
| `weight-fuel-with-covariates` | scatterstats | redirect: regression with covariates (traps) | multiple regression (cyl as a number or a factor) | recipe |

## Tolerances

A reported number is right when |reported - expected| <= max(abs, rel x |expected|):

| Number | abs | rel | Why |
| --- | --- | --- | --- |
| statistic | 0.01 | 0.5% | agents round to 2 or 3 decimals; t, z, r and rho may have either sign (level order) |
| df, df_error | 0.02 | | Welch and Greenhouse-Geisser df are printed to 2 decimals |
| p-value | 0.001 | 5% | rounding to 2 or 3 significant digits |
| n | 0 | | exact, but any of the task's counts listed for the test |
| effect size | 0.01 | | the oracle's effect size or a listed alternative |
| interval bounds | 0.02 | | noncentral and normal intervals from other tools differ slightly; a mirrored interval counts when the sign is reversed (the oracle's effect size only) |

The rank statistics may be reported for either group or direction (`alt` values:
U or n1 n2 - U, V or its complement, S or rho, t or r, z for a normal
approximation). A test with one df may give it in `df` or in `df_error`.

How numbers are read: a JSON number, or text whose number follows the first `=`,
`<` or `>` ("t(9) = -4.06", "p < .001"); `2.83 × 10^-3`, `2.83 x 10⁻³` and
`2.8e-03` are 0.00283; a one-element list is its element; thousands separators
are dropped. A p-value given as a bound ("p < .01") is right when the true p is
below the bound and the bound is .001 or smaller (above .001 APA style reports the
exact p) or within the p tolerance of the true p; so "< 2.2e-16" is right for
p = 5e-41 and "p < .01" is wrong for p = .0028. The bound is recorded with the
number in `score.json`, never read silently as a value. A "p > b" counts only
when b is within the tolerance of the true p.

## Reading the scores

`results/scores.json` has one row per run; the printed table has, per arm, the share
of runs that ran, chose a correct test, got the numbers and the effect right, were
clean (inferplot arm) and avoided the trap (over the runs whose trap is scored),
the number of runs whose trap is to be read, and the runs that used inferplot,
with the median turns and the total output tokens. Under the table it lists each
trap to read with what to look for (`read_for`): the caution a five-value sample
calls for, and the small group of four plants. Read them together: a correct test
with wrong numbers usually means a mistake in the data (pairs, missing values) or
an unlisted variant (the score names the variant the numbers were compared with,
`variant`, and each number's expected and reported value); a trap names the
mistake. `contaminated` marks a baseline run that imported, installed or ran
inferplot, with the snippets in `inferplot_uses`. With one repeat, a difference of
one or two tasks between arms is noise; use `--repeats 3` or more before drawing
conclusions.

Limits: `answer.json` is what the agent says it found; the numbers are not
recomputed from its code. Drawing problems in the inferplot arm are those the agent
reports from `fig.check()` (a saved figure keeps the result but not the data, so the
figure cannot be checked again here). Pairwise comparisons are not scored. The
traps marked `manual` are not scored at all: they are listed for a person to read.
