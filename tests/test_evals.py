"""The agent-evaluation task set and harness (evals/README.md, SPEC.md section 13).

Every task's dry-run validation runs here, so the expected numbers stay true as the
package changes. The scorer is tested on hand-written transcripts and answers, and
the harness is shown to refuse model calls without the explicit flag. No test runs
`claude` or needs the network: the one function that calls the model is replaced.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import inferplot as ip

ROOT = Path(__file__).resolve().parents[1]


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location(
        "evals_run", ROOT / "evals" / "run.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evals_run"] = module
    spec.loader.exec_module(module)
    return module


run = _load_harness()
TASKS = run.load_tasks()
BY_ID = {task.id: task for task in TASKS}


def _forbid_model(*_args: object, **_kwargs: object) -> Any:
    raise AssertionError("a model call was attempted")


@pytest.fixture
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any model call, environment build or subprocess fail the test."""
    monkeypatch.setattr(run, "_call_claude", _forbid_model)
    monkeypatch.setattr(run, "_build_envs", _forbid_model)
    monkeypatch.setattr(subprocess, "run", _forbid_model)
    monkeypatch.setattr(subprocess, "Popen", _forbid_model)


# --- the task set ------------------------------------------------------------------


def test_task_set_covers_the_functions_and_traps() -> None:
    assert 28 <= len(TASKS) <= 36
    assert {t.function for t in TASKS} == set(run.FUNCTIONS)
    traps = " ".join(t.trap for t in TASKS)
    for needed in (
        "subject column",  # pairing from the subject, not the row order
        "paired yes/no",  # paired counts
        "stated value",  # one-sample test against a value
        "unequal variances",
        "heavy ties",
        "small sample",
        "missing",
        "table of counts",
        "significance",  # no hunting across test types
        "regression with covariates",  # redirect
        "v0.1 does not support",  # redirect
        "refuses",  # the package raises
    ):
        assert needed in traps, needed


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_task_is_valid(task: Any) -> None:
    # Dataset equal to its source, every number equal to its fixture and
    # reproduced by inferplot or by its recipe within the task's tolerance, the
    # expected check() codes (the dry run's check, task by task).
    assert run.validate_task(task) == []


def test_requests_do_not_name_the_answer() -> None:
    words = (
        "t-test",
        "anova",
        "wilcoxon",
        "mann",
        "kruskal",
        "friedman",
        "mcnemar",
        "chi-square",
        "chi square",
        "pearson",
        "spearman",
        "inferplot",
        "welch",
    )
    for task in TASKS:
        lowered = task.request.lower()
        assert not [w for w in words if w in lowered], task.id


def test_every_fixture_number_is_checked_against_inferplot_or_a_recipe() -> None:
    for task in TASKS:
        for block in (*task.accepted, *task.traps):
            if block.source == "fixture":
                assert block.kwargs is not None, (task.id, block.label)
            if block.source != "none":
                assert block.numbers, (task.id, block.label)


def test_validation_catches_a_wrong_number() -> None:
    task = BY_ID["sleep-two-drugs"]
    block = task.accepted[0]
    wrong = dict(block.numbers)
    wrong["p_value"] = dataclasses.replace(wrong["p_value"], value=0.0383)
    broken = dataclasses.replace(
        task, accepted=(dataclasses.replace(block, numbers=wrong), *task.accepted[1:])
    )
    problems = run.validate_task(broken)
    assert any("p_value" in p and "fixture" in p for p in problems)
    assert any("inferplot gives p_value" in p for p in problems)


def test_validation_catches_a_changed_dataset(tmp_path: Path) -> None:
    task = BY_ID["sleep-two-drugs"]
    data = pd.read_csv(task.data_file)
    data.loc[0, "extra_sleep_h"] = 0.8
    changed = tmp_path / "data.csv"
    data.to_csv(changed, index=False)
    problems = run.validate_task(dataclasses.replace(task, data_file=changed))
    assert problems == [
        f"sleep-two-drugs: dataset differs from {task.data_source} "
        "with the stated renames"
    ]


def test_bad_task_files_are_refused(tmp_path: Path) -> None:
    source = (BY_ID["sleep-two-drugs"].path / "task.toml").read_text(encoding="utf-8")
    cases = {
        "unknown-family": source.replace('family = "paired_t"', 'family = "t_test"'),
        "unknown-key": source + "\nexpected_p = 0.1\n",
        "missing-request": source.replace("request = ", "question = "),
    }
    for name, text in cases.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "task.toml").write_text(
            text.replace("sleep-two-drugs", name), encoding="utf-8"
        )
        with pytest.raises(run.TaskError):
            run.load_task(folder / "task.toml")


# --- the dry run and the refusal to spend usage ------------------------------------


def test_dry_run_makes_no_model_call(no_model: None, tmp_path: Path) -> None:
    lines: list[str] = []
    ids = "sleep-two-drugs,bayes-factor-three-groups"
    code = run.main(["--tasks", ids, "--workdir", str(tmp_path)], out=lines.append)
    assert code == 0
    text = "\n".join(lines)
    assert "no model call was made. 2 of 2 tasks valid." in text
    assert "4 claude sessions (2 tasks x 2 arms x 1 repeat)" in text
    # The arms: data only in the baseline, data and the skill in the inferplot arm,
    # and nothing of the task file (the expected numbers) in either.
    baseline = tmp_path / "baseline" / "sleep-two-drugs"
    with_skill = tmp_path / "inferplot" / "sleep-two-drugs"
    assert sorted(p.name for p in baseline.rglob("*")) == ["data.csv"]
    assert (with_skill / ".claude" / "skills" / "inferplot" / "SKILL.md").is_file()
    assert not list(tmp_path.rglob("task.toml"))
    prompt = (tmp_path / "prompts" / "sleep-two-drugs.txt").read_text(encoding="utf-8")
    assert "answer.json" in prompt
    assert "paired" not in prompt.lower()
    # The exact commands are printed, the environments first.
    assert "uv venv" in text
    assert text.count("--output-format stream-json") == 4


def test_real_mode_needs_the_flag(no_model: None, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run.main(["--real", "--workdir", str(tmp_path)], out=lambda _: None)
    with pytest.raises(SystemExit):
        run.main(
            ["--dry-run", run.SPEND_FLAG, "--workdir", str(tmp_path)],
            out=lambda _: None,
        )
    # Without any flag the harness only does the dry run.
    assert (
        run.main(
            ["--tasks", "sleep-two-drugs", "--workdir", str(tmp_path)],
            out=lambda _: None,
        )
        == 0
    )


def test_real_mode_is_refused_in_ci(
    no_model: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    with pytest.raises(SystemExit, match="CI"):
        run.main([run.SPEND_FLAG, "--workdir", str(tmp_path)], out=lambda _: None)


def test_workdir_inside_the_repository_is_refused(no_model: None) -> None:
    with pytest.raises(SystemExit):
        run.main(["--workdir", str(ROOT / "evals" / "work")], out=lambda _: None)


# --- the scorer on hand-written transcripts ----------------------------------------


def _transcript(
    *, turns: int = 6, is_error: bool = False, commands: tuple[str, ...] = ()
) -> str:
    events: list[dict[str, Any]] = [{"type": "system", "subtype": "init"}]
    for command in commands:
        events.append(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": command},
                        },
                    ]
                },
            }
        )
    events.append(
        {
            "type": "result",
            "subtype": "error_during_execution" if is_error else "success",
            "is_error": is_error,
            "num_turns": turns,
            "duration_ms": 41_000,
            "total_cost_usd": 0.12,
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 3400,
                "cache_read_input_tokens": 50_000,
                "cache_creation_input_tokens": 9000,
            },
        }
    )
    return "\n".join(json.dumps(e) for e in events) + "\n"


SLEEP_ANSWER = {
    "test": "Paired t-test",
    "statistic": -4.06,
    "df": 9,
    "df_error": None,
    "p_value": 0.0028,
    "effect_size": "Hedges' g",
    "estimate": -1.17,
    "conf_low": -1.94,
    "conf_high": -0.38,
    "n": 10,
    "tests_run": ["Paired t-test"],
    "figure": "sleep.svg",
    "check_issues": [],
    "notes": "Drug B gave more extra sleep.",
}


def _write(path: Path, answer: dict[str, Any] | None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if answer is not None:
        (path / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    return path


def _sleep_figure(path: Path) -> None:
    task = BY_ID["sleep-two-drugs"]
    data = pd.read_csv(task.data_file)
    fig = ip.withinstats(data, "drug", "extra_sleep_h", subject_id="patient")
    fig.save(path / "sleep.svg")


def test_score_of_a_correct_run(tmp_path: Path) -> None:
    folder = _write(tmp_path, SLEEP_ANSWER)
    _sleep_figure(folder)
    score = run.score_run(
        BY_ID["sleep-two-drugs"], "inferplot", _transcript(), folder, exit_code=0
    )
    assert score.ran
    assert score.correct_test
    assert score.family == "paired_t"
    assert score.numbers_ok
    assert score.effect_ok
    assert score.trap is None
    assert score.check_clean, score.check_reason
    assert score.check_reason == "clean"
    assert (score.turns, score.output_tokens, score.cost_usd, score.duration_s) == (
        6,
        3400,
        0.12,
        41.0,
    )


def test_score_of_a_run_with_the_wrong_test(tmp_path: Path) -> None:
    # The classic mistake on these data: an unpaired test (t = -1.86, p = .079).
    answer = SLEEP_ANSWER | {
        "test": "Welch Two Sample t-test",
        "statistic": -1.8608,
        "df": 17.776,
        "p_value": 0.0794,
        "estimate": -0.80,
        "conf_low": -1.70,
        "conf_high": 0.10,
        "n": 20,
    }
    score = run.score_run(
        BY_ID["sleep-two-drugs"],
        "baseline",
        _transcript(),
        _write(tmp_path, answer),
        exit_code=0,
    )
    assert score.ran
    assert score.correct_test is False
    assert score.test_reason == "rejected: Welch's t-test"
    assert score.numbers_ok is None
    assert score.trap == "unpaired Welch's t-test"
    assert score.check_clean is None  # scored only in the inferplot arm


def test_score_of_a_run_with_a_wrong_number(tmp_path: Path) -> None:
    answer = SLEEP_ANSWER | {"p_value": 0.0057}  # a one-sided p doubled, say
    score = run.score_run(
        BY_ID["sleep-two-drugs"],
        "baseline",
        _transcript(),
        _write(tmp_path, answer),
        exit_code=0,
    )
    assert score.correct_test
    assert score.numbers_ok is False
    assert score.numbers["p_value"] == {
        "expected": 0.00283289019738427,
        "reported": 0.0057,
        "ok": False,
    }
    assert score.numbers["statistic"]["ok"]
    assert score.effect_ok


def test_score_of_a_crash(tmp_path: Path) -> None:
    folder = _write(tmp_path, None)
    score = run.score_run(
        BY_ID["sleep-two-drugs"], "baseline", "Traceback ...\n", folder, exit_code=1
    )
    assert not score.ran
    assert score.failure == "exit code 1"
    assert score.correct_test is None
    errored = run.score_run(
        BY_ID["sleep-two-drugs"],
        "baseline",
        _transcript(is_error=True),
        folder,
        exit_code=0,
    )
    assert errored.failure == "session ended with error_during_execution"
    late = run.score_run(
        BY_ID["sleep-two-drugs"], "baseline", "", folder, exit_code=None, timed_out=True
    )
    assert late.failure == "time limit"


def test_score_reads_numbers_written_as_text(tmp_path: Path) -> None:
    # F(2, 17.13) written in one field, p as an inequality, sign of t reversed.
    answer = {
        "test": "Welch's ANOVA",
        "statistic": "5.18",
        "df": "2, 17.13",
        "p_value": "p = .017",
        "estimate": 0.29,
        "conf_low": 0.0,
        "conf_high": 1.0,
        "n": "30",
    }
    score = run.score_run(
        BY_ID["diets-three-groups"],
        "baseline",
        _transcript(),
        _write(tmp_path, answer),
        exit_code=0,
    )
    assert score.numbers_ok
    assert score.effect_ok
    bugs = {
        "test": "repeated measures ANOVA (Greenhouse-Geisser)",
        "statistic": 20.59,
        "df": 2.63,
        "df_error": 228.95,
        "p_value": "< .001",
        "n": 88,
    }
    score = run.score_run(
        BY_ID["bugs-four-situations"],
        "baseline",
        _transcript(),
        _write(tmp_path / "bugs", bugs),
        exit_code=0,
    )
    assert score.numbers_ok
    uncorrected = bugs | {
        "test": "repeated measures ANOVA",
        "df": 3,
        "df_error": 261,
        "p_value": 1.5e-11,
    }
    score = run.score_run(
        BY_ID["bugs-four-situations"],
        "baseline",
        _transcript(),
        _write(tmp_path / "bugs2", uncorrected),
        exit_code=0,
    )
    assert score.numbers_ok is False
    assert score.trap == "repeated-measures ANOVA without the sphericity correction"


def test_score_of_variants_and_redirects(tmp_path: Path) -> None:
    # SciPy's default chi-square on a 2 x 2 table has Yates' correction: the right
    # test with its own numbers.
    yates = {
        "test": "Chi-square test of independence (Yates)",
        "statistic": 0.3475,
        "df": 1,
        "p_value": 0.5555,
        "n": 32,
    }
    score = run.score_run(
        BY_ID["engine-transmission-two-by-two"],
        "baseline",
        _transcript(),
        _write(tmp_path / "yates", yates),
        exit_code=0,
    )
    assert score.correct_test
    assert score.numbers_ok
    # A refused Bayes ANOVA, and a frequentist test silently put in its place.
    refused = {"test": None, "notes": "inferplot 0.1 has no Bayes factor for 3 groups."}
    score = run.score_run(
        BY_ID["bayes-factor-three-groups"],
        "inferplot",
        _transcript(),
        _write(tmp_path / "refused", refused),
        exit_code=0,
    )
    assert score.correct_test
    assert score.check_clean is None
    switched = {"test": "Welch's ANOVA", "statistic": 5.18, "p_value": 0.017}
    score = run.score_run(
        BY_ID["bayes-factor-three-groups"],
        "inferplot",
        _transcript(),
        _write(tmp_path / "switched", switched),
        exit_code=0,
    )
    assert score.correct_test is False
    assert score.test_reason == "rejected: Welch's ANOVA"
    # A correlation where the covariates call for a regression.
    corr = {"test": "Pearson correlation", "statistic": -0.868, "p_value": 1.29e-10}
    score = run.score_run(
        BY_ID["weight-fuel-with-covariates"],
        "inferplot",
        _transcript(),
        _write(tmp_path / "corr", corr),
        exit_code=0,
    )
    assert score.correct_test is False
    assert score.trap == "simple correlation ignoring the covariates"


def test_check_clean_needs_the_figure_and_the_expected_issues(tmp_path: Path) -> None:
    task = BY_ID["strains-unbalanced-small-group"]
    assert "small_n" in task.check_codes
    answer = {"test": "paired t-test", "figure": "f.svg", "check_issues": []}
    folder = _write(tmp_path, answer)
    clean, reason = run._check_clean(task, answer, folder)
    assert (clean, reason) == (False, "no saved figure")
    (folder / "f.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
    )
    clean, reason = run._check_clean(task, answer, folder)
    assert (clean, reason) == (False, "not reported: small_n")
    answer["check_issues"] = ["[warning] small_n: strain_1 has 4 plants"]
    clean, reason = run._check_clean(task, answer, folder)
    assert clean
    assert reason.startswith("self-reported")
    answer["check_issues"] = ["small_n", "[warning] label_overlap: two texts overlap"]
    clean, reason = run._check_clean(task, answer, folder)
    assert (clean, reason) == (False, "figure problems left: label_overlap")


def test_baseline_that_uses_inferplot_is_marked(tmp_path: Path) -> None:
    transcript = _transcript(commands=("pip install inferplot",))
    score = run.score_run(
        BY_ID["sleep-two-drugs"],
        "baseline",
        transcript,
        _write(tmp_path, SLEEP_ANSWER),
        exit_code=0,
    )
    assert score.contaminated
    assert score.tool_calls == 1


@pytest.mark.parametrize(
    ("family", "name", "expected"),
    [
        ("welch_t", "Welch Two Sample t-test", True),
        ("welch_t", "Welch's ANOVA", False),
        ("welch_anova", "One-way analysis of means (Welch)", True),
        ("student_t", "Two-sample t-test (equal_var=True)", True),
        ("student_t", "Welch t-test (not assuming equal variances)", False),
        ("paired_t", "scipy.stats.ttest_rel", True),
        ("paired_t", "Wilcoxon signed-rank test", False),
        ("mann_whitney", "Wilcoxon rank sum test", True),
        ("wilcoxon_signed_rank", "Wilcoxon rank sum test", False),
        ("rm_anova", "Repeated-measures ANOVA (Greenhouse-Geisser)", True),
        ("rm_anova", "statsmodels AnovaRM", True),
        ("oneway_anova", "Repeated-measures ANOVA", False),
        ("chi_square_independence", "McNemar's chi-squared test", False),
        ("chi_square_independence", "scipy chi2_contingency", True),
        ("pearson", "Pearson's chi-squared test", False),
        ("regression", "OLS multiple regression", True),
        ("regression", "film", False),
        ("no_test", None, True),
        ("no_test", "Not run", True),
        ("no_test", "paired t-test", False),
    ],
)
def test_family_matching(family: str, name: str | None, expected: bool) -> None:
    assert run.family_matches(family, name) is expected


def test_real_run_with_a_fake_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real mode end to end, with the model call and the environment build
    # replaced: each fake session writes an answer, as an agent would.
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(run, "_build_envs", lambda _options: None)
    monkeypatch.setattr(run.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[tuple[str, ...]] = []

    def fake_claude(
        command: Any, cwd: Path, env: Any, timeout: int
    ) -> tuple[int, str, bool]:
        calls.append(tuple(command))
        assert "envs" in env["PATH"].split(os.pathsep)[0]
        assert timeout == 300
        (cwd / "answer.json").write_text(json.dumps(SLEEP_ANSWER), encoding="utf-8")
        return 0, _transcript(), False

    monkeypatch.setattr(run, "_call_claude", fake_claude)
    lines: list[str] = []
    code = run.main(
        [
            run.SPEND_FLAG,
            "--tasks",
            "sleep-two-drugs",
            "--time-limit",
            "300",
            "--workdir",
            str(tmp_path),
        ],
        out=lines.append,
    )
    assert code == 0
    assert len(calls) == 2
    assert calls[0][:2] == ("claude", "-p")
    scores = json.loads(
        (tmp_path / "results" / "scores.json").read_text(encoding="utf-8")
    )
    assert [(s["arm"], s["correct_test"], s["numbers_ok"]) for s in scores] == [
        ("baseline", True, True),
        ("inferplot", True, True),
    ]
    assert (
        tmp_path / "results" / "baseline" / "sleep-two-drugs" / "transcript.jsonl"
    ).is_file()
    assert "baseline" in "\n".join(lines)
