"""The agent-evaluation task set and harness (evals/README.md, SPEC.md section 13).

Every task's dry-run validation runs here, so the expected numbers stay true as the
package changes. The scorer is tested on transcripts and answers shaped like real
ones: tool calls that carry paths, answers a careful SciPy analyst would write
(each number computed here from the task's data with SciPy and NumPy, not typed),
and the answers an inferplot-arm agent would copy from ``fig.result``. The harness
is shown to refuse model calls without the explicit flag. No test runs `claude` or
needs the network: the one function that calls the model is replaced.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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


def _dirs(tmp_path: Path) -> list[str]:
    return ["--workdir", str(tmp_path / "work"), "--results", str(tmp_path / "out")]


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
    # reproduced by inferplot or by its recipe within the task's tolerance, every
    # count recomputed, the expected check() codes (the dry run's check).
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
        "kendall",
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


def test_validation_catches_a_wrong_recipe_number_count_and_effect() -> None:
    task = BY_ID["headaches-before-after-paired"]
    exact = next(b for b in task.accepted if b.recipe == "mcnemar_exact")
    numbers = dict(exact.numbers)
    numbers["statistic"] = dataclasses.replace(numbers["statistic"], alt=(11.0,))
    effect = dataclasses.replace(exact.effects[0], estimate=run.Expected(0.5, (2.0,)))
    changed = dataclasses.replace(exact, numbers=numbers, effects=(effect,))
    broken = dataclasses.replace(
        task,
        accepted=tuple(changed if b is exact else b for b in task.accepted),
        counts={**task.counts, "responses": 81.0},
    )
    problems = run.validate_task(broken)
    assert any("statistic alternatives" in p for p in problems)
    assert any("effect odds_ratio" in p for p in problems)
    assert any("count responses is 80" in p for p in problems)


def test_recipes_agree_with_the_oracle_and_scipy() -> None:
    # The Greenhouse-Geisser recipe (whose epsilon also gives Huynh-Feldt's)
    # reproduces ggstatsplot's corrected df; Fisher's exact test for r x 2 tables
    # reproduces SciPy's on a 2 x 2 table.
    for task_id in ("bugs-four-situations", "noise-four-conditions-sphericity"):
        task = BY_ID[task_id]
        fixture = task.accepted[0]
        got = run.recipe_rm_anova(
            pd.read_csv(task.data_file),
            x=task.args[0],
            y=task.args[1],
            subject=task.kwargs["subject_id"],
            correction="gg",
        )
        for name in ("statistic", "df", "df_error", "p_value"):
            assert got[name] == pytest.approx(fixture.numbers[name].value, rel=1e-6)
    task = BY_ID["engine-transmission-two-by-two"]
    table = run._table(pd.read_csv(task.data_file), "transmission", "engine")
    assert run.fisher_exact_p(table) == pytest.approx(
        stats.fisher_exact(table).pvalue, rel=1e-10
    )


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
        "unknown-count": source.replace("pairs = 10", "patients = 10"),
        "n-also-not-a-count": source.replace(
            'n_also = ["observations"]', 'n_also = ["rows"]'
        ),
        "manual-without-read-for": source.replace(
            'trap_scoring = "traps"', 'trap_scoring = "manual"'
        ),
    }
    for name, text in cases.items():
        assert text != source, name
        folder = tmp_path / name
        folder.mkdir()
        (folder / "task.toml").write_text(
            text.replace("sleep-two-drugs", name), encoding="utf-8"
        )
        with pytest.raises(run.TaskError):
            run.load_task(folder / "task.toml")


def test_one_factor_tasks_have_one_factor() -> None:
    # E5: the supplement tasks are one-factor designs: no dose column, no word of
    # the doses in the request; the dose task keeps it.
    for task_id in (
        "supplement-no-significance-hunting",
        "supplement-registered-direction",
    ):
        task = BY_ID[task_id]
        assert list(pd.read_csv(task.data_file).columns) == [
            "tooth_length",
            "supplement",
        ]
        assert "dose" not in task.request.lower()
    dose = BY_ID["dose-by-supplement-interaction"]
    assert "dose_mg" in pd.read_csv(dose.data_file).columns


def test_every_trap_is_scored_or_listed_for_reading() -> None:
    for task in TASKS:
        assert task.trap_scoring in run.TRAP_SCORING
        if task.trap_scoring == "manual":
            assert task.read_for, task.id
    manual = {t.id for t in TASKS if t.trap_scoring == "manual"}
    assert manual == {
        "caffeine-pilot-five-people",
        "recipe-tasters-five",
        "strains-unbalanced-small-group",
    }


# --- the dry run and the refusal to spend usage ------------------------------------


def test_dry_run_makes_no_model_call(no_model: None, tmp_path: Path) -> None:
    lines: list[str] = []
    ids = "sleep-two-drugs,bayes-factor-three-groups"
    code = run.main(["--tasks", ids, *_dirs(tmp_path)], out=lines.append)
    assert code == 0
    text = "\n".join(lines)
    assert "no model call was made. 2 of 2 tasks valid." in text
    assert "4 claude sessions (2 tasks x 2 arms x 1 repeat)" in text
    # The arms: data only in the baseline, data and the skill in the inferplot arm,
    # and nothing of the task file (the expected numbers) in either. Where each arm
    # is, is written to the results root only.
    work, out = tmp_path / "work", tmp_path / "out"
    runs = json.loads((out / "runs.json").read_text(encoding="utf-8"))
    places = {(v["arm"], v["task"]): work / k for k, v in runs["runs"].items()}
    baseline = places[("baseline", "sleep-two-drugs")]
    with_skill = places[("inferplot", "sleep-two-drugs")]
    assert sorted(p.name for p in baseline.rglob("*")) == ["data.csv"]
    assert (with_skill / ".claude" / "skills" / "inferplot" / "SKILL.md").is_file()
    assert not list(work.rglob("task.toml"))
    prompt = (out / "prompts" / "sleep-two-drugs.txt").read_text(encoding="utf-8")
    assert "answer.json" in prompt
    assert "paired" not in prompt.lower()
    # The exact commands are printed, the environments first.
    assert "uv venv" in text
    assert text.count("--output-format stream-json") == 4


def test_agent_visible_paths_are_neutral(no_model: None, tmp_path: Path) -> None:
    # E1, E6: nothing an agent can see names the package, the arm or the task,
    # except the skill inside the inferplot arm's own run directories; the results
    # root is not under the agents' root.
    run.main(
        ["--tasks", "sleep-two-drugs,diets-three-groups", *_dirs(tmp_path)],
        out=lambda _: None,
    )
    work = tmp_path / "work"
    for path in work.rglob("*"):
        relative = path.relative_to(work).as_posix()
        if ".claude/skills/inferplot" in relative:
            continue
        lowered = relative.lower()
        for word in ("inferplot", "baseline", "sleep", "diets", "eval"):
            assert word not in lowered, relative
    assert sorted(p.name for p in work.iterdir()) == ["arm-a", "arm-b"]
    assert not (tmp_path / "out").resolve().is_relative_to(work.resolve())


def test_default_roots_are_neutral_and_apart(
    no_model: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    made: list[str] = []

    def mkdtemp(prefix: str = "") -> str:
        made.append(prefix)
        folder = tmp_path / f"{prefix}{len(made)}"
        folder.mkdir()
        return str(folder)

    monkeypatch.setattr(run.tempfile, "mkdtemp", mkdtemp)
    assert run.main(["--tasks", "sleep-two-drugs"], out=lambda _: None) == 0
    assert made == ["agent-eval-", "eval-results-"]
    assert not any("inferplot" in p for p in made)


def test_results_inside_the_workdir_are_refused(no_model: None, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run.main(
            ["--workdir", str(tmp_path), "--results", str(tmp_path / "results")],
            out=lambda _: None,
        )


def test_inferplot_is_installed_from_a_wheel_in_the_arm(tmp_path: Path) -> None:
    # E6: the environment records the wheel's path in the inferplot arm, never the
    # repository; the wheel is removed once installed.
    options = run.Options(workdir=tmp_path / "w", results=tmp_path / "r")
    commands = run.env_commands(options)
    install = commands["inferplot"][-1]
    wheel = run.wheel_path(options)
    assert wheel.is_relative_to(run.arm_root(options, "inferplot"))
    assert install[-1] == f"inferplot[pandas] @ {wheel.as_uri()}"
    assert not any(ROOT.as_uri() in part for part in install)
    assert commands["inferplot"][0][:2] == ["uv", "build"]
    baseline = " ".join(" ".join(c) for c in commands["baseline"])
    assert "inferplot" not in baseline.replace(str(tmp_path), "")


def test_agents_environment_names_neither_the_repository_nor_the_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    options = run.Options(workdir=tmp_path / "w", results=tmp_path / "r")
    venv_bin = str(ROOT / ".venv" / "bin")
    monkeypatch.setenv("PATH", os.pathsep.join([venv_bin, "/usr/bin", "/bin"]))
    monkeypatch.setenv("VIRTUAL_ENV", str(ROOT / ".venv"))
    monkeypatch.setenv("SOME_TOOL_HOME", str(ROOT / "tools"))
    monkeypatch.setenv("RESULTS_HINT", str(tmp_path / "r" / "scores.json"))
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    cwd = run.arm_root(options, "baseline") / "run-01"
    env = run._run_env(options, "baseline", cwd)
    text = json.dumps(env)
    assert str(ROOT) not in text
    assert str(tmp_path / "r") not in text
    assert "PYTHONPATH" not in env
    assert env["PATH"].split(os.pathsep)[0] == str(
        run.env_dir(options, "baseline") / "bin"
    )
    assert env["VIRTUAL_ENV"] == str(run.env_dir(options, "baseline"))
    assert env["PWD"] == str(cwd)


def test_real_mode_needs_the_flag(no_model: None, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run.main(["--real", *_dirs(tmp_path)], out=lambda _: None)
    with pytest.raises(SystemExit):
        run.main(["--dry-run", run.SPEND_FLAG, *_dirs(tmp_path)], out=lambda _: None)
    # Without any flag the harness only does the dry run.
    assert (
        run.main(["--tasks", "sleep-two-drugs", *_dirs(tmp_path)], out=lambda _: None)
        == 0
    )


def test_real_mode_is_refused_in_ci(
    no_model: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    with pytest.raises(SystemExit, match="CI"):
        run.main([run.SPEND_FLAG, *_dirs(tmp_path)], out=lambda _: None)


def test_workdir_inside_the_repository_is_refused(no_model: None) -> None:
    with pytest.raises(SystemExit):
        run.main(["--workdir", str(ROOT / "evals" / "work")], out=lambda _: None)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers; it is dead for our purpose once reaped by init.
    try:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return True
    return bool(state) and not state.startswith("Z")


def test_a_session_leaves_no_process_behind(tmp_path: Path) -> None:
    # E10: the time limit and the end of a session end the session's process
    # group, including what its shell started in the background. (No model: the
    # "session" is a shell.)
    script = 'sleep 60 & echo $! > child.pid; echo "{}"; '
    if os.name == "nt":
        with pytest.raises(SystemExit, match="POSIX"):
            run._call_claude(["cmd", "/c", "echo"], tmp_path, dict(os.environ), 5)
        return
    started = time.monotonic()
    code, out, timed_out = run._call_claude(
        ["sh", "-c", script + "exit 0"], tmp_path, dict(os.environ), 30
    )
    assert (code, out.strip(), timed_out) == (0, "{}", False)
    assert time.monotonic() - started < 20  # not held by the background child
    assert not _alive(int((tmp_path / "child.pid").read_text()))
    code, out, timed_out = run._call_claude(
        ["sh", "-c", script + "sleep 60"], tmp_path, dict(os.environ), 1
    )
    assert (code, timed_out) == (None, True)
    assert out.strip() == "{}"
    assert not _alive(int((tmp_path / "child.pid").read_text()))


# --- transcripts shaped like real ones ----------------------------------------------


def _event(kind: str, **body: Any) -> dict[str, Any]:
    return {"type": kind, **body}


def _tool(name: str, **tool_input: Any) -> dict[str, Any]:
    return _event(
        "assistant",
        message={"content": [{"type": "tool_use", "name": name, "input": tool_input}]},
    )


def _transcript(
    *, turns: int = 6, is_error: bool = False, calls: tuple[dict[str, Any], ...] = ()
) -> str:
    events: list[dict[str, Any]] = [{"type": "system", "subtype": "init"}, *calls]
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


# The old default layout put the package's name in every path an agent saw.
OLD_RUN = "/private/var/folders/tn/T/inferplot-evals-k2j9/baseline/sleep-two-drugs"
SCIPY_SCRIPT = (
    "import pandas as pd\nfrom scipy import stats\n"
    "d = pd.read_csv('data.csv')\nprint(stats.ttest_rel(d.a, d.b))\n"
)
PATH_ONLY_CALLS = (
    _event("system", subtype="init", cwd=OLD_RUN, tools=["Bash", "Read"]),
    _tool("Bash", command=f"cd {OLD_RUN} && ls -la && head data.csv"),
    _tool("Read", file_path=f"{OLD_RUN}/data.csv"),
    _tool("Glob", pattern="**/*.csv", path=OLD_RUN),
    _tool("Grep", pattern="patient", path=f"{OLD_RUN}/data.csv"),
    _tool("Write", file_path=f"{OLD_RUN}/analysis.py", content=SCIPY_SCRIPT),
    _tool("Edit", file_path=f"{OLD_RUN}/analysis.py", old_string="a", new_string="b"),
    _tool("Bash", command=f"python {OLD_RUN}/analysis.py > {OLD_RUN}/out.txt"),
    _tool("Bash", command="pip list | grep -i scipy; ls /tmp/inferplot-evals-k2j9"),
    _event(
        "user",
        message={
            "content": [{"type": "tool_result", "content": f"{OLD_RUN}/data.csv"}]
        },
    ),
)
USE_CALLS: dict[str, dict[str, Any]] = {
    "python -c import": _tool("Bash", command='python -c "import inferplot as ip"'),
    "heredoc from-import": _tool(
        "Bash", command="python - <<'EOF'\nfrom inferplot import betweenstats\nEOF"
    ),
    "written script": _tool(
        "Write", file_path="analysis.py", content="import numpy\nimport inferplot\n"
    ),
    "edited script": _tool(
        "Edit",
        file_path="a.py",
        old_string="import scipy",
        new_string="import scipy\nimport inferplot as ip",
    ),
    "pip install": _tool("Bash", command="pip install inferplot"),
    "uv pip install extra": _tool("Bash", command="uv pip install 'inferplot[pandas]'"),
    "wheel from the other arm": _tool(
        "Bash",
        command="pip install ../../arm-b/wheels/inferplot-0.1.0-py3-none-any.whl",
    ),
    "from github": _tool(
        "Bash", command="pip install git+https://github.com/sinkomr/inferplot"
    ),
    "importlib": _tool(
        "Write", file_path="x.py", content="m = importlib.import_module('inferplot')"
    ),
    "module run": _tool("Bash", command="python -m inferplot --help"),
}


def _write(path: Path, answer: dict[str, Any] | None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if answer is not None:
        (path / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    return path


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


@pytest.mark.parametrize("arm", ["baseline", "inferplot"])
def test_paths_that_contain_the_word_are_not_use(arm: str, tmp_path: Path) -> None:
    folder = _write(tmp_path / "inferplot-evals-x" / "baseline", SLEEP_ANSWER)
    (folder / "analysis.py").write_text(SCIPY_SCRIPT, encoding="utf-8")
    score = run.score_run(
        BY_ID["sleep-two-drugs"],
        arm,
        _transcript(calls=PATH_ONLY_CALLS),
        folder,
        exit_code=0,
    )
    assert score.tool_calls == 8
    assert score.inferplot_uses == []
    assert not score.used_inferplot
    assert not score.contaminated


@pytest.mark.parametrize("label", list(USE_CALLS))
def test_real_use_is_detected_in_both_arms(label: str, tmp_path: Path) -> None:
    calls = (*PATH_ONLY_CALLS, USE_CALLS[label])
    for arm in ("baseline", "inferplot"):
        score = run.score_run(
            BY_ID["sleep-two-drugs"],
            arm,
            _transcript(calls=calls),
            _write(tmp_path / arm, SLEEP_ANSWER),
            exit_code=0,
        )
        assert score.used_inferplot, (label, arm)
        assert score.contaminated is (arm == "baseline")


def test_use_in_a_script_left_in_the_run_directory_is_detected(tmp_path: Path) -> None:
    folder = _write(tmp_path, SLEEP_ANSWER)
    (folder / "make_figure.py").write_text(
        "import pandas as pd\nimport inferplot as ip\n", encoding="utf-8"
    )
    score = run.score_run(
        BY_ID["sleep-two-drugs"], "baseline", _transcript(), folder, exit_code=0
    )
    assert score.contaminated
    assert score.inferplot_uses == ["import inferplot"]


# --- the scorer -------------------------------------------------------------------


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
    assert score.effect_matched == "the oracle's effect size"
    assert score.trap is None
    assert score.trap_avoided is True
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
    assert score.trap_avoided is False
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
    assert score.trap_avoided is None
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
        "statistic": "F = 5.18",
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
    assert score.numbers["p_value"]["bound"] == "<"
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
    assert score.trap_avoided is False


@pytest.mark.parametrize(
    ("text", "value", "bound"),
    [
        ("2.83 \u00d7 10^-3", 0.00283, None),
        ("2.83 x 10^-3", 0.00283, None),
        ("2.83\u00d710\u207b\u00b3", 0.00283, None),
        ("2.83 * 10**(-3)", 0.00283, None),
        ("2.8e-03", 0.0028, None),
        ("p = .0028", 0.0028, None),
        ("p<.01", 0.01, "<"),
        ("p \u2264 0.001", 0.001, "<"),
        ("< 2.2e-16", 2.2e-16, "<"),
        ("p > .05", 0.05, ">"),
        ("t(9) = \u22124.06", -4.06, None),
        ("n = 10 pairs (20 observations)", 10.0, None),
        ("2,201", 2201.0, None),
        ("1,234,567", 1234567.0, None),
        ([0.0028], 0.0028, None),
        ("not computed", None, None),
        (math.nan, None, None),
    ],
)
def test_parse_number(text: Any, value: float | None, bound: str | None) -> None:
    got, got_bound = run.parse_number(text)
    if value is None:
        assert got is None
    else:
        assert got == pytest.approx(value, rel=1e-12)
    assert got_bound == bound


def test_p_written_as_a_bound(tmp_path: Path) -> None:
    # E10: "< b" is right when the true p is below b and b is .001 or below (or
    # within the p tolerance); "p<.01" for p = .0028 is not as precise as
    # conventional reporting and is marked wrong, with the bound recorded.
    task = BY_ID["sleep-two-drugs"]
    cases = {"p<.01": False, "< .003": True, "< .002": False}
    for i, (p, ok) in enumerate(cases.items()):
        score = run.score_run(
            task,
            "baseline",
            _transcript(),
            _write(tmp_path / str(i), SLEEP_ANSWER | {"p_value": p}),
            exit_code=0,
        )
        assert score.numbers["p_value"]["ok"] is ok, p
        assert score.numbers["p_value"]["bound"] == "<"
    titanic = {
        "test": "Pearson's chi-squared test",
        "statistic": 190.4,
        "df": 3,
        "p_value": "< 2.2e-16",
        "n": 2201,
    }
    score = run.score_run(
        BY_ID["titanic-table-of-counts"],
        "baseline",
        _transcript(),
        _write(tmp_path / "titanic", titanic),
        exit_code=0,
    )
    assert score.numbers_ok


def test_exact_mcnemar_is_scored_on_its_own_numbers(tmp_path: Path) -> None:
    # E2: statsmodels' default McNemar (exact; statistic = the smaller discordant
    # count) on the headache data: right as a whole for the exact variant, although
    # its statistic equals R's chi-square.
    answer = {
        "test": "McNemar's test (exact)",
        "statistic": 4.0,
        "p_value": 0.0768,
        "n": 40,
    }
    score = run.score_run(
        BY_ID["headaches-before-after-paired"],
        "baseline",
        _transcript(),
        _write(tmp_path, answer),
        exit_code=0,
    )
    assert score.correct_test
    assert score.numbers_ok, score.numbers
    assert score.variant == "exact McNemar (binomial) test"
    assert score.trap is None


def test_qualified_variant_needs_its_word_or_its_numbers(tmp_path: Path) -> None:
    task = BY_ID["reaction-times-real-outliers"]
    groups = [
        np.log(g["reaction_time_s"].to_numpy(dtype=float))
        for _, g in pd.read_csv(task.data_file).groupby("condition")
    ]
    welch = _welch_anova(groups)
    by_numbers = {"test": "Welch's ANOVA", **welch, "n": 30}
    by_name = by_numbers | {"test": "Welch ANOVA of log-transformed times"}
    raw = {"test": "Welch's ANOVA", "statistic": 3.54, "p_value": 0.054, "n": 30}
    for name, answer, correct in (
        ("numbers", by_numbers, True),
        ("name", by_name, True),
        ("raw", raw, False),
    ):
        score = run.score_run(
            task,
            "baseline",
            _transcript(),
            _write(tmp_path / name, answer),
            exit_code=0,
        )
        assert score.correct_test is correct, name
    assert score.test_reason == "rejected: Welch's ANOVA"
    assert score.trap == "Welch's ANOVA driven by the two extreme values"


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
    assert score.trap_avoided is True
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
    assert score.trap_avoided is False
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


def test_manual_traps_are_not_counted_as_avoided(tmp_path: Path) -> None:
    task = BY_ID["recipe-tasters-five"]
    x = pd.read_csv(task.data_file)["score"].dropna().to_numpy(dtype=float)
    res = stats.ttest_1samp(x, 3)
    answer = {
        "test": "One-sample t-test",
        "statistic": float(res.statistic),
        "df": 4,
        "p_value": float(res.pvalue),
        "n": 5,
    }
    score = run.score_run(
        task, "baseline", _transcript(), _write(tmp_path, answer), exit_code=0
    )
    assert score.numbers_ok
    assert score.trap_avoided is None
    assert score.read_for == task.read_for
    summary = run.summarize([score])
    assert "trap avoided" in summary
    assert "0/0" in summary.splitlines()[1]
    assert f"recipe-tasters-five: {task.read_for}" in summary


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
    seen: list[list[str]] = []

    def fake_claude(
        command: Any, cwd: Path, env: Any, timeout: int
    ) -> tuple[int, str, bool]:
        calls.append(tuple(command))
        assert env["PATH"].split(os.pathsep)[0].endswith(os.path.join("env", "bin"))
        assert str(ROOT) not in json.dumps(env)
        assert timeout == 300
        # No earlier run's directory is left next to this one.
        seen.append(sorted(p.name for p in cwd.parent.iterdir()))
        assert sorted(p.name for p in cwd.parent.parent.iterdir()) == ["arm-a", "arm-b"]
        (cwd / "answer.json").write_text(json.dumps(SLEEP_ANSWER), encoding="utf-8")
        return 0, _transcript(), False

    monkeypatch.setattr(run, "_call_claude", fake_claude)
    lines: list[str] = []
    code = run.main(
        [
            run.SPEND_FLAG,
            "--tasks",
            "sleep-two-drugs,diets-three-groups",
            "--time-limit",
            "300",
            *_dirs(tmp_path),
        ],
        out=lines.append,
    )
    assert code == 0
    assert len(calls) == 4
    assert calls[0][:2] == ("claude", "-p")
    assert seen[0] == ["run-01", "run-02"]  # both of this arm's runs, before either
    assert seen[-1] == ["run-02"]  # the finished runs have moved to the results
    out = tmp_path / "out"
    scores = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    assert [(s["arm"], s["task"], s["correct_test"]) for s in scores][:2] == [
        ("baseline", "diets-three-groups", False),
        ("inferplot", "diets-three-groups", False),
    ]
    assert [(s["arm"], s["numbers_ok"]) for s in scores][2:] == [
        ("baseline", True),
        ("inferplot", True),
    ]
    assert (out / "baseline" / "sleep-two-drugs" / "transcript.jsonl").is_file()
    assert (out / "baseline" / "sleep-two-drugs" / "answer.json").is_file()
    assert not list((tmp_path / "work").rglob("answer.json"))
    assert "baseline" in "\n".join(lines)


# --- the test-name matcher ------------------------------------------------------------

SPELLINGS: list[tuple[str | None, set[str]]] = [
    # Welch's t-test
    ("Welch Two Sample t-test", {"welch_t"}),
    ("Welch's t-test", {"welch_t"}),
    ("welch_t", {"welch_t"}),
    ("t-test (unequal variances)", {"welch_t"}),
    ("Two-sample t-test, equal_var=False", {"welch_t"}),
    ("Independent t-test (not assuming equal variances)", {"welch_t"}),
    ("Welch\u2013Satterthwaite t-test", {"welch_t"}),
    ("WELCH T TEST", {"welch_t"}),
    # Student's t-test
    ("Student's t-test", {"student_t"}),
    ("Two Sample t-test (var.equal = TRUE)", {"student_t"}),
    ("independent samples t-test, equal variances assumed", {"student_t"}),
    ("pooled-variance t-test", {"student_t"}),
    ("Student t\u2011test, one-sided", {"student_t"}),
    # ANOVAs
    ("Welch's ANOVA", {"welch_anova"}),
    ("One-way analysis of means (not assuming equal variances)", {"welch_anova"}),
    ("welch_anova", {"welch_anova"}),
    ("Welch F test", {"welch_anova"}),
    ("heteroscedastic one-way ANOVA (Welch)", {"welch_anova"}),
    ("One-way ANOVA", {"oneway_anova"}),
    ("scipy.stats.f_oneway", {"oneway_anova"}),
    ("Analysis of variance (F-test)", {"oneway_anova"}),
    ("one-way analysis of means", {"oneway_anova"}),
    ("Kruskal\u2013Wallis H-test", {"kruskal_wallis"}),
    ("kruskal_wallis", {"kruskal_wallis"}),
    ("\u03c7\u00b2 Kruskal\u2013Wallis", {"kruskal_wallis"}),
    # two independent groups, ranks
    ("Mann\u2013Whitney U", {"mann_whitney"}),
    ("Wilcoxon rank-sum test", {"mann_whitney"}),
    ("mannwhitneyu", {"mann_whitney"}),
    ("Wilcoxon rank sum test with continuity correction", {"mann_whitney"}),
    # paired and one-sample
    ("Paired t\u2013test", {"paired_t"}),
    ("paired_t", {"paired_t"}),
    ("Paired samples t-test", {"paired_t"}),
    ("scipy.stats.ttest_rel", {"paired_t"}),
    ("dependent t-test", {"paired_t"}),
    ("repeated-measures t-test", {"paired_t"}),
    ("matched-pairs t test", {"paired_t"}),
    ("Wilcoxon signed-rank test", {"wilcoxon_signed_rank"}),
    ("wilcoxon_signed_rank", {"wilcoxon_signed_rank"}),
    ("Wilcoxon signed rank test with continuity correction", {"wilcoxon_signed_rank"}),
    ("Wilcoxon test (paired)", {"wilcoxon_signed_rank"}),
    ("Sign test", {"sign_test"}),
    ("exact sign test (binomial)", {"sign_test"}),
    ("One Sample t-test", {"one_sample_t"}),
    ("one_sample_t", {"one_sample_t"}),
    ("scipy.stats.ttest_1samp", {"one_sample_t"}),
    ("one-sample Student's t test against \u03bc = 20", {"one_sample_t"}),
    ("Bootstrap-t test of the 20% trimmed mean", {"trimmed_bootstrap"}),
    ("Bootstrap-t method for one-sample test", {"trimmed_bootstrap"}),
    # repeated measures
    ("rmANOVA (GG)", {"rm_anova"}),
    ("RM-ANOVA", {"rm_anova"}),
    ("Repeated-measures ANOVA (Greenhouse-Geisser)", {"rm_anova"}),
    ("Greenhouse\u2013Geisser corrected ANOVA", {"rm_anova"}),
    ("statsmodels AnovaRM", {"rm_anova"}),
    ("within-subjects ANOVA", {"rm_anova"}),
    ("Repeated measures ANOVA, Huynh-Feldt correction", {"rm_anova"}),
    ("ANOVA estimation for factorial designs using 'afex'", {"rm_anova"}),
    ("Friedman rank sum test", {"friedman"}),
    ("Friedman \u03c7\u00b2", {"friedman"}),
    # correlations
    ("Pearson's r", {"pearson"}),
    ("Pearson product-moment correlation", {"pearson"}),
    ("Pearson correlation (Fisher z CI)", {"pearson"}),
    ("Spearman's \u03c1", {"spearman"}),
    ("Spearman rank-order correlation", {"spearman"}),
    ("Kendall's \u03c4-b", {"kendall"}),
    ("Kendall rank correlation (tau)", {"kendall"}),
    ("Winsorized Pearson correlation (20% winsorized)", {"winsorized"}),
    # robust between
    ("Yuen's t-test (20% trimmed means)", {"yuen"}),
    ("Trimmed-means ANOVA (20% trimmed means)", {"trimmed_anova"}),
    ("WRS2 t1way", {"trimmed_anova"}),
    ("A heteroscedastic one-way ANOVA for trimmed means", {"trimmed_anova"}),
    # counts
    ("McNemar's Chi-squared test", {"mcnemar"}),
    ("Exact binomial test (McNemar)", {"mcnemar"}),
    ("Pearson's chi-square test of independence", {"chi_square_independence"}),
    ("chi2_contingency", {"chi_square_independence"}),
    ("\u03c7\u00b2 test of independence (Yates)", {"chi_square_independence"}),
    ("Chi-squared test", {"chi_square_independence", "chi_square_gof"}),
    ("\u03c7\u00b2 test", {"chi_square_independence", "chi_square_gof"}),
    ("Chi-squared test for given probabilities", {"chi_square_gof"}),
    ("scipy.stats.chisquare", {"chi_square_gof"}),
    ("goodness of fit test", {"chi_square_gof"}),
    ("Fisher's exact test", {"fisher"}),
    ("Fisher exact (Freeman\u2013Halton)", {"fisher"}),
    # models
    ("OLS multiple regression", {"regression"}),
    ("linear model (lm)", {"regression"}),
    ("ANCOVA", {"regression"}),
    ("Two-way ANOVA", {"two_way_anova"}),
    ("factorial ANOVA (2 x 3)", {"two_way_anova"}),
    ("linear model with interaction term", {"regression", "two_way_anova"}),
    ("Bayes factor (JZS)", {"bayes_factor"}),
    ("BF10 from a Bayesian ANOVA", {"bayes_factor"}),
    # no test
    (None, {"no_test"}),
    ("Not run", {"no_test"}),
    ("N/A", {"no_test"}),
    # wrong or not specific enough: no family
    ("t-test", set()),
    ("Independent samples t-test", set()),
    ("scipy.stats.ttest_ind", set()),
    ("Kendall's W", set()),
    ("Levene's test", set()),
    ("Shapiro\u2013Wilk test", set()),
    ("Brown-Forsythe", set()),
    ("Mixed ANOVA", set()),
    ("Cohen's d", set()),
    ("film", set()),
]


@pytest.mark.parametrize(
    ("name", "families"), SPELLINGS, ids=[str(s[0]) for s in SPELLINGS]
)
def test_test_names_are_read_robustly(name: str | None, families: set[str]) -> None:
    assert run.families_of(name) == families


def test_the_spelling_table_covers_every_family() -> None:
    assert len(SPELLINGS) >= 60
    named = set().union(*(f for _, f in SPELLINGS))
    assert named == {*run.FAMILIES, run.NO_TEST}
    assert sum(1 for _, f in SPELLINGS if not f) >= 8


def _inferplot_blocks() -> list[tuple[Any, Any]]:
    return [
        (task, block)
        for task in TASKS
        for block in task.accepted
        if block.kwargs is not None
    ]


@pytest.mark.parametrize(
    ("task", "block"),
    _inferplot_blocks(),
    ids=[f"{t.id}-{b.family}" for t, b in _inferplot_blocks()],
)
def test_inferplot_names_are_recognized(task: Any, block: Any) -> None:
    # E4: an inferplot-arm agent may copy result.test.name, .label or .method.
    result = run._analyze(task, pd.read_csv(task.data_file), block.kwargs)
    for name in (result.test.name, result.test.label, result.test.method):
        assert run.family_matches(block.family, name), name


# --- acceptance: answers a careful SciPy analyst would give -------------------------
# Each number below is computed here from the task's data with SciPy and NumPy
# (SciPy's defaults, or statsmodels' defaults recomputed), rounded as a person
# reports it, and must score as the correct test with the correct numbers.


def _data(task_id: str) -> pd.DataFrame:
    return pd.read_csv(BY_ID[task_id].data_file)


def _g(value: float, digits: int = 3) -> float:
    return float(f"{value:.{digits}g}")


def _r(value: float, places: int = 2) -> float:
    return round(float(value), places)


def _p(value: float) -> Any:
    return "p < .001" if value < 0.001 else _g(value)


def _groups(df: pd.DataFrame, x: str, y: str) -> list[np.ndarray]:
    clean = df[[x, y]].dropna()
    return [s.to_numpy(dtype=float) for _, s in clean.groupby(x)[y]]


def _welch_anova(groups: list[np.ndarray]) -> dict[str, Any]:
    # statsmodels' anova_oneway(use_var="unequal"), written out (Welch, 1951).
    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    weights = n / np.array([np.var(g, ddof=1) for g in groups])
    means = np.array([np.mean(g) for g in groups])
    grand = np.sum(weights * means) / np.sum(weights)
    lam = np.sum((1 - weights / np.sum(weights)) ** 2 / (n - 1))
    f = np.sum(weights * (means - grand) ** 2) / (k - 1)
    f /= 1 + 2 * (k - 2) / (k**2 - 1) * lam
    df2 = (k**2 - 1) / (3 * lam)
    return {
        "statistic": _r(f),
        "df": k - 1,
        "df_error": _r(df2),
        "p_value": _p(stats.f.sf(f, k - 1, df2)),
    }


def _paired(
    task_id: str, x: str, y: str, subject: str
) -> tuple[np.ndarray, np.ndarray]:
    wide = _data(task_id).pivot_table(index=subject, columns=x, values=y).dropna()
    return wide["before" if "before" in wide else wide.columns[0]].to_numpy(
        dtype=float
    ), wide["after" if "after" in wide else wide.columns[1]].to_numpy(dtype=float)


def _gg_rm_anova(wide: np.ndarray) -> dict[str, Any]:
    # The Greenhouse-Geisser epsilon from the eigenvalues of the covariance of
    # orthonormal contrasts (pingouin's and afex's construction).
    n, k = wide.shape
    contrasts = np.linalg.qr(np.vstack([np.ones(k), np.eye(k)[:-1]]).T)[0][:, 1:]
    eig = np.linalg.eigvalsh(np.cov(wide @ contrasts, rowvar=False))
    eps = eig.sum() ** 2 / ((k - 1) * (eig**2).sum())
    grand = wide.mean()
    ss_cond = n * ((wide.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((wide.mean(axis=1) - grand) ** 2).sum()
    ss_err = ((wide - grand) ** 2).sum() - ss_cond - ss_subj
    f = (ss_cond / (k - 1)) / (ss_err / ((k - 1) * (n - 1)))
    df1, df2 = eps * (k - 1), eps * (k - 1) * (n - 1)
    return {
        "statistic": _r(f),
        "df": _r(df1),
        "df_error": _r(df2),
        "p_value": _p(stats.f.sf(f, df1, df2)),
        "n": n,
    }


def _ols_t(df: pd.DataFrame, factor_cyl: bool) -> dict[str, Any]:
    columns = [np.ones(len(df)), df["wt"].to_numpy(float), df["hp"].to_numpy(float)]
    if factor_cyl:
        columns += [(df["cyl"] == level).to_numpy(float) for level in (6, 8)]
    else:
        columns.append(df["cyl"].to_numpy(float))
    design = np.column_stack(columns)
    target = df["mpg"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    dof = len(df) - design.shape[1]
    se = np.sqrt(resid @ resid / dof * np.linalg.inv(design.T @ design)[1, 1])
    t = coef[1] / se
    return {
        "statistic": _r(t),
        "df": dof,
        "p_value": _p(2 * stats.t.sf(abs(t), dof)),
        "estimate": _r(coef[1]),
        "n": len(df),
    }


def _a_transmission() -> list[dict[str, Any]]:
    df = _data("transmission-unequal-variances")
    a = df.loc[df.transmission == "automatic", "mpg"].to_numpy(float)
    b = df.loc[df.transmission == "manual", "mpg"].to_numpy(float)
    welch = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / 30)
    mw = stats.mannwhitneyu(a, b)
    return [
        {
            "test": "Welch Two Sample t-test",
            "statistic": _r(welch.statistic),
            "df": _r(welch.df),
            "p_value": _p(welch.pvalue),
            "effect_size": "Cohen's d",
            "estimate": _r((a.mean() - b.mean()) / pooled),
            "n": 32,
        },
        {
            "test": "Mann\u2013Whitney U test",
            "statistic": float(mw.statistic),
            "p_value": _p(mw.pvalue),
            "effect_size": "probability of superiority",
            "estimate": _r(mw.statistic / (len(a) * len(b))),
            "n": 32,
        },
    ]


def _a_diets() -> list[dict[str, Any]]:
    groups = _groups(_data("diets-three-groups"), "treatment", "dry_weight")
    f = stats.f_oneway(*groups)
    k = stats.kruskal(*groups)
    values = np.concatenate(groups)
    ss_b = sum(len(g) * (g.mean() - values.mean()) ** 2 for g in groups)
    eta2 = ss_b / ((values - values.mean()) ** 2).sum()
    return [
        {
            "test": "Welch's ANOVA (statsmodels anova_oneway, use_var='unequal')",
            **_welch_anova(groups),
            "n": 30,
        },
        {
            "test": "One-way ANOVA (scipy.stats.f_oneway)",
            "statistic": _r(f.statistic),
            "df": [2, 27],
            "p_value": _p(f.pvalue),
            "effect_size": "eta squared",
            "estimate": _r(eta2),
            "n": 30,
        },
        {
            "test": "Kruskal\u2013Wallis H-test",
            "statistic": _r(k.statistic),
            "df": 2,
            "p_value": _p(k.pvalue),
            "n": 30,
        },
    ]


def _a_registered() -> list[dict[str, Any]]:
    df = _data("supplement-registered-direction")
    oj = df.loc[df.supplement == "orange_juice", "tooth_length"].to_numpy(float)
    vc = df.loc[df.supplement == "ascorbic_acid", "tooth_length"].to_numpy(float)
    welch = stats.ttest_ind(oj, vc, equal_var=False, alternative="greater")
    student = stats.ttest_ind(oj, vc, alternative="greater")
    mw = stats.mannwhitneyu(oj, vc, alternative="greater")
    return [
        {
            "test": "Welch's t-test (one-sided, OJ > VC)",
            "statistic": _r(welch.statistic),
            "df": _r(welch.df),
            "p_value": _p(welch.pvalue),
            "n": 60,
        },
        {
            "test": "Student's t-test, one-sided (equal variances assumed)",
            "statistic": _r(student.statistic),
            "df": 58,
            "p_value": _p(student.pvalue),
            "n": 60,
        },
        {
            "test": "Mann-Whitney U (one-sided)",
            "statistic": float(mw.statistic),
            "p_value": _p(mw.pvalue),
            "n": 60,
        },
    ]


def _a_reaction_and_yield() -> list[tuple[str, dict[str, Any]]]:
    rt = _groups(_data("reaction-times-real-outliers"), "condition", "reaction_time_s")
    k = stats.kruskal(*rt)
    yields = _groups(_data("yield-missing-values"), "site", "yield_t_ha")
    f = stats.f_oneway(*yields)
    return [
        (
            "reaction-times-real-outliers",
            {
                "test": "Kruskal-Wallis rank sum test",
                "statistic": _r(k.statistic),
                "df": 2,
                "p_value": _p(k.pvalue),
                "n": 30,
            },
        ),
        (
            "yield-missing-values",
            {
                "test": "one-way ANOVA",
                "statistic": _r(f.statistic),
                "df": "2, 23",
                "p_value": _p(f.pvalue),
                "n": 26,
            },
        ),
    ]


def _a_paired() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    sleep = _data("sleep-two-drugs").pivot_table(
        index="patient", columns="drug", values="extra_sleep_h"
    )
    a, b = sleep["drug_A"].to_numpy(float), sleep["drug_B"].to_numpy(float)
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    d = a - b
    sign = stats.binomtest(int((d[d != 0] > 0).sum()), int((d != 0).sum()))
    out += [
        (
            "sleep-two-drugs",
            {
                "test": "Paired t\u2013test",
                "statistic": _r(t.statistic),
                "df": 9,
                "p_value": _p(t.pvalue),
                "effect_size": "Cohen's d_z",
                "estimate": _r(d.mean() / d.std(ddof=1)),
                "n": "20 observations (10 pairs)",
            },
        ),
        (
            "sleep-two-drugs",
            {
                "test": "Wilcoxon signed-rank test",
                "statistic": float(w.statistic),
                "p_value": _p(w.pvalue),
                "n": int((d != 0).sum()),
            },
        ),
        (
            "sleep-two-drugs",
            {
                "test": "Sign test",
                "statistic": int((d > 0).sum()),
                "p_value": _p(sign.pvalue),
                "n": 10,
            },
        ),
    ]
    for task_id, x, y, subject in (
        ("grip-strength-missed-visits", "visit", "grip_kg", "patient"),
        ("training-before-after-shuffled", "time", "score", "student"),
        ("anxiety-ratings-before-after", "session", "anxiety", "child"),
    ):
        before, after = _paired(task_id, x, y, subject)
        w = stats.wilcoxon(before, after)
        out.append(
            (
                task_id,
                {
                    "test": "Wilcoxon signed rank (scipy.stats.wilcoxon)",
                    "statistic": float(w.statistic),
                    "p_value": _p(w.pvalue),
                    "n": len(before),
                },
            )
        )
        if task_id != "anxiety-ratings-before-after":
            t = stats.ttest_rel(before, after)
            out.append(
                (
                    task_id,
                    {
                        "test": "paired samples t-test (scipy.stats.ttest_rel)",
                        "statistic": _r(t.statistic),
                        "df": len(before) - 1,
                        "p_value": _p(t.pvalue),
                        "n": len(before),
                    },
                )
            )
    before, after = _paired(
        "anxiety-ratings-before-after", "session", "anxiety", "child"
    )
    d = before - after
    sign = stats.binomtest(int((d > 0).sum()), int((d != 0).sum()))
    out.append(
        (
            "anxiety-ratings-before-after",
            {
                "test": "Sign test (exact binomial)",
                "statistic": int((d > 0).sum()),
                "p_value": _p(sign.pvalue),
                "n": int((d != 0).sum()),
            },
        )
    )
    bugs = (
        _data("bugs-four-situations")
        .pivot_table(index="subject", columns="situation", values="desire")
        .dropna()
        .to_numpy(float)
    )
    fr = stats.friedmanchisquare(*bugs.T)
    out += [
        (
            "bugs-four-situations",
            {
                "test": "Repeated-measures ANOVA, Greenhouse\u2013Geisser corrected",
                **_gg_rm_anova(bugs),
            },
        ),
        (
            "bugs-four-situations",
            {
                "test": "Friedman \u03c7\u00b2 test",
                "statistic": _r(fr.statistic),
                "df": 3,
                "p_value": _p(fr.pvalue),
                "n": bugs.size,
            },
        ),
    ]
    return out


def _a_one_sample() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for task_id, column, mu in (
        ("fuel-economy-against-20", "mpg", 20.0),
        ("bag-weights-missing", "weight_kg", 10.0),
        ("sensor-calibration-outlier", "reading", 10.0),
    ):
        x = _data(task_id)[column].dropna().to_numpy(float)
        w = stats.wilcoxon(x - mu)
        out.append(
            (
                task_id,
                {
                    "test": f"Wilcoxon signed-rank test (one-sample, vs {mu:g})",
                    "statistic": float(w.statistic),
                    "p_value": _p(w.pvalue),
                    "n": len(x),
                },
            )
        )
        if task_id != "sensor-calibration-outlier":
            t = stats.ttest_1samp(x, mu)
            out.append(
                (
                    task_id,
                    {
                        "test": f"One-sample t-test against \u03bc = {mu:g}",
                        "statistic": _r(t.statistic),
                        "df": len(x) - 1,
                        "p_value": _p(t.pvalue),
                        "effect_size": "Cohen's d",
                        "estimate": _r((x.mean() - mu) / x.std(ddof=1)),
                        "n": len(x),
                    },
                )
            )
    return out


def _a_correlations() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for task_id, x, y in (
        ("weight-fuel-correlation", "wt", "mpg"),
        ("satisfaction-engagement-ordinal", "satisfaction", "engagement"),
        ("height-leaf-area-missing-pairs", "height_cm", "leaf_area_cm2"),
        ("reading-vocabulary-outlier", "reading_hours", "vocabulary"),
    ):
        df = _data(task_id)[[x, y]].dropna()
        rho = stats.spearmanr(df[x], df[y])
        tau = stats.kendalltau(df[x], df[y])
        out += [
            (
                task_id,
                {
                    "test": "Spearman's \u03c1",
                    "statistic": _r(rho.statistic, 3),
                    "p_value": _p(rho.pvalue),
                    "effect_size": "rho",
                    "estimate": _r(rho.statistic, 3),
                    "n": len(df),
                },
            ),
            (
                task_id,
                {
                    "test": "Kendall's \u03c4-b",
                    "statistic": _r(tau.statistic, 3),
                    "p_value": _p(tau.pvalue),
                    "estimate": _r(tau.statistic, 3),
                    "n": len(df),
                },
            ),
        ]
        if task_id in ("weight-fuel-correlation", "height-leaf-area-missing-pairs"):
            r = stats.pearsonr(df[x], df[y])
            ci = r.confidence_interval()
            out.append(
                (
                    task_id,
                    {
                        "test": "Pearson correlation",
                        "statistic": _r(r.statistic, 3),
                        "p_value": _p(r.pvalue),
                        "effect_size": "Pearson's r",
                        "estimate": _r(r.statistic, 3),
                        "conf_low": _r(ci.low, 3),
                        "conf_high": _r(ci.high, 3),
                        "n": len(df),
                    },
                )
            )
    cars = _data("weight-fuel-with-covariates")
    out += [
        (
            "weight-fuel-with-covariates",
            {
                "test": "OLS multiple regression (cyl as factor)",
                **_ols_t(cars, factor_cyl=True),
            },
        ),
        (
            "weight-fuel-with-covariates",
            {"test": "Multiple linear regression", **_ols_t(cars, factor_cyl=False)},
        ),
    ]
    return out


def _a_counts() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    cars = _data("engine-transmission-two-by-two")
    engine = pd.crosstab(cars["transmission"], cars["engine"])
    chi = stats.chi2_contingency(engine)
    fisher = stats.fisher_exact(engine)
    plain = stats.chi2_contingency(engine, correction=False)
    out += [
        (
            "engine-transmission-two-by-two",
            {
                "test": "Chi-squared test of independence (Yates' correction)",
                "statistic": _r(chi.statistic, 3),
                "df": 1,
                "p_value": _p(chi.pvalue),
                "n": 32,
            },
        ),
        (
            "engine-transmission-two-by-two",
            {
                "test": "Fisher's exact test",
                "statistic": _r(fisher.statistic, 3),
                "p_value": _p(fisher.pvalue),
                "effect_size": "odds ratio",
                "estimate": _r(fisher.statistic, 3),
                "n": 32,
            },
        ),
        (
            "engine-transmission-two-by-two",
            {
                "test": "Pearson's \u03c7\u00b2 test",
                "statistic": _r(plain.statistic, 3),
                "df": 1,
                "p_value": _p(plain.pvalue),
                "effect_size": "phi",
                "estimate": _r(np.sqrt(plain.statistic / 32), 3),
                "n": 32,
            },
        ),
    ]
    patients = _data("headaches-before-after-paired")
    paired = pd.crosstab(
        patients["headache_before"], patients["headache_after"]
    ).to_numpy()
    b, c = int(paired[0, 1]), int(paired[1, 0])
    corrected = (abs(b - c) - 1) ** 2 / (b + c)
    out += [
        (
            "headaches-before-after-paired",
            {
                "test": "McNemar's test (exact binomial)",
                "statistic": float(min(b, c)),
                "p_value": _p(stats.binomtest(min(b, c), b + c).pvalue),
                "n": 40,
            },
        ),
        (
            "headaches-before-after-paired",
            {
                "test": "McNemar's test with continuity correction",
                "statistic": _r(corrected, 3),
                "df": 1,
                "p_value": _p(stats.chi2.sf(corrected, 1)),
                "n": "80 responses",
            },
        ),
    ]
    titanic = _data("titanic-table-of-counts").pivot_table(
        index="Class", columns="Survived", values="Freq", aggfunc="sum"
    )
    t = stats.chi2_contingency(titanic)
    n = int(titanic.to_numpy().sum())
    out.append(
        (
            "titanic-table-of-counts",
            {
                "test": "Pearson's \u03c7\u00b2 test of independence",
                "statistic": _r(t.statistic),
                "df": int(t.dof),
                "p_value": "< 2.2e-16",
                "effect_size": "Cram\u00e9r's V",
                "estimate": _r(np.sqrt(t.statistic / n), 3),
                "n": f"{n:,}",
            },
        )
    )
    cyl = (
        _data("cylinders-planned-shares")["cyl"].value_counts().sort_index().to_numpy()
    )
    g = stats.chisquare(cyl, np.array([0.5, 0.25, 0.25]) * cyl.sum())
    out.append(
        (
            "cylinders-planned-shares",
            {
                "test": "Chi-square goodness-of-fit test (scipy.stats.chisquare)",
                "statistic": _r(g.statistic, 3),
                "df": 2,
                "p_value": _p(g.pvalue),
                "effect_size": "Cohen's w",
                "estimate": _r(np.sqrt(g.statistic / cyl.sum()), 3),
                "n": int(cyl.sum()),
            },
        )
    )
    survey = _data("survey-regions-missing").dropna(subset=["region", "answer"])
    s = stats.chi2_contingency(pd.crosstab(survey.region, survey.answer))
    out.append(
        (
            "survey-regions-missing",
            {
                "test": "Pearson chi-square test",
                "statistic": _r(s.statistic, 3),
                "df": int(s.dof),
                "p_value": _p(s.pvalue),
                "n": len(survey),
            },
        )
    )
    return out


def _analyst_answers() -> list[tuple[str, dict[str, Any]]]:
    answers: list[tuple[str, dict[str, Any]]] = []
    answers += [("transmission-unequal-variances", a) for a in _a_transmission()]
    answers += [("diets-three-groups", a) for a in _a_diets()]
    answers += [("supplement-registered-direction", a) for a in _a_registered()]
    answers += _a_reaction_and_yield()
    answers += _a_paired()
    answers += _a_one_sample()
    answers += _a_correlations()
    answers += _a_counts()
    return answers


ANALYST = _analyst_answers()


def test_the_analyst_set_spans_the_functions() -> None:
    tasks = {task_id for task_id, _ in ANALYST}
    assert len(tasks) >= 12
    assert {BY_ID[t].function for t in tasks} == set(run.FUNCTIONS)


@pytest.mark.parametrize(
    ("task_id", "answer"),
    ANALYST,
    ids=[f"{t}-{a['test']}" for t, a in ANALYST],
)
def test_a_careful_scipy_analysis_scores_as_correct(
    task_id: str, answer: dict[str, Any], tmp_path: Path
) -> None:
    score = run.score_run(
        BY_ID[task_id],
        "baseline",
        _transcript(calls=PATH_ONLY_CALLS),
        _write(tmp_path, answer),
        exit_code=0,
    )
    assert score.correct_test, score.test_reason
    assert score.numbers_ok, score.numbers
    assert score.trap is None
    assert not score.contaminated
    if answer.get("estimate") is not None:
        assert score.effect_ok, (score.variant, answer)


# --- acceptance: the answers an inferplot-arm agent copies from fig.result ----------


def _natural_answer(result: Any, name: Callable[[Any], str]) -> dict[str, Any]:
    test, effect = result.test, result.effect_size
    return {
        "test": name(test),
        "statistic": test.statistic,
        "df": test.df,
        "df_error": test.df_error,
        "p_value": test.p_value,
        "effect_size": effect.label if effect else None,
        "estimate": effect.estimate if effect else None,
        "conf_low": effect.conf_low if effect else None,
        "conf_high": effect.conf_high if effect else None,
        "n": result.n_obs,
        "check_issues": [
            f"[{i.severity}] {i.code}: {i.message}" for i in result.check()
        ],
    }


NAMERS: dict[str, Callable[[Any], str]] = {
    "name": lambda test: test.name,
    "label": lambda test: test.label,
    "method": lambda test: test.method,
}


@pytest.mark.parametrize(
    ("task", "block"),
    _inferplot_blocks(),
    ids=[f"{t.id}-{b.family}" for t, b in _inferplot_blocks()],
)
def test_inferplots_own_answer_scores_as_correct(
    task: Any, block: Any, tmp_path: Path
) -> None:
    result = run._analyze(task, pd.read_csv(task.data_file), block.kwargs)
    for key, namer in NAMERS.items():
        score = run.score_run(
            task,
            "inferplot",
            _transcript(),
            _write(tmp_path / key, _natural_answer(result, namer)),
            exit_code=0,
        )
        assert score.correct_test, (key, score.test_reason)
        assert score.numbers_ok, (key, score.numbers)
        assert score.trap is None
        if "estimate" in block.numbers:
            assert score.effect_ok, (key, score.numbers)
