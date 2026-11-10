"""Packaging and import-time guarantees."""

import importlib.metadata
import subprocess
import sys
import textwrap

import inferplot

# Imports every inferplot module, so a module that only some code path loads is
# checked too.
IMPORT_ALL = """
import importlib, pkgutil, sys
import inferplot
for module in pkgutil.walk_packages(inferplot.__path__, "inferplot."):
    importlib.import_module(module.name)
"""


def _run(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_version_matches_installed_metadata() -> None:
    assert inferplot.__version__ == importlib.metadata.version("inferplot")


def test_import_does_not_load_pyplot() -> None:
    # CLAUDE.md rule 6: library code never imports matplotlib.pyplot.
    _run(IMPORT_ALL + "assert 'matplotlib.pyplot' not in sys.modules\n")


def test_import_works_without_optional_backends() -> None:
    # CLAUDE.md rule 5: pandas, polars and pyarrow are optional extras. With all
    # three unimportable (a None entry in sys.modules makes `import` raise
    # ImportError), every inferplot module must still import.
    block = "import sys\nfor name in ('pandas', 'polars', 'pyarrow'):\n"
    block += "    sys.modules[name] = None\n"
    _run(block + IMPORT_ALL)


def test_public_api() -> None:
    assert sorted(inferplot.__all__) == [
        "Issue",
        "StatsFigure",
        "StatsResult",
        "__version__",
        "analyze",
        "barstats",
        "betweenstats",
        "histostats",
        "render",
        "scatterstats",
        "withinstats",
    ]
    for name in inferplot.__all__:
        assert hasattr(inferplot, name)
    functions = [
        "barstats",
        "betweenstats",
        "histostats",
        "scatterstats",
        "withinstats",
    ]
    assert inferplot.analyze.__all__ == functions
    for name in functions:
        assert callable(getattr(inferplot.analyze, name))
        assert callable(getattr(inferplot.render, name))
        assert callable(getattr(inferplot, name))
    assert set(inferplot.__all__) <= set(dir(inferplot))


def test_render_and_statsfigure_load_matplotlib_lazily() -> None:
    # `import inferplot` stays free of matplotlib; the names that need it load it
    # on first access, and an unknown name still raises AttributeError.
    _run(
        """
        import sys
        import inferplot
        assert "matplotlib" not in sys.modules
        assert callable(inferplot.betweenstats)
        assert "matplotlib" not in sys.modules, "ip.betweenstats is not lazy"
        from inferplot.render.figure import StatsFigure
        assert inferplot.StatsFigure is StatsFigure
        assert inferplot.render.betweenstats.__name__ == "betweenstats"
        assert "matplotlib.pyplot" not in sys.modules
        try:
            inferplot.not_a_name
        except AttributeError as error:
            assert "not_a_name" in str(error)
        else:
            raise AssertionError("no AttributeError")
        """
    )


def test_analysis_does_not_load_matplotlib() -> None:
    # SPEC.md section 2: inferplot.analyze must not import matplotlib, and neither
    # must `import inferplot` nor running an analysis.
    _run(
        """
        import sys
        import inferplot
        assert "matplotlib" not in sys.modules, "import inferplot"
        import inferplot.analyze
        assert "matplotlib" not in sys.modules, "import inferplot.analyze"
        import pandas as pd
        two = pd.DataFrame({"g": list("aaabbb"), "y": [1.0, 2.0, 4.0, 3.0, 5.0, 8.0]})
        three = pd.DataFrame(
            {"g": list("aaabbbccc"), "y": [1.0, 2, 4, 3, 5, 8, 6, 9, 13]}
        )
        for data in (two, three):
            result = inferplot.analyze.betweenstats(data, "g", "y")
            result.to_json()
            result.check()
            str(result)
        paired = pd.DataFrame(
            {"s": [1, 2, 3, 1, 2, 3], "g": list("aaabbb"), "y": [1.0, 2, 4, 3, 5, 8]}
        )
        inferplot.analyze.withinstats(paired, "g", "y", subject_id="s")
        pairs = pd.DataFrame({"a": [1.0, 2, 3, 5], "b": [2.0, 1, 4, 3]})
        for result in (
            inferplot.analyze.histostats(pairs, "a", test_value=1),
            inferplot.analyze.scatterstats(pairs, "a", "b"),
        ):
            result.to_json()
            result.check()
            str(result)
        assert "matplotlib" not in sys.modules, "running an analysis"
        """
    )
