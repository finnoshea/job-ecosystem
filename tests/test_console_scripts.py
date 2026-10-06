"""Tests for the console-script declarations in pyproject.toml.

A broken entry point fails only when someone runs the command -- which for a
cron job means silently, at 3am. These tests resolve each declared target
without installing the package, so a rename or typo is caught here instead.

The other half of the point: tests must keep working without an editable
install. Nothing in this file requires the package to be installed, and neither
does the rest of the suite; see ``conftest.py``.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scripts(config) -> dict[str, str]:
    return config.get("project", {}).get("scripts", {})


# ---------------------------------------------------------------------------
# the file is well formed
# ---------------------------------------------------------------------------

def test_pyproject_is_valid_toml(config):
    assert config["project"]["name"] == "jobecosystem"


def test_project_scripts_table_exists(scripts):
    assert scripts, "pyproject.toml should declare at least one console script"


def test_script_names_are_prefixed(scripts):
    # Distinguishable from unrelated commands on PATH.
    for name in scripts:
        assert name.startswith("jobecosystem-"), name


# ---------------------------------------------------------------------------
# every declared target resolves
# ---------------------------------------------------------------------------

def test_every_script_target_is_module_colon_function(scripts):
    for name, target in scripts.items():
        assert ":" in target, f"{name} -> {target!r} has no ':function' part"
        module, _, function = target.partition(":")
        assert module and function, f"{name} -> {target!r} is malformed"


def test_every_script_target_imports(scripts):
    for name, target in scripts.items():
        module_name, _, function_name = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            pytest.fail(f"{name}: cannot import {module_name!r}: {error}")
        assert hasattr(module, function_name), (
            f"{name}: {module_name} has no attribute {function_name!r}"
        )


def test_every_script_target_is_callable(scripts):
    for name, target in scripts.items():
        module_name, _, function_name = target.partition(":")
        function = getattr(importlib.import_module(module_name), function_name)
        assert callable(function), f"{name}: {target!r} is not callable"


def test_script_targets_take_an_optional_argv(scripts):
    # The generated launcher calls the function with no arguments, so it must be
    # callable without one. Passing argv explicitly is what tests rely on.
    for name, target in scripts.items():
        module_name, _, function_name = target.partition(":")
        function = getattr(importlib.import_module(module_name), function_name)
        required = [
            p for p in inspect.signature(function).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert not required, (
            f"{name}: {function_name} requires {[p.name for p in required]},"
            " so the installed command would fail without arguments"
        )


# ---------------------------------------------------------------------------
# the scraper entry point specifically
# ---------------------------------------------------------------------------

def test_scrape_script_points_at_ingest_cli(scripts):
    assert scripts.get("jobecosystem-scrape") == "jobecosystem.ingest.cli:main"


def test_describe_script_points_at_describe_cli(scripts):
    assert scripts.get("jobecosystem-describe") == (
        "jobecosystem.ingest.describe_cli:main"
    )


def test_embed_script_points_at_embed_cli(scripts):
    assert scripts.get("jobecosystem-embed") == (
        "jobecosystem.ingest.embed_cli:main"
    )


def test_discover_script_points_at_discover_cli(scripts):
    assert scripts.get("jobecosystem-discover") == (
        "jobecosystem.ingest.discover_cli:main"
    )


def test_tailor_script_points_at_tailor_cli(scripts):
    assert scripts.get("jobecosystem-tailor") == (
        "jobecosystem.tailor.cli:main"
    )


def test_scrape_entry_point_imports_without_the_tui():
    # The scraper must not drag in the TUI: they are separate commands.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import jobecosystem.ingest.cli;"
            " assert 'textual' not in sys.modules, 'TUI imported by scraper';"
            " print('ok')",
        ],
        capture_output=True,
        text=True,
        cwd=str(PYPROJECT.parent),
        env={"PYTHONPATH": str(PYPROJECT.parent / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------

def test_package_discovery_uses_the_src_layout(config):
    assert config["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_runtime_dependencies_are_narrow(config):
    # Heavy or optional libraries belong in an extra, not the base install.
    dependencies = config["project"]["dependencies"]
    assert not any("torch" in d or "sentence" in d for d in dependencies)
    assert any("httpx" in d for d in dependencies)


def test_embedding_libraries_are_an_extra(config):
    extras = config["project"]["optional-dependencies"]
    assert any("voyageai" in d for d in extras["embed"])


def test_tui_is_an_extra(config):
    extras = config["project"]["optional-dependencies"]
    assert any("textual" in d for d in extras["tui"])


def test_triage_script_points_at_triage_cli(scripts):
    assert scripts.get("jobecosystem-triage") == "jobecosystem.triage.cli:main"


def test_scraper_and_triage_are_separate_entry_points(scripts):
    # Two commands, not one dispatcher: they have nothing in common
    # operationally, and the scraper must not import the TUI.
    assert scripts["jobecosystem-scrape"] != scripts["jobecosystem-triage"]
    assert "ingest" in scripts["jobecosystem-scrape"]
    assert "triage" in scripts["jobecosystem-triage"]


def test_ingest_cli_does_not_import_the_tui():
    # The scraper runs on a timer; it must not require textual.
    import importlib
    import sys

    # Purging and re-importing creates duplicate module objects: any test that
    # imported `jobecosystem...` at collection time would then hold classes that
    # no longer match the freshly imported ones (breaking isinstance and
    # pytest.raises checks). Snapshot first and restore afterwards.
    saved = {
        name: module for name, module in sys.modules.items()
        if name.startswith("jobecosystem")
    }
    try:
        for name in saved:
            del sys.modules[name]
        importlib.import_module("jobecosystem.ingest.cli")
        assert "jobecosystem.triage.tui.app" not in sys.modules
    finally:
        sys.modules.update(saved)
