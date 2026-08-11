"""Regression tests for ``scripts/quality_gate.sh`` argument resolution.

The gate's step selection is only observable through which tools it runs, so a
silently skipped step looks exactly like a passing one. That is how ``--coverage``
came to drop pyright, isort, and black from the merge gate without anyone
noticing. ``--dry-run`` makes the resolved plan observable; these tests pin it.
"""

import os
import pathlib
import shutil
import subprocess

import pydantic
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GATE_SCRIPT = _REPO_ROOT / "scripts" / "quality_gate.sh"
_MERGE_SCRIPT = _REPO_ROOT / "scripts" / "merge_worktree.sh"
_BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(_BASH is None, reason="bash is not on PATH")


class GatePlan(pydantic.BaseModel):
    """The resolved gate plan reported by ``quality_gate.sh --dry-run``."""

    steps: list[str]
    full_gate: bool
    target: str
    pyright_args: str
    format_args: str
    pytest_args: str


def _dry_run(*flags: str, workers: str | None = None) -> GatePlan:
    """Resolve ``flags`` through the gate's ``--dry-run`` path."""
    assert _BASH is not None
    env = dict(os.environ)
    if workers is not None:
        env["WINGSPAN_PYTEST_WORKERS"] = workers
    completed = subprocess.run(
        [_BASH, str(_GATE_SCRIPT), *flags, "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return GatePlan(
        steps=fields["steps"].split(),
        full_gate=fields["full_gate"] == "true",
        target=fields["target"],
        pyright_args=fields["pyright_args"],
        format_args=fields["format_args"],
        pytest_args=fields["pytest_args"],
    )


def _merge_gate_flags() -> list[str]:
    """Extract the flags ``merge_worktree.sh`` passes to the quality gate."""
    for line in _MERGE_SCRIPT.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "quality_gate.sh" not in stripped:
            continue
        _, _, tail = stripped.partition("quality_gate.sh")
        return [
            token for token in tail.replace('"', "").split() if token.startswith("--")
        ]
    return []


def test_bare_gate_runs_pyright_format_and_pytest() -> None:
    """No flags means the full gate."""
    plan = _dry_run()
    assert plan.steps == ["pyright", "format", "pytest"]
    assert plan.full_gate is True


def test_coverage_alone_runs_the_full_gate() -> None:
    """--coverage is a modifier: it must not suppress the type and format steps."""
    plan = _dry_run("--coverage")
    assert plan.steps == ["pyright", "format", "pytest", "coverage"]
    assert plan.full_gate is True


def test_merge_gate_type_checks_and_format_checks() -> None:
    """Whatever flags merge_worktree.sh passes must still run pyright and black."""
    flags = _merge_gate_flags()
    assert flags, "no quality_gate.sh invocation found in merge_worktree.sh"
    plan = _dry_run(*flags)
    assert "pyright" in plan.steps
    assert "format" in plan.steps


def test_explicit_pytest_with_coverage_stays_a_narrow_run() -> None:
    """Naming --pytest explicitly still selects a coverage-only run."""
    plan = _dry_run("--pytest", "--coverage")
    assert plan.steps == ["pytest", "coverage"]
    assert plan.full_gate is False


@pytest.mark.parametrize(
    ("flags", "expected_steps"),
    [
        (("--pyright",), ["pyright"]),
        (("--format",), ["format"]),
        (("--pytest",), ["pytest"]),
        (("--pyright", "--pytest"), ["pyright", "pytest"]),
        (("--pytest", "--pyright"), ["pyright", "pytest"]),
    ],
)
def test_section_flags_select_only_their_steps(
    flags: tuple[str, ...], expected_steps: list[str]
) -> None:
    """Section flags select exactly their steps, in canonical order."""
    plan = _dry_run(*flags)
    assert plan.steps == expected_steps
    assert plan.full_gate is False


def test_coverage_pytest_args_are_serial_and_measured() -> None:
    """The coverage run is serial so term-missing output stays readable."""
    plan = _dry_run("--coverage")
    assert "-p no:xdist" in plan.pytest_args
    assert "--cov" in plan.pytest_args


def test_default_pytest_args_are_parallel() -> None:
    """The default full-suite run fans out across xdist workers."""
    plan = _dry_run(workers="8")
    assert "-n" in plan.pytest_args.split()
    assert "--dist" in plan.pytest_args


def test_zero_workers_runs_serially() -> None:
    """WINGSPAN_PYTEST_WORKERS=0 drops the xdist flags."""
    plan = _dry_run(workers="0")
    assert "-n" not in plan.pytest_args.split()


def test_explicit_pytest_args_replace_the_defaults() -> None:
    """Explicit pytest args replace the default set entirely."""
    plan = _dry_run("--pytest", "tests/test_smoke.py")
    assert plan.pytest_args == "tests/test_smoke.py"


def test_dry_run_precedes_preflight() -> None:
    """A nonexistent target still resolves: the plan is reported before the venv check."""
    plan = _dry_run("/nonexistent/target/dir", "--pytest")
    assert plan.target == "/nonexistent/target/dir"
    assert plan.steps == ["pytest"]
