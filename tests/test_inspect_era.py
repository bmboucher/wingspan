"""Tests for the descriptor-routed reporting seam and the static architecture
diagram.

Three guarantees from the same change:

* The descriptor-driven builders in ``runmeta`` (``build_inspect_report`` /
  ``build_model_summary_html``) are the single code path behind both the
  run-start writers and ``wingspan inspect`` — the JSON / HTML artifacts a run
  leaves behind are reproducible from its run-config descriptor byte for byte.
* ``arch_diagram.render_static`` (the ``wingspan inspect`` ARCHITECTURE panel)
  renders without the interactive configurator state — including the separate
  setup-net box — and shows the *caller-supplied* descriptor-routed
  choice-encoder widths, not live-encoder recomputations.
* The ``wingspan inspect`` CLI itself runs end to end against the no-dir
  baseline, reports a missing descriptor cleanly, and its ``--html`` mode
  reproduces the run-start ``model_summary.html`` rather than clobbering it.

No pre-1.0 compat shims remain (dropped at the 1.0 MAJOR bump), so every
loadable same-MAJOR descriptor resolves to the live encoder geometry; the
descriptor seam is exercised here at that single live era.
"""

from __future__ import annotations

import io
import pathlib
import sys

import pytest

pytest.importorskip("torch")

from wingspan import architecture, encode, setup_model, version
from wingspan.reporting import inspect_cli
from wingspan.training import artifacts, config, runmeta, setup_runmeta
from wingspan.training.configure import arch_diagram

_DIAGRAM_WIDTH = 60
_CONSOLE_WIDTH = 130


def _baseline_descriptor(**overrides: object) -> runmeta.ModelConfig:
    """A current-era descriptor at the live baseline dims, fields overridable."""
    fields: dict[str, object] = {
        "run_name": "era-test",
        "state_dim": encode.state_size(),
        "choice_dim": encode.CHOICE_FEATURE_DIM,
        "family_order": ("PLAY_BIRD", "GAIN_FOOD"),
        "architecture": architecture.ModelArchitecture(),
        "include_setup": encode.DEFAULT_SPEC.include_setup,
        "version": version.MODEL_VERSION,
    }
    fields.update(overrides)
    return runmeta.ModelConfig.model_validate(fields)


def _render_plain(descriptor: runmeta.ModelConfig, *, use_setup_model: bool) -> str:
    """Render the static diagram for ``descriptor`` and join the row text."""
    rows = arch_diagram.render_static(
        descriptor.architecture,
        state_dim=descriptor.state_dim,
        choice_dim=descriptor.choice_dim,
        family_order=descriptor.family_order,
        choice_in=runmeta.choice_input_dim_for(descriptor),
        choice_extra=runmeta.choice_extra_for(descriptor),
        use_setup_model=use_setup_model,
        setup_arch=setup_model.SetupArchitecture(),
        width=_DIAGRAM_WIDTH,
    )
    assert rows, "expected a non-empty diagram"
    return "\n".join(row.plain for row in rows)


def _run_inspect(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    """Drive ``main_inspect`` with stdout swapped for a plain ``StringIO``
    (no ``.buffer`` attribute, so ``_utf8_stdout`` falls back to it directly)
    and return the exit code plus the captured console text."""
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    code = inspect_cli.main_inspect(argv)
    return code, stream.getvalue()


#### Static architecture diagram ####


def test_render_static_draws_the_live_baseline():
    """The focus-free diagram renders the baseline network — the path the
    interactive configurator never exercises — with the choice encoder's
    caller-supplied input width in its caption and no setup box."""
    descriptor = _baseline_descriptor()
    plain = _render_plain(descriptor, use_setup_model=False)
    assert f"in {runmeta.choice_input_dim_for(descriptor)}" in plain
    assert "setup model · off" in plain


def test_render_static_draws_the_setup_box():
    """With the separate setup model active, the static diagram includes its
    two-tower boxes — a run trained with ``use_setup_model`` shows its full topology."""
    plain = _render_plain(_baseline_descriptor(), use_setup_model=True)
    assert "SETUP STATE" in plain
    assert "SETUP CHOICE" in plain


#### Run-start artifacts reproduce from the descriptor ####


def test_run_start_html_matches_the_descriptor_rebuild(tmp_path: pathlib.Path):
    """``write_model_summary_html`` (the cfg path) and
    ``build_model_summary_html`` on the read-back descriptor produce identical
    documents — the by-construction consistency contract."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(run_name="html-parity"),
    )
    _write_run_config(tmp_path, cfg)
    original = runmeta.write_model_summary_html(str(tmp_path), cfg).read_text(
        encoding="utf-8"
    )
    descriptor = runmeta.read_model_config(str(tmp_path))
    setup_cfg = setup_runmeta.read_setup_config(str(tmp_path))
    assert (
        runmeta.build_model_summary_html(
            descriptor, setup_cfg.setup_arch, setup_cfg.setup_encoding
        )
        == original
    )


def test_run_start_inspect_json_matches_the_descriptor_rebuild(
    tmp_path: pathlib.Path,
):
    """``model_inspect.json`` reproduces exactly from the run's descriptor."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(run_name="json-parity"),
    )
    _write_run_config(tmp_path, cfg)
    written = runmeta.write_inspect_report(str(tmp_path), cfg).read_text(
        encoding="utf-8"
    )
    descriptor = runmeta.read_model_config(str(tmp_path))
    rebuilt = runmeta.build_inspect_report(descriptor)
    assert rebuilt == runmeta.InspectReport.model_validate_json(written)


#### The inspect CLI end to end ####


def test_inspect_baseline_prints_all_sections(monkeypatch: pytest.MonkeyPatch):
    code, out = _run_inspect(["--width", str(_CONSOLE_WIDTH)], monkeypatch)
    assert code == 0
    for section in ("STATE VECTOR", "CHOICE VECTOR", "ARCHITECTURE", "PARAMETERS"):
        assert section in out


def test_inspect_missing_descriptor_is_a_clean_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    code, out = _run_inspect(
        ["--checkpoint-dir", str(tmp_path / "not-a-run")], monkeypatch
    )
    assert code == 1
    assert "not a run directory" in out


def test_inspect_html_reproduces_the_run_start_report(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """``wingspan inspect --html`` into a run directory regenerates the exact
    ``model_summary.html`` the run wrote at startup — same builder, same
    descriptor — instead of clobbering it with a live-encoder view."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(run_name="html-regen"),
    )
    _write_run_config(tmp_path, cfg)
    original = runmeta.write_model_summary_html(str(tmp_path), cfg).read_text(
        encoding="utf-8"
    )
    code, out = _run_inspect(["--checkpoint-dir", str(tmp_path), "--html"], monkeypatch)
    assert code == 0
    assert "HTML report written" in out
    regenerated = (tmp_path / artifacts.MODEL_SUMMARY_HTML).read_text(encoding="utf-8")
    assert regenerated == original


def _write_run_config(tmp_path: pathlib.Path, cfg: config.RunConfig) -> None:
    """Write the run's unified ``run_config_<stamp>.json`` descriptor sidecar."""
    runmeta.write_run_config(
        str(tmp_path),
        cfg,
        stamp="t0",
        started_at="t0",
        git_sha=None,
        resumed_from_iteration=0,
    )
