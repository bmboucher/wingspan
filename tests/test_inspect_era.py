"""Tests for the era-routed reporting seam and the static architecture diagram.

Three guarantees from the same change:

* The descriptor-driven builders in ``runmeta`` (``build_inspect_report`` /
  ``build_model_summary_html``) are the single code path behind both the
  run-start writers and ``wingspan inspect`` — the JSON / HTML artifacts a run
  leaves behind are reproducible from its ``model_config.json`` byte for byte.
* ``arch_diagram.render_static`` (the ``wingspan inspect`` ARCHITECTURE panel)
  renders without the interactive configurator state — including the separate
  setup-net box — and shows the *caller-supplied* era-routed choice-encoder
  widths, not live-encoder recomputations.
* The ``wingspan inspect`` CLI itself runs end to end against the no-dir
  baseline and a pre-0.1 run directory (the v0.0 compat fixture's descriptor —
  plain JSON, no checkpoint load), and its ``--html`` mode reproduces the
  run-start ``model_summary.html`` rather than clobbering it.

The per-era routing *values* (frozen v0.0 vs live widths, param totals matching
``sum(p.numel())`` of the loaded nets) are asserted against the pinned fixture
checkpoints in ``test_compat_v0_0.py`` / ``test_compat_v0_1.py``.
"""

from __future__ import annotations

import io
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")

from wingspan import architecture, encode, setup_model, version
from wingspan.compat import v0_0
from wingspan.reporting import inspect_cli
from wingspan.training import artifacts, config, runmeta
from wingspan.training.configure import arch_diagram

V0_0_FIXTURE_DIR = pathlib.Path(__file__).parent / "data" / "compat" / "v0.0"

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
    box — a run trained with ``use_setup_model`` shows its full topology."""
    plain = _render_plain(_baseline_descriptor(), use_setup_model=True)
    assert "SETUP MODEL" in plain


def test_render_static_shows_the_frozen_era_widths():
    """A pre-0.1 descriptor's diagram carries the v0.0 choice-encoder input
    width — different from what the live formula would claim for it."""
    spec = encode.DEFAULT_SPEC
    descriptor = _baseline_descriptor(
        choice_dim=v0_0.choice_feature_dim(spec),
        version=version.PRE_VERSIONING_VERSION,
    )
    frozen_in = runmeta.choice_input_dim_for(descriptor)
    live_in = encode.choice_input_dim(
        descriptor.choice_dim,
        descriptor.architecture.card_embed_dim,
        include_setup=descriptor.include_setup,
    )
    assert frozen_in != live_in
    assert f"in {frozen_in}" in _render_plain(descriptor, use_setup_model=True)


#### Run-start artifacts reproduce from the descriptor ####


def test_run_start_html_matches_the_descriptor_rebuild(tmp_path: pathlib.Path):
    """``write_model_summary_html`` (the cfg path) and
    ``build_model_summary_html`` on the read-back descriptor produce identical
    documents — the by-construction consistency contract."""
    cfg = config.TrainConfig(device="cpu", run_name="html-parity")
    runmeta.write_model_config(str(tmp_path), cfg)
    original = runmeta.write_model_summary_html(str(tmp_path), cfg).read_text(
        encoding="utf-8"
    )
    descriptor = runmeta.read_model_config(str(tmp_path))
    assert runmeta.build_model_summary_html(descriptor, cfg.setup_arch) == original


def test_run_start_inspect_json_matches_the_descriptor_rebuild(
    tmp_path: pathlib.Path,
):
    """``model_inspect.json`` reproduces exactly from the run's descriptor."""
    cfg = config.TrainConfig(device="cpu", run_name="json-parity")
    runmeta.write_model_config(str(tmp_path), cfg)
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


def test_inspect_v0_0_run_shows_the_frozen_geometry(
    monkeypatch: pytest.MonkeyPatch,
):
    """Pointed at the pinned v0.0 run directory (descriptor only — no
    checkpoint load), the choice table resurrects the habitat stripe and the
    diagram captions the v0.0 choice-encoder input width."""
    code, out = _run_inspect(
        ["--checkpoint-dir", str(V0_0_FIXTURE_DIR), "--width", str(_CONSOLE_WIDTH)],
        monkeypatch,
    )
    descriptor = runmeta.read_model_config(str(V0_0_FIXTURE_DIR))
    assert code == 0
    assert "habitat" in out
    assert f"in {runmeta.choice_input_dim_for(descriptor)}" in out


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
    cfg = config.TrainConfig(device="cpu", run_name="html-regen")
    runmeta.write_model_config(str(tmp_path), cfg)
    original = runmeta.write_model_summary_html(str(tmp_path), cfg).read_text(
        encoding="utf-8"
    )
    code, out = _run_inspect(["--checkpoint-dir", str(tmp_path), "--html"], monkeypatch)
    assert code == 0
    assert "HTML report written" in out
    regenerated = (tmp_path / artifacts.MODEL_SUMMARY_HTML).read_text(encoding="utf-8")
    assert regenerated == original
