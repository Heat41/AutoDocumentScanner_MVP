"""Synthetic opt-in PR6 automatic-button and CLI integration tests.

The legacy branch is exercised as a control. No KTP data or real UI window.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SUGGESTED = [[30., 30.], [350., 30.], [350., 260.], [30., 260.]]


def _review(status):
    return {
        "status": status,
        "action": "open_four_corner_editor",
        "candidate_source": "v2_0" if status == "preview_review" else None,
        "suggested_corners": [p[:] for p in SUGGESTED] if status == "preview_review" else None,
        "reason_codes": [] if status == "preview_review" else ["unresolved_geometry_conflict"],
        "requires_explicit_confirmation": True,
        "auto_select": None,
        "auto_crop_allowed": False,
        "crop_performed": False,
        "file_write_performed": False,
        "image_shape": [300, 400],
        "mode": "experimental_runtime_review_only",
    }


def _fake_ui(tmp_path):
    from autodocscanner.ui.safe import SafeScannerUI

    source = tmp_path / "synthetic.png"
    source.write_bytes(b"not decoded: analyzer is monkeypatched")
    observed = {"status": [], "buttons": [], "selected": []}
    output_dir = tmp_path / "output"

    fake = SimpleNamespace(
        files=[source], output_dir=output_dir,
        _corners_by_file={}, _output_by_file={}, _metadata_by_file={},
        _failure_by_file={}, _pr6_runtime_review_by_file={},
        scanner=SimpleNamespace(last_detection={"selected_source": "legacy"},
                                last_quality={}, last_validation={},
                                last_corners=None),
        _processing_active=True,
        after=lambda _delay, callback, *args: callback(*args),
        status_text=SimpleNamespace(set=lambda msg: observed["status"].append(msg)),
        process_button=SimpleNamespace(configure=lambda **kw: observed["buttons"].append(kw)),
        listbox=SimpleNamespace(curselection=lambda: (0,)),
        _show_selected=lambda i: observed["selected"].append(i),
        _refresh_ktp_action_state=lambda: None,
        _failure_summary=SafeScannerUI._failure_summary,
    )
    return fake, source, observed


@pytest.mark.parametrize("status", ["manual_review", "preview_review"])
def test_proses_otomatis_pr6_never_calls_old_crop_or_writes_file(
        tmp_path, monkeypatch, status):
    from autodocscanner.ui import stage2

    fake, source, observed = _fake_ui(tmp_path)
    monkeypatch.setattr(stage2, "experimental_pr6_runtime_enabled", lambda: True)
    fake._pr6_review_requested = True
    monkeypatch.setattr(stage2, "evaluate_pr6_runtime_file", lambda path: _review(status))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy auto crop must not run")
    monkeypatch.setattr(stage2, "process_ktp_output", forbidden)
    stage2.Stage2ScannerUI._process_worker(fake, "color")

    key = str(source)
    assert key not in fake._output_by_file
    assert key in fake._failure_by_file
    assert fake._pr6_runtime_review_by_file[key]["status"] == status
    assert fake._metadata_by_file[key]["pr6_runtime_review"]["auto_crop_allowed"] is False
    if status == "preview_review":
        assert fake._corners_by_file[key].tolist() == SUGGESTED
    else:
        assert key not in fake._corners_by_file
    assert not any(fake.output_dir.iterdir())
    assert not fake._processing_active
    assert observed["selected"] == [0]
    assert any("menunggu review" in msg for msg in observed["status"])


def test_manual_review_drops_old_candidate_and_old_output_reference(tmp_path, monkeypatch):
    from autodocscanner.ui import stage2

    fake, source, observed = _fake_ui(tmp_path)
    old_file = tmp_path / "old_scanned.jpg"
    old_file.write_bytes(b"previous successful crop")
    before = old_file.read_bytes()
    key = str(source)
    fake._corners_by_file[key] = np.asarray(SUGGESTED, dtype=np.float32)
    fake._output_by_file[key] = old_file
    monkeypatch.setattr(stage2, "experimental_pr6_runtime_enabled", lambda: True)
    fake._pr6_review_requested = True
    monkeypatch.setattr(stage2, "evaluate_pr6_runtime_file",
                        lambda path: _review("manual_review"))
    monkeypatch.setattr(stage2, "process_ktp_output",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("must not crop")
                        ))
    stage2.Stage2ScannerUI._process_worker(fake, "color")

    assert key not in fake._corners_by_file
    assert key not in fake._output_by_file
    assert old_file.read_bytes() == before


def test_analyzer_exception_fails_closed_without_legacy_fallback(
        tmp_path, monkeypatch):
    from autodocscanner.ui import stage2

    fake, source, observed = _fake_ui(tmp_path)
    monkeypatch.setattr(stage2, "experimental_pr6_runtime_enabled", lambda: True)
    fake._pr6_review_requested = True
    def broken(_path):
        raise ValueError("invalid synthetic input")
    monkeypatch.setattr(stage2, "evaluate_pr6_runtime_file", broken)
    monkeypatch.setattr(stage2, "process_ktp_output", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(AssertionError("must not crop")))
    monkeypatch.setattr(stage2.messagebox, "showwarning", lambda *_a, **_k: None)

    stage2.Stage2ScannerUI._process_worker(fake, "color")
    key = str(source)
    assert key not in fake._output_by_file
    assert key not in fake._corners_by_file
    assert "Pemeriksaan PR6 gagal" in fake._failure_by_file[key]["message"]
    assert list(fake.output_dir.iterdir()) == []


def test_default_legacy_worker_unchanged_when_flag_disabled(
        tmp_path, monkeypatch):
    from autodocscanner.ui import stage2

    fake, source, observed = _fake_ui(tmp_path)
    monkeypatch.setattr(stage2, "experimental_pr6_runtime_enabled", lambda: False)
    monkeypatch.setattr(stage2, "evaluate_pr6_runtime_file", lambda path:
                        (_ for _ in ()).throw(AssertionError("PR6 disabled")))
    calls = []
    def legacy(scanner, path, output_dir, **kwargs):
        calls.append((path, kwargs))
        return {
            "output_path": output_dir / "synthetic_scanned.jpg",
            "corners": np.asarray(SUGGESTED, dtype=np.float32),
        }
    monkeypatch.setattr(stage2, "process_ktp_output", legacy)
    stage2.Stage2ScannerUI._process_worker(fake, "color")

    assert len(calls) == 1
    assert fake._output_by_file[str(source)].name == "synthetic_scanned.jpg"
    assert str(source) not in fake._failure_by_file
    assert not fake._pr6_runtime_review_by_file


def test_legacy_auto_runs_even_with_pr6_feature_flag_on_when_not_requested(
        tmp_path, monkeypatch):
    from autodocscanner.ui import stage2

    fake, source, observed = _fake_ui(tmp_path)
    fake._pr6_review_requested = False
    monkeypatch.setattr(stage2, "experimental_pr6_runtime_enabled", lambda: True)
    monkeypatch.setattr(
        stage2, "evaluate_pr6_runtime_file",
        lambda path: (_ for _ in ()).throw(
            AssertionError("Review PR6 must be a separate button")
        ),
    )
    calls = []
    def legacy(scanner, path, output_dir, **kwargs):
        calls.append(path)
        return {
            "output_path": output_dir / "synthetic_scanned.jpg",
            "corners": np.asarray(SUGGESTED, dtype=np.float32),
        }
    monkeypatch.setattr(stage2, "process_ktp_output", legacy)
    stage2.Stage2ScannerUI._process_worker(fake, "color")

    assert calls == [source]
    assert fake._output_by_file[str(source)].name == "synthetic_scanned.jpg"
    assert str(source) not in fake._failure_by_file
    assert fake._pr6_review_requested is False
    assert not fake._pr6_runtime_review_by_file


def test_pr6_run_request_is_one_shot(tmp_path, monkeypatch):
    from autodocscanner.ui import stage2

    fake, source, observed = _fake_ui(tmp_path)
    fake._pr6_review_requested = True
    monkeypatch.setattr(stage2, "experimental_pr6_runtime_enabled", lambda: True)
    monkeypatch.setattr(
        stage2, "evaluate_pr6_runtime_file",
        lambda path: _review("manual_review"),
    )
    monkeypatch.setattr(
        stage2, "process_ktp_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("PR6 review should not crop")
        ),
    )
    stage2.Stage2ScannerUI._process_worker(fake, "color")
    assert fake._pr6_review_requested is False
    assert str(source) in fake._pr6_runtime_review_by_file


def test_final_ui_review_action_requires_explicit_flag_and_idle():
    from autodocscanner.ui.final import FinalScannerUI
    fake = SimpleNamespace(
        files=[Path("synthetic.png")],
        _processing_active=False,
        _pr6_review_requested=False,
        status_text=SimpleNamespace(set=lambda *_a: None),
        _start_ktp_processing=lambda: "started",
    )
    import autodocscanner.ui.final as final
    from unittest.mock import patch
    with patch.object(final, "experimental_pr6_runtime_enabled", return_value=False):
        assert FinalScannerUI.run_pr6_review(fake) is None
        assert fake._pr6_review_requested is False
    with patch.object(final, "experimental_pr6_runtime_enabled", return_value=True):
        assert FinalScannerUI.run_pr6_review(fake) == "started"
        assert fake._pr6_review_requested is True
        fake._processing_active = True
        fake._pr6_review_requested = False
        assert FinalScannerUI.run_pr6_review(fake) is None
        assert fake._pr6_review_requested is False


def test_safe_ui_distinguishes_pr6_review_from_ordinary_rejection():
    from autodocscanner.ui.safe import SafeScannerUI

    status, message = SafeScannerUI._failure_summary({
        "message": "Periksa empat sudut di editor.",
        "metadata": {"pr6_runtime_review": {
            "status": "manual_review", "auto_crop_allowed": False,
        }},
    })
    assert "Menunggu konfirmasi" in status
    assert "empat sudut" in message
    ordinary_status, _ = SafeScannerUI._failure_summary({
        "message": "Detector error", "metadata": {},
    })
    assert ordinary_status == "Status: Ditolak"


def test_cli_pr6_flag_runs_review_instead_of_batch_crop(tmp_path, monkeypatch, capsys):
    import sys
    import app
    from autodocscanner.core import pr6_runtime_review as runtime

    source = tmp_path / "synthetic.png"
    source.write_bytes(b"fake")
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.setenv("AUTODOCSCANNER_PR6_RUNTIME_REVIEW", "1")
    monkeypatch.setattr(app, "get_input_images", lambda folder: [source])
    monkeypatch.setattr(runtime, "evaluate_pr6_runtime_file",
                        lambda path: _review("manual_review"))
    monkeypatch.setattr(app, "BatchScanRunner", lambda *_a, **_k:
                        (_ for _ in ()).throw(AssertionError("must not call old crop")))
    app.main()
    stdout = capsys.readouterr().out
    assert "TANPA CROP" in stdout
    assert "manual_review" in stdout
    assert "Tidak ada gambar hasil yang disimpan." in stdout
