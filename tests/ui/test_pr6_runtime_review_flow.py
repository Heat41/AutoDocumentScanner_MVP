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
    assert any("Lihat Kandidat" in msg for msg in observed["status"])


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
    monkeypatch.setattr(sys, "argv", ["app.py", "--pr6-review"])
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



def test_cli_default_still_runs_legacy_batch_when_pr6_flag_on(
        tmp_path, monkeypatch, capsys):
    import sys
    import app

    source = tmp_path / "synthetic.png"
    source.write_bytes(b"fake")
    monkeypatch.setattr(sys, "argv", ["app.py"])
    monkeypatch.setenv("AUTODOCSCANNER_PR6_RUNTIME_REVIEW", "1")
    monkeypatch.setattr(app, "get_input_images", lambda folder: [source])
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path / "out")
    runner_calls = []
    class FakeBatchRunner:
        def __init__(self, scanner):
            runner_calls.append("created")
        def run_many(self, images, output_dir, output_mode, mode):
            runner_calls.append((list(images), mode))
            return {
                "total_count": 1,
                "success_count": 0,
                "failed_count": 1,
                "results": [{
                    "input_path": source, "success": False,
                    "error": "synthetic detector declined",
                }],
            }
    monkeypatch.setattr(app, "BatchScanRunner", FakeBatchRunner)
    app.main()
    output = capsys.readouterr().out
    assert runner_calls == ["created", ([source], "ktp")]
    assert "AUTO DOCUMENT SCANNER" in output
    assert "PR6 REVIEW EKSPERIMEN" not in output


def test_cli_explicit_pr6_review_requires_opt_in(monkeypatch):
    import sys
    import app

    monkeypatch.setattr(sys, "argv", ["app.py", "--pr6-review"])
    monkeypatch.delenv("AUTODOCSCANNER_PR6_RUNTIME_REVIEW", raising=False)
    with pytest.raises(SystemExit) as exc:
        app.main()
    assert exc.value.code == 2



def test_final_ui_main_auto_action_clears_pending_pr6_request():
    from autodocscanner.ui.final import FinalScannerUI

    recorded = []
    fake = SimpleNamespace(
        files=[Path("synthetic.png")],
        _processing_active=False,
        _pr6_review_requested=True,
        _choose_ktp_processing_mode=lambda: {"accepted": True, "mode": "color"},
        output_mode=SimpleNamespace(set=lambda mode: recorded.append(mode)),
        status_text=SimpleNamespace(set=lambda msg: recorded.append(msg)),
        _ktp_mode_label=lambda mode: "Warna",
        _start_ktp_processing=lambda: "legacy_started",
    )
    assert FinalScannerUI.process_files(fake) == "legacy_started"
    assert fake._pr6_review_requested is False
    assert recorded[0] == "color"


def test_final_ui_refresh_keeps_pr6_button_independent_from_auto_button(monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    from autodocscanner.ui import final

    states = {"legacy": [], "pr6": []}
    fake = SimpleNamespace(
        files=[Path("synthetic.png")],
        _processing_active=False,
        _processed_image_paths=lambda: [],
        process_button=SimpleNamespace(
            configure=lambda **kwargs: states["legacy"].append(kwargs["state"])
        ),
        pr6_review_button=SimpleNamespace(
            configure=lambda **kwargs: states["pr6"].append(kwargs["state"])
        ),
    )
    monkeypatch.setattr(final, "experimental_pr6_runtime_enabled", lambda: True)
    FinalScannerUI._refresh_ktp_action_state(fake)
    assert states["legacy"][-1] == "normal"
    assert states["pr6"][-1] == "normal"

    fake._processing_active = True
    FinalScannerUI._refresh_ktp_action_state(fake)
    assert states["legacy"][-1] == "disabled"
    assert states["pr6"][-1] == "disabled"

    fake._processing_active = False
    monkeypatch.setattr(final, "experimental_pr6_runtime_enabled", lambda: False)
    FinalScannerUI._refresh_ktp_action_state(fake)
    assert states["legacy"][-1] == "normal"
    assert states["pr6"][-1] == "disabled"



def test_visual_debug_button_available_only_after_selected_photo_review(monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    from autodocscanner.ui import final

    source = Path("synthetic.png")
    statuses = []
    fake = SimpleNamespace(
        files=[source], _processing_active=False,
        _processed_image_paths=lambda: [],
        _active_file_index=lambda: 0,
        _pr6_runtime_review_by_file={},
        pr6_candidates_button=SimpleNamespace(
            configure=lambda **kw: statuses.append(kw["state"])
        ),
    )
    monkeypatch.setattr(final, "experimental_pr6_runtime_enabled", lambda: True)
    FinalScannerUI._refresh_ktp_action_state(fake)
    assert statuses[-1] == "disabled"

    fake._pr6_runtime_review_by_file[str(source)] = {
        "status": "manual_review",
        "visual_debug": {
            "status": "offline_visual_diagnostic_only",
            "auto_select": None, "auto_crop_allowed": False,
            "candidates": [],
        },
    }
    FinalScannerUI._refresh_ktp_action_state(fake)
    assert statuses[-1] == "normal"
    fake._processing_active = True
    FinalScannerUI._refresh_ktp_action_state(fake)
    assert statuses[-1] == "disabled"

    fake._processing_active = False
    fake._active_file_index = lambda: None
    FinalScannerUI._refresh_ktp_action_state(fake)
    assert statuses[-1] == "disabled"


def test_open_pr6_visual_debug_for_active_photo_only_and_no_scan(monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    from autodocscanner.ui import final

    photo = Path("synthetic.png")
    overlay = {
        "status": "offline_visual_diagnostic_only",
        "candidates": [],
    }
    calls = []
    fake = SimpleNamespace(
        files=[photo],
        _active_file_index=lambda: 0,
        _processing_active=False,
        status_text=SimpleNamespace(set=lambda msg: calls.append(("status", msg))),
        _pr6_runtime_review_by_file={
            str(photo): {"visual_debug": overlay}
        },
    )
    monkeypatch.setattr(final, "experimental_pr6_runtime_enabled", lambda: True)
    monkeypatch.setattr(
        final, "open_pr6_candidate_debug_dialog",
        lambda parent, path, diagnostics: calls.append(
            ("open", path, diagnostics)
        ) or "read_only_dialog",
    )
    assert FinalScannerUI.open_pr6_candidate_debug(fake) == "read_only_dialog"
    assert calls == [("open", photo, overlay)]

    fake._processing_active = True
    assert FinalScannerUI.open_pr6_candidate_debug(fake) is None
    assert len(calls) == 1
    fake._processing_active = False
    fake._pr6_runtime_review_by_file.clear()
    assert FinalScannerUI.open_pr6_candidate_debug(fake) is None
    assert "Klik Review PR6" in calls[-1][1]


def test_review_worker_schedules_visual_dialog_only_after_finishing(tmp_path, monkeypatch):
    from autodocscanner.ui import stage2

    fake, source, observed = _fake_ui(tmp_path)
    observed["debug"] = []
    fake.open_pr6_candidate_debug = lambda: observed["debug"].append("opened")
    fake._hide_loading_popup = lambda: observed["debug"].append("closed_loading")
    fake._pr6_review_requested = True
    monkeypatch.setattr(stage2, "experimental_pr6_runtime_enabled", lambda: True)
    monkeypatch.setattr(stage2, "evaluate_pr6_runtime_file",
                        lambda path: _review("manual_review"))
    stage2.Stage2ScannerUI._process_worker(fake, "color")
    assert observed["debug"] == ["closed_loading", "opened"]
    assert str(source) in fake._pr6_runtime_review_by_file


def test_legacy_worker_never_opens_pr6_visual_dialog(tmp_path, monkeypatch):
    from autodocscanner.ui import stage2

    fake, source, observed = _fake_ui(tmp_path)
    observed["debug"] = []
    fake.open_pr6_candidate_debug = lambda: observed["debug"].append("opened")
    fake._pr6_review_requested = False
    monkeypatch.setattr(stage2, "experimental_pr6_runtime_enabled", lambda: False)
    monkeypatch.setattr(stage2, "process_ktp_output",
                        lambda scanner, path, output_dir, **kwargs: {
                            "output_path": output_dir / "synthetic_scanned.jpg",
                            "corners": np.asarray(SUGGESTED, np.float32),
                        })
    stage2.Stage2ScannerUI._process_worker(fake, "color")
    assert observed["debug"] == []
