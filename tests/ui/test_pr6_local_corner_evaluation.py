"""Tk read-only PR6 error summary tests, with anonymous synthetic geometry."""
import os
import tkinter as tk
import pytest

from autodocscanner.core.pr6_local_corner_evaluation import (
    evaluate_pr6_candidates_against_manual as evaluate,
)
from autodocscanner.ui.pr6_local_evaluation_dialog import (
    format_local_pr6_evaluation, open_pr6_local_evaluation_dialog,
)


def _result():
    points = [[80, 50], [230, 50], [230, 150], [80, 150]]
    snapshot = {
        "status": "offline_visual_diagnostic_only",
        "image_shape": [200, 320],
        "auto_select": None, "auto_crop_allowed": False,
        "candidates": [
            {"source": "stable", "family": "Stable", "corners": points},
        ],
    }
    return evaluate(snapshot, points, (200, 320))


def test_text_discloses_that_best_is_post_annotation_oracle():
    result = format_local_pr6_evaluation(_result())
    assert "ORACLE" in result and "BUKAN pemilihan otomatis" in result
    assert "0.00%" in result and "stable" in result
    assert "Tidak ada gambar, koordinat acuan" in result


def test_invalid_evaluation_is_never_displayed():
    result = _result()
    result["auto_crop_allowed"] = True
    with pytest.raises(ValueError):
        format_local_pr6_evaluation(result)


def test_real_tk_dialog_displays_results_without_saving_or_changing_source():
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("Display or Xvfb required")
    root = tk.Tk()
    root.withdraw()
    dialog = tk.Toplevel(root)
    dialog.grab_set()
    try:
        popup = open_pr6_local_evaluation_dialog(dialog, _result())
        popup.update()
        def texts(widget):
            found = []
            for child in widget.winfo_children():
                if isinstance(child, tk.Text):
                    found.append(child)
                found.extend(texts(child))
            return found
        views = texts(popup)
        assert len(views) == 1
        assert "stable" in views[0].get("1.0", "end")
        assert str(views[0].cget("state")) == "disabled"
        assert popup.grab_current() == popup
        popup.destroy()
    finally:
        dialog.destroy()
        root.destroy()



def test_report_shows_per_corner_named_errors_and_generator_counts():
    snapshot = {
        "status": "offline_visual_diagnostic_only",
        "image_shape": [200, 320],
        "auto_select": None, "auto_crop_allowed": False,
        "generated_side_recovery_count": 4,
        "generated_paired_alignment_count": 0,
        "independent_sides_count": 0,
        "paired_alignment_parents_examined": 4,
        "independent_sides_parents_examined": 4,
        "source_hypotheses_hidden": 2,
        "candidates": [{
            "source": "v2_3", "family": "Outer Region",
            "corners": [[85, 50], [230, 50], [230, 150], [80, 150]],
        }],
    }
    truth = [[80, 50], [230, 50], [230, 150], [80, 150]]
    text = format_local_pr6_evaluation(evaluate(snapshot, truth, (200, 320)))
    assert "Kiri-atas" in text and "5.00 px" in text
    assert "dx +5.00" in text
    assert "Paired Alignment: 0" in text
    assert "Independent Sides: 0" in text
    assert "2 hipotesis tidak masuk tampilan" in text



def test_read_only_evaluation_explains_skipped_off_photo_independent_sides():
    points = [[80, 50], [230, 50], [230, 150], [80, 150]]
    snap = {
        "status": "offline_visual_diagnostic_only",
        "image_shape": [200, 320],
        "auto_crop_allowed": False, "auto_select": None,
        "candidates": [
            {"source": "stable", "family": "Stable", "corners": points},
            {"source": "independent_sides_0", "family": "Independent Sides",
             "corners": [[-4, 50], [230, 50], [230, 150], [80, 150]]},
        ],
    }
    summary = format_local_pr6_evaluation(
        evaluate(snap, points, (200, 320))
    )
    assert "Kandidat terukur: 1; dilewati: 1" in summary
    assert "ALASAN KANDIDAT DILEWATI" in summary
    assert "independent_sides_0" in summary
    assert "Ada sudut di luar foto asli" in summary
    assert "KANDIDAT DI LUAR FOTO" in summary
    assert "BUKAN CROP" in summary
    assert "Independent Sides" in summary
    assert "0.00%" in summary
