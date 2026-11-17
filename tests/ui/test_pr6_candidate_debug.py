"""Read-only PR6 candidate overlay GUI tests with anonymous synthetic pixels."""
import os
import tkinter as tk
from tkinter import ttk

from PIL import Image
import pytest

from autodocscanner.ui.pr6_candidate_debug import (
    TONE_GOOD, TONE_UNKNOWN, TONE_WEAK,
    _side_tone, draw_pr6_candidate_overlay,
    format_pr6_candidate_details,
    open_pr6_candidate_debug_dialog,
)


def _diag():
    return {
        "status": "offline_visual_diagnostic_only",
        "image_shape": [240, 360],
        "candidates": [
            {
                "source": "stable", "family": "Stable",
                "corners": [[20, 30], [340, 20], [335, 215], [25, 212]],
                "ranking_score": .81, "geometry_valid": True,
                "selection_representative": "stable",
                "side_evidence": [
                    {
                        "external_support": .88, "local_coverage": .91,
                        "median_offset_fraction": .08, "offset_spread": .11,
                        "inner_outer_contrast": 10.0, "contrast_coverage": .9,
                    },
                    {}, {}, {},
                ],
                "candidate_warnings": ["weak_inner_outer_separation"],
                "ranking_warnings": [],
                "assessment_reasons": ["incomplete_alignment"],
            },
            {
                "source": "v2_0", "family": "Four-Line",
                "corners": [[32, 33], [310, 36], [312, 195], [35, 193]],
                "ranking_score": .60, "geometry_valid": True,
                "selection_representative": "v2_0",
                "side_evidence": [{}, {}, {}, {}],
                "candidate_warnings": ["poor_edge_alignment"],
                "ranking_warnings": ["weak_side"],
                "assessment_reasons": [],
            },
        ],
        "review_decision": "manual_review",
        "decision_reason_codes": [
            "pr6_candidate_decision_shadow:requires_manual_review"
        ],
        "auto_select": None, "auto_crop_allowed": False,
    }


def test_overlay_draws_distinct_hypothesized_sides_without_mutating_source():
    source = Image.new("RGB", (360, 240), "#B7C6D8")
    before = source.tobytes()
    overlay, offset = draw_pr6_candidate_overlay(
        source, _diag()["candidates"][0], canvas_size=(600, 420)
    )
    assert overlay.size == (600, 420)
    assert overlay.tobytes() != before
    assert source.tobytes() == before
    assert offset[0] >= 0 and offset[1] >= 0


def test_overlay_preserves_portrait_mapping_and_image_dimensions():
    source = Image.new("RGB", (240, 360), "#CBD5E1")
    row = _diag()["candidates"][0].copy()
    row["corners"] = [[20, 30], [216, 25], [219, 325], [22, 321]]
    output, _ = draw_pr6_candidate_overlay(source, row, canvas_size=(420, 550))
    assert output.size == (420, 550)


def test_side_tones_are_diagnostic_only_not_confidence():
    good = _diag()["candidates"][0]["side_evidence"][0]
    assert _side_tone(good) == TONE_GOOD
    weak = dict(good, external_support=.05)
    assert _side_tone(weak) == TONE_WEAK
    assert _side_tone({}) == TONE_UNKNOWN


def test_details_show_four_side_metrics_and_refusal_without_personal_info():
    diag = _diag()
    result = format_pr6_candidate_details(diag["candidates"][0], diag)
    assert "Atas" in result and "Bawah" in result
    assert "Kontras dalam/luar" in result
    assert "weak_inner_outer_separation" in result
    assert "pr6_candidate_decision_shadow:requires_manual_review" in result
    assert "bukan probabilitas" in result
    assert "tidak ada crop" in result


@pytest.mark.parametrize("points", [
    None, [],
    [[1, 2], [3, 4]],
    [[1, 2], [3, 4], [5, 6], [float("nan"), 2]],
    [[1, 2], [3, 4], [5, 6], [10000, 4]],
])
def test_invalid_points_are_never_drawn_as_verified_boundaries(points):
    source = Image.new("RGB", (360, 240), "#B7C6D8")
    candidate = _diag()["candidates"][0].copy()
    candidate["corners"] = points
    with pytest.raises(ValueError):
        draw_pr6_candidate_overlay(source, candidate)


def _tk_root():
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("Desktop display or Xvfb required")
    return tk.Tk()


def _children(widget, target):
    found = []
    for child in widget.winfo_children():
        if isinstance(child, target):
            found.append(child)
        found.extend(_children(child, target))
    return found


def test_real_tk_visual_dialog_switches_candidates_and_never_writes(
        tmp_path):
    path = tmp_path / "anonymous_synthetic.png"
    Image.new("RGB", (360, 240), "#B7C6D8").save(path)
    before = path.read_bytes()
    root = _tk_root()
    root.withdraw()
    try:
        popup = open_pr6_candidate_debug_dialog(root, path, _diag())
        popup.update()
        pickers = _children(popup, ttk.Combobox)
        assert len(pickers) == 1
        picker = pickers[0]
        assert len(picker["values"]) == 2
        texts = _children(popup, tk.Text)
        assert len(texts) == 1
        assert "Stable" in texts[0].get("1.0", "end")
        picker.current(1)
        picker.event_generate("<<ComboboxSelected>>")
        popup.update()
        assert "Four-Line" in texts[0].get("1.0", "end")
        assert "poor_edge_alignment" in texts[0].get("1.0", "end")
        assert path.read_bytes() == before
        popup.destroy()
    finally:
        root.destroy()


def test_real_tk_no_candidate_dialog_has_manual_fallback(tmp_path):
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (360, 240), "#B7C6D8").save(path)
    diagnostic = _diag()
    diagnostic["candidates"] = []
    root = _tk_root()
    root.withdraw()
    try:
        popup = open_pr6_candidate_debug_dialog(root, path, diagnostic)
        popup.update()
        texts = _children(popup, tk.Text)
        assert "Koreksi Manual" in texts[0].get("1.0", "end")
        popup.destroy()
    finally:
        root.destroy()


def test_real_tk_refuses_orientation_mismatch(tmp_path):
    path = tmp_path / "synthetic.png"
    Image.new("RGB", (240, 360), "#B7C6D8").save(path)
    root = _tk_root()
    root.withdraw()
    try:
        with pytest.raises(ValueError):
            open_pr6_candidate_debug_dialog(root, path, _diag())
    finally:
        root.destroy()
