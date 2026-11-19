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



def test_side_recovery_details_show_changed_side_and_no_automatic_crop():
    diagnostic = _diag()
    candidate = dict(diagnostic["candidates"][0])
    candidate["source"] = "side_recovery_0"
    candidate["family"] = "Side Recovery"
    candidate["ranking_score"] = None
    candidate["side_recovery_details"] = {
        "parent_source": "v2_6",
        "changed_side_index": 0,
        "shift_fraction_of_parent_gap": .36,
        "side_score_before": 3.4,
        "side_score_after": 29.2,
    }
    rendered = format_pr6_candidate_details(candidate, diagnostic)
    assert "Side Recovery" in rendered
    assert "Kandidat asal: v2_6" in rendered
    assert "Sisi yang dicari ulang: 1 dari 4" in rendered
    assert "bukan probabilitas" in rendered
    assert "tidak ikut ranking, Review Gate, atau auto-crop" in rendered
    assert "tidak ada crop" in rendered



def test_zero_side_recovery_message_is_visible_in_real_tk(tmp_path):
    image_path = tmp_path / "anonymous_synthetic.png"
    Image.new("RGB", (360, 240), "#B7C6D8").save(image_path)
    report = _diag()
    report.update({
        "generated_side_recovery_count": 0,
        "side_recovery_parents_examined": 10,
        "side_recovery_reason_codes": [
            "no_side_shift_passed_visual_evidence_and_geometry"
        ],
    })
    root = _tk_root()
    root.withdraw()
    try:
        popup = open_pr6_candidate_debug_dialog(root, image_path, report)
        popup.update()
        labels = [str(x.cget("text")) for x in _children(popup, ttk.Label)]
        assert any(
            "10 kandidat asal diperiksa" in x
            and "belum ditemukan perpindahan" in x
            for x in labels
        )
        popup.destroy()
    finally:
        root.destroy()



def test_paired_alignment_details_identify_both_sides_without_auto_crop():
    diagnostic = _diag()
    candidate = dict(diagnostic["candidates"][0])
    candidate["family"] = "Paired Alignment"
    candidate["source"] = "paired_alignment_0"
    candidate["ranking_score"] = None
    candidate["paired_alignment_details"] = {
        "parent_source": "side_recovery_3",
        "outer_region_source": "v2_7",
        "paired_side_indices": [0, 2],
        "shift_fraction_of_pair_gap": -.13,
        "translation_original_xy": [0, -26],
        "side_scores_before": [1.5, 2.1],
        "side_scores_after": [27.2, 30.8],
    }
    rendered = format_pr6_candidate_details(candidate, diagnostic)
    assert "PENYESUAIAN DUA SISI" in rendered
    assert "side_recovery_3" in rendered
    assert "v2_7" in rendered
    assert "1 dan 3 dari 4" in rendered
    assert "Bukti kedua tepi" in rendered
    assert "KEDUA sisi harus menunjukkan peningkatan" in rendered
    assert "TIDAK ikut ranking, Review Gate, atau auto-crop" in rendered


def test_real_tk_no_paired_alignment_explains_two_side_evidence(tmp_path):
    path = tmp_path / "anonymous_synthetic.png"
    Image.new("RGB", (360, 240), "#B7C6D8").save(path)
    report = _diag()
    report.update({
        "generated_paired_alignment_count": 0,
        "paired_alignment_parents_examined": 4,
        "paired_alignment_reason_codes": ["no_opposite_sides_both_improved"],
    })
    root = _tk_root()
    root.withdraw()
    try:
        popup = open_pr6_candidate_debug_dialog(root, path, report)
        popup.update()
        labels = [str(w.cget("text")) for w in _children(popup, ttk.Label)]
        assert any(
            "4 kandidat diperiksa" in text
            and "memperkuat dua sisi berhadapan" in text
            for text in labels
        )
        popup.destroy()
    finally:
        root.destroy()



def test_missing_paired_alignment_explains_blocking_edge_for_selected_source():
    diagnostic = _diag()
    diagnostic["paired_alignment_parent_checks"] = [{
        "source": "side_recovery_3",
        "status": "rejected",
        "failure_reason": "no_opposite_sides_both_improved",
        "paired_side_indices": [0, 2],
        "attempts_evaluated": 60,
        "best_attempt": {
            "shift_fraction_of_pair_gap": -.14,
            "side_scores_before": [8.0, 14.0],
            "side_scores_after": [20.0, 17.0],
            "required_side_scores": [15.0, 21.0],
            "blocking_side_index": 2,
            "both_pass": False,
        },
    }]
    candidate = dict(diagnostic["candidates"][0])
    candidate["source"] = "side_recovery_3"
    candidate["family"] = "Side Recovery"
    candidate["side_recovery_details"] = {
        "parent_source": "v2_7",
        "changed_side_index": 0,
    }
    explanation = format_pr6_candidate_details(candidate, diagnostic)
    assert "ALASAN PAIRED ALIGNMENT" in explanation
    assert "Jumlah perpindahan diperiksa: 60" in explanation
    assert "Atas: 8.00 → 20.00; minimum 15.00" in explanation
    assert "Bawah: 14.00 → 17.00; minimum 21.00" in explanation
    assert "Sisi yang paling menghambat: Bawah" in explanation
    assert "bukan ketepatan sudut" in explanation


def test_missing_paired_alignment_diagnostics_not_wrongly_shown_for_other_source():
    diagnostic = _diag()
    diagnostic["paired_alignment_parent_checks"] = [{
        "source": "side_recovery_3",
        "failure_reason": "no_opposite_sides_both_improved",
        "attempts_evaluated": 60,
    }]
    text = format_pr6_candidate_details(
        diagnostic["candidates"][0], diagnostic,
    )
    assert "ALASAN PAIRED ALIGNMENT" not in text



def test_independent_sides_details_show_both_independent_offsets_and_no_auto_crop():
    report = _diag()
    candidate = dict(report["candidates"][0])
    candidate["source"] = "independent_sides_0"
    candidate["family"] = "Independent Sides"
    candidate["ranking_score"] = None
    candidate["independent_sides_details"] = {
        "parent_source": "side_recovery_3",
        "outer_region_source": "v2_7",
        "paired_side_indices": [0, 2],
        "independent_offset_fractions": [-.13, -.24],
        "side_scores_before": [11.84, 0.0],
        "side_scores_after": [20.0, 14.0],
    }
    text = format_pr6_candidate_details(candidate, report)
    assert "DUA SISI DIPERBAIKI TERPISAH" in text
    assert "side_recovery_3" in text and "v2_7" in text
    assert "Atas: perpindahan -0.13" in text
    assert "Bawah: perpindahan -0.24" in text
    assert "bukan probabilitas" in text
    assert "Tidak ikut ranking, Review Gate, atau auto-crop" in text


def test_independent_sides_rejection_reports_edge_peaks_for_its_parent():
    report = _diag()
    report["independent_sides_parent_checks"] = [{
        "source": "side_recovery_3",
        "status": "rejected",
        "failure_reason": "unsupported_boundary_for_one_or_both_sides",
        "side_checks": [
            {
                "side_index": 0, "score_before": 11.84,
                "strongest_score": 13.5, "required_score": 18.84,
                "strongest_offset_fraction": -.15,
            },
            {
                "side_index": 2, "score_before": 0,
                "strongest_score": 3.14, "required_score": 11.0,
                "strongest_offset_fraction": .21,
            },
        ],
    }]
    candidate = dict(report["candidates"][0])
    candidate["source"] = "side_recovery_3"
    candidate["side_recovery_details"] = {"parent_source": "v2_7"}
    explanation = format_pr6_candidate_details(candidate, report)
    assert "PEMERIKSAAN INDEPENDENT SIDES" in explanation
    assert "Atas: bukti awal 11.84" in explanation
    assert "Bawah: bukti awal 0.00" in explanation
    assert "puncak 3.14" in explanation
    assert "Puncak bukti tidak otomatis menjadi kandidat crop" in explanation


def test_independent_sides_rejection_not_attributed_to_wrong_parent():
    report = _diag()
    report["independent_sides_parent_checks"] = [{
        "source": "side_recovery_3",
        "status": "rejected", "side_checks": [],
    }]
    candidate = dict(report["candidates"][0])
    candidate["source"] = "side_recovery_2"
    candidate["side_recovery_details"] = {"parent_source": "v2_7"}
    assert "PEMERIKSAAN INDEPENDENT SIDES" not in (
        format_pr6_candidate_details(candidate, report)
    )


def test_real_tk_independent_sides_zero_message(tmp_path):
    path = tmp_path / "anonymous_synthetic.png"
    Image.new("RGB", (360, 240), "#B7C6D8").save(path)
    report = _diag()
    report.update({
        "independent_sides_count": 0,
        "independent_sides_parents_examined": 4,
        "independent_sides_reason_codes": [
            "no_independent_two_side_hypothesis_passed"
        ],
    })
    root = _tk_root()
    root.withdraw()
    try:
        popup = open_pr6_candidate_debug_dialog(root, path, report)
        popup.update()
        labels = [str(w.cget("text")) for w in _children(popup, ttk.Label)]
        assert any(
            "Independent Sides: 4 kandidat awal diperiksa" in label
            for label in labels
        )
        popup.destroy()
    finally:
        root.destroy()



def test_outer_region_shows_both_direct_side_checks():
    report = _diag()
    candidate = dict(report["candidates"][0])
    candidate["family"] = "Outer Region"
    candidate["source"] = "v2_2"
    report["independent_sides_parent_checks"] = [
        {
            "source": "v2_2", "parent_family": "outer_region",
            "paired_side_indices": [0, 2],
            "status": "rejected",
            "failure_reason": "unsupported_boundary_for_one_or_both_sides",
            "side_checks": [{
                "side_index": 0, "score_before": 8,
                "strongest_score": 12, "required_score": 15,
            }],
        },
        {
            "source": "v2_2", "parent_family": "outer_region",
            "paired_side_indices": [1, 3],
            "status": "rejected", "side_checks": [],
        },
    ]
    details = format_pr6_candidate_details(candidate, report)
    assert "INDEPENDENT SIDES LANGSUNG DARI OUTER REGION" in details
    assert "Atas + Bawah" in details
    assert "Kanan + Kiri" in details
    assert "puncak 12.00" in details
    assert "syarat pindah 15.00" in details
    assert "izin crop" in details



def test_held_independent_side_is_explicitly_labelled_in_review():
    report = _diag()
    candidate = dict(report["candidates"][0])
    candidate["family"] = "Independent Sides"
    candidate["source"] = "independent_sides_0"
    candidate["independent_sides_details"] = {
        "parent_source": "v2_2",
        "outer_region_source": "v2_2",
        "paired_side_indices": [0, 2],
        "anchored_side_indices": [0],
        "refinement_kind": "one_side_held_one_shifted",
        "independent_offset_fractions": [0., -.11],
        "side_scores_before": [15., 1.],
        "side_scores_after": [14., 18.],
    }
    description = format_pr6_candidate_details(candidate, report)
    assert "sisi lawannya digeser" in description
    assert "(DITAHAN)" in description
    assert "tidak ikut ranking" in description or "Tidak ikut ranking" in description
