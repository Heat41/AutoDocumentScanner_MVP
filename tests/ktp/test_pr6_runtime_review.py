"""Pure PR6 runtime review tests. Synthetic pixels and geometry only."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from autodocscanner.core.pr6_runtime_review import (
    RUNTIME_FLAG,
    experimental_pr6_runtime_enabled,
    evaluate_pr6_runtime_image,
    evaluate_pr6_runtime_file,
)


IMAGE = np.full((300, 400, 3), 125, dtype=np.uint8)
CORNERS = [[30.0, 30.0], [350.0, 30.0],
           [350.0, 260.0], [30.0, 260.0]]


def _report(*, preview=True, source="stable"):
    action = "preview_suggested_corners" if preview else "manual_corners"
    status = "preview_review" if preview else "manual_review"
    return {
        "stable": {"corners": deepcopy(CORNERS)},
        "adaptive": {"corners": deepcopy(CORNERS)},
        "v2": [{"corners": deepcopy(CORNERS)}],
        "pr6_review_consolidation_shadow": {
            "status": status,
            "action": action,
            "mode": "offline_shadow_review_only",
            "candidate_source": source if preview else None,
            "reason_codes": [] if preview else ["boundary_evidence_incomplete"],
            "physical_boundary_verified": False,
            "auto_select": None,
            "auto_crop_allowed": False,
            "requires_explicit_confirmation": True,
        },
        "pr6_manual_fallback_shadow": {
            "status": status,
            "action": action,
            "status_scope": "offline_diagnostic_only",
            "ui_integration_status": "not_connected_shadow_only",
            "candidate_source": source if preview else None,
            "suggested_corners": deepcopy(CORNERS) if preview else None,
            "source_image_shape": [300, 400],
            "source_image_policy": "unmodified_original_image_required",
            "auto_select": None,
            "auto_crop_allowed": False,
            "file_write_performed": False,
            "requires_explicit_confirmation": True,
        },
    }


def test_opt_in_is_exact_and_not_activated_by_manual_magnifier_flag():
    assert not experimental_pr6_runtime_enabled({})
    assert not experimental_pr6_runtime_enabled({
        "AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE": "1"
    })
    assert not experimental_pr6_runtime_enabled({RUNTIME_FLAG: "yes"})
    assert experimental_pr6_runtime_enabled({RUNTIME_FLAG: "1"})


def test_preview_supplies_only_original_four_corners_and_no_crop():
    report = _report()
    before = deepcopy(report)
    image_before = IMAGE.copy()
    result = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda image, *, expected=None: report
    )
    assert result["status"] == "preview_review"
    assert result["action"] == "open_four_corner_editor"
    assert result["suggested_corners"] == CORNERS
    assert result["candidate_source"] == "stable"
    assert result["auto_select"] is None
    assert result["auto_crop_allowed"] is False
    assert result["requires_explicit_confirmation"] is True
    assert result["file_write_performed"] is False
    assert result["crop_performed"] is False
    assert report == before
    assert np.array_equal(IMAGE, image_before)


def test_analyzer_never_receives_ground_truth():
    observed = []
    def checker(image, *, expected=None):
        observed.append(expected)
        return _report(preview=False)
    evaluate_pr6_runtime_image(IMAGE, analyzer=checker)
    assert observed == [None]


def test_manual_review_disposes_of_all_suggested_corners():
    result = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_args, **_kw: _report(preview=False)
    )
    assert result["status"] == "manual_review"
    assert result["suggested_corners"] is None
    assert result["candidate_source"] is None
    assert result["reason_codes"] == ["boundary_evidence_incomplete"]


def test_v2_candidate_is_supported_for_preview():
    result = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_args, **_kw: _report(source="v2_0")
    )
    assert result["status"] == "preview_review"
    assert result["candidate_source"] == "v2_0"


def test_preview_never_uses_proposed_corners_different_from_original_candidate():
    report = _report()
    report["pr6_manual_fallback_shadow"]["suggested_corners"][0][0] += 18
    result = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_a, **_kw: report
    )
    assert result["status"] == "manual_review"
    assert result["suggested_corners"] is None


def test_preview_mismatch_gate_vs_plan_source_fails_closed():
    report = _report()
    report["pr6_manual_fallback_shadow"]["candidate_source"] = "adaptive"
    result = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_a, **_kw: report
    )
    assert result["status"] == "manual_review"


def test_pretending_preview_without_verified_quad_fails_closed():
    report = _report()
    report["stable"]["corners"] = [[0, 0]] * 4
    result = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_a, **_kw: report
    )
    assert result["status"] == "manual_review"


@pytest.mark.parametrize("section", [
    "pr6_review_consolidation_shadow", "pr6_manual_fallback_shadow",
])
def test_missing_gate_section_rejected(section):
    report = _report()
    report.pop(section)
    with pytest.raises(ValueError):
        evaluate_pr6_runtime_image(
            IMAGE, analyzer=lambda *_a, **_kw: report
        )


@pytest.mark.parametrize("path,value", [
    (("pr6_review_consolidation_shadow", "auto_crop_allowed"), True),
    (("pr6_manual_fallback_shadow", "auto_crop_allowed"), True),
    (("pr6_manual_fallback_shadow", "file_write_performed"), True),
    (("pr6_review_consolidation_shadow", "auto_select"), "stable"),
    (("pr6_manual_fallback_shadow", "auto_select"), "stable"),
    (("pr6_review_consolidation_shadow", "physical_boundary_verified"), True),
    (("pr6_manual_fallback_shadow", "requires_explicit_confirmation"), False),
    (("pr6_review_consolidation_shadow", "mode"), "production_auto_crop"),
    (("pr6_manual_fallback_shadow", "source_image_shape"), [400, 300]),
])
def test_unsafe_contract_cannot_yield_unconfirmed_preview(path, value):
    report = _report()
    report[path[0]][path[1]] = value
    result = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_a, **_kw: report
    )
    assert result["status"] == "manual_review"
    assert result["auto_crop_allowed"] is False


def test_preview_has_unresolved_reasons_never_promotes():
    report = _report()
    report["pr6_review_consolidation_shadow"]["reason_codes"] = [
        "unresolved_nested_geometry"
    ]
    result = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_a, **_kw: report
    )
    assert result["status"] == "manual_review"
    assert result["suggested_corners"] is None


@pytest.mark.parametrize("image", [
    None, np.zeros((4, 4, 3), dtype=np.uint8),
    np.zeros((300, 400), dtype=np.uint8),
    np.zeros((300, 400, 4), dtype=np.uint8),
    np.zeros((300, 400, 3), dtype=np.float32),
])
def test_invalid_image_input_rejected(image):
    with pytest.raises(ValueError):
        evaluate_pr6_runtime_image(
            image, analyzer=lambda *_a, **_kw: _report()
        )


def test_eval_file_no_output_writes(tmp_path):
    input_path = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(input_path), IMAGE)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = evaluate_pr6_runtime_file(
        input_path, analyzer=lambda *_a, **_kw: _report(preview=False)
    )
    assert result["status"] == "manual_review"
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_nonexistent_source_file_does_not_pretend_success(tmp_path):
    with pytest.raises(ValueError):
        evaluate_pr6_runtime_file(tmp_path / "not_there.png")


def test_annotations_and_detector_quality_are_not_runtime_authority():
    report = _report(preview=False)
    before = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_a, **_kw: report
    )
    report["expected"] = [[0, 0]] * 4
    report["normalized_corner_error"] = 0
    report["pr6_multi_refinement_ground_truth_evaluation"] = {
        "accepted": True,
    }
    report["stable"]["warp_quality"] = 99
    after = evaluate_pr6_runtime_image(
        IMAGE, analyzer=lambda *_a, **_kw: report
    )
    assert before == after
