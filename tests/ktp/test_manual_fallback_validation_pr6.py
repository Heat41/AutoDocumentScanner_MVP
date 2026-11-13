"""PR6 fallback contract; exclusively synthetic geometry, no private images."""
from copy import deepcopy

import numpy as np
import pytest

from autodocscanner.core.manual_fallback_validation_pr6 import (
    prepare_manual_fallback_pr6, evaluate_manual_confirmation_pr6,
)


SHAPE = (300, 400, 3)
QUAD = [[30, 30], [350, 30], [350, 260], [30, 260]]


def base(status="manual_review"):
    return {
        "stable": {"corners": deepcopy(QUAD)},
        "pr6_review_consolidation_shadow": {
            "status": status,
            "action": "manual_corners" if status == "manual_review" else "preview_suggested_corners",
            "candidate_source": None if status == "manual_review" else "stable",
            "preview_geometry_origin": None if status == "manual_review" else "original_detector_candidate",
            "reason_codes": ["pr6_candidate_decision_shadow:requires_manual_review"] if status == "manual_review" else [],
            "fallback_editor": "existing_manual_four_corners",
            "requires_explicit_confirmation": True,
            "auto_select": None,
            "auto_crop_allowed": False,
            "mode": "offline_shadow_review_only",
        },
        "v3_manual_handoff": {
            "action": "preview_suggested_corners",
            "auto_crop_allowed": False,
            "requires_explicit_confirmation": True,
            "suggested_corners": deepcopy(QUAD),
        },
    }


def test_manual_suppresses_legacy_preview_and_stale_corners():
    report = base()
    old = deepcopy(report)
    plan = prepare_manual_fallback_pr6(report, SHAPE)
    assert plan["status"] == "manual_review"
    assert plan["suggested_corners"] is None
    assert plan["candidate_source"] is None
    assert "legacy_v3_preview_differs_from_consolidated_manual" in plan["ui_gap_codes"]
    assert plan["auto_crop_allowed"] is False
    assert report == old


def test_preview_uses_only_original_detector_geometry():
    plan = prepare_manual_fallback_pr6(base("preview_review"), SHAPE)
    assert plan["status"] == "preview_review"
    assert plan["suggested_corners"] == QUAD
    assert plan["auto_select"] is None
    assert plan["file_write_performed"] is False


def test_preview_result_copied_not_referencing_report():
    report = base("preview_review")
    result = prepare_manual_fallback_pr6(report, SHAPE)
    result["suggested_corners"][0][0] = 40
    assert report["stable"]["corners"] == QUAD


def test_preview_rejects_mismatched_handoff():
    report = base("preview_review")
    report["v3_manual_handoff"]["suggested_corners"][0][0] += 20
    plan = prepare_manual_fallback_pr6(report, SHAPE)
    assert plan["status"] == "manual_review"
    assert plan["suggested_corners"] is None
    assert "handoff_does_not_match_original_candidate" in plan["reason_codes"]


@pytest.mark.parametrize("corners", [None, [], [[1, 2]], [[1, 2]] * 4,
                                      [[30, 30], [350, 30], [350, 260], [float("nan"), 260]]])
def test_invalid_original_preview_corners_block(corners):
    report = base("preview_review")
    report["stable"]["corners"] = corners
    assert prepare_manual_fallback_pr6(report, SHAPE)["status"] == "manual_review"


def test_wrong_candidate_source_blocks_preview():
    data = base("preview_review")
    data["pr6_review_consolidation_shadow"]["candidate_source"] = "v2_40"
    plan = prepare_manual_fallback_pr6(data, SHAPE)
    assert "missing_original_candidate" in plan["reason_codes"]


def test_read_v2_original_candidate_preview():
    data = base("preview_review")
    data["pr6_review_consolidation_shadow"]["candidate_source"] = "v2_0"
    data["v2"] = [{"corners": deepcopy(QUAD)}]
    plan = prepare_manual_fallback_pr6(data, SHAPE)
    assert plan["status"] == "preview_review"
    assert plan["candidate_source"] == "v2_0"


def test_preview_requires_consistent_origin_and_handoff():
    for case in ("preview_geometry_origin", "action"):
        data = base("preview_review")
        data["pr6_review_consolidation_shadow"][case] = "invalid"
        assert prepare_manual_fallback_pr6(data, SHAPE)["status"] == "manual_review"


def test_manual_never_promotes_missing_gate():
    plan = prepare_manual_fallback_pr6({}, SHAPE)
    assert plan["status"] == "manual_review"
    assert "missing_or_invalid_consolidated_gate" in plan["reason_codes"]
    assert plan["suggested_corners"] is None


def test_manual_no_candidate_still_has_review_instruction():
    data = base()
    del data["stable"]
    plan = prepare_manual_fallback_pr6(data, SHAPE)
    assert plan["action"] == "manual_corners"
    assert plan["manual_corner_entry"] == "user_must_supply_four_original_image_points"
    assert "auto_ktp_manual_editor_requires_existing_detector_corners" in plan["ui_gap_codes"]
    assert plan["ui_integration_status"] == "not_connected_shadow_only"


def test_invalid_image_shape_cannot_confirm():
    for shape in ((), (0, 500), (300, -1), (True, 500), (500,), (None, 400)):
        plan = prepare_manual_fallback_pr6(base(), shape)
        assert "invalid_source_image_shape" in plan["reason_codes"]
        assert evaluate_manual_confirmation_pr6(plan, shape, action="confirm", corners=QUAD)["status"] == "rejected"


def test_cancel_does_not_allow_write_even_if_bad_corners():
    plan = prepare_manual_fallback_pr6(base(), SHAPE)
    before = deepcopy(plan)
    cancelled = evaluate_manual_confirmation_pr6(plan, SHAPE, action="cancel", corners=[[1, 2]])
    assert cancelled["status"] == "cancelled"
    assert cancelled["approved_corners"] is None
    assert cancelled["file_write_performed"] is False
    assert cancelled["crop_performed"] is False
    assert plan == before


def test_explicit_confirm_returns_validated_corners_not_executed_crop():
    plan = prepare_manual_fallback_pr6(base(), SHAPE)
    result = evaluate_manual_confirmation_pr6(plan, SHAPE, action="confirm", corners=QUAD)
    assert result["status"] == "validated_for_manual_apply"
    assert result["approved_corners"] == QUAD
    assert result["file_write_performed"] is False
    assert result["crop_performed"] is False
    assert result["requires_existing_ui_confirmation"] is True


def test_preview_confirmation_requires_explicit_points_not_auto_apply():
    plan = prepare_manual_fallback_pr6(base("preview_review"), SHAPE)
    assert evaluate_manual_confirmation_pr6(plan, SHAPE, action="confirm")["status"] == "rejected"
    result = evaluate_manual_confirmation_pr6(plan, SHAPE, action="confirm", corners=QUAD)
    assert result["status"] == "validated_for_manual_apply"


@pytest.mark.parametrize("corners", [None, [], [[1, 2], [3, 4]],
                                      [[30, 30], [350, 30], [350, 260], [float("inf"), 260]],
                                      [[30, 30], [350, 260], [350, 30], [30, 260]],
                                      [[30, 30], [30, 260], [350, 260], [350, 30]],
                                      [[30, 30], [350, 30], [350, 260], [-2, 260]],
                                      [[30, 30], [350, 30], [350, 260], [30, 999]]])
def test_invalid_manual_submit_fails_closed(corners):
    plan = prepare_manual_fallback_pr6(base(), SHAPE)
    result = evaluate_manual_confirmation_pr6(plan, SHAPE, action="confirm", corners=corners)
    assert result["status"] == "rejected"
    assert result["approved_corners"] is None


def test_confirmation_must_match_source_image_shape():
    plan = prepare_manual_fallback_pr6(base(), SHAPE)
    assert evaluate_manual_confirmation_pr6(plan, (600, 800), action="confirm", corners=QUAD)["status"] == "rejected"


def test_invalid_action_is_not_confirmation():
    plan = prepare_manual_fallback_pr6(base(), SHAPE)
    for action in (None, "approve", "auto_crop", "save", "preview"):
        assert evaluate_manual_confirmation_pr6(plan, SHAPE, action=action, corners=QUAD)["status"] == "rejected"


def test_tampered_plan_cannot_authorize_crop():
    for key, value in (("auto_crop_allowed", True), ("requires_explicit_confirmation", False),
                       ("ui_integration_status", "production_connected"),
                       ("editor", "auto_crop")):
        plan = prepare_manual_fallback_pr6(base(), SHAPE)
        plan[key] = value
        assert evaluate_manual_confirmation_pr6(plan, SHAPE, action="confirm", corners=QUAD)["status"] == "rejected"


def test_portrait_and_landscape_manual_corners():
    for shape, quad in (((300, 400), QUAD), ((400, 300), [[30, 30], [260, 30], [260, 350], [30, 350]])):
        plan = prepare_manual_fallback_pr6(base(), shape)
        result = evaluate_manual_confirmation_pr6(plan, shape, action="confirm", corners=quad)
        assert result["status"] == "validated_for_manual_apply"


def test_frame_adjacent_manual_quad_allowed_if_inside():
    shape = (300, 400)
    corners = [[0, 0], [399, 0], [399, 299], [0, 299]]
    plan = prepare_manual_fallback_pr6(base(), shape)
    assert evaluate_manual_confirmation_pr6(plan, shape, action="confirm", corners=corners)["status"] == "validated_for_manual_apply"


def test_ground_truth_changes_do_not_affect_plan_or_acceptance():
    data = base()
    original = prepare_manual_fallback_pr6(data, SHAPE)
    data["normalized_corner_error"] = 0.999
    data["expected"] = [[0, 0], [3, 0], [3, 3], [0, 3]]
    data["pr6_multi_refinement_ground_truth_evaluation"] = {"score": 0}
    assert prepare_manual_fallback_pr6(data, SHAPE) == original


def test_fallback_has_no_io_or_user_image_data():
    plan = prepare_manual_fallback_pr6(base(), SHAPE)
    assert plan["file_write_performed"] is False
    assert plan["source_image_policy"] == "unmodified_original_image_required"
    assert plan["source_image_shape"] == [300, 400]
    assert "image_bytes" not in plan and "output_path" not in plan


def test_manual_retains_consolidated_blocker_reasons():
    plan = prepare_manual_fallback_pr6(base(), SHAPE)
    assert plan["consolidated_gate_reason_codes"] == [
        "pr6_candidate_decision_shadow:requires_manual_review"]


def test_consolidated_preview_with_blockers_fails_closed():
    data = base("preview_review")
    data["pr6_review_consolidation_shadow"]["reason_codes"] = ["conflicting_geometry"]
    plan = prepare_manual_fallback_pr6(data, SHAPE)
    assert plan["status"] == "manual_review"
    assert "consolidated_preview_has_blockers" in plan["reason_codes"]


def test_requesting_manual_action_keeps_original_report_unchanged():
    data = base()
    original = deepcopy(data)
    plan = prepare_manual_fallback_pr6(data, SHAPE)
    evaluate_manual_confirmation_pr6(plan, SHAPE, action="confirm", corners=QUAD)
    assert data == original
