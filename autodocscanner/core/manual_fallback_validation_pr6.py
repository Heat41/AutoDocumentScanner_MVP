"""Shadow-only PR6 manual fallback planning and confirmation contract.

No GUI calls, disk writes, crop, network, output mutation, or runtime wiring.
Original image is owned by the existing UI. Human annotations and detector
accuracy metrics never influence a routing decision.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def _section(report, key):
    obj = report.get(key)
    return obj if isinstance(obj, dict) else {}


def _shape(image_shape):
    if not isinstance(image_shape, (tuple, list)) or len(image_shape) < 2:
        return None
    height, width = image_shape[:2]
    if type(height) not in (int, np.int32, np.int64) or type(width) not in (int, np.int32, np.int64):
        return None
    if not (2 <= height <= 100_000 and 2 <= width <= 100_000):
        return None
    return [int(height), int(width)]


def _quad(corners, shape, *, strictly_inside=False):
    try:
        points = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    if (np.max(np.abs(points)) > 1_000_000):
        return None
    valid, _, _ = validate_quad(
        points, shape, max_area_ratio=1.0 if strictly_inside else 0.985
    )
    if not valid:
        return None
    # Manual editor labels TL, TR, BR, BL. Reject crossed or permuted points,
    # even if sorting the same coordinates could construct a valid polygon.
    polygon = points.astype(np.float32)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon, oriented=True) <= 0:
        return None
    height, width = shape
    if strictly_inside and (np.any(points[:, 0] < 0) or np.any(points[:, 0] > width - 1) or
                            np.any(points[:, 1] < 0) or np.any(points[:, 1] > height - 1)):
        return None
    return points.tolist()


def prepare_manual_fallback_pr6(report, image_shape):
    """Produce a review plan. Never send an unapproved quad to the scanner.

    A manual decision suppresses even a *legacy* V3 preview suggestion.
    For preview, verify exact original-detector geometry and handoff again.
    """
    report = report if isinstance(report, dict) else {}
    shape = _shape(image_shape)
    gate = _section(report, "pr6_review_consolidation_shadow")
    handoff = _section(report, "v3_manual_handoff")
    reasons = set()
    status = "manual_review"
    source = None
    corners = None
    if shape is None:
        reasons.add("invalid_source_image_shape")
    if gate.get("mode") != "offline_shadow_review_only":
        reasons.add("missing_or_invalid_consolidated_gate")
    if gate.get("auto_select", "invalid") is not None or gate.get("auto_crop_allowed") is not False:
        reasons.add("unsafe_consolidated_gate_contract")
    if gate.get("requires_explicit_confirmation") is not True or gate.get("fallback_editor") != "existing_manual_four_corners":
        reasons.add("invalid_confirmation_or_editor_contract")

    requested = gate.get("status")
    if requested not in ("manual_review", "preview_review"):
        reasons.add("invalid_consolidated_decision")
    if requested == "preview_review" and gate.get("reason_codes") != []:
        reasons.add("consolidated_preview_has_blockers")
    if requested == "preview_review" and shape is not None and not reasons:
        source = gate.get("candidate_source")
        if not isinstance(source, str) or not source:
            reasons.add("missing_preview_source")
        if gate.get("action") != "preview_suggested_corners" or gate.get("preview_geometry_origin") != "original_detector_candidate":
            reasons.add("invalid_preview_provenance")
        if handoff.get("action") != "preview_suggested_corners" or handoff.get("auto_crop_allowed") is not False or handoff.get("requires_explicit_confirmation") is not True:
            reasons.add("invalid_legacy_handoff")
        candidate = (report.get(source) if source in ("stable", "adaptive") else None)
        if candidate is None and isinstance(source, str) and source.startswith("v2_") and source[3:].isdigit():
            choices = report.get("v2")
            index = int(source[3:])
            candidate = choices[index] if isinstance(choices, list) and index < len(choices) else None
        if not isinstance(candidate, dict):
            reasons.add("missing_original_candidate")
        else:
            original = _quad(candidate.get("corners"), shape)
            proposed = _quad(handoff.get("suggested_corners"), shape)
            if original is None or proposed is None or not np.allclose(original, proposed, rtol=0, atol=1e-3):
                reasons.add("handoff_does_not_match_original_candidate")
            else:
                corners = [p[:] for p in original]
        if not reasons:
            status = "preview_review"
    elif requested == "manual_review":
        if gate.get("action") != "manual_corners":
            reasons.add("inconsistent_manual_action")
        # No stale V3 preview, no detector quad, no auto-filled manual corners.
        source = None
        corners = None
    else:
        if requested == "preview_review" and shape is not None and reasons:
            reasons.add("preview_blocked_by_invalid_context")

    # The actual Auto KTP dialog requires detector corners (final.py).
    # It has not been adapted to PR6's no-candidate/manual-only routing.
    ui_gaps = ["pr6_shadow_not_connected_to_production_ui"]
    if status == "manual_review":
        ui_gaps.append("auto_ktp_manual_editor_requires_existing_detector_corners")
    if handoff.get("action") == "preview_suggested_corners" and status == "manual_review":
        ui_gaps.append("legacy_v3_preview_differs_from_consolidated_manual")
    return {
        "status": status,
        "action": "preview_suggested_corners" if status == "preview_review" else "manual_corners",
        "candidate_source": source if status == "preview_review" else None,
        "suggested_corners": corners if status == "preview_review" else None,
        "source_image_shape": shape,
        "source_image_policy": "unmodified_original_image_required",
        "editor": "existing_manual_four_corners",
        "manual_corner_entry": "user_must_supply_four_original_image_points" if status == "manual_review" else "user_may_adjust_suggestion",
        "reason_codes": sorted(reasons),
        "consolidated_gate_reason_codes": (
            list(gate["reason_codes"])
            if isinstance(gate.get("reason_codes"), list) and
            all(isinstance(x, str) for x in gate["reason_codes"]) else []
        ),
        "ui_gap_codes": ui_gaps,
        "ui_integration_status": "not_connected_shadow_only",
        "requires_explicit_confirmation": True,
        "auto_select": None,
        "auto_crop_allowed": False,
        "file_write_performed": False,
        "status_scope": "offline_diagnostic_only",
    }


def evaluate_manual_confirmation_pr6(plan, image_shape, *, action, corners=None):
    """Simulate user cancel/confirm. Returns data only; no crop, save or UI action.

    The UI must call the existing scanner itself *after* a real user action;
    this function does not authorize the shadow module to operate the scanner.
    """
    result = {
        "status": "rejected", "approved_corners": None,
        "reason": "invalid_action", "file_write_performed": False,
        "crop_performed": False, "requires_existing_ui_confirmation": True,
        "auto_crop_allowed": False,
    }
    if action == "cancel":
        result.update(status="cancelled", reason="pending_review_discarded")
        return result
    if action != "confirm":
        return result
    shape = _shape(image_shape)
    if not isinstance(plan, dict) or shape is None or plan.get("source_image_shape") != shape:
        result["reason"] = "source_image_context_mismatch"
        return result
    if (plan.get("status_scope") != "offline_diagnostic_only" or
            plan.get("status") not in ("manual_review", "preview_review") or
            plan.get("action") != ("manual_corners" if plan["status"] == "manual_review" else "preview_suggested_corners") or
            plan.get("auto_crop_allowed") is not False or
            plan.get("requires_explicit_confirmation") is not True or
            plan.get("editor") != "existing_manual_four_corners" or
            plan.get("ui_integration_status") != "not_connected_shadow_only" or
            plan.get("source_image_policy") != "unmodified_original_image_required"):
        result["reason"] = "invalid_plan_contract"
        return result
    accepted = _quad(corners, shape, strictly_inside=True)
    if accepted is None:
        result["reason"] = "invalid_or_out_of_bounds_four_corners"
        return result
    result.update(status="validated_for_manual_apply", approved_corners=accepted,
                  reason="explicit_user_confirmation_validated")
    return result
