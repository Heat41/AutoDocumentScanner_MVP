"""Consolidated PR6 review advice, without replacing existing gates or UI.

Conservatively reconciles V3, PR6 selection, Candidate Decision, and physical
refinement context. Original-candidate previews do not require a successful
refinement, but a conflicting or unverified *accepted* refinement blocks them.
Never reads annotation/ground truth, chooses a crop, or mutates inputs.
"""
from __future__ import annotations

import math


def _dict(report, name):
    value = report.get(name)
    return value if isinstance(value, dict) else {}


def _reasons(value):
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(x, str) for x in reasons):
        return ["invalid_reasons_contract"]
    return reasons


def _source(value):
    return value if isinstance(value, str) and value else None


def consolidate_review_gate(report):
    """Suggest only confirmed *review* of original corners, or manual corners.

    No output contains executable crop instructions or geometry. The existing
    four-corner handoff remains unchanged; connecting this to the UI is a
    separate, explicitly validated stage.
    """
    if not isinstance(report, dict):
        report = {}
    baseline = _dict(report, "v3_review_gate")
    pr6 = _dict(report, "pr6_review_gate_shadow")
    decision = _dict(report, "pr6_candidate_decision_shadow")
    physical = _dict(report, "pr6_physical_refinement_validation_shadow")
    multi = _dict(report, "pr6_multi_refinement_shadow")
    selection = _dict(report, "pr6_selection_shadow")

    blockers = set()
    trace = {}
    gates = (
        ("v3_review_gate", baseline),
        ("pr6_review_gate_shadow", pr6),
        ("pr6_candidate_decision_shadow", decision),
    )
    for label, gate in gates:
        status = gate.get("status")
        gate_reasons = _reasons(gate)
        trace[label] = {
            "status": status if status in ("no_candidate", "manual_review", "preview_review") else "invalid",
            "reasons": list(gate_reasons),
        }
        if status != "preview_review" or gate_reasons:
            blockers.add(label + ":requires_manual_review")
        intended_action = (
            "preview_suggested_corners" if status == "preview_review" else "manual_corners"
        )
        if gate.get("action") != intended_action:
            blockers.add(label + ":inconsistent_action")
        if gate.get("auto_crop_allowed") is not False:
            blockers.add(label + ":unsafe_crop_contract")
        if gate.get("confidence_calibrated") is not False:
            blockers.add(label + ":invalid_confidence_contract")
    if decision.get("mode") != "offline_shadow_review_only":
        blockers.add("candidate_decision:invalid_shadow_mode")
    if decision.get("auto_select", "invalid") is not None:
        blockers.add("candidate_decision:unexpected_auto_selection")
    if decision.get("requires_explicit_confirmation") is not True:
        blockers.add("candidate_decision:confirmation_required")
    if pr6.get("selection_status") != "pr6_shadow_review_only":
        blockers.add("pr6_review_gate:invalid_shadow_mode")

    primary = _source(baseline.get("candidate_source"))
    if not primary or _source(pr6.get("candidate_source")) != primary:
        blockers.add("review_gates:candidate_source_mismatch")
    if baseline.get("status") == "no_candidate" or not primary:
        blockers.add("review_gates:no_candidate")

    # All preview sources must resolve to the *same selection-group proposal*.
    groups = selection.get("groups")
    matched = []
    if selection.get("status") != "offline_diagnostic_only" or not isinstance(groups, list) or not groups:
        blockers.add("selection:missing_or_invalid_groups")
        groups = []
    seen_reps = set()
    for group in groups:
        if not isinstance(group, dict):
            blockers.add("selection:invalid_group")
            continue
        representative = _source(group.get("representative"))
        members = group.get("sources")
        if (not representative or not isinstance(members, list) or not members
                or any(not _source(x) for x in members) or len(members) != len(set(members))):
            blockers.add("selection:invalid_group")
            continue
        if representative in seen_reps:
            blockers.add("selection:duplicate_representative")
        seen_reps.add(representative)
        if representative not in members:
            blockers.add("selection:invalid_representative")
        if primary in members:
            matched.append((representative, set(members)))
    if len(groups) != 1:
        blockers.add("selection:distinct_geometry_groups")
    if len(matched) != 1:
        blockers.add("selection:ambiguous_source_group")

    target = matched[0][0] if len(matched) == 1 else None
    allowed_sources = matched[0][1] if len(matched) == 1 else set()
    if _source(decision.get("suggested_for_review")) != target:
        blockers.add("candidate_decision:source_mismatch")
    assessments = decision.get("candidate_assessments")
    if not isinstance(assessments, list):
        blockers.add("candidate_decision:invalid_assessments")
        assessments = []
    matching_assessments = [x for x in assessments if isinstance(x, dict) and x.get("source") == target]
    if (len(matching_assessments) != 1 or
            matching_assessments[0].get("evidence_state") != "complete_for_review" or
            matching_assessments[0].get("reasons") != []):
        blockers.add("candidate_decision:incomplete_candidate_evidence")

    # Physical validation describes refinements, not the original detector quad.
    # When refinement was declined, this is *not* an additional veto on an
    # otherwise safe original-candidate preview. Accepted refinements that
    # remain ambiguous MUST block preview, even if the original gates passed.
    refinements = multi.get("candidates")
    physical_rows = physical.get("candidate_assessments")
    physical_contract = (
        multi.get("status") == "offline_diagnostic_only"
        and physical.get("status") == "offline_diagnostic_only"
        and physical.get("auto_crop_allowed") is False
        and physical.get("auto_select") is None
        and physical.get("requires_explicit_confirmation") is True
        and physical.get("physical_boundary_verified") is False
        and isinstance(refinements, list)
        and isinstance(physical_rows, list)
    )
    if not physical_contract:
        blockers.add("physical_refinement:invalid_or_missing_context")
    if not isinstance(refinements, list):
        refinements = []
    if not isinstance(physical_rows, list):
        physical_rows = []
    refinement_by_source = {}
    for row in refinements:
        if not isinstance(row, dict) or not _source(row.get("source")):
            blockers.add("physical_refinement:invalid_refinement_row")
            continue
        name = row["source"]
        if name in refinement_by_source:
            blockers.add("physical_refinement:duplicate_refinement")
        refinement_by_source[name] = row
    physical_by_source = {}
    for row in physical_rows:
        if not isinstance(row, dict) or not _source(row.get("source")):
            blockers.add("physical_refinement:invalid_assessment_row")
            continue
        name = row["source"]
        if name in physical_by_source:
            blockers.add("physical_refinement:duplicate_assessment")
        physical_by_source[name] = row
    if set(refinement_by_source) != set(physical_by_source):
        blockers.add("physical_refinement:assessment_coverage_mismatch")
    accepted = [row for row in refinement_by_source.values() if row.get("accepted_for_shadow") is True]
    if type(physical.get("geometrically_accepted_count")) is not int or physical["geometrically_accepted_count"] != len(accepted):
        blockers.add("physical_refinement:accepted_count_mismatch")
    if any(row.get("physical_boundary_verified") is not False for row in physical_rows if isinstance(row, dict)):
        blockers.add("physical_refinement:invalid_verification_claim")
    eligible_count = physical.get("preview_review_eligible_count")
    if (type(eligible_count) is not int or not 0 <= eligible_count <= len(accepted)
            or eligible_count != sum(
                row.get("eligible_for_preview_review") is True
                for row in physical_rows if isinstance(row, dict))):
        blockers.add("physical_refinement:eligibility_count_mismatch")
    refinement_context = "no_accepted_refinement_original_only"
    if len(accepted) > 1:
        blockers.add("physical_refinement:multiple_accepted_refinements")
        refinement_context = "unresolved_accepted_refinement"
    elif len(accepted) == 1:
        accepted_name = accepted[0]["source"]
        assessment = physical_by_source.get(accepted_name) or {}
        refinement_context = "accepted_refinement_requires_context_review"
        if accepted_name != target or accepted_name not in allowed_sources:
            blockers.add("physical_refinement:accepted_source_mismatch")
        if not (
            assessment.get("geometric_refinement_accepted") is True
            and assessment.get("eligible_for_preview_review") is True
            and assessment.get("requires_manual_review") is False
            and assessment.get("independent_four_side_evidence") is True
            and assessment.get("post_refinement_four_side_evidence") is True
            and assessment.get("physical_boundary_verified") is False
            and assessment.get("reasons") == []
        ):
            blockers.add("physical_refinement:unresolved_accepted_boundary")
        if not any(x.startswith("physical_refinement:") for x in blockers):
            refinement_context = "accepted_refinement_corroborated_not_verified"
    trace["physical_refinement"] = {
        "accepted_count": len(accepted),
        "context": refinement_context,
        "physical_boundary_verified": False,
    }

    # A V3 preview decision alone does not guarantee the handoff has valid
    # original-image corners. Do not advertise a preview without four finite
    # vertices already validated by the existing manual handoff subsystem.
    handoff = _dict(report, "v3_manual_handoff")
    corners = handoff.get("suggested_corners")
    points_ready = (
        isinstance(corners, list) and len(corners) == 4 and
        all(isinstance(point, (list, tuple)) and len(point) == 2 and
            all(isinstance(coord, (int, float)) and not isinstance(coord, bool)
                and math.isfinite(coord) for coord in point)
            for point in corners)
    )
    if (handoff.get("action") != "preview_suggested_corners" or
            handoff.get("auto_crop_allowed") is not False or
            handoff.get("requires_explicit_confirmation") is not True or
            not points_ready):
        blockers.add("manual_handoff:preview_not_ready")

    preview = not blockers
    # The original detector proposal remains the only possible preview at this
    # stage. Refined vertices are NEVER silently substituted.
    route = "preview_review" if preview else "manual_review"
    return {
        "status": route,
        "action": "preview_suggested_corners" if preview else "manual_corners",
        "candidate_source": primary if preview else None,
        "preview_geometry_origin": "original_detector_candidate" if preview else None,
        "reason_codes": sorted(blockers),
        "gate_trace": trace,
        "legacy_handoff_differs": (
            _dict(report, "v3_manual_handoff").get("action") !=
            ("preview_suggested_corners" if preview else "manual_corners")
            if isinstance(report.get("v3_manual_handoff"), dict) else None
        ),
        "fallback_editor": "existing_manual_four_corners",
        "requires_explicit_confirmation": True,
        "auto_select": None,
        "auto_crop_allowed": False,
        "confidence_calibrated": False,
        "physical_boundary_verified": False,
        "mode": "offline_shadow_review_only",
    }
