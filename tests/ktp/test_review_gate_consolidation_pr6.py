"""Synthetic PR6 review consolidation tests, no KTP images/identities."""
from copy import deepcopy

import pytest

from autodocscanner.core.review_gate_consolidation_pr6 import consolidate_review_gate


def report(*, aliases=("adaptive",), candidate="adaptive"):
    return {
        "v3_review_gate": {"status": "preview_review", "candidate_source": candidate,
                           "auto_crop_allowed": False, "confidence_calibrated": False, "reasons": [],
                           "action": "preview_suggested_corners"},
        "pr6_review_gate_shadow": {"status": "preview_review", "candidate_source": candidate,
                                   "auto_crop_allowed": False, "confidence_calibrated": False,
                                   "selection_status": "pr6_shadow_review_only", "reasons": [],
                                   "action": "preview_suggested_corners"},
        "pr6_candidate_decision_shadow": {"status": "preview_review", "action": "preview_suggested_corners",
            "suggested_for_review": aliases[0], "auto_select": None,
            "auto_crop_allowed": False, "confidence_calibrated": False,
            "requires_explicit_confirmation": True, "mode": "offline_shadow_review_only",
            "reasons": [], "candidate_assessments": [
                {"source": aliases[0], "evidence_state": "complete_for_review", "reasons": []}]},
        "pr6_selection_shadow": {"status": "offline_diagnostic_only", "groups": [
            {"representative": aliases[0], "sources": list(aliases), "score": .87}]},
        "pr6_multi_refinement_shadow": {"status": "offline_diagnostic_only", "candidates": [
            {"source": aliases[0], "accepted_for_shadow": False, "reason": "weak_edge"}]},
        "pr6_physical_refinement_validation_shadow": {
            "status": "offline_diagnostic_only", "auto_crop_allowed": False,
            "auto_select": None, "requires_explicit_confirmation": True,
            "physical_boundary_verified": False, "geometrically_accepted_count": 0,
            "preview_review_eligible_count": 0,
            "candidate_assessments": [
                {"source": aliases[0], "geometric_refinement_accepted": False,
                 "eligible_for_preview_review": False, "requires_manual_review": True,
                 "reason": "geometric_refinement_not_accepted", "physical_boundary_verified": False}]},
        "v3_manual_handoff": {"action": "preview_suggested_corners", "editor": "existing_manual_four_corners",
            "auto_crop_allowed": False, "requires_explicit_confirmation": True,
            "suggested_corners": [[40, 30], [340, 30], [340, 190], [40, 190]]},
    }


def test_clean_agreement_original_candidate_preview_only():
    data = report()
    before = deepcopy(data)
    decision = consolidate_review_gate(data)
    assert decision["status"] == "preview_review"
    assert decision["candidate_source"] == "adaptive"
    assert decision["preview_geometry_origin"] == "original_detector_candidate"
    assert decision["reason_codes"] == []
    assert decision["legacy_handoff_differs"] is False
    assert decision["auto_select"] is None and decision["auto_crop_allowed"] is False
    assert decision["requires_explicit_confirmation"] is True
    assert data == before


def test_refinement_decline_does_not_override_safe_original_review():
    decision = consolidate_review_gate(report())
    assert decision["status"] == "preview_review"
    assert decision["gate_trace"]["physical_refinement"]["context"] == "no_accepted_refinement_original_only"


def test_alias_same_selection_group_can_preview_original():
    decision = consolidate_review_gate(report(aliases=("adaptive", "stable"), candidate="stable"))
    assert decision["status"] == "preview_review"
    assert decision["candidate_source"] == "stable"


@pytest.mark.parametrize("gate", ["v3_review_gate", "pr6_review_gate_shadow", "pr6_candidate_decision_shadow"])
def test_any_prior_manual_gate_blocks_preview(gate):
    data = report()
    data[gate]["status"] = "manual_review"
    decision = consolidate_review_gate(data)
    assert decision["status"] == "manual_review"
    assert decision["candidate_source"] is None
    assert gate + ":requires_manual_review" in decision["reason_codes"]


def test_previous_preview_handoff_cannot_upgrade_pr6_manual():
    data = report()
    data["pr6_candidate_decision_shadow"]["status"] = "manual_review"
    decision = consolidate_review_gate(data)
    assert decision["status"] == "manual_review"
    assert decision["legacy_handoff_differs"] is True
    assert decision["fallback_editor"] == "existing_manual_four_corners"


def test_nocandidate_routes_manual_fallback():
    data = report()
    data["v3_review_gate"].update(status="no_candidate", candidate_source=None, reasons=["no_valid_candidate"])
    data["pr6_review_gate_shadow"]["candidate_source"] = None
    decision = consolidate_review_gate(data)
    assert decision["status"] == "manual_review"
    assert "review_gates:no_candidate" in decision["reason_codes"]


def test_pr6_and_v3_mismatched_candidate_blocks():
    data = report()
    data["pr6_review_gate_shadow"]["candidate_source"] = "v2_0"
    assert "review_gates:candidate_source_mismatch" in consolidate_review_gate(data)["reason_codes"]


def test_candidate_decision_source_mismatch_blocks():
    data = report()
    data["pr6_candidate_decision_shadow"]["suggested_for_review"] = "v2_0"
    assert "candidate_decision:source_mismatch" in consolidate_review_gate(data)["reason_codes"]


def test_weak_consensus_evidence_blocks():
    data = report()
    data["pr6_candidate_decision_shadow"]["candidate_assessments"][0]["evidence_state"] = "requires_manual_verification"
    assert "candidate_decision:incomplete_candidate_evidence" in consolidate_review_gate(data)["reason_codes"]


def test_distinct_geometry_groups_block_even_if_all_gate_flags_preview():
    data = report()
    data["pr6_selection_shadow"]["groups"].append({"representative":"v2_0", "sources":["v2_0"]})
    d = consolidate_review_gate(data)
    assert d["status"] == "manual_review"
    assert "selection:distinct_geometry_groups" in d["reason_codes"]


def test_conflicting_source_group_membership_blocks():
    data = report(aliases=("adaptive", "stable"), candidate="stable")
    data["pr6_selection_shadow"]["groups"].append({"representative": "stable", "sources": ["stable"]})
    assert "selection:ambiguous_source_group" in consolidate_review_gate(data)["reason_codes"]


def test_broken_selection_membership_blocks():
    data = report()
    data["pr6_selection_shadow"]["groups"][0]["sources"] = ["stable"]
    assert consolidate_review_gate(data)["status"] == "manual_review"


def accepted(data, *, eligible=True):
    source = data["pr6_selection_shadow"]["groups"][0]["representative"]
    data["pr6_multi_refinement_shadow"]["candidates"] = [
        {"source": source, "accepted_for_shadow": True, "reason": "geometrically_stable_refinement"}]
    data["pr6_physical_refinement_validation_shadow"]["geometrically_accepted_count"] = 1
    data["pr6_physical_refinement_validation_shadow"]["preview_review_eligible_count"] = int(eligible)
    data["pr6_physical_refinement_validation_shadow"]["candidate_assessments"] = [{
        "source": source, "geometric_refinement_accepted": True,
        "eligible_for_preview_review": eligible,
        "requires_manual_review": not eligible,
        "independent_four_side_evidence": True,
        "post_refinement_four_side_evidence": True,
        "reasons": [] if eligible else ["unresolved_nested_contour_conflict"],
        "physical_boundary_verified": False,
    }]
    return data


def test_accepted_refinement_with_strong_context_allows_original_preview_only():
    d = consolidate_review_gate(accepted(report()))
    assert d["status"] == "preview_review"
    assert d["preview_geometry_origin"] == "original_detector_candidate"
    assert d["gate_trace"]["physical_refinement"]["context"] == "accepted_refinement_corroborated_not_verified"


def test_accepted_inner_contour_denied_by_physical_validation():
    d = consolidate_review_gate(accepted(report(), eligible=False))
    assert d["status"] == "manual_review"
    assert "physical_refinement:unresolved_accepted_boundary" in d["reason_codes"]


def test_incomplete_post_refinement_boundary_blocks():
    data = accepted(report())
    data["pr6_physical_refinement_validation_shadow"]["candidate_assessments"][0]["post_refinement_four_side_evidence"] = False
    assert consolidate_review_gate(data)["status"] == "manual_review"


def test_mismatched_refinement_sources_block():
    data = accepted(report())
    data["pr6_multi_refinement_shadow"]["candidates"][0]["source"] = "v2_0"
    d = consolidate_review_gate(data)
    assert d["status"] == "manual_review"
    assert "physical_refinement:assessment_coverage_mismatch" in d["reason_codes"]


def test_duplicate_refinement_rows_block():
    data = report()
    data["pr6_multi_refinement_shadow"]["candidates"].append(deepcopy(data["pr6_multi_refinement_shadow"]["candidates"][0]))
    assert "physical_refinement:duplicate_refinement" in consolidate_review_gate(data)["reason_codes"]


def test_accepted_count_mismatch_block():
    data = accepted(report())
    data["pr6_physical_refinement_validation_shadow"]["geometrically_accepted_count"] = 0
    assert "physical_refinement:accepted_count_mismatch" in consolidate_review_gate(data)["reason_codes"]


def test_physical_verification_claim_fails_closed():
    data = report()
    data["pr6_physical_refinement_validation_shadow"]["physical_boundary_verified"] = True
    assert consolidate_review_gate(data)["status"] == "manual_review"


@pytest.mark.parametrize("key", ["v3_review_gate", "pr6_review_gate_shadow", "pr6_candidate_decision_shadow", "pr6_selection_shadow", "pr6_multi_refinement_shadow", "pr6_physical_refinement_validation_shadow"])
def test_missing_or_malformed_section_denies_preview(key):
    for value in (None, [], "wrong", {}):
        data = report()
        data[key] = value
        assert consolidate_review_gate(data)["status"] == "manual_review"


def test_invalid_reasons_contract_fails_closed():
    data = report()
    data["v3_review_gate"]["reasons"] = None
    assert consolidate_review_gate(data)["status"] == "manual_review"


def test_any_upstream_auto_crop_permission_is_error():
    for key in ("v3_review_gate", "pr6_review_gate_shadow", "pr6_candidate_decision_shadow"):
        data = report()
        data[key]["auto_crop_allowed"] = True
        d = consolidate_review_gate(data)
        assert d["status"] == "manual_review"
        assert d["auto_crop_allowed"] is False


def test_annotations_and_ranking_scores_do_not_participate():
    data = report()
    initial = consolidate_review_gate(data)
    data["expected"] = [[-10,0], [0,2], [3,4], [5,6]]
    data["normalized_corner_error"] = 0.99
    data["pr6_multi_refinement_ground_truth_evaluation"] = {"candidates": [{"after_normalized_corner_error": .99}]}
    data["pr6_selection_shadow"]["groups"][0]["score"] = -50
    assert consolidate_review_gate(data) == initial


def test_manual_mode_has_no_suggested_corners_or_crop():
    data = accepted(report(), eligible=False)
    d = consolidate_review_gate(data)
    assert "suggested_corners" not in d
    assert d["action"] == "manual_corners"
    assert d["requires_explicit_confirmation"] is True
    assert d["auto_crop_allowed"] is False


def test_preview_requires_existing_four_valid_handoff_corners():
    for corners in (None, [], [[1, 2]], [[1, 2], [3, 4], [5, 6], [float("nan"), 8]]):
        data = report()
        data["v3_manual_handoff"]["suggested_corners"] = corners
        result = consolidate_review_gate(data)
        assert result["status"] == "manual_review"
        assert "manual_handoff:preview_not_ready" in result["reason_codes"]


def test_conflicting_gate_action_fails_closed():
    data = report()
    data["v3_review_gate"]["action"] = "manual_corners"
    assert "v3_review_gate:inconsistent_action" in consolidate_review_gate(data)["reason_codes"]


def test_physical_eligibility_count_disagreement_fails_closed():
    data = accepted(report())
    data["pr6_physical_refinement_validation_shadow"]["preview_review_eligible_count"] = 0
    assert "physical_refinement:eligibility_count_mismatch" in consolidate_review_gate(data)["reason_codes"]


def test_detector_comparison_adds_consolidated_shadow_without_changing_old_gates(monkeypatch):
    import numpy as np
    from autodocscanner.core import detector_comparison as comparison

    image = np.full((440, 640, 3), 120, dtype=np.uint8)
    points = np.asarray([[90, 90], [540, 90], [540, 360], [90, 360]], dtype=np.float32)

    class FakeEngine:
        def detect(self, img):
            return points.copy(), {"score": .5, "selected_source": "synthetic"}
        def order_points(self, p):
            return np.asarray(p, dtype=np.float32)
        def warp(self, img, p):
            return img
        def _warp_quality(self, img):
            return .75

    monkeypatch.setattr(comparison, "StableBaselinePerspectiveEngine", FakeEngine)
    monkeypatch.setattr(comparison, "AutoPerspectiveEngine", FakeEngine)
    monkeypatch.setattr(comparison, "contour_candidates", lambda *args, **kwargs: [])
    previous_handoff = comparison.prepare_manual_handoff

    result = comparison.compare_detectors(image)
    consolidated = result["pr6_review_consolidation_shadow"]
    assert consolidated["auto_select"] is None
    assert consolidated["auto_crop_allowed"] is False
    assert consolidated["requires_explicit_confirmation"] is True
    assert consolidated["mode"] == "offline_shadow_review_only"
    assert result["v3_manual_handoff"] == previous_handoff(result, image.shape)
    assert result["v3_review_gate"]["auto_crop_allowed"] is False
    assert result["pr6_review_gate_shadow"]["auto_crop_allowed"] is False


def test_extreme_unrepresentable_corner_coordinate_fails_closed():
    data = report()
    data["v3_manual_handoff"]["suggested_corners"][0][0] = 10 ** 1000
    result = consolidate_review_gate(data)
    assert result["status"] == "manual_review"
    assert "manual_handoff:preview_not_ready" in result["reason_codes"]
