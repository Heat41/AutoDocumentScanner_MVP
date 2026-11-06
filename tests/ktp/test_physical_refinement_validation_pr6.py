"""Synthetic PR6 physical-boundary context tests. No real KTP data or ground truth."""
from copy import deepcopy

import numpy as np

from autodocscanner.core import physical_refinement_validation_pr6 as validator
from autodocscanner.core.physical_refinement_validation_pr6 import (
    validate_physical_refinement_context,
)


def _candidate(name="v2_0", *, complete=True, warnings=None):
    if warnings is None:
        warnings = [] if complete else ["poor_edge_alignment"]
    return {
        "source": name,
        "complete_independent_evidence": complete,
        "warnings": warnings,
        "side_checks": {
            "supported_external": 4,
            "localized_aligned": 4 if complete else 2,
            "measurable_contrast": 4,
        },
    }


def _physical():
    return {"measurable": True, "side_measurements": [
        {"contrast": 12.0, "coverage": 1.0} for _ in range(4)
    ]}


def _localized():
    return {"measurable": True, "edge_aligned": True, "side_localization": [
        {"coverage": 1.0, "median_offset_fraction": 0.0, "offset_spread": 0.0}
        for _ in range(4)
    ]}


def _accepted(name="v2_0"):
    return {
        "source": name, "accepted_for_shadow": True,
        "reason": "geometrically_stable_refinement",
        "after_corners": [[20., 20.], [180., 20.], [180., 120.], [20., 120.]],
        "selection_group_sources": [name],
    }


def _report():
    return {
        "pr6_multi_refinement_shadow": {
            "status": "offline_diagnostic_only", "candidates": [_accepted()]},
        "pr6_combined_candidate_shadow": {
            "status": "offline_diagnostic_only", "candidates": [_candidate()]},
        "pr6_spatial_relationship_shadow": {
            "status": "offline_diagnostic_only", "relationships": []},
        "pr6_evidence_consensus_shadow": {
            "status": "offline_diagnostic_only",
            "consensus_groups": [{"members": ["v2_0"]}]},
        "pr6_physical_boundary_shadow": {"v2_0": _physical()},
        "pr6_boundary_localization_shadow": {"v2_0": _localized()},
        "pr6_candidate_decision_shadow": {
            "mode": "offline_shadow_review_only", "status": "preview_review",
            "reasons": [], "suggested_for_review": "v2_0",
            "consensus_assessments": [
                {"group_id": 0, "boundary_offset_disagreement": False}]},
        "v3_review_gate": {
            "status": "preview_review", "candidate_source": "v2_0", "reasons": []},
        "pr6_review_gate_shadow": {
            "status": "preview_review", "candidate_source": "v2_0", "reasons": []},
    }


def _add_competitor(report, *, nested=True, smaller="v2_0", larger="adaptive",
                    competitor_complete=False):
    report["pr6_combined_candidate_shadow"]["candidates"].append(
        _candidate("adaptive", complete=competitor_complete))
    report["pr6_evidence_consensus_shadow"]["consensus_groups"].append(
        {"members": ["adaptive"]})
    report["pr6_spatial_relationship_shadow"]["relationships"] = [{
        "candidate_a": "v2_0", "candidate_b": "adaptive",
        "nested_geometry_risk": nested,
        "smaller_source": smaller, "larger_source": larger,
        "smaller_coverage": 1., "smaller_area_fraction": .1,
    }]
    report["pr6_candidate_decision_shadow"]["status"] = "manual_review"
    report["pr6_candidate_decision_shadow"]["suggested_for_review"] = None
    report["pr6_review_gate_shadow"]["status"] = "manual_review"


def test_one_supported_refinement_only_eligible_for_explicit_preview(monkeypatch):
    report = _report()
    original = deepcopy(report)
    monkeypatch.setattr(
        validator, "_post_refinement_check",
        lambda image, corners: (True, "post_refinement_four_sides_observed"))
    image = np.zeros((200, 240, 3), dtype=np.uint8)
    result = validate_physical_refinement_context(report, image)
    assert result["geometrically_accepted_count"] == 1
    assert result["corroborated_for_review_count"] == 1
    assert result["preview_review_eligible_count"] == 1
    assert result["candidate_assessments"][0]["boundary_context_status"] == (
        "corroborated_boundary_signals_not_verified")
    assert result["physical_boundary_verified"] is False
    assert result["auto_select"] is None and result["auto_crop_allowed"] is False
    assert result["requires_explicit_confirmation"] is True
    assert report == original


def test_strong_inner_rectangle_is_ambiguous_not_physically_accepted():
    report = _report()
    _add_competitor(report, competitor_complete=True)
    result = validate_physical_refinement_context(report)
    row = result["candidate_assessments"][0]
    assert row["geometric_refinement_accepted"] is True
    assert row["independent_four_side_evidence"] is True
    assert row["boundary_context_status"] == "unresolved_nested_geometry"
    assert "inside_larger_candidate" in row["spatial_roles"]
    assert "unresolved_nested_contour_conflict" in row["reasons"]
    assert not row["eligible_for_preview_review"]
    assert not result["physical_boundary_verified"]


def test_large_frame_candidate_does_not_automatically_validate_smaller_candidate():
    report = _report()
    _add_competitor(report)
    report["pr6_combined_candidate_shadow"]["candidates"][1]["warnings"] = [
        "frame_or_extrapolation_risk"]
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert row["boundary_context_status"] == "unresolved_nested_geometry"
    assert row["physical_boundary_verified"] is False


def test_large_contour_is_also_ambiguous_not_winner():
    report = _report()
    _add_competitor(report, smaller="adaptive", larger="v2_0")
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert row["boundary_context_status"] == "unresolved_nested_geometry"
    assert row["spatial_roles"] == ["encloses_smaller_candidate"]
    assert row["requires_manual_review"]


def test_distinct_competitor_blocks_preview_without_nesting():
    report = _report()
    _add_competitor(report, nested=False)
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert row["boundary_context_status"] == "unresolved_competing_geometry"
    assert "unresolved_distinct_candidate_conflict" in row["reasons"]


def test_missing_spatial_relation_fails_closed():
    report = _report()
    _add_competitor(report)
    report["pr6_spatial_relationship_shadow"]["relationships"] = []
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert "missing_pairwise_spatial_evidence" in row["reasons"]
    assert not row["eligible_for_preview_review"]


def test_equivalent_group_with_disagreeing_offsets_fails_closed():
    report = _report()
    report["pr6_combined_candidate_shadow"]["candidates"].append(_candidate("stable"))
    report["pr6_evidence_consensus_shadow"]["consensus_groups"] = [
        {"members": ["v2_0", "stable"]}]
    report["pr6_spatial_relationship_shadow"]["relationships"] = [{
        "candidate_a": "v2_0", "candidate_b": "stable",
        "nested_geometry_risk": False, "intersection_over_union": .96}]
    report["pr6_candidate_decision_shadow"]["consensus_assessments"][0][
        "boundary_offset_disagreement"] = True
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert "near_equivalent_offset_disagreement" in row["reasons"]


def test_weak_side_stays_incomplete_even_with_clean_geometry():
    report = _report()
    report["pr6_combined_candidate_shadow"]["candidates"][0] = _candidate(
        complete=False)
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert row["boundary_context_status"] == "insufficient_physical_boundary_evidence"
    assert not row["independent_four_side_evidence"]


def test_raw_boundary_measurements_are_required_not_just_flags():
    report = _report()
    report["pr6_physical_boundary_shadow"]["v2_0"]["side_measurements"][2]["contrast"] = 0.
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert "independent_four_side_evidence_incomplete" in row["reasons"]


def test_nan_or_boolean_measurement_fails_closed():
    for bad in (float("nan"), True, None):
        report = _report()
        report["pr6_boundary_localization_shadow"]["v2_0"]["side_localization"][0][
            "offset_spread"] = bad
        assert validate_physical_refinement_context(report)[
            "candidate_assessments"][0]["eligible_for_preview_review"] is False


def test_manual_gate_is_never_upgraded():
    report = _report()
    report["pr6_candidate_decision_shadow"]["status"] = "manual_review"
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert "existing_review_gate_requires_manual" in row["reasons"]
    assert row["requires_manual_review"]


def test_mismatching_preview_candidate_source_blocks():
    report = _report()
    report["v3_review_gate"]["candidate_source"] = "stable"
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert "review_gate_candidate_mismatch" in row["reasons"]


def test_mismatching_candidate_decision_blocks():
    report = _report()
    report["pr6_candidate_decision_shadow"]["suggested_for_review"] = "adaptive"
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert "candidate_decision_disagreement" in row["reasons"]


def test_unrefined_candidate_is_not_promoted():
    report = _report()
    item = report["pr6_multi_refinement_shadow"]["candidates"][0]
    item["accepted_for_shadow"] = False
    item["reason"] = "inconsistent_edge"
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert row["boundary_context_status"] == "not_evaluated_refinement_declined"
    assert not row["eligible_for_preview_review"]
    assert validate_physical_refinement_context(report)["geometrically_accepted_count"] == 0


def test_missing_or_invalid_report_is_fail_closed():
    for report in ({}, None, {"pr6_multi_refinement_shadow": {"candidates": "bad"}}):
        result = validate_physical_refinement_context(report)
        assert result["auto_select"] is None
        assert result["auto_crop_allowed"] is False
        assert result.get("reason") == "invalid_or_missing_evidence_sections"


def test_duplicate_candidate_source_is_context_error():
    report = _report()
    report["pr6_combined_candidate_shadow"]["candidates"].append(_candidate())
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert not row["eligible_for_preview_review"]
    assert "inconsistent_or_missing_shadow_context" in row["reasons"]


def test_duplicate_group_membership_is_context_error():
    report = _report()
    report["pr6_evidence_consensus_shadow"]["consensus_groups"].append(
        {"members": ["v2_0"]})
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert "inconsistent_or_missing_shadow_context" in row["reasons"]


def test_ground_truth_fields_and_scores_never_change_review_classification():
    report = _report()
    before = validate_physical_refinement_context(report)
    report["pr6_multi_refinement_ground_truth_evaluation"] = {
        "candidates": [{"source": "v2_0", "after_normalized_corner_error": .8}]}
    report["normalized_corner_error"] = .00001
    report["v2"] = [{"normalized_corner_error": .99, "warp_quality": .01}]
    report["pr6_combined_candidate_shadow"]["candidates"][0]["ranking_score"] = 9999.
    assert validate_physical_refinement_context(report) == before


def test_frame_warning_never_becomes_proof_of_outer_or_inner():
    report = _report()
    _add_competitor(report, smaller="adaptive", larger="v2_0")
    report["pr6_combined_candidate_shadow"]["candidates"][0]["warnings"] = [
        "frame_or_extrapolation_risk"]
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert row["boundary_context_status"] == "unresolved_nested_geometry"
    assert not row["independent_four_side_evidence"]
    assert row["physical_boundary_verified"] is False


def test_more_than_one_refinement_accepted_cannot_choose_winner():
    report = _report()
    _add_competitor(report, nested=False)
    report["pr6_multi_refinement_shadow"]["candidates"].append(_accepted("adaptive"))
    row = validate_physical_refinement_context(report)["candidate_assessments"][0]
    assert "multiple_geometric_refinements_accepted" in row["reasons"]
    assert validate_physical_refinement_context(report)["auto_select"] is None


def test_missing_image_never_passes_post_refinement_gate():
    result = validate_physical_refinement_context(_report())
    row = result["candidate_assessments"][0]
    assert row["boundary_context_status"] == "insufficient_physical_boundary_evidence"
    assert row["post_refinement_four_side_evidence"] is False
    assert row["post_refinement_evidence_reason"] == "source_image_unavailable"
    assert row["requires_manual_review"] is True


def test_blank_image_does_not_claim_a_physical_boundary():
    image = np.zeros((200, 240, 3), dtype=np.uint8)
    result = validate_physical_refinement_context(_report(), image)
    row = result["candidate_assessments"][0]
    assert row["post_refinement_four_side_evidence"] is False
    assert not row["eligible_for_preview_review"]
    assert result["auto_crop_allowed"] is False


def test_post_refinement_evidence_is_required_even_with_original_perfect(monkeypatch):
    monkeypatch.setattr(
        validator, "_post_refinement_check",
        lambda img, pts: (False, "weak_post_refinement_external_side"))
    image = np.zeros((200, 240, 3), dtype=np.uint8)
    row = validate_physical_refinement_context(_report(), image)["candidate_assessments"][0]
    assert row["independent_four_side_evidence"] is True
    assert row["post_refinement_four_side_evidence"] is False
    assert "post_refinement_four_side_evidence_incomplete" in row["reasons"]


def test_invalid_after_corner_coordinates_fail_closed():
    report = _report()
    report["pr6_multi_refinement_shadow"]["candidates"][0]["after_corners"] = [
        [float("nan"), 0], [180, 20], [180, 120], [20, 120]]
    row = validate_physical_refinement_context(
        report, np.zeros((200, 240, 3), dtype=np.uint8))["candidate_assessments"][0]
    assert row["post_refinement_evidence_reason"] == "invalid_refined_corners"
    assert not row["eligible_for_preview_review"]
