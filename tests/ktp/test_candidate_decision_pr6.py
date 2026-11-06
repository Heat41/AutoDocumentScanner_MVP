"""Synthetic PR6 candidate decision tests: no real KTP data or annotations."""
from autodocscanner.core.candidate_decision_pr6 import decide_candidate_review


def _candidate(name, *, clean=True, frame=False):
    warnings = ["frame_or_extrapolation_risk"] if frame else ([] if clean else ["poor_edge_alignment"])
    return {"source": name,
            "warnings": warnings,
            "complete_independent_evidence": clean and not frame,
            "side_checks": {"supported_external": 4, "localized_aligned": 4 if clean else 2,
                            "measurable_contrast": 4}}


def _interpretation(name, classification="aligned_consistent", offsets=None):
    if offsets is None:
        offsets = [0, -.125, 0, -.125]
    return {"source": name,
            "offset_evidence": {"classification": classification, "coverage_sufficient": True,
                                "coherent_side_count": 4, "offsets": offsets}}


def _fixture(names=("stable",), *, groups=None, relations=None, gate="preview_review"):
    if groups is None:
        groups = [[name] for name in names]
    if relations is None:
        relations = []
    return {
        "pr6_combined_candidate_shadow": {
            "status": "offline_diagnostic_only",
            "candidates": [_candidate(name) for name in names]},
        "pr6_evidence_consensus_shadow": {
            "status": "offline_diagnostic_only",
            "consensus_groups": [{"members": members} for members in groups],
            "candidate_interpretations": [_interpretation(name) for name in names]},
        "pr6_spatial_relationship_shadow": {"status": "offline_diagnostic_only",
                                                 "relationships": relations},
        "pr6_comparative_evidence_shadow": {"status": "offline_diagnostic_only"},
        "pr6_review_gate_shadow": {"status": gate, "reasons": []},
    }


def test_single_well_supported_candidate_is_only_a_confirmed_preview():
    result = decide_candidate_review(_fixture())
    assert result["status"] == "preview_review"
    assert result["suggested_for_review"] == "stable"
    assert result["requires_explicit_confirmation"]
    assert result["auto_crop_allowed"] is False
    assert result["auto_select"] is None


def test_prior_manual_gate_cannot_be_upgraded():
    result = decide_candidate_review(_fixture(gate="manual_review"))
    assert result["status"] == "manual_review"
    assert result["suggested_for_review"] is None
    assert "prior_review_gate_requires_manual" in result["reasons"]


def test_single_weak_candidate_requires_manual():
    data = _fixture()
    data["pr6_combined_candidate_shadow"]["candidates"][0] = _candidate("stable", clean=False)
    result = decide_candidate_review(data)
    assert result["status"] == "manual_review"
    assert result["candidate_assessments"][0]["evidence_state"] == "requires_manual_verification"


def test_close_quads_have_one_group_but_conflicting_offsets_require_review():
    data = _fixture(("stable", "v2_0"), groups=[["stable", "v2_0"]])
    second = data["pr6_evidence_consensus_shadow"]["candidate_interpretations"][1]
    second["offset_evidence"]["classification"] = "coherent_displaced_boundary"
    second["offset_evidence"]["offsets"] = [-.375, -.5, -.375, -.5]
    result = decide_candidate_review(data)
    assert len(result["consensus_assessments"]) == 1
    assert result["status"] == "manual_review"
    assert "within_group_boundary_disagreement" in result["reasons"]


def test_two_equivalent_clean_quads_can_preview_but_not_crop():
    data = _fixture(("stable", "adaptive"), groups=[["stable", "adaptive"]])
    result = decide_candidate_review(data)
    assert result["status"] == "preview_review"
    assert result["suggested_for_review"] == "stable"
    assert result["auto_crop_allowed"] is False


def test_two_distinct_plausible_quads_are_unresolved():
    relation = {"candidate_a": "stable", "candidate_b": "adaptive", "nested_geometry_risk": False}
    result = decide_candidate_review(_fixture(("stable", "adaptive"), relations=[relation]))
    assert result["status"] == "manual_review"
    assert result["suggested_for_review"] is None
    assert result["conflicts"][0]["kind"] == "distinct_geometry_conflict"


def test_strong_small_nested_shape_is_not_assumed_correct():
    relation = {"candidate_a": "stable", "candidate_b": "v2_0",
                "nested_geometry_risk": True,
                "smaller_source": "v2_0", "larger_source": "stable",
                "smaller_area_fraction": .1}
    result = decide_candidate_review(_fixture(("stable", "v2_0"), relations=[relation]))
    assert result["status"] == "manual_review"
    assert result["conflicts"][0]["kind"] == "nested_extent_conflict"
    assert result["conflicts"][0]["well_supported_inner"] is True
    assert result["conflicts"][0]["winner"] is None


def test_outer_frame_warning_does_not_automatically_approve_inner():
    relation = {"candidate_a": "stable", "candidate_b": "v2_0",
                "nested_geometry_risk": True, "smaller_source": "v2_0", "larger_source": "stable"}
    data = _fixture(("stable", "v2_0"), relations=[relation])
    data["pr6_combined_candidate_shadow"]["candidates"][0] = _candidate("stable", frame=True)
    result = decide_candidate_review(data)
    assert result["status"] == "manual_review"
    assert result["conflicts"][0]["outer_frame_warning"] is True
    assert result["suggested_for_review"] is None


def test_missing_pairwise_relation_fails_closed():
    result = decide_candidate_review(_fixture(("stable", "adaptive")))
    assert "missing_spatial_relation" in result["reasons"]
    assert result["conflicts"][0]["kind"] == "missing_spatial_relation"


def test_incomplete_consensus_membership_fails_closed():
    result = decide_candidate_review(_fixture(("stable", "adaptive"), groups=[["stable"]]))
    assert result["status"] == "manual_review"
    assert "incomplete_consensus_coverage" in result["reasons"]


def test_no_input_always_manual():
    result = decide_candidate_review({})
    assert result["status"] == "manual_review"
    assert result["suggested_for_review"] is None
    assert result["auto_crop_allowed"] is False


def test_annotation_and_score_changes_do_not_change_decisions():
    data = _fixture()
    original = decide_candidate_review(data)
    data["stable"] = {"normalized_corner_error": .9, "mean_corner_error_px": 500}
    data["pr6_combined_candidate_shadow"]["candidates"][0]["ranking_score"] = -.99
    data["expected"] = [[1, 2], [3, 4], [5, 6], [7, 8]]
    modified = decide_candidate_review(data)
    assert original == modified
