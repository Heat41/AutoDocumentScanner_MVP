"""Synthetic tests for PR6 comparative candidate evidence."""
from autodocscanner.core.comparative_evidence_pr6 import compare_candidate_evidence


def _fixture(delta=False, iou=.1, nested=True):
    return {
        "pr6_combined_candidate_shadow": {"candidates": [
            {"source": "adaptive", "side_checks": {"supported_external": 4,
             "localized_aligned": 4, "measurable_contrast": 4}, "warnings": ["frame_or_extrapolation_risk"]},
            {"source": "v2_0", "side_checks": {"supported_external": 4,
             "localized_aligned": 2 if delta else 4, "measurable_contrast": 4}, "warnings": []},
        ]},
        "pr6_spatial_relationship_shadow": {"relationships": [
            {"candidate_a": "adaptive", "candidate_b": "v2_0",
             "intersection_over_union": iou,
             "nested_geometry_risk": nested, "smaller_source": "v2_0",
             "larger_source": "adaptive"},
        ]},
    }


def test_complete_small_internal_contour_is_not_autoselected():
    out = compare_candidate_evidence(_fixture())
    pair = out["pairwise_comparisons"][0]
    assert pair["geometry_relation"] == "nested_geometry_ambiguous"
    assert pair["evidence_balance"] == "equal_side_evidence"
    assert pair["requires_review"]
    assert pair["winner"] is None
    assert out["auto_select"] is None


def test_independent_side_advantage_is_descriptive_only():
    out = compare_candidate_evidence(_fixture(delta=True))
    assert out["pairwise_comparisons"][0]["evidence_balance"] == "candidate_a_more_supported"
    assert out["auto_crop_allowed"] is False


def test_nearly_identical_candidate_geometry_identified():
    out = compare_candidate_evidence(_fixture(iou=.96, nested=False))
    assert out["pairwise_comparisons"][0]["geometry_relation"] == "near_equivalent_geometry"


def test_missing_side_count_is_not_assumed_zero():
    data = _fixture()
    del data["pr6_combined_candidate_shadow"]["candidates"][1]["side_checks"]["localized_aligned"]
    out = compare_candidate_evidence(data)
    assert out["pairwise_comparisons"][0]["evidence_balance"] == "insufficient_evidence"


def test_no_comparison_fails_closed():
    data = _fixture()
    data["pr6_spatial_relationship_shadow"]["relationships"] = []
    out = compare_candidate_evidence(data)
    assert out["compared_pair_count"] == 0
    assert set(out["uncompared_candidates"]) == {"adaptive", "v2_0"}
    assert out["suggested_for_review"] is None


def test_unknown_spatial_source_is_ignored():
    data = _fixture()
    data["pr6_spatial_relationship_shadow"]["relationships"][0]["candidate_b"] = "missing"
    out = compare_candidate_evidence(data)
    assert out["pairwise_comparisons"] == []
    assert out["auto_crop_allowed"] is False
