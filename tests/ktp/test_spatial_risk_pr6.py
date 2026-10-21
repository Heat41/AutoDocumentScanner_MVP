"""Synthetic PR6 spatial-risk assessment tests; no real KTP data."""
from autodocscanner.core.spatial_risk_pr6 import assess_spatial_risk


def _report():
    return {
        "pr6_combined_candidate_shadow": {"candidates": [
            {"source": "outer", "warnings": ["frame_or_extrapolation_risk"],
             "complete_independent_evidence": False},
            {"source": "inner", "warnings": [],
             "complete_independent_evidence": True},
        ]},
        "pr6_spatial_relationship_shadow": {"relationships": [{
            "smaller_source": "inner", "larger_source": "outer",
            "nested_geometry_risk": True
        }], "invalid_or_missing_sources": []}
    }


def test_nested_complete_inner_candidate_still_requires_review():
    result = assess_spatial_risk(_report())
    inner = result["candidate_assessments"][1]
    assert inner["complete_boundary_evidence"] is True
    assert inner["requires_human_validation"] is True
    assert "nested_within_other_proposal" in inner["reasons"]
    assert result["auto_select"] is None


def test_outer_frame_suspect_is_not_assumed_correct():
    result = assess_spatial_risk(_report())
    outer = result["candidate_assessments"][0]
    assert "encloses_smaller_proposal" in outer["reasons"]
    assert "frame_or_extrapolation_risk" in outer["reasons"]
    assert outer["requires_human_validation"] is True


def test_no_nested_pairs_and_good_boundary_are_not_auto_approved():
    report = _report()
    report["pr6_spatial_relationship_shadow"]["relationships"] = []
    report["pr6_combined_candidate_shadow"]["candidates"][1]["warnings"] = []
    result = assess_spatial_risk(report)
    assert result["candidate_assessments"][1]["observation"] == "no_observed_risk"
    assert result["auto_crop_allowed"] is False
    assert result["suggested_for_review"] is None


def test_missing_evidence_fails_closed():
    report = {"pr6_combined_candidate_shadow": {"candidates": [{
        "source": "v2_0", "warnings": ["incomplete_edge_localization"],
        "complete_independent_evidence": False,
    }]}}
    result = assess_spatial_risk(report)
    assert result["candidate_assessments"][0]["requires_human_validation"]
    assert result["auto_select"] is None


def test_empty_report_safe_and_not_authorized():
    result = assess_spatial_risk({})
    assert result["candidate_assessments"] == []
    assert result["auto_crop_allowed"] is False
