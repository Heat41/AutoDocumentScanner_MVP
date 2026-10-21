"""Synthetic-only checks of the PR6 combined diagnostics contract."""
from autodocscanner.core.combined_candidate_pr6 import compare_combined_evidence


def _example():
    return {
        "stable": {"geometry_valid": True, "geometry": {"area_ratio": .55},
                   "boundary_evidence": {"evidence_complete": True,
                                         "side_support": [.8, .7, .9, .6]}},
        "v2": [{"geometry_valid": True, "geometry": {"area_ratio": .04},
                "boundary_evidence": {"evidence_complete": True,
                                      "side_support": [.9, .9, .9, .9]}}],
        "pr6_selection_shadow": {"groups": [
            {"representative": "stable", "sources": ["stable", "adaptive"], "score": .8},
            {"representative": "v2_0", "sources": ["v2_0"], "score": .65}],
            "warnings": ["different_candidate_extent"]},
        "pr6_physical_boundary_shadow": {
            "stable": {"measurable": True, "side_measurements":
                       [{"contrast": 25, "coverage": 1}] * 4},
            "v2_0": {"measurable": True, "side_measurements":
                     [{"contrast": 25, "coverage": 1}] * 4}},
        "pr6_boundary_localization_shadow": {
            "stable": {"measurable": True, "side_localization":
                       [{"coverage": 1, "median_offset_fraction": 0,
                         "offset_spread": 0}] * 4},
            "v2_0": {"measurable": True, "side_localization":
                     [{"coverage": 1, "median_offset_fraction": 0,
                       "offset_spread": 0}] * 4}},
    }


def test_complete_evidence_still_cannot_auto_select():
    report = _example()
    result = compare_combined_evidence(report)
    assert result["candidates"][0]["complete_independent_evidence"]
    assert result["candidates"][1]["complete_independent_evidence"]
    assert result["auto_select"] is None
    assert result["suggested_for_review"] is None
    assert result["auto_crop_allowed"] is False


def test_missing_boundary_measurements_are_not_assumed_good():
    report = _example()
    del report["pr6_boundary_localization_shadow"]["stable"]
    result = compare_combined_evidence(report)
    assert not result["candidates"][0]["complete_independent_evidence"]
    assert "incomplete_edge_localization" in result["candidates"][0]["warnings"]


def test_frame_risk_and_weak_side_are_exposed():
    report = _example()
    report["stable"]["geometry"]["frame_suspect"] = True
    report["stable"]["boundary_evidence"]["side_support"][1] = .02
    result = compare_combined_evidence(report)
    reasons = result["candidates"][0]["warnings"]
    assert "frame_or_extrapolation_risk" in reasons
    assert "weak_external_sides" in reasons


def test_empty_candidates_fail_closed():
    result = compare_combined_evidence({})
    assert result["candidates"] == []
    assert result["auto_select"] is None


def test_invalid_v2_reference_does_not_crash():
    report = _example()
    report["pr6_selection_shadow"]["groups"] = [
        {"representative": "v2_99", "sources": ["v2_99"], "score": .9}]
    result = compare_combined_evidence(report)
    assert "invalid_geometry" in result["candidates"][0]["warnings"]
