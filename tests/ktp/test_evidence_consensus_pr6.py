"""Synthetic-only PR6 consensus tests."""
from autodocscanner.core.evidence_consensus_pr6 import interpret_candidate_consensus


def _fixture():
    return {
        "pr6_combined_candidate_shadow": {"candidates": [
            {"source": source, "warnings": [], "complete_independent_evidence": True}
            for source in ("stable", "v2_0", "adaptive")]},
        "pr6_spatial_relationship_shadow": {"relationships": [
            {"candidate_a": "stable", "candidate_b": "v2_0",
             "intersection_over_union": .96, "nested_geometry_risk": False},
            {"candidate_a": "stable", "candidate_b": "adaptive",
             "intersection_over_union": .45, "nested_geometry_risk": True,
             "smaller_source": "adaptive", "larger_source": "stable"},
            {"candidate_a": "v2_0", "candidate_b": "adaptive",
             "intersection_over_union": .45, "nested_geometry_risk": True,
             "smaller_source": "adaptive", "larger_source": "v2_0"},
        ]},
        "pr6_boundary_localization_shadow": {
            "stable": {"side_localization": [
                {"coverage": 1, "median_offset_fraction": 0, "offset_spread": .1}] * 4},
            "v2_0": {"side_localization": [
                {"coverage": 1, "median_offset_fraction": -.4, "offset_spread": .1}] * 4},
        },
    }


def test_close_candidates_are_grouped_without_selecting():
    out = interpret_candidate_consensus(_fixture())
    assert out["consensus_groups"][0]["members"] == ["stable", "v2_0"]
    assert len(out["consensus_groups"]) == 2
    assert out["auto_select"] is None and out["auto_crop_allowed"] is False


def test_coherent_offset_is_not_rejected_or_claimed_accurate():
    out = interpret_candidate_consensus(_fixture())
    rows = {x["source"]: x for x in out["candidate_interpretations"]}
    assert rows["stable"]["offset_evidence"]["classification"] == "aligned_consistent"
    assert rows["v2_0"]["offset_evidence"]["classification"] == "coherent_displaced_boundary"
    assert rows["adaptive"]["offset_evidence"]["classification"] == "insufficient_measurements"


def test_chain_overlap_does_not_merge_unrelated_geometries():
    data = _fixture()
    data["pr6_spatial_relationship_shadow"]["relationships"] = [
        {"candidate_a": "stable", "candidate_b": "v2_0", "intersection_over_union": .94},
        {"candidate_a": "v2_0", "candidate_b": "adaptive", "intersection_over_union": .93},
        {"candidate_a": "stable", "candidate_b": "adaptive", "intersection_over_union": .7},
    ]
    out = interpret_candidate_consensus(data)
    assert all(len(group["members"]) <= 2 for group in out["consensus_groups"])


def test_nested_candidates_remain_ambiguous_even_when_both_complete():
    out = interpret_candidate_consensus(_fixture())
    assert len(out["nested_conflicts"]) == 2
    assert all(row["winner"] is None for row in out["nested_conflicts"])


def test_invalid_and_incomplete_offsets_are_safe():
    data = _fixture()
    data["pr6_boundary_localization_shadow"]["stable"]["side_localization"] = [
        {"coverage": .1, "median_offset_fraction": 0, "offset_spread": 0}] * 4
    out = interpret_candidate_consensus(data)
    assert out["candidate_interpretations"][0]["offset_evidence"]["classification"] == "insufficient_measurements"


def test_empty_report_is_fail_closed():
    out = interpret_candidate_consensus({})
    assert out["consensus_groups"] == []
    assert out["auto_select"] is None
    assert not out["auto_crop_allowed"]
