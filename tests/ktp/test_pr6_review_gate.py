"""Safe PR6 review: sample-independent synthetic geometry checks."""
from autodocscanner.core.pr6_review_gate import assess_pr6_review_gate
from autodocscanner.core.candidate_selection_pr6 import compare_selection


def _base():
    return {"status": "preview_review", "candidate_source": "stable", "reasons": []}


def test_competing_shapes_downgrade_preview_to_manual():
    report = {"v3_review_gate": _base(),
              "pr6_selection_shadow": {"groups": [
                  {"score": .55}, {"score": .43}],
                  "warnings": ["competing_geometries", "different_candidate_extent"]}}
    gate = assess_pr6_review_gate(report)
    assert gate["status"] == "manual_review"
    assert "competing_geometries" in gate["reasons"]
    assert not gate["auto_crop_allowed"]


def test_duplicates_do_not_create_false_ambiguity():
    report = {"v3_review_gate": _base(),
              "pr6_selection_shadow": {"groups": [
                  {"score": .72, "sources": ["stable", "adaptive"]}], "warnings": []}}
    gate = assess_pr6_review_gate(report)
    assert gate["status"] == "preview_review"
    assert gate["auto_crop_allowed"] is False


def test_existing_manual_gate_is_never_upgraded():
    report = {"v3_review_gate": {"status": "manual_review", "candidate_source": "stable",
                                   "reasons": ["weak_side"]},
              "pr6_selection_shadow": {"groups": [{"score": .9}], "warnings": []}}
    gate = assess_pr6_review_gate(report)
    assert gate["status"] == "manual_review"
    assert gate["reasons"] == ["weak_side"]


def test_missing_shadow_fails_closed():
    gate = assess_pr6_review_gate({"v3_review_gate": _base()})
    assert gate["auto_crop_allowed"] is False
    # Missing diagnostics should not grant preview permission.
    assert gate["status"] == "manual_review"


def test_per_corner_difference_not_hidden_by_low_average():
    outer = [[10, 10], [390, 10], [390, 240], [10, 240]]
    offset = [[10, 10], [390, 10], [390, 240], [28, 240]]
    report = {"stable": {"geometry_valid": True, "corners": outer,
                          "geometry": {"area_ratio": .8}},
              "adaptive": {"geometry_valid": True, "corners": offset,
                            "geometry": {"area_ratio": .78}},
              "v2": [], "v3_ranking": {"ranking": [
                  {"source": "stable", "score": .8},
                  {"source": "adaptive", "score": .75}]}}
    result = compare_selection(report, (260, 400))
    assert len(result["groups"]) == 2
