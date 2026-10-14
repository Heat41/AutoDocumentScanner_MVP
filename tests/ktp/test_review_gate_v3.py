"""PR5 gate tests: decision safety, not classifier accuracy."""
from autodocscanner.core.review_gate_v3 import assess_review_gate


def candidate(*, sides=(.8, .8, .8, .8), complete=True, extra=False):
    return {
        "geometry_valid": True,
        "geometry": {"slight_extrapolation": extra},
        "warp_quality": .8,
        "corners": [[20, 20], [350, 20], [350, 225], [20, 225]],
        "boundary_evidence": {
            "valid": True,
            "evidence_complete": complete,
            "frame_review_needed": False,
            "side_support": list(sides),
        },
    }


def report(entries, **data):
    return {"v3_ranking": {"ranking": entries}, **data}


def rank(source, score=.7, reasons=()):
    return {"source": source, "score": score,
            "review_required": bool(reasons), "review_reasons": list(reasons)}


def test_no_candidates_goes_manual():
    result = assess_review_gate(report([]), (300, 400))
    assert result["status"] == "no_candidate"
    assert result["action"] == "manual_corners"
    assert result["auto_crop_allowed"] is False


def test_clean_candidate_gets_preview_but_never_auto_crop():
    result = assess_review_gate(report([rank("stable")], stable=candidate()), (300, 400))
    assert result["status"] == "preview_review"
    assert result["action"] == "preview_suggested_corners"
    assert not result["auto_crop_allowed"]
    assert not result["confidence_calibrated"]


def test_missing_edge_requires_manual():
    result = assess_review_gate(
        report([rank("stable")], stable=candidate(complete=False)), (300, 400))
    assert result["action"] == "manual_corners"
    assert "incomplete_evidence" in result["reasons"]


def test_extrapolation_requires_manual():
    result = assess_review_gate(
        report([rank("stable")], stable=candidate(extra=True)), (300, 400))
    assert "extrapolated_corner" in result["reasons"]


def test_similar_scores_different_quads_are_ambiguous():
    other = candidate()
    other["corners"] = [[55, 55], [385, 55], [385, 260], [55, 260]]
    result = assess_review_gate(report(
        [rank("stable", .65), rank("adaptive", .63)],
        stable=candidate(), adaptive=other), (300, 400))
    assert "ambiguous_candidates" in result["reasons"]


def test_same_corners_not_ambiguity_even_when_scores_tie():
    result = assess_review_gate(report(
        [rank("stable", .65), rank("adaptive", .65)],
        stable=candidate(), adaptive=candidate()), (300, 400))
    assert "ambiguous_candidates" not in result["reasons"]


def test_unrecognized_candidate_fail_closed():
    result = assess_review_gate(report([rank("unknown")]), (300, 400))
    assert result["action"] == "manual_corners"
    assert not result["auto_crop_allowed"]
