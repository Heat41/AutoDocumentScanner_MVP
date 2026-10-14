"""Offline selection safety tests. No KTP identities or production data."""
from autodocscanner.core.candidate_ranking_v3 import rank_candidates


def item(quality, sides, *, frame=False, complete=True, valid=True):
    return {
        "found": True,
        "geometry_valid": valid,
        "geometry": {"frame_suspect": frame},
        "warp_quality": quality,
        "boundary_evidence": {
            "side_support": sides,
            "side_coverage": [1.0] * len(sides),
            "evidence_complete": complete,
            "frame_suspect": frame,
        },
    }


def test_rank_good_four_sides_above_frame():
    ranked = rank_candidates([
        ("frame", item(.5, [.7, .8, .8, .8], frame=True)),
        ("card", item(.7, [.7, .8, .8, .8])),
    ])
    assert ranked["ranking"][0]["source"] == "card"
    assert ranked["auto_select"] is None


def test_missing_edge_is_unknown_and_requires_review():
    ranked = rank_candidates([("candidate", item(.9, [.8, .7, None, .7], complete=False))])
    assert ranked["ranking"][0]["review_required"]
    assert "incomplete_boundary" in ranked["ranking"][0]["review_reasons"]


def test_invalid_geometry_never_ranked():
    assert rank_candidates([("bad", item(.9, [1, 1, 1, 1], valid=False))])["ranking"] == []


def test_no_calibrated_automatic_winner():
    ranked = rank_candidates([("strong", item(1.0, [1, 1, 1, 1]))])
    assert ranked["auto_select"] is None
    assert ranked["selection_status"] == "offline_review_only"


def test_weak_side_requires_review():
    ranked = rank_candidates([("weak", item(.8, [1, 1, 1, .01]))])
    assert "weak_side" in ranked["ranking"][0]["review_reasons"]
