"""PR6 shadow selection: independent geometry, not duplicated detector labels."""
from autodocscanner.core.candidate_selection_pr6 import compare_selection


def _item(points, area):
    return {"geometry_valid": True, "corners": points, "geometry": {"area_ratio": area}}


def _report():
    card = [[25, 30], [370, 40], [365, 245], [20, 240]]
    internal = [[180, 130], [340, 130], [340, 220], [180, 220]]
    return {
        "stable": _item(card, .66),
        "adaptive": _item(card, .66),
        "v2": [_item(internal, .08)],
        "v3_ranking": {"ranking": [
            {"source": "stable", "score": .71, "review_required": False},
            {"source": "adaptive", "score": .70, "review_required": False},
            {"source": "v2_0", "score": .62, "review_required": False},
        ]},
    }


def test_duplicate_stable_adaptive_count_as_single_geometry():
    result = compare_selection(_report(), (280, 400))
    assert len(result["groups"]) == 2
    assert set(result["groups"][0]["sources"]) == {"stable", "adaptive"}
    assert "different_candidate_extent" in result["warnings"]
    assert result["auto_select"] is None


def test_competing_similar_scores_are_reported_not_auto_selected():
    report = _report()
    report["v3_ranking"]["ranking"][2]["score"] = .69
    result = compare_selection(report, (280, 400))
    assert "competing_geometries" in result["warnings"]
    assert result["status"] == "offline_diagnostic_only"


def test_no_candidates_fails_closed():
    result = compare_selection({"v3_ranking": {"ranking": []}}, (280, 400))
    assert result["warnings"] == ["no_valid_candidate"]
    assert result["auto_select"] is None


def test_rotated_card_same_geometry_is_grouped():
    report = _report()
    original = report["stable"]["corners"]
    report["adaptive"]["corners"] = original[2:] + original[:2]
    assert len(compare_selection(report, (280, 400))["groups"]) == 2
