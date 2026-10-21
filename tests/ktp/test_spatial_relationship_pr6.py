"""PR6 spatial relationship tests use synthetic polygons only."""
from autodocscanner.core.spatial_relationship_pr6 import compare_spatial_relationships


def _report(polygons):
    return {
        "pr6_selection_shadow": {"groups": [
            {"representative": name} for name in polygons]},
        **{name: {"corners": points} for name, points in polygons.items()},
    }


def test_nested_inner_rectangle_is_flagged_but_not_rejected():
    data = _report({
        "stable": [[0, 0], [300, 0], [300, 200], [0, 200]],
        "adaptive": [[80, 60], [150, 60], [150, 120], [80, 120]],
    })
    result = compare_spatial_relationships(data)
    assert result["nested_pair_count"] == 1
    pair = result["relationships"][0]
    assert pair["smaller_source"] == "adaptive"
    assert pair["larger_source"] == "stable"
    assert result["auto_select"] is None
    assert not result["auto_crop_allowed"]


def test_distant_candidates_do_not_trigger_nesting():
    data = _report({
        "stable": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "adaptive": [[150, 0], [250, 0], [250, 100], [150, 100]],
    })
    result = compare_spatial_relationships(data)
    assert result["nested_pair_count"] == 0
    assert result["relationships"][0]["intersection_over_union"] == 0


def test_identical_geometry_is_not_nesting():
    pts = [[0, 0], [200, 0], [200, 100], [0, 100]]
    result = compare_spatial_relationships(_report({"stable": pts, "adaptive": pts}))
    assert result["nested_pair_count"] == 0
    assert result["relationships"][0]["intersection_over_union"] == 1


def test_rotated_geometry_supports_overlap():
    data = _report({
        "stable": [[20, 0], [140, 20], [120, 100], [0, 80]],
        "adaptive": [[40, 35], [70, 40], [65, 65], [35, 60]],
    })
    result = compare_spatial_relationships(data)
    assert result["nested_pair_count"] == 1


def test_missing_invalid_candidates_fail_closed():
    data = _report({"stable": [[1, 1]] * 4, "adaptive": [[0, 0], [10, 0], [10, 10], [0, 10]]})
    result = compare_spatial_relationships(data)
    assert result["invalid_or_missing_sources"] == ["stable"]
    assert result["relationships"] == []
    assert result["auto_select"] is None


def test_empty_report_safe():
    result = compare_spatial_relationships({})
    assert result["relationships"] == []
    assert result["auto_crop_allowed"] is False
