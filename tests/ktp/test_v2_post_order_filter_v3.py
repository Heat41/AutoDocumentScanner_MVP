"""Regression: invalid post-order V2 candidates must never reach ranking."""
from autodocscanner.core.detector_comparison import _filter_valid_candidates


def test_invalid_candidate_removed_after_metric_ordering():
    evaluated = [
        {"geometry_valid": True, "corners": [[1, 1], [2, 1], [2, 2], [1, 2]]},
        {"geometry_valid": False, "geometry_reasons": ["short_or_duplicate_side", "non_convex"]},
        {"geometry_valid": False, "geometry_reasons": ["outside_image"]},
    ]
    valid, rejected = _filter_valid_candidates(evaluated)
    assert len(valid) == 1
    assert rejected == 2
    assert all(item["geometry_valid"] for item in valid)
