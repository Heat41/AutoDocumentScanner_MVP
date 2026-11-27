"""PR9 shadow-only tests: no V2 or ground-truth dependence."""
import copy

import cv2
import numpy as np

from autodocscanner.core.pr9_boundary_audit import (audit_v3_boundaries_pr9, diagnose_v3_image_pr9)


def scene():
    image = np.full((240, 320, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (60, 45), (260, 190), (218, 226, 234), thickness=-1)
    corners = [[60, 45], [260, 45], [260, 190], [60, 190]]
    return image, corners


def report(stable, adaptive=None):
    if adaptive is None:
        adaptive = stable
    return {"stable": {"found": True, "geometry_valid": True, "corners": stable},
            "adaptive": {"found": True, "geometry_valid": True, "corners": adaptive}}


def test_clean_rectangle_has_four_aligned_sides_but_never_auto_crops():
    image, corners = scene()
    result = audit_v3_boundaries_pr9(image, report(corners))
    assert [x["state"] for x in result["candidates"]["stable"]["side_observations"]] == ["aligned"] * 4
    assert result["candidates"]["stable"]["fully_aligned_for_review"] is True
    assert result["auto_crop_allowed"] is False
    assert result["auto_select"] is None
    assert result["suggested_for_review"] is None


def test_flat_image_is_not_mistaken_for_boundary():
    image = np.full((240, 320, 3), 96, np.uint8)
    _, corners = scene()
    result = audit_v3_boundaries_pr9(image, report(corners))
    assert result["candidates"]["stable"]["fully_aligned_for_review"] is False
    assert all(x["state"] == "insufficient_edge_coverage" for x in result["candidates"]["stable"]["side_observations"])


def test_misaligned_rectangle_triggers_review():
    image, corners = scene()
    offset = [[x+9, y] for x, y in corners]
    result = audit_v3_boundaries_pr9(image, report(offset))
    assert result["candidates"]["stable"]["fully_aligned_for_review"] is False


def test_distinct_candidates_raise_conflict_even_if_one_is_good():
    image, corners = scene()
    shifted = [[x+12, y] for x, y in corners]
    result = audit_v3_boundaries_pr9(image, report(corners, shifted))
    assert result["distinct_geometry_conflict"] is True
    assert all("unresolved_v3_candidate_conflict" in c["reason_codes"] for c in result["candidates"].values())
    assert all(not c["fully_aligned_for_review"] for c in result["candidates"].values())


def test_near_frame_though_bounded_still_requires_manual_review():
    image = np.full((240, 320, 3), 20, np.uint8)
    cv2.rectangle(image, (2, 2), (317, 237), (235, 240, 245), -1)
    result = audit_v3_boundaries_pr9(image, report([[2,2],[317,2],[317,237],[2,237]]))
    assert "frame_or_extrapolation_risk" in result["candidates"]["stable"]["reason_codes"]
    assert result["candidates"]["stable"]["fully_aligned_for_review"] is False


def test_v2_and_ground_truth_scores_have_zero_influence():
    image, corners = scene()
    baseline = report(corners)
    changed = copy.deepcopy(baseline)
    changed["v2"] = [{"found": True, "geometry_valid": True, "corners": corners,
                      "normalized_corner_error": 0., "detector_score": 9999.}]
    changed["pr6_selection_shadow"] = {"groups": [{"representative": "v2_0"}]}
    changed["expected"] = corners
    changed["stable"]["normalized_corner_error"] = 1.
    changed["adaptive"]["detector_score"] = -9999.
    assert audit_v3_boundaries_pr9(image, baseline) == audit_v3_boundaries_pr9(image, changed)


def test_invalid_geometry_is_rejected_and_not_cropped():
    image, corners = scene()
    bad = [[60,45],[260,45],[260,190],[260,190]]
    output = audit_v3_boundaries_pr9(image, report(corners, bad))
    assert output["candidates"]["adaptive"]["status"] == "invalid_geometry"
    assert output["auto_crop_allowed"] is False


def test_missing_candidate_handled_without_v2_fallback():
    image, corners = scene()
    output = audit_v3_boundaries_pr9(image, {"stable":{"found":True,"corners":corners},"v2":[{"found":True,"corners":corners}]})
    assert output["candidates"]["adaptive"]["status"] == "missing_candidate"
    assert output["auto_select"] is None


def test_input_image_is_not_mutated():
    image, corners = scene()
    before = image.copy()
    audit_v3_boundaries_pr9(image, report(corners))
    assert np.array_equal(before, image)


def test_invalid_input_fail_closed():
    _, corners = scene()
    for img in [None, np.zeros((20,20,3), np.uint8), np.zeros((240,320), np.uint8)]:
        output = audit_v3_boundaries_pr9(img, report(corners))
        assert output["reason"] == "invalid_image"
        assert output["auto_crop_allowed"] is False
    output = audit_v3_boundaries_pr9(scene()[0], report(corners), samples=0)
    assert output["reason"] == "invalid_samples"


def test_gradient_handles_chroma_only_boundary():
    # Similar gray intensities, different chromaticity.
    background = (105, 105, 105)
    foreground = (255, 0, 255)
    image = np.full((240, 320, 3), background, np.uint8)
    cv2.rectangle(image, (60, 45), (260, 190), foreground, -1)
    corners = [[60,45],[260,45],[260,190],[60,190]]
    output = audit_v3_boundaries_pr9(image, report(corners))
    assert output["candidates"]["stable"]["fully_aligned_for_review"] is True
    assert output["auto_crop_allowed"] is False


def test_end_to_end_v3_only_runner_never_touches_v2():
    image, corners = scene()
    calls = []
    class FakeDetector:
        def __init__(self, name): self.name = name
        def detect(self, img):
            calls.append(self.name)
            return corners, {"score": 1.}
    output = diagnose_v3_image_pr9(image, detectors={
        "stable": FakeDetector("stable"),
        "adaptive": FakeDetector("adaptive"),
        "v2": FakeDetector("v2"),
    })
    assert calls == ["stable", "adaptive"]
    assert output["auto_crop_allowed"] is False


def test_end_to_end_handles_detector_failure_as_missing():
    image, corners = scene()
    class CrashingDetector:
        def detect(self, img): raise ValueError("failure")
    class GoodDetector:
        def detect(self, img): return corners, None
    output = diagnose_v3_image_pr9(image, detectors={
        "stable": CrashingDetector(), "adaptive": GoodDetector()})
    assert output["candidates"]["stable"]["status"] == "missing_candidate"
    assert output["candidates"]["adaptive"]["status"] == "four_side_alignment_observed"
    assert output["auto_crop_allowed"] is False
