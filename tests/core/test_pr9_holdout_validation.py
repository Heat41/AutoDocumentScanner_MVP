"""Synthetic regression tests for the evaluation harness, not real-photo accuracy."""
import json
import cv2
import numpy as np
import pytest
from autodocscanner.tools.pr9_holdout_validation import evaluate_photo, evaluate_folder, summarize

QUAD = np.array([[60, 45], [260, 45], [260, 190], [60, 190]], float)
IMAGE = np.zeros((240, 320, 3), np.uint8)


class Detector:
    def __init__(self, corners, mutate=False):
        self.corners = corners
        self.mutate = mutate
    def detect(self, image):
        assert not image.any()
        if self.mutate:
            image[:] = 255
        return self.corners, {"detector_score": 999}


def audit(image, observations):
    assert not image.any()
    assert set(observations) == {"stable", "adaptive"}
    assert all(set(c) == {"found", "corners"} for c in observations.values())
    return {"auto_crop_allowed": False, "auto_select": None, "suggested_for_review": None,
            "candidates": {s: {"side_observations": [{"state": "aligned"}]*4,
                              "fully_aligned_for_review": s == "stable"} for s in observations}}


def localizer(image, corners):
    assert not image.any()
    return {"edge_aligned": True}


def evaluate(stable=QUAD, adaptive=QUAD, annotation=QUAD, **kwargs):
    return evaluate_photo(IMAGE, annotation, detectors={
        "stable": Detector(stable, mutate=True), "adaptive": Detector(adaptive),
        "v2": Detector(QUAD)}, auditor=audit, localizer=localizer, **kwargs)


def test_identical_reference_and_cyclic_permutation():
    row = evaluate(stable=np.roll(QUAD, 2, axis=0), adaptive=QUAD[::-1])
    assert row["candidates"]["stable"]["mean_error"] == 0
    assert row["candidates"]["adaptive"]["max_error"] == 0
    assert not IMAGE.any()


def test_original_diagonal_normalization_and_max():
    displaced = QUAD.copy(); displaced[0, 0] += 20
    row = evaluate(stable=displaced)["candidates"]["stable"]
    assert row["mean_error"] == pytest.approx(.0125)
    assert row["max_error"] == pytest.approx(.05)


def test_annotation_does_not_change_evidence():
    first = evaluate()
    second = evaluate(annotation=QUAD+15)
    for source in ("stable", "adaptive"):
        a, b = first["candidates"][source], second["candidates"][source]
        for key in ("pr6_edge_aligned", "pr9_four_sides_aligned", "pr9_review_evidence"):
            assert a[key] == b[key]
        assert a["mean_error"] != b["mean_error"]


def test_missing_and_off_photo_not_measured():
    row = evaluate(stable=None, adaptive=QUAD-70)
    assert row["candidates"]["stable"]["status"] == "missing_candidate"
    assert row["candidates"]["adaptive"]["status"] == "invalid_or_off_photo"
    out = summarize([row], "development")
    assert out["oracle_photos_with_candidate_within_2pct"] == 0
    assert out["source_metrics"]["stable"]["mean_corner_error_fraction"]["p90"] is None


def test_invalid_reference_and_image_are_explicitly_excluded():
    assert evaluate(annotation=None)["status"] == "missing_or_invalid_annotation"
    assert evaluate_photo(None, QUAD)["status"] == "unreadable_or_invalid_image"
    assert evaluate(annotation=QUAD-70)["status"] == "missing_or_invalid_annotation"


def test_supported_bad_candidates_counted_not_hidden():
    rows = [evaluate(), evaluate(stable=QUAD+25, adaptive=QUAD)]
    stats = summarize(rows, "development")
    metrics = stats["source_metrics"]["stable"]
    assert metrics["within_2pct"] == 1
    assert metrics["evidence"]["pr9_review_evidence"]["supported_over_2pct"] == 1
    assert metrics["evidence"]["pr9_review_evidence"]["support_precision_at_2pct"] == .5
    assert stats["oracle_photos_with_candidate_within_2pct"] == 2
    assert not stats["auto_crop_allowed"]


def test_folder_counts_unannotated_and_unreadable_without_leaking_names(tmp_path):
    cv2.imwrite(str(tmp_path / "private_name.jpg"), IMAGE)
    cv2.imwrite(str(tmp_path / "another_name.png"), IMAGE)
    (tmp_path / "unreadable.jpg").write_bytes(b"bad data")
    def runner(image, annotation):
        if image is None or annotation is None:
            return evaluate_photo(image, annotation)
        return evaluate(annotation=annotation)
    out = evaluate_folder(tmp_path, {"private_name.jpg": QUAD.tolist()}, dataset_role="development", evaluator=runner)
    assert out["photo_count"] == 3
    assert out["evaluated_photo_count"] == 1
    assert out["excluded_by_status"] == {"unreadable_or_invalid_image":1, "missing_or_invalid_annotation":1}
    serialized = json.dumps(out, allow_nan=False)
    assert "private_name" not in serialized and "another_name" not in serialized
    assert "corners" not in serialized


def test_detector_failure_not_a_success():
    class Bad:
        def detect(self, image): raise ValueError("bad")
    row = evaluate_photo(IMAGE, QUAD, detectors={"stable": Bad(), "adaptive": Detector(QUAD)}, auditor=audit, localizer=localizer)
    assert row["candidates"]["stable"]["status"] == "detector_error"
    assert row["candidates"]["adaptive"]["mean_error"] == 0


def test_unsafe_audit_contract_fails_closed():
    def unsafe(image, observations):
        return {"auto_crop_allowed": True}
    with pytest.raises(ValueError, match="safety contract"):
        evaluate_photo(IMAGE, QUAD, detectors={"stable": Detector(QUAD), "adaptive": Detector(QUAD)}, auditor=unsafe, localizer=localizer)


def test_real_pr6_and_pr9_diagnostics_with_conflicting_geometry():
    image = np.full((240, 320, 3), 35, np.uint8)
    cv2.rectangle(image, (60, 45), (260, 190), (218, 226, 234), -1)
    class Fixed:
        def __init__(self, points): self.points = points
        def detect(self, img): return self.points, {}
    result = evaluate_photo(image, QUAD, detectors={"stable": Fixed(QUAD), "adaptive": Fixed(QUAD + [12, 0])})
    stable, adaptive = (result["candidates"][s] for s in ("stable", "adaptive"))
    assert stable["pr6_edge_aligned"] is True
    assert stable["pr9_four_sides_aligned"] is True
    assert adaptive["pr9_four_sides_aligned"] is False
    assert result["geometry_conflict"] is True
    assert not stable["pr9_review_evidence"] and not adaptive["pr9_review_evidence"]
    assert stable["mean_error"] == 0
    assert adaptive["mean_error"] == pytest.approx(.03)
    assert result["auto_crop_allowed"] is False
