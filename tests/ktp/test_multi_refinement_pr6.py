"""Synthetic multi-proposal refinement tests; no KTP photos or identity data."""
from copy import deepcopy

import cv2
import numpy as np

from autodocscanner.core import multi_refinement_pr6 as mod


class FakeEngine:
    def __init__(self, regression=False):
        self.regression = regression
        self.last = None

    def warp(self, image, corners):
        self.last = np.asarray(corners, np.float32).copy()
        return image

    def _warp_quality(self, image):
        if self.regression and self.last[0, 0] > 80:
            return .65
        return .8


def quad(x=80, y=80, w=470, h=280):
    return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]


def data(quads=None):
    if quads is None:
        quads = [quad(), quad(210, 170, 150, 120)]
    candidates = [{"geometry_valid": True, "corners": q} for q in quads]
    return {
        "stable": candidates[0],
        "adaptive": candidates[1] if len(candidates) > 1 else candidates[0],
        "v2": candidates[2:],
        "pr6_selection_shadow": {"groups": [
            {"representative": "stable", "sources": ["stable"]},
            *([{"representative": "adaptive", "sources": ["adaptive"]}] if len(candidates) > 1 else []),
            *[{"representative": f"v2_{i}", "sources": [f"v2_{i}"]}
              for i in range(max(0, len(candidates)-2))],
        ]},
    }


def image():
    picture = np.full((440, 640, 3), 30, dtype=np.uint8)
    cv2.rectangle(picture, (80, 80), (550, 360), (220, 220, 220), -1)
    return picture


def move_one_pixel(_image, corners):
    points = corners.copy()
    points[:, 0] += 1
    return points, {"accepted": True, "reason": "line_fit"}


def test_distinct_proposals_evaluated_independently():
    report = data()
    old = deepcopy(report)
    result = mod.evaluate_multi_candidate_refinement(
        image(), report, FakeEngine(), refine_fn=move_one_pixel)
    assert [r["source"] for r in result["candidates"]] == ["stable", "adaptive"]
    assert result["evaluated_count"] == 2
    assert result["plausible_refinement_count"] == 2
    assert all(r["accepted_for_shadow"] for r in result["candidates"])
    assert report == old
    assert result["auto_select"] is None and result["auto_crop_allowed"] is False


def test_no_candidate_ranking_changed_by_annotation_content():
    report = data()
    plain = mod.evaluate_multi_candidate_refinement(
        image(), report, FakeEngine(), refine_fn=move_one_pixel)
    report["expected"] = [[1, 2], [3, 4], [5, 6], [7, 8]]
    report["stable"]["normalized_corner_error"] = .999
    report["stable"]["mean_corner_error_px"] = 999
    assert mod.evaluate_multi_candidate_refinement(
        image(), report, FakeEngine(), refine_fn=move_one_pixel) == plain


def test_frame_suspect_is_skipped_and_not_refined():
    report = data([quad(3, 3, 632, 430)])

    def forbidden(*args):
        raise AssertionError("Frame candidate should not reach refinement")

    result = mod.evaluate_multi_candidate_refinement(
        image(), report, FakeEngine(), refine_fn=forbidden)
    assert result["evaluated_count"] == 0
    assert result["candidates"][0]["reason"] == "frame_or_extrapolation_risk"


def test_blank_image_real_refiner_fails_closed():
    report = data([quad()])
    blank = np.full((440, 640, 3), 120, np.uint8)
    result = mod.evaluate_multi_candidate_refinement(blank, report, FakeEngine())
    assert result["plausible_refinement_count"] == 0
    assert result["candidates"][0]["before_corners"] == result["candidates"][0]["after_corners"]


def test_bad_warp_quality_refuses_even_when_refiner_accepts():
    result = mod.evaluate_multi_candidate_refinement(
        image(), data([quad()]), FakeEngine(regression=True), refine_fn=move_one_pixel)
    assert result["candidates"][0]["reason"] == "warp_quality_regression"
    assert not result["candidates"][0]["accepted_for_shadow"]


def test_large_shift_is_rejected():
    def move_far(_image, corners):
        points = corners.copy()
        points[:, 0] += 25
        return points, {"accepted": True}

    result = mod.evaluate_multi_candidate_refinement(
        image(), data([quad()]), FakeEngine(), refine_fn=move_far)
    assert result["candidates"][0]["reason"] == "perspective_stability_regression"


def test_vertex_reordering_is_rejected():
    def reorder(_image, corners):
        return np.roll(corners, 1, axis=0), {"accepted": True}

    result = mod.evaluate_multi_candidate_refinement(
        image(), data([quad()]), FakeEngine(), refine_fn=reorder)
    assert not result["candidates"][0]["accepted_for_shadow"]


def test_finite_and_invalid_geometry_are_safely_handled():
    report = data([[[10, 10], [10, 10], [300, 300], [40, 300]]])
    result = mod.evaluate_multi_candidate_refinement(
        image(), report, FakeEngine(), refine_fn=move_one_pixel)
    assert result["candidates"][0]["reason"] == "invalid_original_geometry"
    report["stable"]["corners"] = [[float("nan"), 0], [10, 10], [20, 20], [30, 30]]
    assert mod.evaluate_multi_candidate_refinement(
        image(), report, FakeEngine())["candidates"][0]["reason"] == "missing_or_invalid_corners"


def test_cpu_budget_reports_all_without_executing_extra():
    quads = [quad(30+i, 40+i, 400, 280) for i in range(15)]
    report = data(quads)
    count = [0]

    def tracked(img, corners):
        count[0] += 1
        return corners.copy(), {"accepted": False, "reason": "insufficient_evidence"}

    result = mod.evaluate_multi_candidate_refinement(
        image(), report, FakeEngine(), max_candidates=6, refine_fn=tracked)
    assert result["candidate_count"] == 15
    assert result["evaluated_count"] == 6 and count[0] == 6
    assert result["not_evaluated_count"] == 9
    assert result["candidates"][-1]["reason"] == "cpu_budget_exhausted"


def test_missing_or_unusable_report_fails_closed():
    for item in (None, {}, {"pr6_selection_shadow": {"groups": "bad"}}):
        result = mod.evaluate_multi_candidate_refinement(image(), item, FakeEngine())
        assert result["auto_select"] is None
        assert result["auto_crop_allowed"] is False
    assert mod.evaluate_multi_candidate_refinement(
        np.empty((0, 0), np.uint8), data(), FakeEngine())["reason"] == "invalid_input"


def test_portrait_and_grayscale_dimensions_are_allowed():
    gray = np.full((700, 440), 100, np.uint8)
    report = data([quad(40, 40, 340, 610)])
    result = mod.evaluate_multi_candidate_refinement(
        gray, report, FakeEngine(), refine_fn=move_one_pixel)
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["accepted_for_shadow"]


def test_per_side_evidence_regression_blocks_refinement(monkeypatch):
    calls = [0]

    def changed(img, corners):
        calls[0] += 1
        return {"evidence_complete": True,
                "side_support": [.9, .9, .9, .9] if calls[0] % 2 else [.9, .9, .3, .9]}

    monkeypatch.setattr(mod, "boundary_evidence", changed)
    result = mod.evaluate_multi_candidate_refinement(
        image(), data([quad()]), FakeEngine(), refine_fn=move_one_pixel)
    assert result["candidates"][0]["reason"] == "per_side_evidence_regression"


def test_missing_side_support_never_means_good_edge(monkeypatch):
    monkeypatch.setattr(
        mod, "boundary_evidence",
        lambda *_: {"evidence_complete": True, "side_support": [.8, None, .8, .8]})
    result = mod.evaluate_multi_candidate_refinement(
        image(), data([quad()]), FakeEngine(), refine_fn=move_one_pixel)
    assert result["candidates"][0]["reason"] == "unmeasured_side_support"


def test_refiner_exception_is_isolated_to_candidate():
    def broken(*args):
        raise ValueError("synthetic")

    result = mod.evaluate_multi_candidate_refinement(
        image(), data(), FakeEngine(), refine_fn=broken)
    assert len(result["candidates"]) == 2
    assert all(r["reason"] == "refinement_evaluation_error" for r in result["candidates"])


def test_duplicate_source_only_one_run():
    report = data([quad()])
    report["pr6_selection_shadow"]["groups"].append(
        {"representative": "stable", "sources": ["stable"]})
    result = mod.evaluate_multi_candidate_refinement(
        image(), report, FakeEngine(), refine_fn=move_one_pixel)
    assert result["candidate_count"] == 1
