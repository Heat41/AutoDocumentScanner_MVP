"""Independent opposite-side PR6 recovery — anonymous synthetic fixtures.

Testing candidate recall and failure handling, not real KTP accuracy.
"""
import math
import cv2
import numpy as np
import pytest

from autodocscanner.core.independent_side_alignment_pr6 import (
    propose_independent_opposite_sides,
)
from autodocscanner.core.side_boundary_recovery_pr6 import (
    _order, prepare_side_recovery_lab,
)


def _normal(quad, side):
    quad = _order(quad)
    a, b = quad[side], quad[(side + 1) % 4]
    edge = b - a
    n = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
    if n @ (quad.mean(axis=0) - (a + b) / 2) < 0:
        n = -n
    return n


def _scene(rotation=0, card_color=(75, 155, 196)):
    h, w = 620, 830
    rng = np.random.default_rng(191)
    pixels = np.clip(
        135 + rng.normal(0, 4, (h, w, 1)), 0, 255,
    ).astype(np.uint8)
    image = np.repeat(pixels, 3, axis=2)
    card = np.full((210, 344, 3), card_color, np.uint8)
    cv2.rectangle(card, (230, 53), (310, 171), (84, 86, 90), -1)
    for index in range(5):
        y = 50 + index * 25
        cv2.line(card, (20, y), (190, y), (64, 87, 105), 1)
    angle = math.radians(rotation)
    transform = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    centered = np.asarray([
        [-172, -105], [172, -105], [172, 105], [-172, 105],
    ], float)
    target = (centered @ transform.T + [w/2, h/2]).astype(np.float32)
    src = np.asarray([[0, 0], [343, 0], [343, 209], [0, 209]], np.float32)
    warp = cv2.getPerspectiveTransform(src, target)
    warped = cv2.warpPerspective(card, warp, (w, h))
    mask = cv2.warpPerspective(
        np.full((210, 344), 255, np.uint8), warp, (w, h)
    )
    image[mask > 128] = warped[mask > 128]
    return image, _order(target)


def _offset_both(true, first=24., second=18., *, pair=0):
    pts = _order(true).copy()
    for idx, distance in ((pair, first), (pair + 2, second)):
        n = _normal(pts, idx)
        pts[idx] += n * distance
        pts[(idx + 1) % 4] += n * distance
    return pts


def _error(found, truth):
    # Cyclic vertex ordering is irrelevant to evaluation.
    found = np.asarray(found, float)
    expected = _order(truth)
    return min(
        np.linalg.norm(found - np.roll(expected, s, axis=0), axis=1).mean()
        for s in range(4)
    )


@pytest.mark.parametrize("rotation", [0, 19, 51, 95, 142])
@pytest.mark.parametrize("pair", [0, 1])
def test_independent_edge_offsets_recover_shrunk_card_across_rotations(rotation, pair):
    image, truth = _scene(rotation)
    wrong = _offset_both(truth, 24, 18, pair=pair)
    original_error = _error(wrong, truth)
    assert original_error >= 17
    before = image.copy()
    diagnostic = {}
    results = propose_independent_opposite_sides(
        image, wrong, pair_side_index=pair,
        max_results=4, diagnostics=diagnostic,
    )
    assert results, diagnostic
    assert min(_error(row["corners"], truth) for row in results) < 12
    assert diagnostic["status"] == "proposed"
    assert diagnostic["failure_reason"] is None
    assert len(diagnostic["side_checks"]) == 2
    assert all(row["auto_select"] is None and not row["auto_crop_allowed"]
               for row in results)
    assert np.array_equal(before, image)


def test_different_independent_displacements_change_card_size():
    image, true = _scene(22)
    wrong = _offset_both(true, 37., 13.)
    result = propose_independent_opposite_sides(
        image, wrong, pair_side_index=0, max_results=4,
    )
    assert result
    assert min(_error(row["corners"], true) for row in result) < 12
    assert any(abs(row["independent_offset_fractions"][0] -
                   row["independent_offset_fractions"][1]) > .035
               for row in result)


def test_correct_card_does_not_manufacture_two_edge_shift():
    image, true = _scene(29)
    assert propose_independent_opposite_sides(image, true) == []


def test_featureless_background_fails_closed_and_reports_both_sides():
    image = np.full((420, 640, 3), 127, np.uint8)
    quad = [[90, 85], [440, 85], [440, 320], [90, 320]]
    diagnostic = {}
    assert propose_independent_opposite_sides(
        image, quad, diagnostics=diagnostic
    ) == []
    assert diagnostic["failure_reason"] == "unsupported_boundary_for_one_or_both_sides"
    assert len(diagnostic["side_checks"]) == 2
    assert all(side["has_supported_alternative"] is False
               for side in diagnostic["side_checks"])
    assert all(side["strongest_score"] == 0
               for side in diagnostic["side_checks"])
    assert "image" not in str(diagnostic)


@pytest.mark.parametrize("source", [
    None, [],
    np.zeros((250, 350), np.uint8),
    np.zeros((250, 350, 4), np.uint8),
    np.zeros((250, 350, 3), np.float32),
])
def test_invalid_images_are_rejected(source):
    diagnostic = {}
    assert propose_independent_opposite_sides(
        source, [[0, 0]] * 4, diagnostics=diagnostic
    ) == []
    assert diagnostic["status"] == "not_evaluated"
    assert "corners" not in diagnostic


def test_invalid_geometry_and_controls_are_rejected():
    image, points = _scene()
    assert propose_independent_opposite_sides(image, [[0, 0]]*4) == []
    assert propose_independent_opposite_sides(image, points, pair_side_index=3) == []
    assert propose_independent_opposite_sides(image, points, max_results=0) == []


def test_reuse_scaled_lab_without_modifying_source():
    image, true = _scene(rotation=28)
    scaled = cv2.resize(image, (1660, 1240))
    wrong = _offset_both(true, 24., 18.) * 2
    prepared = prepare_side_recovery_lab(scaled, max_longest=900)
    assert prepared[1] < 1.
    before = scaled.copy()
    results = propose_independent_opposite_sides(
        scaled, wrong, prepared=prepared, max_results=4,
    )
    assert results
    assert min(_error(row["corners"], true * 2) for row in results) < 22
    assert np.array_equal(scaled, before)


def test_independent_sides_do_not_change_selection_or_manual_fallback(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, _ = _scene(18)
    original = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    assert "pr6_independent_sides_shadow" in original
    assert original["pr6_independent_sides_shadow"]["auto_crop_allowed"] is False
    monkeypatch.setattr(
        comparison, "propose_independent_opposite_sides",
        lambda *_args, **_kwargs: [],
    )
    disabled = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    for key in (
        "pr6_review_consolidation_shadow",
        "pr6_candidate_decision_shadow",
        "pr6_manual_fallback_shadow",
        "v3_ranking",
        "pr6_side_recovery_shadow",
        "pr6_paired_alignment_shadow",
    ):
        assert original[key] == disabled[key]
    assert disabled["pr6_independent_sides_shadow"]["candidate_count"] == 0


def test_snapshot_exposes_independent_side_warning_without_auto_crop():
    from autodocscanner.core.pr6_visual_diagnostics import (
        build_pr6_visual_diagnostics,
    )

    image, true = _scene()
    wrong = _offset_both(true)
    result = propose_independent_opposite_sides(
        image, wrong, max_results=4,
    )
    assert result
    report = {
        "stable": {}, "adaptive": {}, "v2": [],
        "pr6_independent_sides_shadow": {
            "status": "offline_visual_hypotheses_only",
            "candidates": [{
                **result[0],
                "parent_source": "side_recovery_3",
                "outer_region_source": "v2_7",
            }],
            "parent_checks": [], "auto_select": None,
            "auto_crop_allowed": False,
        }
    }
    visual = build_pr6_visual_diagnostics(report, image.shape, image=image)
    assert visual["family_counts"]["Independent Sides"] == 1
    candidate = next(x for x in visual["candidates"]
                     if x["family"] == "Independent Sides")
    assert candidate["independent_sides_details"]["parent_source"] == "side_recovery_3"
    assert candidate["ranking_score"] is None
    assert visual["auto_select"] is None
    assert visual["auto_crop_allowed"] is False



def test_original_outer_region_examined_on_both_axes_without_side_recovery(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, quad = _scene(20)
    monkeypatch.setattr(comparison, "contour_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(
        comparison, "region_outer_quad_candidates",
        lambda *_a, **_k: [quad.copy()],
    )
    monkeypatch.setattr(
        comparison, "recover_one_side_hypotheses", lambda *_a, **_k: [],
    )
    visited = []

    def check(_image, points, *, pair_side_index, diagnostics=None, **_):
        visited.append(pair_side_index)
        if diagnostics is not None:
            diagnostics.update({
                "status": "rejected",
                "paired_side_indices": [pair_side_index, pair_side_index+2],
                "failure_reason": "unsupported_boundary_for_one_or_both_sides",
            })
        return []

    monkeypatch.setattr(
        comparison, "propose_independent_opposite_sides", check,
    )
    report = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    independent = report["pr6_independent_sides_shadow"]
    assert visited == [0, 1]
    assert report["pr6_side_recovery_shadow"]["candidate_count"] == 0
    assert independent["direct_outer_region_parents_examined"] == 1
    assert independent["direct_outer_region_pair_checks"] == 2
    assert [item["source"] for item in independent["parent_checks"]] == [
        "v2_0", "v2_0",
    ]
    assert independent["auto_select"] is None
    assert independent["auto_crop_allowed"] is False


def test_small_independent_result_remains_visible_below_old_dedup_range(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, quad = _scene(15)
    monkeypatch.setattr(comparison, "contour_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(
        comparison, "region_outer_quad_candidates",
        lambda *_a, **_k: [quad.copy()],
    )
    monkeypatch.setattr(
        comparison, "recover_one_side_hypotheses", lambda *_a, **_k: [],
    )

    def check(_image, points, *, pair_side_index, **_):
        if pair_side_index == 1:
            return []
        return [{
            "corners": (np.asarray(points, np.float64) + [0, 14]).tolist(),
            "auto_select": None,
            "auto_crop_allowed": False,
            "paired_side_indices": [0, 2],
        }]

    monkeypatch.setattr(
        comparison, "propose_independent_opposite_sides", check,
    )
    result = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    independent = result["pr6_independent_sides_shadow"]
    assert independent["candidate_count"] == 1
    assert independent["candidates"][0]["parent_source"] == "v2_0"
    assert independent["candidates"][0]["parent_family"] == "outer_region"
    assert result["pr6_review_consolidation_shadow"]["auto_crop_allowed"] is False
