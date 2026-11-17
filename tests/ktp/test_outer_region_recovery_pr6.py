"""Region-first PR6 card boundary proposals, synthetic/anonymous CV fixtures.

This is candidate recall, NOT an automatic KTP detection success metric.
No personal documents, ground-truth at runtime, or output image writes.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from autodocscanner.core.outer_region_recovery_pr6 import (
    region_outer_quad_candidates, inspect_outer_perspective_hypothesis,
)


def _scene(*, rotation=17, color=(180, 135, 70), seed=3,
           background=(133, 134, 135), card_scale=1.):
    rng = np.random.default_rng(seed)
    h, w = 520, 720
    noise = rng.normal(0, 5, size=(h, w, 1))
    image = np.clip(
        np.asarray(background, dtype=float)[None, None, :] + noise,
        0, 255,
    ).astype(np.uint8)
    # Background has many edges and unrelated colored objects, so a
    # region-quad proposal must not require a blank studio background.
    for index in range(34):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        dx = int(rng.integers(4, 85))
        cv2.line(image, (x, y), (min(w-1, x+dx), y),
                 tuple(int(a) for a in background), 1)

    card_w, card_h = 324, 205
    card = np.full((card_h, card_w, 3), color, dtype=np.uint8)
    cv2.rectangle(card, (9, 11), (card_w - 11, card_h - 11),
                  (int(color[0] * .9), int(color[1] * .9),
                   int(color[2] * .9)), 2)
    cv2.rectangle(card, (235, 37), (295, 157), (72, 72, 72), -1)
    for index in range(7):
        y = 26 + index * 20
        cv2.line(card, (17, y), (183, y), (96, 96, 96), 1)

    theta = math.radians(rotation)
    rotation_matrix = np.asarray([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])
    relative = np.asarray([
        [-162, -102.5], [162, -102.5], [162, 102.5], [-162, 102.5]
    ], dtype=float)
    true = (
        relative @ rotation_matrix.T * card_scale
        + [w / 2, h / 2]
        + rng.uniform(-8, 8, (4, 2))
    ).astype(np.float32)
    source = np.asarray(
        [[0, 0], [card_w-1, 0], [card_w-1, card_h-1], [0, card_h-1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, true)
    warped = cv2.warpPerspective(card, matrix, (w, h))
    mask = cv2.warpPerspective(
        np.full((card_h, card_w), 255, np.uint8), matrix, (w, h)
    )
    image[mask > 128] = warped[mask > 128]
    return image, true


def _corner_error(candidate, true):
    """Order independent matching for rotated output; only used in tests."""
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(true, dtype=float)
    options = []
    for shifted in range(4):
        options.append(
            np.linalg.norm(a - np.roll(b, shifted, axis=0), axis=1).mean()
        )
        options.append(
            np.linalg.norm(a - np.roll(b[::-1], shifted, axis=0), axis=1).mean()
        )
    return min(options)


@pytest.mark.parametrize("rotation,seed,color", [
    (0, 3, (180, 135, 70)),
    (22, 7, (70, 180, 150)),
    (51, 11, (192, 182, 90)),
    (89, 19, (90, 130, 220)),
    (135, 23, (185, 100, 140)),
])
def test_region_recovers_outer_corners_from_multiscale_colored_card(
        rotation, seed, color):
    image, true = _scene(rotation=rotation, seed=seed, color=color)
    before = image.copy()
    candidates = region_outer_quad_candidates(image)
    assert candidates
    assert len(candidates) <= 12
    assert min(_corner_error(c, true) for c in candidates) < 32
    assert np.array_equal(image, before)


def test_far_card_not_eliminated_by_large_background_regions():
    image, true = _scene(rotation=31, card_scale=.62)
    quads = region_outer_quad_candidates(image)
    assert quads
    assert min(_corner_error(c, true) for c in quads) < 24


def test_neutral_grayscale_card_not_assumed_to_be_blue():
    image, true = _scene(
        rotation=13, color=(205, 205, 205), background=(110, 110, 110),
    )
    result = region_outer_quad_candidates(image)
    assert result
    assert min(_corner_error(c, true) for c in result) < 30


def test_uniform_source_yields_no_region_hypothesis():
    image = np.full((390, 620, 3), 112, dtype=np.uint8)
    assert region_outer_quad_candidates(image) == []


@pytest.mark.parametrize("invalid", [
    None, [], np.zeros((30, 30, 3), np.uint8),
    np.zeros((260, 390), np.uint8),
    np.zeros((260, 390, 4), np.uint8),
    np.zeros((260, 390, 3), np.float32),
])
def test_invalid_source_returns_no_candidate(invalid):
    assert region_outer_quad_candidates(invalid) == []


def test_no_unbounded_candidate_generation():
    image, _ = _scene()
    assert len(region_outer_quad_candidates(image, max_candidates=2)) <= 2
    assert region_outer_quad_candidates(image, max_candidates=0) == []


def test_perspective_inspection_flags_internal_rectangle_even_if_edges_strong():
    image, outer = _scene()
    outer = np.asarray(outer, dtype=float)
    center = outer.mean(axis=0)
    photo_box = center + (outer - center) * .30
    measured = inspect_outer_perspective_hypothesis(
        image, photo_box, other_quads=[outer],
    )
    assert measured["status"] == "manual_review_only"
    assert measured["nested_in_larger_quad"] is True
    assert "possible_internal_rectangle" in measured["reason_codes"]
    assert measured["auto_select"] is None
    assert measured["auto_crop_allowed"] is False


def test_outer_perspective_hypothesis_is_not_an_automatic_crop_authorization():
    image, outer = _scene(rotation=18)
    before = image.copy()
    result = inspect_outer_perspective_hypothesis(
        image, outer,
    )
    assert result["perspective_feasible"] is True
    assert len(result["chromatic_step_by_side"]) == 4
    assert result["auto_crop_allowed"] is False
    assert result["auto_select"] is None
    assert np.array_equal(image, before)


def test_reject_invalid_perspective_geometry_without_warping():
    image, _ = _scene()
    bad = np.asarray([[10, 10]] * 4, np.float32)
    result = inspect_outer_perspective_hypothesis(image, bad)
    assert result["perspective_feasible"] is False
    assert "invalid_geometry" in result["reason_codes"]
    assert result["auto_crop_allowed"] is False


def test_opt_in_region_candidates_only_do_not_change_legacy_comparison(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, _ = _scene(rotation=19)
    calls = []
    original = comparison.region_outer_quad_candidates

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(comparison, "region_outer_quad_candidates", counted)
    base = comparison.compare_detectors(
        image, include_region_candidates=False,
    )
    assert calls == []
    assert base["pr6_region_proposal_count"] == 0
    assert "pr6_outer_region_recovery_shadow" not in base

    enlarged = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    assert calls
    assert enlarged["pr6_region_proposal_count"] >= 1
    assert "outer_region" in enlarged["pr6_v2_proposal_origins"]
    assert enlarged["pr6_outer_region_recovery_shadow"]["auto_select"] is None
    assert enlarged["pr6_outer_region_recovery_shadow"]["auto_crop_allowed"] is False
    assert enlarged["pr6_review_consolidation_shadow"]["auto_crop_allowed"] is False
    assert enlarged["pr6_manual_fallback_shadow"]["auto_crop_allowed"] is False


def test_no_personal_annotations_in_perspective_inspection():
    image, quad = _scene()
    out = inspect_outer_perspective_hypothesis(image, quad)
    assert "source_image" not in out
    assert "ground_truth" not in out
    assert "file_path" not in out



def test_contextual_region_ranking_preserves_card_hypotheses_amid_distractors():
    image, outer = _scene(rotation=18, seed=7)
    # A large, clean, saturated external object can outrank the actual
    # document under area/fill heuristics. Synthetic distractor only.
    cv2.rectangle(image, (25, 25), (235, 145), (200, 65, 60), -1)
    center = np.asarray(outer, float).mean(axis=0)
    inner = center + (np.asarray(outer, float) - center) * .3
    context = [{
        "family": "contour", "corners": inner.tolist(),
    }]
    plain = region_outer_quad_candidates(image, max_candidates=12)
    ranked = region_outer_quad_candidates(
        image, max_candidates=12, context_candidates=context,
        contextual_ranking=True,
    )
    assert plain and ranked
    assert min(_corner_error(c, outer) for c in ranked) < 35
    assert len(ranked) <= 12
    # Reranking is review only, never confirmation of target identity.
    assert len(ranked) == len(plain)


def test_context_ranking_no_document_and_unknown_support_safe():
    image = np.full((340, 570, 3), 120, np.uint8)
    assert region_outer_quad_candidates(
        image, contextual_ranking=True,
        context_candidates=[
            {"family": "contour", "corners": [[0, 0]] * 4},
        ],
    ) == []


def test_contextual_ranking_does_not_write_or_change_image():
    image, outer = _scene(rotation=51, seed=11)
    original = image.copy()
    result = region_outer_quad_candidates(
        image, contextual_ranking=True,
    )
    assert result
    assert np.array_equal(image, original)
