"""PR6 chromatic-edge proposals: synthetic variations only, no personal images.

These tests establish hypothesis recall for low-luminance-contrast cards.
They do NOT validate an automatic crop or any real-world KTP accuracy.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from autodocscanner.core.chroma_quad_proposals_pr6 import (
    chroma_quad_candidates,
)


def _photo(*, seed=12, rotation=20, clutter=True):
    rng = np.random.default_rng(seed)
    height, width = 620, 800
    # The card and neutral fabric have similar grayscale luminance. Spatial
    # tint transitions in Lab a/b, not fixed card colors, reveal the border.
    base = np.clip(
        130 + rng.normal(0, 6, (height, width, 1)), 0, 255
    ).astype(np.uint8)
    image = np.repeat(base, 3, axis=2)
    if clutter:
        for _ in range(40):
            x1, y1 = (int(v) for v in rng.integers([0, 0], [width, height]))
            x2, y2 = (int(v) for v in rng.integers([0, 0], [width, height]))
            cv2.line(image, (x1, y1), (x2, y2), (150, 150, 150), 1)

    card = np.full((200, 335, 3), (55, 120, 180), dtype=np.uint8)
    for index in range(6):
        y = 30 + index * 24
        cv2.line(card, (20, y), (185, y), (80, 110, 150), 1)

    source = np.array(
        [[0, 0], [334, 0], [334, 199], [0, 199]], np.float32
    )
    angle = math.radians(rotation)
    rotation_matrix = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    centered = np.array([
        [-167, -100], [167, -100], [167, 100], [-167, 100],
    ], np.float64)
    target = (centered @ rotation_matrix.T
              + np.array([width / 2, height / 2])
              + rng.uniform(-9, 9, (4, 2)))
    transform = cv2.getPerspectiveTransform(
        source, target.astype(np.float32),
    )
    warped = cv2.warpPerspective(card, transform, (width, height))
    mask = cv2.warpPerspective(
        np.full((200, 335), 255, np.uint8), transform, (width, height),
    )
    image[mask > 100] = warped[mask > 100]
    return image, target


def _corner_error(candidate, truth):
    distances = np.linalg.norm(
        np.asarray(candidate)[:, None] -
        np.asarray(truth)[None, :], axis=2,
    )
    return float(distances.min(axis=1).mean())


@pytest.mark.parametrize("seed,rotation", [
    (12, 0), (12, 20), (12, 45), (12, 72), (12, 120),
    (19, 153), (41, 28),
])
def test_recalls_colored_card_border_on_brightness_ambiguous_background(
        seed, rotation):
    photo, ground_truth = _photo(seed=seed, rotation=rotation)
    before = photo.copy()
    candidates = chroma_quad_candidates(photo)
    assert candidates
    assert min(_corner_error(x, ground_truth) for x in candidates) < 12.
    assert len(candidates) <= 8
    assert np.array_equal(photo, before)


def test_neutral_uniform_image_never_invents_candidate():
    source = np.full((380, 540, 3), 128, np.uint8)
    assert chroma_quad_candidates(source) == []


def test_no_cropping_or_auto_selection_from_generator():
    source, _ = _photo(rotation=68)
    result = chroma_quad_candidates(source, max_candidates=2)
    assert len(result) <= 2
    assert all(isinstance(x, np.ndarray) and x.shape == (4, 2)
               and np.isfinite(x).all() for x in result)


@pytest.mark.parametrize("invalid", [
    None, [], np.zeros((35, 35, 3), dtype=np.uint8),
    np.zeros((280, 410), dtype=np.uint8),
    np.zeros((280, 410, 4), dtype=np.uint8),
    np.zeros((280, 410, 3), dtype=np.float32),
])
def test_invalid_input_returns_no_suggestion(invalid):
    assert chroma_quad_candidates(invalid) == []


def test_max_candidates_is_bounded():
    source, _ = _photo()
    assert len(chroma_quad_candidates(source, max_candidates=1)) <= 1
    assert chroma_quad_candidates(source, max_candidates=0) == []


def test_new_generator_only_runs_in_explicit_optional_pr6_review(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, _ = _photo(seed=12, rotation=20)
    called = []
    real = comparison.chroma_quad_candidates

    def counted(*args, **kwargs):
        called.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(comparison, "chroma_quad_candidates", counted)
    old = comparison.compare_detectors(
        image, include_line_candidates=False, include_chroma_candidates=False
    )
    assert not called
    assert old["pr6_chroma_proposal_count"] == 0

    augmented = comparison.compare_detectors(
        image, include_line_candidates=False, include_chroma_candidates=True
    )
    assert called
    assert augmented["pr6_chroma_proposal_count"] >= 1
    assert "chroma" in augmented["pr6_v2_proposal_origins"]
    assert augmented["pr6_review_consolidation_shadow"]["auto_select"] is None
    assert augmented["pr6_review_consolidation_shadow"]["auto_crop_allowed"] is False
    assert augmented["pr6_manual_fallback_shadow"]["auto_crop_allowed"] is False


def test_runtime_chromatic_hypothesis_remains_read_only():
    from autodocscanner.core.pr6_runtime_review import (
        evaluate_pr6_runtime_image,
    )

    image, _ = _photo(seed=19, rotation=153)
    report = evaluate_pr6_runtime_image(image)
    visual = report["visual_debug"]
    assert visual["candidate_count"] >= 1
    assert visual["generated_chroma_count"] >= 1
    assert visual["family_counts"]["Chroma"] >= 1
    assert report["auto_crop_allowed"] is False
    assert report["auto_select"] is None
    assert report["file_write_performed"] is False
    assert report["crop_performed"] is False
