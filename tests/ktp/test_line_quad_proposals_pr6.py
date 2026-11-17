"""PR6 independent-line proposals: synthetic, diverse images only.

These tests assess *hypothesis recall*, never automatic-crop correctness.
No real KTP data, annotations, OCR, or file names are involved.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from autodocscanner.core.line_quad_proposals_pr6 import line_quad_candidates


def _synthetic_card(*, seed=2, rotation=8, clutter=20):
    random = np.random.default_rng(seed)
    height, width = 660, 880
    photo = np.full((height, width, 3), 175, dtype=np.uint8)
    # Non-card lines and texture may contain false rectangular evidence.
    for _ in range(clutter):
        x1, y1 = (int(x) for x in random.integers([0, 0], [width, height]))
        x2, y2 = (int(x) for x in random.integers([0, 0], [width, height]))
        color = tuple(int(x) for x in random.integers(30, 245, 3))
        cv2.line(photo, (x1, y1), (x2, y2), color, int(random.integers(1, 3)))

    card_width, card_height = 340, 230
    card = np.full((card_height, card_width, 3), (195, 185, 130), dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (card_width - 1, card_height - 1),
                  (40, 50, 65), 3)
    cv2.rectangle(card, (238, 39), (306, 161), (90, 90, 90), 2)
    for index in range(8):
        y = 20 + index * 20
        cv2.line(card, (20, y), (card_width // 2, y), (40, 50, 60), 1)

    angle = math.radians(rotation)
    rotation_matrix = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    source_quad = np.array([[0, 0], [339, 0], [339, 229], [0, 229]], np.float32)
    relative = np.array([
        [-170, -115], [170, -115], [170, 115], [-170, 115],
    ], dtype=np.float64)
    target = relative @ rotation_matrix.T + np.array([width / 2, height / 2])
    target += random.uniform(-12, 12, (4, 2))
    matrix = cv2.getPerspectiveTransform(source_quad, target.astype(np.float32))
    warped = cv2.warpPerspective(card, matrix, (width, height))
    mask = cv2.warpPerspective(
        np.full((card_height, card_width), 255, dtype=np.uint8),
        matrix, (width, height),
    )
    photo[mask > 128] = warped[mask > 128]
    return photo, target


def _unordered_error(proposal, expected):
    distance = np.linalg.norm(
        proposal[:, None, :] - expected[None, :, :], axis=2
    )
    return float(distance.min(axis=1).mean())


@pytest.mark.parametrize("rotation", [8, 21, 39, 66, 130, 155])
def test_recovers_outer_quad_from_four_independent_edges_in_clutter(rotation):
    image, expected = _synthetic_card(rotation=rotation, clutter=20)
    before = image.copy()
    proposals = line_quad_candidates(image)
    assert 1 <= len(proposals) <= 6
    assert min(_unordered_error(p, expected) for p in proposals) < 9.
    assert np.array_equal(image, before)  # no source mutation, even on failure


def test_hypothesis_recall_on_different_backgrounds_includes_hard_cases():
    # Different OpenCV versions produce slightly different Hough segments.
    # Do not silently remove challenging orientations from the fixture:
    # measure recall and retain failed hypotheses for follow-up validation.
    errors = {}
    for rotation in (8, 39, 66, 130, 155):
        image, expected = _synthetic_card(
            seed=7, rotation=rotation, clutter=20
        )
        proposals = line_quad_candidates(image, max_candidates=6)
        errors[rotation] = (
            min(_unordered_error(p, expected) for p in proposals)
            if proposals else float("inf")
        )
    # Proposals are never automatic accepted crops. This checks a measured
    # range of recovery across background variation, not universal accuracy.
    assert sum(error < 10 for error in errors.values()) >= 4, errors


def test_only_returns_proposals_not_acceptance_or_confidence():
    image, _ = _synthetic_card()
    proposed = line_quad_candidates(image, max_candidates=2)
    assert len(proposed) <= 2
    assert all(isinstance(x, np.ndarray) and x.shape == (4, 2)
               and np.isfinite(x).all() for x in proposed)


@pytest.mark.parametrize("bad", [
    None,
    np.zeros((20, 20, 3), dtype=np.uint8),
    np.zeros((300, 400), dtype=np.uint8),
    np.zeros((300, 400, 4), dtype=np.uint8),
    np.zeros((300, 400, 3), dtype=np.float64),
])
def test_missing_or_invalid_image_never_proposes_quad(bad):
    assert line_quad_candidates(bad) == []


def test_uniform_background_does_not_fabricate_a_card():
    image = np.zeros((350, 530, 3), dtype=np.uint8)
    assert line_quad_candidates(image) == []


def test_no_absolute_size_or_fixed_orientation_assumption():
    image, expected = _synthetic_card(rotation=39, clutter=8)
    scaled = cv2.resize(image, (440, 330), interpolation=cv2.INTER_AREA)
    proposals = line_quad_candidates(scaled)
    assert all((p[:, 0] >= -1).all() and (p[:, 0] <= 441).all()
               and (p[:, 1] >= -1).all() and (p[:, 1] <= 331).all()
               for p in proposals)


def test_line_proposals_are_optional_and_do_not_change_baseline_report(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison
    image, expected = _synthetic_card(rotation=39, clutter=8)

    # Contour-only reports continue to have no injected Hough candidates.
    calls = []
    original = comparison.line_quad_candidates

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(comparison, "line_quad_candidates", counted)
    baseline = comparison.compare_detectors(image, include_line_candidates=False)
    assert not calls
    assert baseline["pr6_line_proposal_count"] == 0

    enhanced = comparison.compare_detectors(image, include_line_candidates=True)
    assert calls
    assert enhanced["pr6_line_proposal_count"] >= 0
    assert enhanced["pr6_review_consolidation_shadow"]["auto_crop_allowed"] is False
    assert enhanced["pr6_review_consolidation_shadow"]["auto_select"] is None
    assert enhanced["pr6_manual_fallback_shadow"]["auto_crop_allowed"] is False



def test_offline_cli_can_compare_contour_only_and_long_edges(tmp_path, monkeypatch):
    import json
    import sys
    from autodocscanner.core import detector_comparison as comparison

    image_path = tmp_path / "synthetic.png"
    image, _ = _synthetic_card(rotation=21, clutter=8)
    assert cv2.imwrite(str(image_path), image)
    calls = []

    def fake_compare(image, *, expected=None, max_proposals=24,
                     include_line_candidates=False):
        calls.append((expected, include_line_candidates))
        return {"four_line_enabled": include_line_candidates}

    monkeypatch.setattr(comparison, "compare_detectors", fake_compare)
    old_report = tmp_path / "baseline.json"
    new_report = tmp_path / "four_line.json"
    monkeypatch.setattr(sys, "argv", [
        "detector_comparison.py", str(tmp_path), "--output", str(old_report),
    ])
    comparison.main()
    monkeypatch.setattr(sys, "argv", [
        "detector_comparison.py", str(tmp_path),
        "--include-line-proposals", "--output", str(new_report),
    ])
    comparison.main()

    assert calls == [(None, False), (None, True)]
    baseline = json.loads(old_report.read_text(encoding="utf-8"))
    extended = json.loads(new_report.read_text(encoding="utf-8"))
    assert baseline["synthetic.png"]["four_line_enabled"] is False
    assert extended["synthetic.png"]["four_line_enabled"] is True
