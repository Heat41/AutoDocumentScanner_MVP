"""PR6 side-wise physical border recovery: synthetic, no real KTP data."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from autodocscanner.core.side_boundary_recovery_pr6 import (
    recover_one_side_hypotheses,
)


def _scene(rotation=0, *, grayscale_card=False, textured=True):
    height, width = 620, 830
    background = np.full((height, width, 3), (155, 169, 175), np.uint8)
    random = np.random.default_rng(174)
    if textured:
        # Cluttered but non-document line texture; no KTP-specific colors.
        for _ in range(100):
            a = (
                int(random.integers(0, width)),
                int(random.integers(0, height)),
            )
            b = (
                int(random.integers(0, width)),
                int(random.integers(0, height)),
            )
            cv2.line(background, a, b, (160, 166, 169), 2)

    card_color = (
        (62, 62, 62) if grayscale_card else (62, 139, 205)
    )
    card = np.full((205, 342, 3), card_color, np.uint8)
    for i in range(7):
        y = 25 + 24 * i
        cv2.line(card, (22, y), (188, y), (44, 72, 85), 1)
    cv2.rectangle(card, (248, 34), (311, 162), (70, 72, 76), -1)

    theta = math.radians(rotation)
    matrix = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])
    relative = np.array([
        [-171, -102.5], [171, -102.5],
        [171, 102.5], [-171, 102.5],
    ], dtype=float)
    true = (
        relative @ matrix.T + np.array([width / 2, height / 2])
    ).astype(np.float32)
    source = np.array(
        [[0, 0], [341, 0], [341, 204], [0, 204]],
        np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, true)
    warped = cv2.warpPerspective(card, transform, (width, height))
    mask = cv2.warpPerspective(
        np.full((205, 342), 255, np.uint8),
        transform, (width, height),
    )
    background[mask > 128] = warped[mask > 128]
    return background, true


def _wrong_one_side(true, shift=76.):
    points = np.asarray(true, dtype=np.float64)
    a, b = points[0], points[1]
    vector = b - a
    normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
    center = points.mean(axis=0)
    if np.dot(normal, center - (a + b) / 2) < 0:
        normal = -normal
    wrong = points.copy()
    wrong[0] -= normal * shift
    wrong[1] -= normal * shift
    return wrong


def _matching_corner_error(found, truth):
    found = np.asarray(found, float)
    truth = np.asarray(truth, float)
    variants = [
        np.roll(truth, i, axis=0)
        for i in range(4)
    ] + [
        np.roll(truth[::-1], i, axis=0)
        for i in range(4)
    ]
    return min(
        np.linalg.norm(found - variant, axis=1).mean()
        for variant in variants
    )


@pytest.mark.parametrize("rotation", [0, 22, 51, 93, 139])
def test_one_bad_outer_side_recovered_for_different_rotations(rotation):
    image, true = _scene(rotation)
    before = image.copy()
    wrong = _wrong_one_side(true)
    old_error = _matching_corner_error(wrong, true)
    candidates = recover_one_side_hypotheses(
        image, wrong, max_results=4,
    )
    assert candidates
    new_error = min(
        _matching_corner_error(item["corners"], true)
        for item in candidates
    )
    assert old_error > 30.
    assert new_error < 12.
    assert new_error < old_error / 3.
    assert all(item["auto_select"] is None and
               item["auto_crop_allowed"] is False for item in candidates)
    assert np.array_equal(image, before)


def test_already_aligned_outer_border_is_not_relocated_to_portrait_box():
    image, true = _scene(18)
    assert recover_one_side_hypotheses(image, true) == []


def test_neutral_colored_card_can_recover_a_side_without_blue_prior():
    image, true = _scene(37, grayscale_card=True)
    candidates = recover_one_side_hypotheses(
        image, _wrong_one_side(true, shift=66.),
    )
    assert candidates
    assert min(
        _matching_corner_error(item["corners"], true)
        for item in candidates
    ) < 13.


def test_uniform_no_document_produces_no_recovery():
    image = np.full((600, 810, 3), 150, np.uint8)
    corners = np.asarray([
        [190, 110], [540, 110],
        [540, 430], [190, 430],
    ], dtype=np.float32)
    assert recover_one_side_hypotheses(image, corners) == []


@pytest.mark.parametrize("bad_image", [
    None, [], np.zeros((260, 410), np.uint8),
    np.zeros((260, 410, 4), np.uint8),
    np.zeros((260, 410, 3), np.float32),
])
def test_invalid_image_never_returns_proposal(bad_image):
    assert recover_one_side_hypotheses(
        bad_image, [[0, 0]] * 4,
    ) == []


def test_invalid_corners_or_unbounded_result_request_fail_closed():
    image, true = _scene()
    assert recover_one_side_hypotheses(
        image, [[30, 30]] * 4,
    ) == []
    assert recover_one_side_hypotheses(
        image, true, max_results=0,
    ) == []


def test_shadow_recovery_never_changes_review_gate_or_manual_fallback(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, _ = _scene(22)
    original = comparison.recover_one_side_hypotheses
    actual = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    assert "pr6_side_recovery_shadow" in actual
    recovery = actual["pr6_side_recovery_shadow"]
    assert recovery["auto_select"] is None
    assert recovery["auto_crop_allowed"] is False
    assert recovery["status"] == "offline_visual_hypotheses_only"

    monkeypatch.setattr(
        comparison, "recover_one_side_hypotheses",
        lambda *_args, **_kwargs: [],
    )
    without = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    for key in (
        "pr6_review_consolidation_shadow",
        "pr6_candidate_decision_shadow",
        "pr6_manual_fallback_shadow",
        "v3_ranking",
    ):
        assert actual[key] == without[key]
    assert without["pr6_side_recovery_shadow"]["candidate_count"] == 0
    assert original is not comparison.recover_one_side_hypotheses


def test_visual_snapshot_keeps_recovered_side_in_separate_family():
    from autodocscanner.core.pr6_visual_diagnostics import (
        build_pr6_visual_diagnostics,
    )

    image, true = _scene(0)
    wrong = _wrong_one_side(true)
    hypothesis = recover_one_side_hypotheses(image, wrong)[0]
    report = {
        "stable": {"found": False}, "adaptive": {"found": False},
        "v2": [],
        "pr6_side_recovery_shadow": {
            "status": "offline_visual_hypotheses_only",
            "auto_select": None, "auto_crop_allowed": False,
            "candidates": [
                {**hypothesis, "parent_source": "v2_9"},
            ],
        },
    }
    debug = build_pr6_visual_diagnostics(
        report, image.shape, image=image,
    )
    assert debug["family_counts"]["Side Recovery"] == 1
    assert debug["generated_side_recovery_count"] == 1
    recovered = next(
        row for row in debug["candidates"]
        if row["family"] == "Side Recovery"
    )
    assert recovered["side_recovery_details"]["parent_source"] == "v2_9"
    assert recovered["ranking_score"] is None
    assert recovered["outer_recovery_check"] is None
    assert debug["auto_select"] is None
    assert debug["auto_crop_allowed"] is False



@pytest.mark.parametrize("rotation", [0, 19, 48, 92, 134, 159])
def test_skewed_candidate_side_recovery_with_perspective_and_texture(rotation):
    image, true = _scene(rotation, textured=True)
    wrong = _wrong_one_side(true, shift=110)
    # Make the wrong side nonparallel to the physical card edge. Pure
    # translations previously could not fit the real boundary.
    a, b = true[0].astype(float), true[1].astype(float)
    direction = b - a
    inward = np.array([-direction[1], direction[0]])
    inward /= np.linalg.norm(inward)
    if np.dot(inward, true.mean(axis=0) - (a + b)/2) < 0:
        inward = -inward
    wrong[0] += inward * 18
    wrong[1] -= inward * 18
    before = image.copy()
    candidates = recover_one_side_hypotheses(image, wrong, max_results=4)
    assert candidates
    assert min(
        _matching_corner_error(candidate["corners"], true)
        for candidate in candidates
    ) < 14
    assert any(
        abs(item["side_tilt_fraction_of_parent_gap"]) > 0
        for item in candidates
    )
    assert np.array_equal(image, before)


def test_oversized_background_quadrilateral_can_shrink_below_old_area_gate():
    image, true = _scene(rotation=18, textured=True)
    # Padding keeps a severely displaced side within the source frame.
    padded = cv2.copyMakeBorder(
        image, 160, 160, 160, 160,
        cv2.BORDER_CONSTANT, value=(155, 169, 175),
    )
    true = true + 160
    oversized = _wrong_one_side(true, shift=225)
    original_error = _matching_corner_error(oversized, true)
    assert original_error > 100
    candidates = recover_one_side_hypotheses(
        padded, oversized, max_results=4
    )
    assert candidates
    assert min(
        _matching_corner_error(item["corners"], true)
        for item in candidates
    ) < 15


def test_prepared_scaled_lab_is_reusable_without_changing_original_image():
    from autodocscanner.core.side_boundary_recovery_pr6 import (
        prepare_side_recovery_lab,
    )

    image, true = _scene(20)
    enlarged = cv2.resize(image, (1660, 1240))
    wrong = _wrong_one_side(true) * 2
    prepared = prepare_side_recovery_lab(enlarged, max_longest=950)
    assert prepared is not None
    lab, scale = prepared
    assert lab.dtype == np.float32 and 0 < scale < 1
    assert max(lab.shape[:2]) <= 950
    before = enlarged.copy()
    candidates = recover_one_side_hypotheses(
        enlarged, wrong, prepared=prepared, max_results=4,
    )
    assert candidates
    assert min(
        _matching_corner_error(item["corners"], true * 2)
        for item in candidates
    ) < 25
    assert np.array_equal(enlarged, before)


def test_no_unverified_photo_pixels_or_personal_fields_in_recovery_report():
    image, true = _scene(22)
    proposals = recover_one_side_hypotheses(
        image, _wrong_one_side(true),
    )
    assert proposals
    for proposal in proposals:
        assert proposal["auto_select"] is None
        assert proposal["auto_crop_allowed"] is False
        assert "image" not in proposal
        assert "identity" not in proposal
        assert "expected" not in proposal
        assert "ground_truth" not in proposal



def test_zero_side_recovery_explains_why_after_search(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, true = _scene(rotation=22)
    monkeypatch.setattr(
        comparison, "recover_one_side_hypotheses",
        lambda *_args, **_kwargs: [],
    )
    report = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    details = report["pr6_side_recovery_shadow"]
    assert details["parents_examined"] >= 1
    assert details["hypotheses_before_dedup"] == 0
    assert details["reason_codes"] == [
        "no_side_shift_passed_visual_evidence_and_geometry"
    ]
    assert details["candidate_count"] == 0
    assert details["auto_crop_allowed"] is False


def test_no_outer_region_parent_is_distinguished_from_failed_relocation(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, true = _scene(rotation=22)
    monkeypatch.setattr(
        comparison, "region_outer_quad_candidates",
        lambda *_args, **_kwargs: [],
    )
    report = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    details = report["pr6_side_recovery_shadow"]
    assert details["parents_examined"] == 0
    assert details["reason_codes"] == ["no_outer_region_parent"]
    assert details["candidate_count"] == 0
    assert report["pr6_review_consolidation_shadow"]["auto_crop_allowed"] is False
