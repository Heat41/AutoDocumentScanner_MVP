"""PR6 paired opposing-edge alignment: synthetic diverse images only.

The implementation proposes a common small translation only when both
opposite card-boundary signals strengthen. All results remain review-only.
No real KTP pixels, filenames, annotation files, or personal data.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from autodocscanner.core.paired_boundary_alignment_pr6 import (
    propose_paired_alignment,
)
from autodocscanner.core.side_boundary_recovery_pr6 import (
    prepare_side_recovery_lab, _order,
)


def _scene(*, rotation=0, color=(62, 139, 205), clutter=True):
    h, w = 620, 830
    image = np.full((h, w, 3), (155, 169, 175), np.uint8)
    rng = np.random.default_rng(174)
    if clutter:
        for _ in range(100):
            a = (int(rng.integers(w)), int(rng.integers(h)))
            b = (int(rng.integers(w)), int(rng.integers(h)))
            cv2.line(image, a, b, (160, 166, 169), 2)

    card = np.full((205, 342, 3), color, np.uint8)
    for i in range(7):
        y = 25 + 24 * i
        cv2.line(card, (22, y), (188, y), (44, 72, 85), 1)
    cv2.rectangle(card, (248, 34), (311, 162), (70, 72, 76), -1)
    theta = math.radians(rotation)
    rotation_matrix = np.asarray([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ])
    relatives = np.asarray([
        [-171, -102.5], [171, -102.5],
        [171, 102.5], [-171, 102.5],
    ], float)
    corners = (relatives @ rotation_matrix.T + [w / 2, h / 2]).astype(np.float32)
    source = np.asarray([
        [0, 0], [341, 0], [341, 204], [0, 204],
    ], np.float32)
    transform = cv2.getPerspectiveTransform(source, corners)
    warped = cv2.warpPerspective(card, transform, (w, h))
    mask = cv2.warpPerspective(
        np.full((205, 342), 255, np.uint8), transform, (w, h),
    )
    image[mask > 128] = warped[mask > 128]
    return image, corners


def _inward_normal(corners, index):
    corners = _order(corners)
    a, b = np.asarray(corners[index], float), np.asarray(
        corners[(index + 1) % 4], float,
    )
    tangent = b - a
    normal = np.array([-tangent[1], tangent[0]])
    normal /= np.linalg.norm(normal)
    if normal @ (np.asarray(corners).mean(axis=0) - (a + b) / 2) < 0:
        normal = -normal
    return normal


def _mean_error(predicted, expected):
    predicted = np.asarray(predicted, float)
    expected = np.asarray(expected, float)
    variants = [
        np.roll(expected, offset, axis=0) for offset in range(4)
    ] + [
        np.roll(expected[::-1], offset, axis=0) for offset in range(4)
    ]
    return min(
        float(np.linalg.norm(predicted - variant, axis=1).mean())
        for variant in variants
    )


@pytest.mark.parametrize("rotation", [0, 22, 51, 93, 139, 164])
def test_paired_edges_recover_common_shift_across_rotations(rotation):
    # This covers the reported failure mode: the candidate rectangle is
    # displaced *toward* the card interior, leaving both physical borders
    # above the proposed line in the image-relative normal direction.
    signed_displacement = 30
    image, truth = _scene(rotation=rotation)
    before = image.copy()
    displaced = truth + _inward_normal(truth, 0) * signed_displacement
    candidates = propose_paired_alignment(
        image, displaced, pair_side_index=0, max_results=2,
    )
    assert candidates
    closest = min(_mean_error(row["corners"], truth) for row in candidates)
    assert closest < 5.
    assert all(row["paired_side_indices"] == [0, 2] for row in candidates)
    assert all(row["auto_select"] is None and not row["auto_crop_allowed"]
               for row in candidates)
    assert np.array_equal(image, before)


def test_opposite_side_conflict_kept_as_a_measured_hard_case():
    # Moving an entire rectangle can help a candidate displaced outward,
    # but only if *both* opposite boundaries gain edge evidence. On some
    # rotations, texture or internal structures give the already misplaced
    # edge a stronger score. Those failures are kept in evaluation rather
    # than allowing a speculative auto-crop or removing test orientations.
    outcomes = {}
    for rotation in (0, 22, 51, 93, 139, 164):
        image, truth = _scene(rotation=rotation)
        displaced = truth - _inward_normal(truth, 0) * 28
        suggestions = propose_paired_alignment(
            image, displaced, pair_side_index=0, max_results=2,
        )
        outcomes[rotation] = bool(suggestions)
        for suggestion in suggestions:
            assert _mean_error(suggestion["corners"], truth) < 6.
            assert suggestion["auto_select"] is None
            assert suggestion["auto_crop_allowed"] is False
    # Evaluate recovery across ALL six difficult rotations. This threshold
    # does not claim that every rotation can be safely recovered.
    assert sum(outcomes.values()) >= 4, outcomes


@pytest.mark.parametrize("rotation", [0, 35, 89, 146])
def test_paired_side_correction_is_direction_agnostic(rotation):
    image, truth = _scene(rotation=rotation, color=(72, 72, 72))
    displaced = truth + _inward_normal(truth, 1) * 23
    candidates = propose_paired_alignment(
        image, displaced, pair_side_index=1,
    )
    assert candidates
    assert min(_mean_error(row["corners"], truth) for row in candidates) < 7.
    assert all(row["paired_side_indices"] == [1, 3] for row in candidates)


def test_already_correct_card_corners_not_shifted_to_internal_portrait():
    image, truth = _scene(rotation=17)
    assert propose_paired_alignment(image, truth) == []


def test_moving_one_good_side_to_fix_other_is_not_allowed():
    image, truth = _scene(rotation=20)
    wrong = _order(truth).copy()
    wrong[:2] -= _inward_normal(truth, 0) * 65
    # Opposite edge still follows the true card. A shared translation must
    # not worsen it merely to improve the first side.
    assert propose_paired_alignment(image, wrong) == []


def test_featureless_photo_does_not_manufacture_edges():
    source = np.full((420, 640, 3), 127, np.uint8)
    points = [[90, 85], [440, 85], [440, 320], [90, 320]]
    assert propose_paired_alignment(source, points) == []


@pytest.mark.parametrize("bad_source", [
    None, [],
    np.zeros((260, 350), np.uint8),
    np.zeros((260, 350, 4), np.uint8),
    np.zeros((260, 350, 3), np.float32),
])
def test_invalid_sources_fail_closed(bad_source):
    assert propose_paired_alignment(bad_source, [[0, 0]] * 4) == []


def test_out_of_bounds_quad_and_invalid_controls_fail_closed():
    image, corners = _scene()
    assert propose_paired_alignment(image, [[0, 0]] * 4) == []
    assert propose_paired_alignment(image, corners, pair_side_index=3) == []
    assert propose_paired_alignment(image, corners, max_results=0) == []


def test_scaled_preprocessing_reuse_preserves_original_photo():
    image, corners = _scene(rotation=30)
    enlarged = cv2.resize(image, (1660, 1240))
    before = enlarged.copy()
    truth = corners * 2
    wrong = truth + _inward_normal(truth, 0) * 55
    prepared = prepare_side_recovery_lab(enlarged, max_longest=900)
    candidates = propose_paired_alignment(
        enlarged, wrong, prepared=prepared,
    )
    assert candidates
    assert min(_mean_error(row["corners"], truth) for row in candidates) < 11.
    assert np.array_equal(enlarged, before)


def test_paired_alignment_is_post_gate_and_review_only(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image, _ = _scene(rotation=18)
    first = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    assert "pr6_paired_alignment_shadow" in first
    paired = first["pr6_paired_alignment_shadow"]
    assert paired["status"] == "offline_visual_hypotheses_only"
    assert paired["auto_select"] is None
    assert paired["auto_crop_allowed"] is False

    monkeypatch.setattr(
        comparison, "propose_paired_alignment",
        lambda *_args, **_kwargs: [],
    )
    second = comparison.compare_detectors(
        image, include_region_candidates=True,
    )
    for key in (
        "pr6_review_consolidation_shadow",
        "pr6_candidate_decision_shadow",
        "pr6_manual_fallback_shadow",
        "v3_ranking",
        "pr6_side_recovery_shadow",
    ):
        assert first[key] == second[key]
    assert second["pr6_paired_alignment_shadow"]["candidate_count"] == 0


def test_visual_debug_carries_parent_and_both_edge_scores():
    from autodocscanner.core.pr6_visual_diagnostics import (
        build_pr6_visual_diagnostics,
    )

    image, corners = _scene()
    wrong = corners + _inward_normal(corners, 0) * 28
    proposed = propose_paired_alignment(image, wrong)[0]
    report = {
        "stable": {}, "adaptive": {}, "v2": [],
        "pr6_paired_alignment_shadow": {
            "candidates": [
                {
                    **proposed, "parent_source": "side_recovery_3",
                    "outer_region_source": "v2_7",
                },
            ],
            "auto_select": None,
            "auto_crop_allowed": False,
        },
    }
    snapshot = build_pr6_visual_diagnostics(
        report, image.shape, image=image,
    )
    assert snapshot["family_counts"]["Paired Alignment"] == 1
    row = next(r for r in snapshot["candidates"]
               if r["family"] == "Paired Alignment")
    assert row["ranking_score"] is None
    assert row["paired_alignment_details"]["parent_source"] == "side_recovery_3"
    assert row["paired_alignment_details"]["outer_region_source"] == "v2_7"
    assert len(row["paired_alignment_details"]["side_scores_after"]) == 2
    assert snapshot["auto_select"] is None
    assert snapshot["auto_crop_allowed"] is False



def test_rejected_paired_shift_reports_blocking_side_without_fabricating_quad():
    image = np.full((420, 640, 3), 127, np.uint8)
    points = [[90, 85], [440, 85], [440, 320], [90, 320]]
    evidence = {}
    result = propose_paired_alignment(
        image, points, diagnostics=evidence,
    )
    assert result == []
    assert evidence["status"] == "rejected"
    assert evidence["failure_reason"] == "no_opposite_sides_both_improved"
    assert evidence["attempts_evaluated"] > 0
    best = evidence["best_attempt"]
    assert best["blocking_side_index"] in (0, 2)
    assert best["both_pass"] is False
    assert len(best["required_side_scores"]) == 2
    assert len(best["side_scores_before"]) == 2
    assert len(best["side_scores_after"]) == 2
    assert all(x == 0 for x in best["side_scores_after"])
    assert all(x >= 11 for x in best["required_side_scores"])
    assert "corners" not in evidence
    assert "image" not in evidence
    assert "ground_truth" not in evidence


def test_paired_alignment_proposal_carries_per_side_diagnostics_without_acceptance():
    image, truth = _scene(rotation=22)
    displaced = truth + _inward_normal(truth, 0) * 30
    details = {"should_be_cleared": "not a result"}
    result = propose_paired_alignment(
        image, displaced, diagnostics=details,
    )
    assert result
    assert "should_be_cleared" not in details
    assert details["status"] == "proposed"
    assert details["failure_reason"] is None
    assert details["best_attempt"]["both_pass"] is True
    assert all(x["auto_select"] is None and not x["auto_crop_allowed"]
               for x in result)


def test_invalid_input_diagnostics_remain_fail_closed():
    details = {}
    assert propose_paired_alignment(
        None, [[0, 0]] * 4, diagnostics=details,
    ) == []
    assert details["status"] == "not_evaluated"
    assert details["failure_reason"] == "invalid_input_or_geometry"
    assert details["best_attempt"] is None


def test_visual_snapshot_reports_rejected_parent_evidence_without_private_image():
    from autodocscanner.core.pr6_visual_diagnostics import (
        build_pr6_visual_diagnostics,
    )
    image = np.zeros((200, 320, 3), np.uint8)
    report = {
        "stable": {}, "adaptive": {}, "v2": [],
        "pr6_paired_alignment_shadow": {
            "status": "offline_visual_hypotheses_only",
            "auto_crop_allowed": False, "auto_select": None,
            "parents_examined": 1, "candidates": [],
            "parent_checks": [{
                "source": "side_recovery_3",
                "outer_region_source": "v2_7",
                "status": "rejected",
                "failure_reason": "no_opposite_sides_both_improved",
                "paired_side_indices": [0, 2],
                "attempts_evaluated": 60,
                "best_attempt": {
                    "shift_fraction_of_pair_gap": -.14,
                    "side_scores_before": [8., 14.],
                    "side_scores_after": [20., 17.],
                    "required_side_scores": [15., 21.],
                    "blocking_side_index": 2,
                    "both_pass": False,
                },
            }],
        },
    }
    snapshot = build_pr6_visual_diagnostics(report, image.shape)
    checks = snapshot["paired_alignment_parent_checks"]
    assert len(checks) == 1
    assert checks[0]["source"] == "side_recovery_3"
    assert checks[0]["best_attempt"]["blocking_side_index"] == 2
    assert checks[0]["best_attempt"]["side_scores_after"] == [20., 17.]
    assert snapshot["auto_crop_allowed"] is False
    assert snapshot["auto_select"] is None
    assert "image" not in str(checks)
    assert "ground_truth" not in str(checks)
