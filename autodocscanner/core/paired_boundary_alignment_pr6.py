"""Paired physical-border alignment proposals for offline PR6 visual review.

A one-side recovery can get two corners near a card while leaving the entire
box translated relative to the physical outer border. This module proposes
a SMALL, shared translation of TWO OPPOSING card edges only when evidence
improves on BOTH edges, not merely on the strongest edge.

Nothing here identifies an identity card or authorizes selection, crop,
warp or saving. Pixel data never leaves memory or gets written to a report.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad
from autodocscanner.core.side_boundary_recovery_pr6 import (
    _order, _side_transition, prepare_side_recovery_lab,
)


def propose_paired_alignment(
    image, corners, *, pair_side_index=0, prepared=None, max_results=1,
    diagnostics=None,
):
    """Return bounded original-image four-corner translation hypotheses.

    pair_side_index: 0 means sides 0/2, 1 means sides 1/3 in cyclic order.
    The shared move leaves widths, slopes and perspective unchanged, so it
    cannot by itself repair an incorrectly sized or shaped parent.

    Optional diagnostics receives bounded numeric reasons for *rejected*
    proposals; it never contains pixels, card text, or a fallback crop.
    """
    details = diagnostics if isinstance(diagnostics, dict) else None
    if details is not None:
        details.clear()
        details.update({
            "status": "not_evaluated",
            "failure_reason": "invalid_input_or_geometry",
            "paired_side_indices": (
                [pair_side_index, pair_side_index + 2]
                if pair_side_index in (0, 1) else []
            ),
            "attempts_evaluated": 0,
            "best_attempt": None,
        })
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or type(pair_side_index) is not int
            or pair_side_index not in (0, 1)
            or type(max_results) is not int
            or not 1 <= max_results <= 4):
        return []

    valid, _, _ = validate_quad(corners, image.shape)
    if not valid:
        return []
    if prepared is None:
        prepared = prepare_side_recovery_lab(image)
    if not isinstance(prepared, tuple) or len(prepared) != 2:
        return []
    lab, scale = prepared
    if (not isinstance(lab, np.ndarray) or lab.ndim != 3
            or lab.shape[2] != 3 or lab.dtype != np.float32
            or not isinstance(scale, (int, float))
            or not np.isfinite(scale) or not 0 < scale <= 1):
        return []

    points = _order(corners) * scale
    center = points.mean(axis=0)
    h, w = lab.shape[:2]
    distance = max(2., min(h, w) * .0075)
    opposite_index = pair_side_index + 2
    observed = []
    for index in (pair_side_index, opposite_index):
        first = points[index]
        last = points[(index + 1) % 4]
        vector = last - first
        length = float(np.linalg.norm(vector))
        if length < 15.:
            return []
        inward = np.array([-vector[1], vector[0]]) / length
        if np.dot(inward, center - (first + last) / 2) < 0:
            inward = -inward
        before = _side_transition(
            lab, first, last, inward, 0., distance,
        )
        observed.append((first, last, inward, float(before)))

    primary, opposite = observed
    separation = float(abs(np.dot(
        (opposite[0] + opposite[1] - primary[0] - primary[1]) / 2,
        primary[2],
    )))
    if separation < 25.:
        if details is not None:
            details["failure_reason"] = "insufficient_pair_separation"
        return []

    candidates = []
    best_trial = None
    best_margin = float("-inf")
    attempts = 0
    evidence_passing = 0
    # The scan is relative to the current box size, not a fixed pixel
    # distance or the known location of any example card.
    for fraction in np.linspace(-.30, .30, 61):
        if abs(fraction) < .025:
            continue
        movement = float(fraction * separation) * primary[2]
        scores = [
            _side_transition(
                lab, a + movement, b + movement, inward, 0., distance,
            )
            for a, b, inward, _ in observed
        ]
        before = [item[3] for item in observed]
        required = [
            max(11., initial * 1.25, initial + 7.)
            for initial in before
        ]
        margins = [
            (after - minimum) / max(minimum, 1.)
            for after, minimum in zip(scores, required)
        ]
        attempts += 1
        if min(margins) > best_margin:
            best_margin = min(margins)
            blocking = int(np.argmin(margins))
            best_trial = {
                "shift_fraction_of_pair_gap": round(float(fraction), 4),
                "side_scores_before": [round(x, 3) for x in before],
                "side_scores_after": [round(float(x), 3) for x in scores],
                "required_side_scores": [round(float(x), 3) for x in required],
                "blocking_side_index": int(
                    (pair_side_index, opposite_index)[blocking]
                ),
                "both_pass": bool(all(margin >= 0 for margin in margins)),
            }
        # Moving the whole box can fix two aligned edges but can equally
        # create a false rectangle on patterned backgrounds. Require BOTH
        # opposite edges to independently gain coherent line support.
        if not all(after >= minimum
                   for after, minimum in zip(scores, required)):
            continue
        evidence_passing += 1

        shifted = (points + movement) / scale
        accepted, _, _ = validate_quad(shifted, image.shape)
        if not accepted:
            continue
        improvements = [a - b for a, b in zip(scores, before)]
        balanced_gain = float(min(improvements) + .2 * np.mean(improvements))
        candidates.append((
            balanced_gain, float(fraction), shifted, before, scores, movement,
        ))

    candidates.sort(key=lambda value: -value[0])
    output = []
    for gain, fraction, corners_out, before, after, displacement in candidates:
        if any(
            abs(fraction - item["shift_fraction_of_pair_gap"]) < .035
            for item in output
        ):
            continue
        output.append({
            "corners": np.round(corners_out, 3).tolist(),
            "paired_side_indices": [pair_side_index, opposite_index],
            "shift_fraction_of_pair_gap": round(fraction, 4),
            "translation_original_xy": [
                round(float(x / scale), 3) for x in displacement
            ],
            "side_scores_before": [round(x, 3) for x in before],
            "side_scores_after": [round(x, 3) for x in after],
            "balanced_evidence_gain": round(gain, 3),
            "status": "hypothesis_for_manual_review",
            "auto_select": None,
            "auto_crop_allowed": False,
        })
        if len(output) >= max_results:
            break
    if details is not None:
        details.update({
            "status": "proposed" if output else "rejected",
            "failure_reason": (
                None if output else
                "geometry_rejected" if evidence_passing else
                "no_opposite_sides_both_improved"
            ),
            "attempts_evaluated": attempts,
            "best_attempt": best_trial,
        })
    return output
