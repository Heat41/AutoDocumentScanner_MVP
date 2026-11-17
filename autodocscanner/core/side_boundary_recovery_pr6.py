"""PR6 experimental one-side boundary relocation, visual review ONLY.

Recover a single incorrectly placed edge of a near-card quadrilateral by
sampling coherent Lab transitions along a family of offset and *tilted*
image-relative lines. Unlike ranking whole background regions, this preserves
the other three candidate sides. A shifted line is only a geometric hypothesis.

No identity or color assumptions, OCR, ML, photo-specific rules, file writes,
automatic selection, warp output, or mutation of the input image.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def _order(points):
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1],
                        pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    return np.roll(pts, -int(np.argmin(pts[:, 0] + pts[:, 1])), axis=0)


def _intersection(a, b, c, d):
    first, second = b - a, d - c
    determinant = float(first[0] * second[1] - first[1] * second[0])
    if abs(determinant) < 1e-7:
        return None
    displacement = c - a
    fraction = (
        displacement[0] * second[1]
        - displacement[1] * second[0]
    ) / determinant
    return a + fraction * first


def _side_transition(lab, a, b, inward, offset, sample_distance, tilt=0.):
    """Robust coherent color transition on a proposed physical edge.

    A median Lab vector rejects isolated strong texture strokes. This is
    image-derived *evidence*, not a probability of a document boundary.
    """
    h, w = lab.shape[:2]
    fractions = np.linspace(.08, .92, 48)
    # Endpoint-relative shifts can differ. This handles genuine projective
    # slant where a translated copy of the original edge cannot fit the card.
    displacements = float(offset) + float(tilt) * (2 * fractions - 1)
    midpoints = (
        a[None, :] + fractions[:, None] * (b - a)[None, :]
        + displacements[:, None] * inward[None, :]
    )
    inside = midpoints + inward[None, :] * sample_distance
    outside = midpoints - inward[None, :] * sample_distance
    valid = (
        (inside[:, 0] >= 1) & (inside[:, 0] < w - 2)
        & (inside[:, 1] >= 1) & (inside[:, 1] < h - 2)
        & (outside[:, 0] >= 1) & (outside[:, 0] < w - 2)
        & (outside[:, 1] >= 1) & (outside[:, 1] < h - 2)
    )
    coverage = float(np.mean(valid))
    if coverage < .70:
        return 0.

    def sample(points):
        x = points[:, 0].astype(np.float32).reshape(1, -1)
        y = points[:, 1].astype(np.float32).reshape(1, -1)
        return cv2.remap(
            lab, x, y, interpolation=cv2.INTER_LINEAR,
        ).reshape(-1, 3)

    delta = sample(inside[valid]) - sample(outside[valid])
    median_vector = np.median(delta, axis=0)
    strength = float(np.linalg.norm(median_vector))
    if strength < 5.:
        return 0.
    aligned = delta @ median_vector / max(strength, 1.)
    support = float(np.mean(aligned > max(7., strength * .35)))
    return strength * support ** 1.5 * coverage


def prepare_side_recovery_lab(image, *, max_longest=1150):
    """Prepare Lab once for many candidates, bounded for large phone photos.

    Local image data is returned in memory only and must never be serialized.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or type(max_longest) is not int or max_longest < 200):
        return None
    height, width = image.shape[:2]
    if min(height, width) < 32:
        return None
    scale = min(1., float(max_longest) / max(height, width))
    small = (cv2.resize(
        image, (max(2, round(width * scale)),
                max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    ) if scale < 1 else image)
    lab = cv2.GaussianBlur(
        cv2.cvtColor(small, cv2.COLOR_BGR2LAB), (3, 3), 0
    ).astype(np.float32)
    return lab, scale


def recover_one_side_hypotheses(image, corners, *, max_results=2,
                                prepared=None):
    """Find up to max_results possible single-side shifts in original pixels.

    Each result preserves the other two corners and refits only the two
    adjacent corners by intersections. It is *not* a valid KTP decision.
    The parent hypothesis must already have a valid convex quadrilateral.
    Allow skewed sides and significant inward shrinkage for oversized parents.
    External lines can look identical: this remains a *review-only hypothesis*.

    prepared is an optional (Lab, scale) from prepare_side_recovery_lab to
    avoid repeated full-resolution conversion for multiple candidates.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or type(max_results) is not int or not 1 <= max_results <= 4):
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
    h, w = lab.shape[:2]
    points = _order(corners) * scale
    center = points.mean(axis=0)
    sample_distance = max(2., min(h, w) * .0075)
    old_area = abs(float(cv2.contourArea(points.astype(np.float32))))
    hypotheses = []

    for index in range(4):
        a, b = points[index], points[(index + 1) % 4]
        vector = b - a
        side_length = float(np.linalg.norm(vector))
        if side_length < 15.:
            continue
        inward = np.array([-vector[1], vector[0]]) / side_length
        if np.dot(inward, center - (a + b) * .5) < 0:
            inward = -inward
        opposite = (
            points[(index + 2) % 4] + points[(index + 3) % 4]
        ) * .5
        gap = float(np.dot(opposite - (a + b) * .5, inward))
        if gap < 20.:
            continue

        # A large background box may need to shrink by most of one side's
        # separation. Also search a limited change of orientation: projective
        # card edges need not run parallel to the original background edge.
        offsets = np.unique(np.r_[
            np.linspace(-.10 * gap, .78 * gap, 45), 0.,
        ])
        tilts = gap * np.asarray((-.16, -.08, 0., .08, .16))
        initial_score = _side_transition(
            lab, a, b, inward, 0., sample_distance,
        )
        minimum_score = max(11., 1.5 * initial_score,
                            initial_score + 10.)
        potentials = []
        for shift in offsets:
            if abs(shift) < .035 * gap:
                continue
            for tilt in tilts:
                evidence = _side_transition(
                    lab, a, b, inward, float(shift), sample_distance,
                    tilt=float(tilt),
                )
                if evidence >= minimum_score:
                    potentials.append((float(evidence), float(shift),
                                       float(tilt)))
        # The highest edge score could create an invalid quadrilateral.
        # Try the strongest bounded alternatives before failing closed.
        potentials.sort(key=lambda row: -row[0])
        for best_score, shift, tilt in potentials[:32]:
            shifted_a = a + (shift - tilt) * inward
            shifted_b = b + (shift + tilt) * inward
            left = _intersection(
                shifted_a, shifted_b,
                points[(index - 1) % 4], points[index],
            )
            right = _intersection(
                shifted_a, shifted_b,
                points[(index + 1) % 4], points[(index + 2) % 4],
            )
            if left is None or right is None:
                continue
            proposed = points.copy()
            proposed[index] = left
            proposed[(index + 1) % 4] = right
            new_area = abs(float(cv2.contourArea(proposed.astype(np.float32))))
            # Large reductions are expected when the old box included
            # extensive background; this is NOT permission to crop.
            if not .28 <= new_area / max(old_area, 1.) <= 1.25:
                continue
            original_coords = proposed / scale
            geometry_ok, _, _ = validate_quad(original_coords, image.shape)
            if not geometry_ok:
                continue
            hypotheses.append({
                "corners": np.round(original_coords, 3).tolist(),
                "changed_side_index": int(index),
                "shift_fraction_of_parent_gap": round(shift / gap, 4),
                "side_tilt_fraction_of_parent_gap": round(tilt / gap, 4),
                "side_score_before": round(initial_score, 3),
                "side_score_after": round(best_score, 3),
                "status": "hypothesis_for_manual_review",
                "auto_select": None,
                "auto_crop_allowed": False,
            })
            break

    hypotheses.sort(
        key=lambda entry: -(
            entry["side_score_after"] - entry["side_score_before"]
        )
    )
    return hypotheses[:max_results]
