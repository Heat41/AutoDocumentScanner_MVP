"""PR6 independent opposite-side border hypotheses — offline visual review.

Unlike a common translation, this allows two opposite edges to move by
DIFFERENT distances, changing the candidate extent while preserving the
orientation of the two adjacent edges. Both new edges must independently
show coherent Lab boundary evidence. No KTP identity classification,
automatic crop, photo-specific assumptions, disk/network writes or ML.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad
from autodocscanner.core.side_boundary_recovery_pr6 import (
    _intersection, _order, _side_transition, prepare_side_recovery_lab,
)


def propose_independent_opposite_sides(
    image, corners, *, pair_side_index=0, prepared=None, max_results=2,
    diagnostics=None,
):
    """Propose two independently shifted opposite edges for visual review.

    The two adjacent side *lines* are kept unchanged, and all four corners
    are reconstructed by their intersections. The source image is immutable.

    diagnostics is an optional mutable dict with bounded, aggregate numeric
    evidence and rejection reasons, never pixels or proposed unsafe crops.
    """
    info = diagnostics if isinstance(diagnostics, dict) else None
    if info is not None:
        info.clear()
        info.update({
            "status": "not_evaluated",
            "failure_reason": "invalid_input_or_geometry",
            "paired_side_indices": (
                [pair_side_index, pair_side_index + 2]
                if type(pair_side_index) is int and pair_side_index in (0, 1)
                else []
            ),
            "side_checks": [],
            "pairs_evaluated": 0,
        })
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or type(pair_side_index) is not int
            or pair_side_index not in (0, 1)
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

    points = _order(corners) * scale
    center = points.mean(axis=0)
    side_ids = (pair_side_index, pair_side_index + 2)
    edges = {}
    for idx in range(4):
        a, b = points[idx], points[(idx + 1) % 4]
        vector = b - a
        norm = float(np.linalg.norm(vector))
        if norm < 15.:
            return []
        inward = np.array([-vector[1], vector[0]]) / norm
        if float(inward @ (center - (a + b) / 2)) < 0:
            inward = -inward
        edges[idx] = (a, b, inward)

    p0, p2 = edges[side_ids[0]], edges[side_ids[1]]
    separation = float(abs(
        (p2[0] + p2[1] - p0[0] - p0[1]) @ p0[2] / 2
    ))
    if separation < 25.:
        if info is not None:
            info["status"] = "rejected"
            info["failure_reason"] = "insufficient_pair_separation"
        return []

    sample_distance = max(2., min(lab.shape[:2]) * .0075)
    offset_fractions = np.unique(np.r_[
        np.linspace(-.42, .50, 47), 0.,
    ])
    support_by_side = []
    checks = []
    for idx in side_ids:
        a, b, normal = edges[idx]
        before = float(_side_transition(
            lab, a, b, normal, 0., sample_distance,
        ))
        required = max(11., 1.25 * before, before + 7.)
        alternatives = []
        strongest = (before, 0.)
        for frac in offset_fractions:
            if abs(frac) < .025:
                continue
            delta = float(frac * separation)
            score = float(_side_transition(
                lab, a, b, normal, delta, sample_distance,
            ))
            if score > strongest[0]:
                strongest = (score, float(frac))
            if score >= required:
                alternatives.append((score, float(frac), delta))
        alternatives.sort(key=lambda row: -row[0])
        # Keep peak diversity, not 4 nearly identical samples of one edge.
        diverse = []
        for entry in alternatives:
            if any(abs(entry[1] - previous[1]) < .035
                   for previous in diverse):
                continue
            diverse.append(entry)
            if len(diverse) == 5:
                break
        support_by_side.append(diverse)
        checks.append({
            "side_index": idx,
            "score_before": round(before, 3),
            "required_score": round(required, 3),
            "strongest_score": round(strongest[0], 3),
            "strongest_offset_fraction": round(strongest[1], 4),
            "has_supported_alternative": bool(diverse),
        })

    if info is not None:
        info["status"] = "rejected"
        info["failure_reason"] = "unsupported_boundary_for_one_or_both_sides"
        info["side_checks"] = checks
    if not all(support_by_side):
        return []

    original_area = abs(float(cv2.contourArea(
        points.astype(np.float32)
    )))
    hypotheses = []
    combinations = 0
    for first in support_by_side[0]:
        for second in support_by_side[1]:
            combinations += 1
            moved = {}
            for idx, (_, _, delta) in zip(side_ids, (first, second)):
                a, b, normal = edges[idx]
                moved[idx] = (a + normal * delta, b + normal * delta)
            for idx in range(4):
                if idx not in moved:
                    moved[idx] = edges[idx][:2]
            proposed = []
            for index in range(4):
                before_a, before_b = moved[(index - 1) % 4]
                after_a, after_b = moved[index]
                vertex = _intersection(
                    before_a, before_b, after_a, after_b,
                )
                if vertex is None or not np.isfinite(vertex).all():
                    break
                proposed.append(vertex)
            if len(proposed) != 4:
                continue
            scaled_quad = np.asarray(proposed, dtype=np.float64)
            new_area = abs(float(cv2.contourArea(
                scaled_quad.astype(np.float32)
            )))
            if not .48 <= new_area / max(original_area, 1.) <= 1.60:
                continue
            recovered = scaled_quad / scale
            geometry_valid, _, _ = validate_quad(
                recovered, image.shape,
            )
            if not geometry_valid:
                continue
            # Re-evaluate the fully reconstructed sides with their ACTUAL
            # endpoints; a strong edge on a nearby parallel line is not
            # necessarily aligned with the final joined quadrilateral.
            confirmed = []
            for idx in side_ids:
                a, b = scaled_quad[idx], scaled_quad[(idx + 1) % 4]
                inward = edges[idx][2]
                confirmed.append(float(_side_transition(
                    lab, a, b, inward, 0., sample_distance,
                )))
            if any(score < max(
                10., check["score_before"] * 1.20,
                check["score_before"] + 6.,
            ) for score, check in zip(confirmed, checks)):
                continue

            hypotheses.append({
                "corners": np.round(recovered, 3).tolist(),
                "paired_side_indices": list(side_ids),
                "independent_offset_fractions": [
                    round(first[1], 4), round(second[1], 4),
                ],
                "side_scores_before": [
                    check["score_before"] for check in checks
                ],
                "side_scores_after": [round(x, 3) for x in confirmed],
                "status": "hypothesis_for_manual_review",
                "auto_select": None,
                "auto_crop_allowed": False,
                "_sort_evidence": min(
                    confirmed[0] - checks[0]["score_before"],
                    confirmed[1] - checks[1]["score_before"],
                ),
            })

    hypotheses.sort(key=lambda row: -row["_sort_evidence"])
    output = []
    for row in hypotheses:
        if any(
            np.mean(np.linalg.norm(
                np.asarray(row["corners"]) -
                np.asarray(old["corners"]), axis=1,
            )) < .015 * np.hypot(*image.shape[:2])
            for old in output
        ):
            continue
        row.pop("_sort_evidence")
        output.append(row)
        if len(output) >= max_results:
            break

    if info is not None:
        info["status"] = "proposed" if output else "rejected"
        info["failure_reason"] = (
            None if output else "joined_geometry_or_boundary_rejected"
        )
        info["pairs_evaluated"] = combinations
    return output
