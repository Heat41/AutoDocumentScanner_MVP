"""Read-only document-region context diagnostics for PR6.

A quadrilateral can be a card, a photograph printed on a card, or a
background object. Region compactness/area alone cannot tell them apart.
These cues inspect non-border interior structure and independent geometric
corroboration; neither is card identification or auto-crop authorization.

All pixels are processed locally in memory, no ground truth/OCR/ML/files.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def _cyclic_quad(corners):
    pts = np.asarray(corners, dtype=np.float32)
    center = pts.mean(axis=0)
    indices = np.argsort(np.arctan2(pts[:, 1] - center[1],
                                   pts[:, 0] - center[0]))
    pts = pts[indices]
    return np.roll(pts, -int(np.argmin(pts[:, 0] + pts[:, 1])), axis=0)


def inspect_region_context(image, corners, *, other_candidates=()):
    """Quantify possible card-content structure and independent agreement.

    No portrait/text detection, identity analysis, accepted-crop score, or
    assumption that any given object is the document.
    """
    result = {
        "status": "offline_diagnostic_only",
        "auto_select": None, "auto_crop_allowed": False,
        "interior_edge_density": None,
        "interior_spatial_coverage": None,
        "interior_axis_line_bands": 0,
        "corroborating_families": [],
        "nested_inner_candidates": 0,
        "reason_codes": [],
    }
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8):
        result["reason_codes"] = ["invalid_image"]
        return result
    valid, _, _ = validate_quad(corners, image.shape)
    if not valid:
        result["reason_codes"] = ["invalid_geometry"]
        return result

    quad = _cyclic_quad(corners)
    dst = np.array([[0, 0], [223, 0], [223, 143], [0, 143]],
                   dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad, dst)
    try:
        warped = cv2.warpPerspective(
            image, transform, (224, 144), flags=cv2.INTER_LINEAR
        )
    except cv2.error:
        result["reason_codes"] = ["warp_diagnostic_failed"]
        return result
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Ignore the frame itself. Measuring the border would reward even a
    # featureless book cover or other external rectangular object.
    inner = cv2.GaussianBlur(gray[18:126, 22:202], (3, 3), 0)
    edges = cv2.Canny(inner, 35, 105)
    result["interior_edge_density"] = round(
        float(cv2.countNonZero(edges) / edges.size), 4
    )
    tiles = []
    for band in np.array_split(edges, 3, axis=0):
        for block in np.array_split(band, 4, axis=1):
            tiles.append(cv2.countNonZero(block) / block.size)
    result["interior_spatial_coverage"] = round(
        sum(v >= .012 for v in tiles) / len(tiles), 4
    )
    # Repeated aligned strokes can occur on documents, but also on fabric,
    # packaging and print: this is an *advisory* structure count only.
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=12, minLineLength=18,
        maxLineGap=5,
    )
    positions = set()
    if segments is not None:
        for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4):
            dx, dy = int(x2-x1), int(y2-y1)
            if abs(dx) >= 18 and abs(dy) <= .23 * abs(dx):
                positions.add(round((y1 + y2) / 2 / 12))
            elif abs(dy) >= 18 and abs(dx) <= .23 * abs(dy):
                positions.add(round((x1 + x2) / 2 / 20))
    result["interior_axis_line_bands"] = min(len(positions), 20)

    candidate_area = abs(float(cv2.contourArea(quad)))
    corroborators = set()
    nested_inner = 0
    for item in other_candidates:
        if not isinstance(item, dict):
            continue
        family = item.get("family")
        try:
            other = np.asarray(item.get("corners"), dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            continue
        if other.shape != (4, 2) or not np.isfinite(other).all():
            continue
        other = cv2.convexHull(other).reshape(-1, 2)
        area = abs(float(cv2.contourArea(other)))
        if area <= 0:
            continue
        try:
            common, _ = cv2.intersectConvexConvex(
                cv2.convexHull(quad), cv2.convexHull(other)
            )
        except cv2.error:
            continue
        common = max(0., float(common))
        union = candidate_area + area - common
        iou = common / union if union > 0 else 0.
        if iou >= .58 and isinstance(family, str) and family:
            corroborators.add(family[:40])
        # A smaller region enclosed by this proposal may be a portrait,
        # internal document section, or an unrelated overlapping object.
        if (area / max(candidate_area, 1) <= .36
                and common / area >= .92):
            nested_inner += 1
    result["corroborating_families"] = sorted(corroborators)[:8]
    result["nested_inner_candidates"] = min(nested_inner, 50)

    reasons = []
    if (result["interior_spatial_coverage"] < .25
            and result["interior_axis_line_bands"] < 2):
        reasons.append("sparse_internal_structure")
    if not corroborators:
        reasons.append("no_independent_geometric_corroboration")
    result["reason_codes"] = reasons
    return result
