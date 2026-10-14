"""Shared, conservative four-corner geometry checks for offline KTP experiments.

This module does NOT establish that a quadrilateral actually surrounds a KTP.
It intentionally avoids image-specific coordinates, names, and color assumptions.
"""
from __future__ import annotations

import cv2
import numpy as np


def validate_quad(corners, image_shape, *, min_area_ratio=0.015,
                  max_area_ratio=0.985, min_side_px=12):
    """Return (valid, reasons, measurements) in original-image coordinates.

    Geometry-only checks. Frame-adjacent quads are flagged, not automatically
    rejected: a tightly photographed KTP can genuinely fill the photograph.
    """
    reasons = []
    metrics = {}
    if len(image_shape) < 2:
        return False, ["invalid_image_shape"], metrics
    h, w = map(int, image_shape[:2])
    if h <= 0 or w <= 0:
        return False, ["invalid_image_shape"], metrics
    try:
        quad = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError):
        return False, ["invalid_points"], metrics
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return False, ["invalid_points"], metrics

    # Order cyclically; this also makes convexity and side checks independent
    # of which vertex is supplied first.
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    ordered = quad[np.argsort(angles)]
    lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    metrics["min_side_px"] = float(lengths.min())
    if (lengths < min_side_px).any():
        reasons.append("short_or_duplicate_side")

    quad32 = ordered.astype(np.float32)
    if not cv2.isContourConvex(quad32):
        reasons.append("non_convex")
    area_ratio = abs(float(cv2.contourArea(quad32))) / float(h * w)
    metrics["area_ratio"] = area_ratio
    if area_ratio < min_area_ratio:
        reasons.append("area_too_small")
    if area_ratio > max_area_ratio:
        reasons.append("area_too_large")

    # Allow a small scale-relative extrapolation from line intersections.
    # Record its magnitude separately; never silently clip corners.
    tolerance_x = max(2.0, w * 0.02)
    tolerance_y = max(2.0, h * 0.02)
    excess_x = np.maximum.reduce([np.zeros(4), -quad[:, 0], quad[:, 0] - (w - 1)])
    excess_y = np.maximum.reduce([np.zeros(4), -quad[:, 1], quad[:, 1] - (h - 1)])
    metrics["outside_excess_px"] = float(max(excess_x.max(), excess_y.max()))
    metrics["slight_extrapolation"] = bool((excess_x > 0).any() or (excess_y > 0).any())
    if ((quad[:, 0] < -tolerance_x) | (quad[:, 0] > (w - 1 + tolerance_x)) |
        (quad[:, 1] < -tolerance_y) | (quad[:, 1] > (h - 1 + tolerance_y))).any():
        reasons.append("outside_image")

    # An informative warning, not a reason for rejection.
    margin = min(h, w) * 0.015
    near = ((quad[:, 0] <= margin) | (quad[:, 0] >= w - 1 - margin) |
            (quad[:, 1] <= margin) | (quad[:, 1] >= h - 1 - margin))
    metrics["near_frame_vertex_count"] = int(near.sum())
    metrics["frame_suspect"] = bool(area_ratio > 0.82 and near.sum() >= 2)
    return not reasons, reasons, metrics
