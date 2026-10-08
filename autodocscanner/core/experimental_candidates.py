"""Experimental OpenCV-only KTP corner candidates.

This module deliberately does not replace the released detector. It proposes
quadrilaterals for later scoring against the stable v1.1.1 baseline.
"""
from __future__ import annotations

import cv2
import numpy as np


def _ordered(quad):
    """Return four cyclic vertices starting at the top-left-ish vertex."""
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    return np.roll(points, -start, axis=0)


def contour_candidates(
    image,
    *,
    target_ratio=85.60 / 53.98,
    scales=(640, 1000, 1400),
    max_candidates=24,
):
    """Return candidate quadrilaterals in ORIGINAL-image pixel coordinates.

    No photo-specific thresholds, region names, templates, or identity fields.
    Geometric pre-filtering only; each proposal still needs independent
    candidate scoring/validation before it can supersede the baseline.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []
    h, w = image.shape[:2]
    if min(h, w) < 30:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    proposals = []

    for longest in scales:
        factor = min(1.0, float(longest) / max(h, w))
        sw, sh = max(2, int(round(w * factor))), max(2, int(round(h * factor)))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        edges = cv2.Canny(blur, 45, 130)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(sw * sh)
        for contour in contours:
            area = abs(cv2.contourArea(contour))
            if area < image_area * .015 or area > image_area * .98:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 40:
                continue
            for epsilon in (.012, .025, .045):
                polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue
                quad = _ordered(polygon.reshape(4, 2)) / factor
                lengths = [
                    float(np.linalg.norm(quad[(i + 1) % 4] - quad[i]))
                    for i in range(4)
                ]
                if min(lengths) < 15:
                    continue
                estimated = max(np.mean(lengths[::2]), np.mean(lengths[1::2])) / max(
                    min(np.mean(lengths[::2]), np.mean(lengths[1::2])), 1
                )
                if abs(estimated - target_ratio) / target_ratio > .85:
                    continue
                proposals.append((area / image_area, quad))
                break

    proposals.sort(key=lambda entry: entry[0], reverse=True)
    unique = []
    diagonal = float(np.hypot(w, h))
    for _, proposal in proposals:
        if any(
            float(np.mean(np.linalg.norm(proposal - existing, axis=1))) / diagonal < .025
            for existing in unique
        ):
            continue
        unique.append(proposal)
        if len(unique) >= max_candidates:
            break
    return unique
