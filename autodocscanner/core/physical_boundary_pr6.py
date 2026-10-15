"""Offline PR6 physical-boundary consistency diagnostic (OpenCV only).

Measures how consistently a candidate's sides separate the pixels immediately
inside and outside its quadrilateral. This is not a card classifier, nor
calibrated confidence, and never authorizes automatic crop.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def verify_physical_boundary(image, corners, *, samples=48):
    valid, reasons, _ = validate_quad(corners, image.shape)
    if not valid:
        return {"measurable": False, "reason": "invalid_geometry", "reasons": reasons}
    h, w = image.shape[:2]
    if min(h, w) < 16:
        return {"measurable": False, "reason": "image_too_small"}
    pts = np.asarray(corners, dtype=np.float64)
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(np.asarray(gray, dtype=np.uint8), (5, 5), 0).astype(float)
    distance = max(2., min(h, w) * .012)
    sides = []
    for i in range(4):
        p, q = pts[i], pts[(i + 1) % 4]
        direction = (q-p) / max(np.linalg.norm(q-p), 1e-8)
        # Ordered angular points are counterclockwise in Cartesian coordinates,
        # but image Y grows downwards. Inward is the direction toward centroid.
        n = np.array([-direction[1], direction[0]])
        midpoint = (p+q) / 2
        if np.dot(n, center-midpoint) < 0:
            n = -n
        values = []
        for fraction in np.linspace(.12, .88, max(8, int(samples))):
            middle = p + (q-p)*fraction
            inner, outer = middle + distance*n, middle-distance*n
            if not (1 <= inner[0] < w-1 and 1 <= inner[1] < h-1 and
                    1 <= outer[0] < w-1 and 1 <= outer[1] < h-1):
                continue
            ix, iy = np.rint(inner).astype(int)
            ox, oy = np.rint(outer).astype(int)
            values.append(float(gray[iy, ix]-gray[oy, ox]))
        coverage = len(values) / max(8, int(samples))
        if len(values) < max(8, int(samples)) // 4:
            sides.append({"coverage": round(coverage, 4), "contrast": None,
                          "polarity_consistency": None})
            continue
        v = np.asarray(values)
        contrast = np.median(np.abs(v))
        active = v[np.abs(v) >= max(5., contrast * .5)]
        coherence = float(max(np.count_nonzero(active > 0), np.count_nonzero(active < 0)) / len(active)) if len(active) else None
        sides.append({"coverage": round(coverage, 4),
                      "contrast": round(float(contrast), 3),
                      "polarity_consistency": round(coherence, 4) if coherence is not None else None})
    measured = [x for x in sides if x["polarity_consistency"] is not None]
    return {"measurable": len(measured) == 4,
            "side_measurements": sides,
            "median_polarity_consistency": round(float(np.median([x["polarity_consistency"] for x in measured])), 4) if measured else None,
            "reason": "observed" if len(measured) == 4 else "incomplete_boundary"}
