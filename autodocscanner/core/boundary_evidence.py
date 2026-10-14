"""Image-derived evidence of actual outer-card boundaries (experimental only).

A strong edge is not proof of a KTP. This module measures local evidence on
all four sides without relying on identifying colors or photo-specific data.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def boundary_evidence(image, corners, *, samples=48):
    """Return a conservative per-side edge/contrast diagnostic.

    Offset distances scale with image size. Out-of-frame samples do not count.
    These signals must not be used as standalone proof of a card boundary.
    """
    valid, reasons, metrics = validate_quad(corners, image.shape)
    if not valid:
        return {"valid": False, "reasons": reasons, "side_support": [], "score": 0.0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = np.asarray(gray, dtype=np.uint8)
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    points = np.asarray(corners, dtype=np.float64)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))]
    offset = max(2.0, min(w, h) * 0.012)
    side_support = []
    for i in range(4):
        a, b = points[i], points[(i + 1) % 4]
        tangent = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        normal = np.array([-tangent[1], tangent[0]])
        t = np.linspace(0.1, 0.9, max(8, int(samples)))
        locations = a[None, :] + t[:, None] * (b-a)[None, :]
        # Evaluate normal gradient near estimated physical boundary,
        # not arbitrary internal text edges.
        aligned = []
        contrast = []
        for p in locations:
            inner, outer = p + normal*offset, p - normal*offset
            coords = np.array([p, inner, outer])
            if ((coords[:, 0] < 1) | (coords[:, 0] >= w-1) |
                (coords[:, 1] < 1) | (coords[:, 1] >= h-1)).any():
                continue
            x, y = np.rint(p).astype(int)
            xi, yi = np.rint(inner).astype(int)
            xo, yo = np.rint(outer).astype(int)
            gradient = abs(float(gx[y, x]*normal[0] + gy[y, x]*normal[1]))
            aligned.append(min(1.0, gradient / 95.0))
            contrast.append(min(1.0, abs(float(blurred[yi, xi])-float(blurred[yo, xo]))/35.0))
        support = float(0.5*np.mean(aligned)+0.5*np.mean(contrast)) if aligned else 0.0
        side_support.append(round(support, 4))
    # Geometric mean discourages one strong side hiding three weak ones.
    score = float(np.prod(np.asarray(side_support)+1e-6)**0.25)
    return {"valid": True, "reasons": [], "side_support": side_support,
            "score": round(score, 4),
            "weakest_side": round(min(side_support), 4),
            "frame_suspect": metrics["frame_suspect"]}
