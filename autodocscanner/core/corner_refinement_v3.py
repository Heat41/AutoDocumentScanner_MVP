"""PR6 experimental card-corner refinement using local OpenCV edge evidence.

No learned models, identity-specific templates, ground-truth-driven nudges, or
production modifications. A conservative no-change result is always allowed.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad
from autodocscanner.core.boundary_evidence import boundary_evidence


def refine_corners(image, corners, *, band_fraction=0.018, max_shift_fraction=0.025):
    """Fit four lines to edges near proposed sides, intersect adjacent lines.

    Returns (original-or-refined 4x2 float32, diagnostics). A proposal is
    rejected when edge support, geometry or movement checks are insufficient.
    """
    source = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    info = {"accepted": False, "reason": "insufficient_evidence", "mean_shift_px": 0.0}
    valid, _, _ = validate_quad(source, image.shape)
    if not valid:
        info["reason"] = "invalid_input"
        return source.copy(), info
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    band = max(2.0, min(h, w) * band_fraction)
    lines = []
    for i in range(4):
        p, q = source[i].astype(float), source[(i + 1) % 4].astype(float)
        delta = q - p
        length = float(np.linalg.norm(delta))
        tangent = delta / max(length, 1)
        normal = np.array([-tangent[1], tangent[0]])
        # The highest gradient on each side-normal search strip is only
        # meaningful when a majority of samples agree on the same offset.
        offsets = np.arange(-band, band + 0.5, 1.0)
        locations = p[None, :] + np.linspace(.12, .88, 48)[:, None] * delta[None, :]
        projected = locations[:, None, :] + offsets[None, :, None] * normal
        x = np.rint(projected[:, :, 0]).astype(int)
        y = np.rint(projected[:, :, 1]).astype(int)
        inside = (x >= 1) & (x < w - 1) & (y >= 1) & (y < h - 1)
        if np.sum(np.any(inside, axis=1)) < 32:
            info["reason"] = "side_out_of_view"
            return source.copy(), info
        strength = np.zeros(x.shape, dtype=np.float32)
        xx, yy = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
        strength[inside] = np.abs(
            gradient_x[yy[inside], xx[inside]] * normal[0] +
            gradient_y[yy[inside], xx[inside]] * normal[1]
        )
        strongest = np.argmax(strength, axis=1)
        strength_peaks = strength[np.arange(len(locations)), strongest]
        chosen_offsets = offsets[strongest]
        good = (strength_peaks >= 35) & np.any(inside, axis=1)
        if int(good.sum()) < 32:
            info["reason"] = "weak_edge"
            return source.copy(), info
        median_offset = float(np.median(chosen_offsets[good]))
        consistent = good & (np.abs(chosen_offsets - median_offset) <= max(2, band * .4))
        if int(consistent.sum()) < 28:
            info["reason"] = "inconsistent_edge"
            return source.copy(), info
        pts = locations[consistent] + chosen_offsets[consistent, None] * normal
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).flatten()
        direction = np.array([float(vx), float(vy)])
        lines.append((np.array([float(x0), float(y0)]), direction))
    refined = []
    for i in range(4):
        a, u = lines[(i - 1) % 4]
        b, v = lines[i]
        matrix = np.column_stack((u, -v))
        if abs(float(np.linalg.det(matrix))) < .075:
            info["reason"] = "parallel_lines"
            return source.copy(), info
        t = np.linalg.solve(matrix, b - a)[0]
        refined.append(a + t*u)
    refined = np.asarray(refined, dtype=np.float32)
    good, _, _ = validate_quad(refined, image.shape)
    shift = np.linalg.norm(refined - source, axis=1)
    info["mean_shift_px"] = round(float(shift.mean()), 3)
    if not good or float(shift.max()) > max_shift_fraction * float(np.hypot(h, w)):
        info["reason"] = "geometry_or_shift"
        return source.copy(), info
    before = boundary_evidence(image, source)
    after = boundary_evidence(image, refined)
    # Reject if measurements become incomplete or noticeably weaker.
    if not after.get("evidence_complete"):
        info["reason"] = "incomplete_boundary"
        return source.copy(), info
    old = before.get("score")
    new = after.get("score")
    if new is None or (old is not None and new < old - .04):
        info["reason"] = "weaker_boundary"
        return source.copy(), info
    info["accepted"] = True
    info["reason"] = "line_fit"
    return refined, info
