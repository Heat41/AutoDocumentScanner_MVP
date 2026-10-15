"""PR6 offline boundary localization: where is the nearest coherent edge?

This is an image-derived *diagnostic*, not an automatic selector. Ground-truth
annotations are never read. It intentionally reports ambiguous or missing edges.
"""
from __future__ import annotations

import cv2
import numpy as np
from autodocscanner.core.quad_validation import validate_quad


def localize_candidate_boundary(image, corners, *, samples=32):
    valid, reasons, _ = validate_quad(corners, image.shape)
    if not valid:
        return {"measurable": False, "reason": "invalid_geometry", "reasons": reasons}
    h, w = image.shape[:2]
    pts = np.asarray(corners, dtype=np.float64)
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(np.asarray(gray, dtype=np.uint8), (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    radius = max(3., min(h, w) * .025)
    offsets = np.linspace(-radius, radius, 17)
    n_samples = max(8, int(samples))
    measured = []
    for i in range(4):
        a, b = pts[i], pts[(i+1) % 4]
        tangent = (b-a)/max(np.linalg.norm(b-a), 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        if np.dot(normal, center-(a+b)*.5) < 0:
            normal = -normal
        peaks, strengths, ties = [], [], 0
        for t in np.linspace(.12, .88, n_samples):
            p = a + (b-a)*t
            samples_xy = p + offsets[:, None]*normal
            inside = ((samples_xy[:, 0] >= 2) & (samples_xy[:, 0] < w-2) &
                      (samples_xy[:, 1] >= 2) & (samples_xy[:, 1] < h-2))
            if inside.sum() < 13:
                continue
            x = samples_xy[:, 0].clip(0, w-1).astype(np.float32)
            y = samples_xy[:, 1].clip(0, h-1).astype(np.float32)
            gxx = cv2.remap(gx, x.reshape(1,-1), y.reshape(1,-1), cv2.INTER_LINEAR).reshape(-1)
            gyy = cv2.remap(gy, x.reshape(1,-1), y.reshape(1,-1), cv2.INTER_LINEAR).reshape(-1)
            strength = np.abs(gxx*normal[0] + gyy*normal[1])
            strength[~inside] = 0
            best = int(np.argmax(strength))
            peak = float(strength[best])
            if peak < 15:
                continue
            peaks.append(float(offsets[best]/radius))
            strengths.append(peak)
            if np.count_nonzero(strength >= .85*peak) > 4:
                ties += 1
        coverage = len(peaks)/n_samples
        if peaks:
            median_offset = float(np.median(peaks))
            spread = float(np.median(np.abs(np.asarray(peaks)-median_offset)))
            measured.append({"coverage": round(coverage,4),
                             "median_offset_fraction": round(median_offset,4),
                             "offset_spread": round(spread,4),
                             "median_gradient": round(float(np.median(strengths)),2),
                             "ambiguous_profile_fraction": round(ties/len(peaks),4)})
        else:
            measured.append({"coverage": 0., "median_offset_fraction": None,
                             "offset_spread": None, "median_gradient": None,
                             "ambiguous_profile_fraction": None})
    complete = all(s["coverage"] >= .5 for s in measured)
    aligned = complete and all(abs(s["median_offset_fraction"]) <= .25 and
                               s["offset_spread"] <= .25 for s in measured)
    return {"measurable": complete, "side_localization": measured,
            "edge_aligned": bool(aligned),
            "reason": "observed" if complete else "insufficient_gradient_coverage",
            "auto_select": None}
