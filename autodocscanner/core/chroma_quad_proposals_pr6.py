"""Experimental OpenCV chromatic-boundary quadrilateral proposals for PR6.

Purely image-derived hypotheses from Lab chroma transitions. Complements the
luminance-contour and long-line generators when card and background have
similar brightness but different chromaticity. There is no identity, fixed
color, learned model, OCR, output saving, or automatic-crop authority.

All proposals must still pass ordinary PR6 review/physical-boundary gates.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def _ordered(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    return np.roll(pts, -int(np.argmin(pts[:, 0] + pts[:, 1])), axis=0)


def chroma_quad_candidates(image, *, max_candidates=8, max_longest=1200):
    """Propose quads from Lab a/b edge contours at two scale-relative gaps.

    This does NOT select a KTP. It deliberately includes geometrically
    plausible false proposals when the background also contains chromatic
    rectangles; downstream independent PR6 evidence must resolve them.
    Coordinates are returned in the ORIGINAL image reference frame.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or type(max_candidates) is not int or not 1 <= max_candidates <= 40
            or type(max_longest) is not int or max_longest < 100):
        return []
    height, width = image.shape[:2]
    if min(height, width) < 75:
        return []

    scale = min(1.0, float(max_longest) / max(height, width))
    small = (
        cv2.resize(
            image,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        ) if scale < 1 else image
    )
    h, w = small.shape[:2]
    chroma = cv2.GaussianBlur(
        cv2.cvtColor(small, cv2.COLOR_BGR2LAB), (5, 5), 0,
    )
    first = cv2.Canny(chroma[:, :, 1], 20, 65)
    second = cv2.Canny(chroma[:, :, 2], 20, 65)
    maps = (first, second, cv2.bitwise_or(first, second))

    candidates = []
    frame_area = float(h * w)
    for edge_map in maps:
        for gap_fraction in (.006, .012):
            diameter = max(3, int(round(min(h, w) * gap_fraction)) | 1)
            closed = cv2.morphologyEx(
                edge_map, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (diameter, diameter),
                ),
            )
            contours, _ = cv2.findContours(
                closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                area_ratio = abs(cv2.contourArea(contour)) / frame_area
                if not .01 <= area_ratio <= .8:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter < max(60, .15 * min(h, w)):
                    continue
                for epsilon in (.012, .025, .045, .07):
                    polygon = cv2.approxPolyDP(
                        contour, epsilon * perimeter, True,
                    )
                    if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                        continue
                    proposed = _ordered(polygon[:, 0, :]) / scale
                    lengths = np.linalg.norm(
                        np.roll(proposed, -1, axis=0) - proposed, axis=1,
                    )
                    if float(lengths.min()) < max(
                        15.0, min(height, width) * .04,
                    ):
                        continue
                    long = max(
                        float(np.mean(lengths[::2])),
                        float(np.mean(lengths[1::2])),
                    )
                    short = max(
                        min(
                            float(np.mean(lengths[::2])),
                            float(np.mean(lengths[1::2])),
                        ), 1.0,
                    )
                    if not 1.12 < long / short < 3.4:
                        continue
                    valid, _, _ = validate_quad(proposed, image.shape)
                    if not valid:
                        continue
                    candidates.append((area_ratio, proposed))
                    break

    # Preserve the diverse source geometries as proposals only. This
    # ordering is not physical-boundary evidence or automatic acceptance.
    candidates.sort(key=lambda entry: -entry[0])
    unique = []
    diagonal = max(float(np.hypot(width, height)), 1.0)
    for _, points in candidates:
        if any(
            float(np.linalg.norm(points - existing, axis=1).mean())
            / diagonal < .017
            for existing in unique
        ):
            continue
        unique.append(points)
        if len(unique) >= max_candidates:
            break
    return unique
