"""PR6 region-first outer-document hypotheses and perspective diagnostics.

OpenCV-only, color-agnostic and image-relative. Generate *possible outer*
quadrilaterals from spatially connected threshold regions rather than only
Canny contours, so an internally textured document may survive as one region.

An apparent region is NOT a recognized KTP. No learned model, OCR, ground
truth, private image persistence, output crop, selection or save occurs here.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def _order(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - center[1],
                                   pts[:, 0] - center[0]))]
    return np.roll(pts, -int(np.argmin(pts[:, 0] + pts[:, 1])), axis=0)


def _candidate_from_hull(contour, shape):
    """Recover *outer* corners from a region convex hull; no inner contours."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return []
    results = []
    for eps in (.012, .025, .045):
        polygon = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            results.append(_order(polygon.reshape(4, 2)))
    # Rectangular hull fallback is approximate (rotation only), not a valid
    # projective fit by itself. Keep it only for nearly rectangular regions.
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box_area = abs(float(cv2.contourArea(box)))
    if box_area > 0:
        shape_fill = abs(float(cv2.contourArea(hull))) / box_area
        if .82 <= shape_fill <= 1.03:
            results.append(_order(box))
    return results


def _valid_cardlike(points, image_shape):
    valid, _, metrics = validate_quad(points, image_shape)
    if not valid:
        return False
    lengths = np.linalg.norm(
        np.roll(points, -1, axis=0) - points, axis=1,
    )
    width = (float(lengths[0]) + float(lengths[2])) / 2
    height = (float(lengths[1]) + float(lengths[3])) / 2
    ratio = max(width, height) / max(min(width, height), 1e-6)
    return bool(1.13 <= ratio <= 3.25
                and .015 <= metrics.get("area_ratio", 0) <= .82)


def region_outer_quad_candidates(image, *, max_candidates=12, max_longest=940):
    """Return a bounded set of region-based outer-quad proposals.

    Uses Lab L/a/b and HSV saturation, Otsu and quantile splits, two
    resolution-relative smoothing scales, closing and external components.
    All thresholds are image-derived or resolution-relative; no card color,
    portrait/text location, filename or identity assumptions.

    Does NOT treat a region, rectangle or good contrast as a verified card.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or type(max_candidates) is not int or not 1 <= max_candidates <= 40
            or type(max_longest) is not int or not 150 <= max_longest <= 2400):
        return []
    h0, w0 = image.shape[:2]
    if min(h0, w0) < 75:
        return []

    scale = min(1., float(max_longest) / max(h0, w0))
    small = (
        cv2.resize(
            image, (max(2, int(round(w0 * scale))),
                    max(2, int(round(h0 * scale)))),
            interpolation=cv2.INTER_AREA,
        ) if scale < 1 else image
    )
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    channels = [lab[:, :, i] for i in range(3)] + [hsv[:, :, 1]]
    minimum = min(w, h)
    k_open = max(3, int(round(minimum * .004)) | 1)
    k_close = max(3, int(round(minimum * .017)) | 1)
    opening = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k_open, k_open),
    )
    closing = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k_close, k_close),
    )
    proposals = []
    seen_masks = set()
    for channel in channels:
        for sigma in (1.2, 3.):
            blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
            if float(np.std(blurred)) < 2.5:
                continue
            otsu, _ = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            thresholds = (int(otsu),
                          int(np.percentile(blurred, 35)),
                          int(np.percentile(blurred, 65)))
            for threshold in set(thresholds):
                for invert in (False, True):
                    mask = cv2.compare(
                        blurred, threshold,
                        cv2.CMP_LE if invert else cv2.CMP_GT,
                    )
                    # Fast dedup of uninformative masks (not image data).
                    count = cv2.countNonZero(mask)
                    if not .012 * h * w <= count <= .87 * h * w:
                        continue
                    digest = (count // 250, int(np.mean(mask[::16, ::16])))
                    if digest in seen_masks:
                        continue
                    seen_masks.add(digest)
                    mask = cv2.morphologyEx(
                        mask, cv2.MORPH_CLOSE, closing,
                    )
                    mask = cv2.morphologyEx(
                        mask, cv2.MORPH_OPEN, opening,
                    )
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                    )
                    contours = sorted(
                        contours, key=cv2.contourArea, reverse=True,
                    )[:16]
                    for contour in contours:
                        region_area = abs(float(cv2.contourArea(contour)))
                        area_fraction = region_area / float(h * w)
                        if not .015 <= area_fraction <= .83:
                            continue
                        for quad in _candidate_from_hull(contour, (h, w)):
                            original = quad / scale
                            if not _valid_cardlike(original, image.shape):
                                continue
                            quad_area = abs(float(cv2.contourArea(quad)))
                            if quad_area <= 0:
                                continue
                            # A compact region is more plausible than a
                            # thin/broken background stripe, but not proof.
                            region_fill = min(1., region_area / quad_area)
                            if region_fill < .58:
                                continue
                            proposals.append((
                                round(region_fill, 4), area_fraction, original,
                            ))
                            break

    # Preserve multiple scales and small distant cards instead of ranking
    # only by contour area. Each band gets round-robin representation.
    bands = [[], [], []]
    diagonal = max(float(np.hypot(w0, h0)), 1.)
    for fill, area, quad in sorted(
        proposals, key=lambda v: (-v[0], -v[1])
    ):
        bin_index = 0 if area < .08 else 1 if area < .32 else 2
        if any(
            float(np.linalg.norm(quad - old, axis=1).mean()) / diagonal < .018
            for band in bands for old in band
        ):
            continue
        bands[bin_index].append(quad)
    out = []
    while len(out) < max_candidates and any(bands):
        for band in bands:
            if band and len(out) < max_candidates:
                out.append(band.pop(0))
    return out


def inspect_outer_perspective_hypothesis(image, corners, *, other_quads=()):
    """Check outer-side and projective plausibility without applying a warp.

    LAB inside/outside pixel differences can corroborate four physical sides,
    but a portrait box or a fabric rectangle may produce the same signal.
    Report nested geometry as *ambiguous*, never infer the KTP identity.
    """
    result = {
        "status": "manual_review_only",
        "auto_select": None, "auto_crop_allowed": False,
        "perspective_feasible": False,
        "all_sides_measurable": False,
        "chromatic_step_by_side": [],
        "nested_in_larger_quad": False,
        "reason_codes": [],
    }
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8):
        result["reason_codes"] = ["invalid_image"]
        return result
    valid, reasons, metrics = validate_quad(corners, image.shape)
    if not valid:
        result["reason_codes"] = ["invalid_geometry"] + list(reasons)
        return result
    quad = _order(corners).astype(np.float64)
    h, w = image.shape[:2]
    # Unit-scale homography avoids artifacts from image resolution when
    # calculating a conditioning diagnostic (not a success probability).
    normalized = (quad / [max(w - 1, 1), max(h - 1, 1)]).astype(np.float32)
    target = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    try:
        homography = cv2.getPerspectiveTransform(normalized, target)
        condition = float(np.linalg.cond(homography))
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        condition = float("inf")
    result["perspective_feasible"] = bool(math.isfinite(condition)
                                          and condition < 1e7)
    result["homography_condition"] = (
        round(condition, 3) if math.isfinite(condition) else None
    )
    result["area_ratio"] = round(float(metrics.get("area_ratio", 0)), 4)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    centroid = quad.mean(axis=0)
    distance = max(2., min(h, w) * .012)
    measurable = 0
    for index in range(4):
        a, b = quad[index], quad[(index + 1) % 4]
        tangent = (b - a) / max(np.linalg.norm(b - a), 1e-8)
        normal = np.array([-tangent[1], tangent[0]])
        if np.dot(normal, centroid - (a + b) / 2) < 0:
            normal = -normal
        delta = []
        for t in np.linspace(.12, .88, 32):
            center = a + (b - a) * t
            inner = center + normal * distance
            outer = center - normal * distance
            if not (1 <= inner[0] < w - 1 and 1 <= inner[1] < h - 1
                    and 1 <= outer[0] < w - 1 and 1 <= outer[1] < h - 1):
                continue
            # Nearest source pixels are diagnostic; we do not mutate input.
            ix, iy = np.rint(inner).astype(int)
            ox, oy = np.rint(outer).astype(int)
            delta.append(float(np.linalg.norm(
                lab[iy, ix] - lab[oy, ox]
            )))
        if len(delta) >= 16:
            measurable += 1
        result["chromatic_step_by_side"].append({
            "coverage": round(len(delta) / 32., 4),
            "median_lab_step": (
                round(float(np.median(delta)), 3) if delta else None
            ),
        })
    result["all_sides_measurable"] = measurable == 4

    own_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    for candidate in other_quads:
        try:
            other = _order(candidate)
            bigger = abs(float(cv2.contourArea(other)))
            if bigger < own_area * 1.35:
                continue
            inside = sum(
                cv2.pointPolygonTest(other, tuple(map(float, pt)), False) >= 0
                for pt in quad
            )
            if inside >= 4:
                result["nested_in_larger_quad"] = True
                break
        except (TypeError, ValueError, cv2.error):
            continue
    flags = []
    if not result["perspective_feasible"]:
        flags.append("ill_conditioned_perspective")
    if not result["all_sides_measurable"]:
        flags.append("incomplete_outer_side_observation")
    if result["nested_in_larger_quad"]:
        flags.append("possible_internal_rectangle")
    if any(
        side["median_lab_step"] is None
        for side in result["chromatic_step_by_side"]
    ):
        flags.append("unmeasured_chromatic_boundary")
    result["reason_codes"] = flags
    return result
