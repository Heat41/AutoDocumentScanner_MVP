"""PR9: physical edge alignment audit for V3 Stable/Adaptive, offline only.

This module deliberately examines *only* the two contemporary V3 detector
outputs. Historical V2 candidates, manual ground truth, detector scores and
PR6 candidate rankings are neither consumed nor allowed to select a corner.

Lab-color gradients help diagnose edges with weak grayscale contrast. Profile
peaks are localized against each proposed side, testing whether evidence is
centered rather than at a nearby parallel edge. These measurements are NOT
proof that the surrounding quadrilateral is a document. Ambiguity remains a
manual-review condition, not a license to crop.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad

V3_SOURCES = ("stable", "adaptive")


def _result():
    return {
        "status": "offline_diagnostic_only",
        "mode": "pr9_v3_boundary_shadow",
        "sources_checked": list(V3_SOURCES),
        "candidates": {},
        "distinct_geometry_conflict": False,
        "requires_human_confirmation": True,
        "suggested_for_review": None,
        "auto_select": None,
        "auto_crop_allowed": False,
    }


def _cyclic(points):
    center = points.mean(axis=0)
    angle = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angle)]
    # Stable and Adaptive can supply different first vertices.
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _side_observations(gx, gy, a, b, center, image_shape, *, samples):
    height, width = image_shape[:2]
    direction = b - a
    tangent = direction / max(float(np.linalg.norm(direction)), 1e-8)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    if np.dot(normal, center - (a + b) / 2) < 0:
        normal *= -1
    radius = max(3.0, min(height, width) * .025)
    offsets = np.linspace(-radius, radius, 17)
    peaks, ratios, dispersions, observed_bands = [], [], [], set()
    valid_profiles = 0
    aligned_profiles = 0
    ambiguous_profiles = 0
    for index, fraction in enumerate(np.linspace(.14, .86, samples)):
        location = a + direction * fraction
        scan = location[None, :] + offsets[:, None] * normal[None, :]
        within = ((scan[:, 0] >= 2) & (scan[:, 0] < width - 2) &
                  (scan[:, 1] >= 2) & (scan[:, 1] < height - 2))
        if not within[8] or within.sum() < 11:
            continue
        xx = scan[:, 0].astype(np.float32).reshape(1, -1)
        yy = scan[:, 1].astype(np.float32).reshape(1, -1)
        sample_x = cv2.remap(gx, xx, yy, cv2.INTER_LINEAR).reshape(-1, 3)
        sample_y = cv2.remap(gy, xx, yy, cv2.INTER_LINEAR).reshape(-1, 3)
        projected = sample_x * normal[0] + sample_y * normal[1]
        # L, a, b channels; retain chromatic edges with little luminance change.
        strength = np.maximum(np.abs(projected[:, 0]),
                              1.15 * np.linalg.norm(projected[:, 1:], axis=1))
        strength[~within] = 0.
        peak = float(np.max(strength))
        if peak < 18.:
            continue
        valid_profiles += 1
        peak_at = int(np.argmax(strength))
        fraction_offset = float(offsets[peak_at] / radius)
        central_peak = float(max(strength[6:11]))
        ratio = central_peak / peak
        is_ambiguous = int(np.count_nonzero(strength >= .85 * peak)) >= 5
        aligned = abs(fraction_offset) <= .25 and ratio >= .72 and not is_ambiguous
        if aligned:
            aligned_profiles += 1
            observed_bands.add(min(3, (index * 4) // samples))
        if is_ambiguous:
            ambiguous_profiles += 1
        peaks.append(fraction_offset)
        ratios.append(ratio)
        dispersions.append(peak)
    coverage = valid_profiles / samples
    if peaks:
        median_offset = float(np.median(peaks))
        mad = float(np.median(np.abs(np.array(peaks) - median_offset)))
        centered_ratio = float(np.median(ratios))
        median_strength = float(np.median(dispersions))
    else:
        median_offset = mad = centered_ratio = median_strength = None
    aligned_fraction = aligned_profiles / valid_profiles if valid_profiles else 0.
    ambiguity = ambiguous_profiles / valid_profiles if valid_profiles else 1.
    is_aligned = (coverage >= .65 and aligned_fraction >= .70 and
                  median_offset is not None and abs(median_offset) <= .25 and
                  mad <= .25 and centered_ratio >= .72 and
                  ambiguity <= .45 and len(observed_bands) >= 3)
    reason = ("aligned" if is_aligned else
              "insufficient_edge_coverage" if coverage < .65 else
              "displaced_or_ambiguous_boundary")
    return {
        "state": reason,
        "coverage": round(coverage, 4),
        "aligned_fraction": round(aligned_fraction, 4),
        "median_offset_fraction": round(median_offset, 4) if median_offset is not None else None,
        "offset_mad": round(mad, 4) if mad is not None else None,
        "median_center_ratio": round(centered_ratio, 4) if centered_ratio is not None else None,
        "median_gradient": round(median_strength, 2) if median_strength is not None else None,
        "ambiguous_profile_fraction": round(ambiguity, 4),
        "supported_length_bands": len(observed_bands),
    }


def audit_v3_boundaries_pr9(image, report, *, samples=32):
    """Return bounded numeric diagnostics; never authorize a crop.

    Results are invariant to any V2 proposal, reference annotation, or score
    supplied in `report`. Input image is not modified. This is an exploratory
    image diagnostic; thresholds are not calibrated to a production dataset.
    """
    output = _result()
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 64):
        output["reason"] = "invalid_image"
        return output
    if not isinstance(report, dict):
        output["reason"] = "invalid_report"
        return output
    if type(samples) is not int or not 16 <= samples <= 128:
        output["reason"] = "invalid_samples"
        return output

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    smooth = cv2.GaussianBlur(lab.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    geometries = {}
    for source in V3_SOURCES:
        candidate = report.get(source)
        reasons = []
        if not isinstance(candidate, dict) or candidate.get("found") is not True:
            output["candidates"][source] = {
                "status": "missing_candidate", "reason_codes": ["candidate_missing"],
                "fully_aligned_for_review": False, "side_observations": [],
            }
            continue
        try:
            points = np.asarray(candidate.get("corners"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            points = np.empty((0, 2), dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            output["candidates"][source] = {
                "status": "invalid_geometry", "reason_codes": ["invalid_corner_points"],
                "fully_aligned_for_review": False, "side_observations": [],
            }
            continue
        valid, geometry_reasons, geom = validate_quad(points, image.shape)
        if not valid or candidate.get("geometry_valid") is False:
            output["candidates"][source] = {
                "status": "invalid_geometry",
                "reason_codes": sorted(set(geometry_reasons + (
                    ["upstream_geometry_rejected"] if candidate.get("geometry_valid") is False else []
                ))),
                "fully_aligned_for_review": False, "side_observations": [],
            }
            continue
        ordered = _cyclic(points)
        geometries[source] = ordered
        center = ordered.mean(axis=0)
        sides = [_side_observations(gx, gy, ordered[i], ordered[(i + 1) % 4],
                                    center, image.shape, samples=samples)
                 for i in range(4)]
        if geom.get("slight_extrapolation") or geom.get("near_frame_vertex_count", 0):
            reasons.append("frame_or_extrapolation_risk")
        if not all(side["state"] == "aligned" for side in sides):
            reasons.append("incomplete_or_displaced_physical_sides")
        output["candidates"][source] = {
            "status": "four_side_alignment_observed" if not reasons else "requires_manual_review",
            "reason_codes": reasons,
            "fully_aligned_for_review": not reasons,
            "side_observations": sides,
            "near_frame_vertex_count": int(geom.get("near_frame_vertex_count", 0)),
        }
    if len(geometries) == 2:
        first, second = (geometries[source] for source in V3_SOURCES)
        diag = math.hypot(image.shape[0], image.shape[1])
        mean_distance = float(np.linalg.norm(first-second, axis=1).mean()) / diag
        output["normalized_candidate_disagreement"] = round(mean_distance, 5)
        output["distinct_geometry_conflict"] = mean_distance > .008
        if output["distinct_geometry_conflict"]:
            # A good edge on two distinct rectangles does not resolve nesting.
            for candidate in output["candidates"].values():
                candidate["fully_aligned_for_review"] = False
                if "unresolved_v3_candidate_conflict" not in candidate["reason_codes"]:
                    candidate["reason_codes"].append("unresolved_v3_candidate_conflict")
                candidate["status"] = "requires_manual_review"
    return output


def diagnose_v3_image_pr9(image, *, detectors=None):
    """Run only Stable/Adaptive and return PR9 review diagnostics.

    `detectors` is injectable for synthetic tests. This entry point never calls
    the legacy contour-proposal comparison (V2), writes pixels, or crops.
    """
    if detectors is None:
        from autodocscanner.core.perspective_engine import AutoPerspectiveEngine
        from autodocscanner.core.perspective_engine_baseline import (
            AutoPerspectiveEngine as StableBaselinePerspectiveEngine,
        )
        detectors = {
            "stable": StableBaselinePerspectiveEngine(),
            "adaptive": AutoPerspectiveEngine(),
        }
    if not isinstance(detectors, dict):
        raise TypeError("detectors must be a mapping")
    observations = {}
    for source in V3_SOURCES:
        detector = detectors.get(source)
        if detector is None:
            observations[source] = {"found": False}
            continue
        try:
            corners, _ = detector.detect(image)
        except (cv2.error, ValueError, TypeError, ArithmeticError):
            corners = None
        observations[source] = {
            "found": corners is not None,
            "corners": corners,
        }
    return audit_v3_boundaries_pr9(image, observations)
