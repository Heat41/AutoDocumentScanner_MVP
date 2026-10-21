"""PR6 offline pairwise spatial relationships between distinct candidate quads.

No identity, annotations, or score-derived decision; nesting is contextual
evidence only, never proof that either the inner or outer polygon is a card.
"""
from __future__ import annotations

import cv2
import numpy as np

from autodocscanner.core.combined_candidate_pr6 import _resolve


def _quad(item):
    try:
        pts = np.asarray(item.get("corners"), dtype=np.float32)
        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            return None
        hull = cv2.convexHull(pts).reshape(-1, 2)
        if len(hull) != 4 or cv2.contourArea(hull) <= 1:
            return None
        return hull
    except (TypeError, ValueError, cv2.error):
        return None


def compare_spatial_relationships(report):
    groups = (report.get("pr6_selection_shadow") or {}).get("groups") or []
    valid = []
    missing = []
    for group in groups:
        source = group.get("representative")
        polygon = _quad(_resolve(report, source))
        if polygon is None:
            missing.append(source)
        else:
            valid.append((source, polygon, float(cv2.contourArea(polygon))))
    relationships = []
    for i, (left, a, area_a) in enumerate(valid):
        for right, b, area_b in valid[i+1:]:
            try:
                intersection, _ = cv2.intersectConvexConvex(a, b)
                intersection = max(0.0, float(intersection))
            except cv2.error:
                intersection = 0.0
            small, large = ((left, area_a), (right, area_b)) if area_a <= area_b else ((right, area_b), (left, area_a))
            smaller_covered = min(1., intersection / max(small[1], 1e-9))
            union = area_a + area_b - intersection
            iou = intersection / union if union > 0 else 0.
            nested = smaller_covered >= .9 and small[1] / large[1] <= .7
            relationships.append({
                "candidate_a": left, "candidate_b": right,
                "intersection_over_union": round(iou, 4),
                "smaller_coverage": round(smaller_covered, 4),
                "smaller_area_fraction": round(small[1] / large[1], 4),
                "smaller_source": small[0], "larger_source": large[0],
                "nested_geometry_risk": bool(nested),
                "interpretation": "possible_internal_contour_or_distant_card" if nested
                                  else "no_strong_nesting_evidence",
            })
    return {
        "relationships": relationships,
        "invalid_or_missing_sources": missing,
        "nested_pair_count": sum(1 for p in relationships if p["nested_geometry_risk"]),
        "auto_select": None, "auto_crop_allowed": False,
        "status": "offline_diagnostic_only",
    }
