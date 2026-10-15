"""PR6 shadow-only candidate selection diagnostics, independent of ground truth.

The goal is to distinguish a credible physical-card quadrilateral from
misleading internal rectangles or photo-frame proposals. No auto-crop.
"""
from __future__ import annotations

import numpy as np


def _corners(item):
    try:
        pts = np.asarray(item.get("corners"), dtype=np.float64)
        if pts.shape == (4, 2) and np.isfinite(pts).all():
            return pts
    except (ValueError, TypeError):
        pass
    return None


def _resolve(report, source):
    if source in ("stable", "adaptive"):
        return report.get(source) or {}
    if isinstance(source, str) and source.startswith("v2_") and source[3:].isdigit():
        i = int(source[3:])
        proposals = report.get("v2") or []
        if i < len(proposals):
            return proposals[i]
    return {}


def _equivalent(a, b, diagonal):
    x, y = _corners(a), _corners(b)
    if x is None or y is None:
        return False
    # Averages alone can merge a correctly-fitted border with an offset
    # border whose one corner is substantially misplaced.
    def cross2d(a, b):
        return float(a[0] * b[1] - a[1] * b[0])

    area_x = abs(cross2d(x[1] - x[0], x[2] - x[0])) + abs(cross2d(x[2] - x[0], x[3] - x[0]))
    area_y = abs(cross2d(y[1] - y[0], y[2] - y[0])) + abs(cross2d(y[2] - y[0], y[3] - y[0]))
    if max(area_x, area_y) <= 0 or min(area_x, area_y) / max(area_x, area_y) < .94:
        return False
    for z in (y, y[::-1]):
        for i in range(4):
            distances = np.linalg.norm(x - np.roll(z, i, axis=0), axis=1) / diagonal
            if float(distances.mean()) <= .008 and float(distances.max()) <= .015:
                return True
    return False


def compare_selection(report, image_shape):
    """Report competing candidate groups and unresolved selection risks.

    This diagnostic never recommends automatic saving. Score bonuses only
    identify consensus with *independent geometries*, not repeated labels
    for identical stable/adaptive detector outputs.
    """
    ranked = (report.get("v3_ranking") or {}).get("ranking") or []
    diag = max(float(np.hypot(*image_shape[:2])), 1.)
    groups = []
    for row in ranked:
        source = row["source"]
        item = _resolve(report, source)
        if not item.get("geometry_valid") or _corners(item) is None:
            continue
        group = next((g for g in groups if _equivalent(item, _resolve(report, g["representative"]), diag)), None)
        if group is None:
            group = {"representative": source, "sources": [], "score": row["score"],
                     "review_required": bool(row.get("review_required"))}
            groups.append(group)
        group["sources"].append(source)
        if row["score"] > group["score"]:
            group["score"] = row["score"]
            group["representative"] = source
        group["review_required"] = group["review_required"] or bool(row.get("review_required"))
    groups.sort(key=lambda group: -group["score"])
    warnings = []
    if not groups:
        warnings.append("no_valid_candidate")
    elif len(groups) > 1:
        if groups[0]["score"] - groups[1]["score"] < .15:
            warnings.append("competing_geometries")
        # A distinct geometry with significantly different extent can be
        # a physical-card candidate despite a superior-looking outer frame.
        p = _resolve(report, groups[0]["representative"])
        b = _resolve(report, groups[1]["representative"])
        area_a = (p.get("geometry") or {}).get("area_ratio") or 0
        area_b = (b.get("geometry") or {}).get("area_ratio") or 0
        if max(area_a, area_b) > 0 and min(area_a, area_b) / max(area_a, area_b) < .7:
            warnings.append("different_candidate_extent")
    if groups and groups[0]["review_required"]:
        warnings.append("leader_requires_review")
    return {"groups": groups, "warnings": warnings, "auto_select": None,
            "status": "offline_diagnostic_only"}
