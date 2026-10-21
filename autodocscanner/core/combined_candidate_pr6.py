"""PR6 combined candidate evidence, offline and independent of annotations.

Evaluates geometrical validity, externally observed sides, boundary-localization
coverage and competing candidate groups. No calibration or automatic selection.
"""
from __future__ import annotations

import math


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _resolve(report, source):
    if source in ("stable", "adaptive"):
        return report.get(source) or {}
    if isinstance(source, str) and source.startswith("v2_") and source[3:].isdigit():
        idx = int(source[3:])
        candidates = report.get("v2") or []
        return candidates[idx] if 0 <= idx < len(candidates) else {}
    return {}


def compare_combined_evidence(report):
    selection = report.get("pr6_selection_shadow") or {}
    physical = report.get("pr6_physical_boundary_shadow") or {}
    localized = report.get("pr6_boundary_localization_shadow") or {}
    groups = selection.get("groups") or []
    results = []
    for group in groups:
        source = group.get("representative")
        item = _resolve(report, source)
        boundary = item.get("boundary_evidence") or {}
        geometry = item.get("geometry") or {}
        physical_item = physical.get(source) or {}
        localized_item = localized.get(source) or {}
        local_sides = localized_item.get("side_localization") or []
        physical_sides = physical_item.get("side_measurements") or []
        boundary_sides = boundary.get("side_support") or []

        reasons = []
        if not item.get("geometry_valid"):
            reasons.append("invalid_geometry")
        if geometry.get("frame_suspect") or geometry.get("slight_extrapolation"):
            reasons.append("frame_or_extrapolation_risk")
        if not boundary.get("evidence_complete"):
            reasons.append("incomplete_external_boundary")
        if not localized_item.get("measurable"):
            reasons.append("incomplete_edge_localization")
        if not physical_item.get("measurable"):
            reasons.append("incomplete_inner_outer_contrast")

        supported = sum(1 for value in boundary_sides if _finite(value) is not None and value >= .20)
        aligned = sum(1 for side in local_sides
                      if _finite(side.get("coverage")) is not None and
                      side["coverage"] >= .5 and
                      _finite(side.get("median_offset_fraction")) is not None and
                      abs(side["median_offset_fraction"]) <= .25 and
                      _finite(side.get("offset_spread")) is not None and
                      side["offset_spread"] <= .25)
        contrast = sum(1 for side in physical_sides
                       if _finite(side.get("contrast")) is not None and side["contrast"] >= 5 and
                       _finite(side.get("coverage")) is not None and side["coverage"] >= .5)
        if supported < 4:
            reasons.append("weak_external_sides")
        if aligned < 4:
            reasons.append("poor_edge_alignment")
        if contrast < 4:
            reasons.append("weak_inner_outer_separation")
        results.append({
            "source": source, "detector_sources": list(group.get("sources") or []),
            "ranking_score": _finite(group.get("score")),
            "area_ratio": _finite(geometry.get("area_ratio")),
            "side_checks": {"supported_external": supported, "localized_aligned": aligned,
                            "measurable_contrast": contrast, "total": 4},
            "warnings": sorted(set(reasons)),
            "complete_independent_evidence": not reasons,
        })
    return {
        "candidates": results,
        "candidate_count": len(results),
        "distinct_geometry_count": len(groups),
        "selection_warnings": list(selection.get("warnings") or []),
        "suggested_for_review": None,
        "auto_select": None,
        "auto_crop_allowed": False,
        "status": "offline_diagnostic_only",
    }
