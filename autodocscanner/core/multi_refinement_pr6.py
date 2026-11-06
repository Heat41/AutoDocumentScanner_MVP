"""Evaluate corner refinement for several independent PR6 proposals, offline only.

A proposed refinement is NOT evidence of the correct physical KTP contour.
This module reads image-derived geometry/evidence only, never ground truth,
never changes original detector outputs, rankings, review gates, or auto-crop.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from autodocscanner.core.boundary_evidence import boundary_evidence
from autodocscanner.core.corner_refinement_v3 import refine_corners
from autodocscanner.core.quad_validation import validate_quad


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _corners(value):
    try:
        quad = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        return None
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None
    return quad.copy()


def _resolve(report, source):
    if source in ("stable", "adaptive"):
        item = report.get(source)
        return item if isinstance(item, dict) else {}
    if isinstance(source, str) and source.startswith("v2_") and source[3:].isdigit():
        items = report.get("v2") or []
        index = int(source[3:])
        if isinstance(items, list) and index < len(items) and isinstance(items[index], dict):
            return items[index]
    return {}


def _quality(engine, image, quad):
    quality = _number(engine._warp_quality(engine.warp(image, quad)))
    if quality is None:
        raise ValueError("warp quality is not finite")
    return quality


def _boundary_drop(before, after):
    """Require measured four sides and forbid material degradation on any side."""
    if not (before.get("evidence_complete") and after.get("evidence_complete")):
        return "incomplete_boundary_evidence"
    old = before.get("side_support") or []
    new = after.get("side_support") or []
    if len(old) != 4 or len(new) != 4:
        return "incomplete_side_support"
    old_values = [_number(x) for x in old]
    new_values = [_number(x) for x in new]
    if any(x is None for x in old_values + new_values):
        return "unmeasured_side_support"
    if any(y + .04 < x for x, y in zip(old_values, new_values)):
        return "per_side_evidence_regression"
    return None


def evaluate_multi_candidate_refinement(image, report, engine, *, max_candidates=12,
                                        refine_fn=None):
    """Review every distinct PR6 selection-group representative up to a CPU budget.

    Distinct geometries are retained even if consensus groups have high IoU.
    Stable/adaptive duplicates within one selection group are evaluated once.
    No annotations, labels, scores, or previous refinement decisions participate
    in acceptance. The method never chooses between candidates.
    """
    result = {
        "candidates": [], "candidate_count": 0, "evaluated_count": 0,
        "plausible_refinement_count": 0, "not_evaluated_count": 0,
        "suggested_for_review": None, "auto_select": None,
        "auto_crop_allowed": False, "status": "offline_diagnostic_only",
    }
    if not isinstance(report, dict) or not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        result["reason"] = "invalid_input"
        return result
    selection = report.get("pr6_selection_shadow") or {}
    groups = selection.get("groups") or [] if isinstance(selection, dict) else []
    if not isinstance(groups, list):
        result["reason"] = "invalid_selection_groups"
        return result
    refine = refine_fn if refine_fn is not None else refine_corners
    budget = max(0, min(int(max_candidates), 24))
    used_sources = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        source = group.get("representative")
        if not isinstance(source, str) or not source or source in used_sources:
            continue
        used_sources.add(source)
        item = _resolve(report, source)
        before = _corners(item.get("corners"))
        entry = {
            "source": source,
            "selection_group_sources": list(group.get("sources") or []),
            "accepted_for_shadow": False,
            "reason": "not_evaluated",
            "before_corners": before.tolist() if before is not None else None,
            "attempted_corners": None,
            "after_corners": before.tolist() if before is not None else None,
        }
        result["candidates"].append(entry)
        if len(result["candidates"]) > budget:
            entry["reason"] = "cpu_budget_exhausted"
            continue
        if before is None:
            entry["reason"] = "missing_or_invalid_corners"
            continue
        valid, _, geometry = validate_quad(before, image.shape)
        if not valid:
            entry["reason"] = "invalid_original_geometry"
            continue
        # A cropped/photo-frame proposal is not a reliable seed for local fitting.
        if geometry.get("frame_suspect") or geometry.get("slight_extrapolation"):
            entry["reason"] = "frame_or_extrapolation_risk"
            continue
        result["evaluated_count"] += 1
        try:
            after_proposal, diagnostic = refine(image, before.copy())
            proposed = _corners(after_proposal)
            if proposed is not None:
                entry["attempted_corners"] = proposed.tolist()
            if not isinstance(diagnostic, dict) or diagnostic.get("accepted") is not True:
                entry["reason"] = (diagnostic.get("reason", "refiner_declined")
                                   if isinstance(diagnostic, dict) else "invalid_refiner_response")
                continue
            if proposed is None:
                entry["reason"] = "invalid_refined_corners"
                continue
            allowed, _, refined_geometry = validate_quad(proposed, image.shape)
            if not allowed or refined_geometry.get("frame_suspect") or refined_geometry.get("slight_extrapolation"):
                entry["reason"] = "refined_geometry_invalid_or_frame_risk"
                continue
            # The vertex order must be preserved; otherwise warp mapping may flip.
            area_before = float(cv2.contourArea(before, oriented=True))
            area_after = float(cv2.contourArea(proposed, oriented=True))
            if area_before * area_after <= 0:
                entry["reason"] = "vertex_order_or_winding_changed"
                continue
            h, w = image.shape[:2]
            diagonal = max(float(np.hypot(w, h)), 1.)
            shift = np.linalg.norm(proposed - before, axis=1)
            max_shift = float(shift.max()) / diagonal
            mean_shift = float(shift.mean()) / diagonal
            area_change = abs(abs(area_after) / abs(area_before) - 1.)
            old_lengths = np.linalg.norm(np.roll(before, -1, axis=0) - before, axis=1)
            new_lengths = np.linalg.norm(np.roll(proposed, -1, axis=0) - proposed, axis=1)
            max_side_change = float(np.max(np.abs(new_lengths / old_lengths - 1.)))
            entry["geometric_stability"] = {
                "max_corner_shift_fraction": round(max_shift, 6),
                "mean_corner_shift_fraction": round(mean_shift, 6),
                "relative_area_change": round(area_change, 6),
                "max_relative_side_change": round(max_side_change, 6),
            }
            if max_shift < 1e-6:
                entry["reason"] = "no_corner_change"
                continue
            if max_shift > .015 or area_change > .04 or max_side_change > .05:
                entry["reason"] = "perspective_stability_regression"
                continue
            before_edge = boundary_evidence(image, before)
            after_edge = boundary_evidence(image, proposed)
            edge_problem = _boundary_drop(before_edge, after_edge)
            if edge_problem:
                entry["reason"] = edge_problem
                continue
            old_quality = _quality(engine, image, before)
            new_quality = _quality(engine, image, proposed)
            entry["warp_quality"] = {"before": round(old_quality, 4), "attempted": round(new_quality, 4)}
            if new_quality + .03 < old_quality:
                entry["reason"] = "warp_quality_regression"
                continue
            # Stable as an experimental line adjustment, not correct KTP detection.
            entry["accepted_for_shadow"] = True
            entry["reason"] = "geometrically_stable_refinement"
            entry["after_corners"] = proposed.tolist()
            result["plausible_refinement_count"] += 1
        except (ValueError, TypeError, OverflowError, cv2.error, np.linalg.LinAlgError):
            entry["reason"] = "refinement_evaluation_error"
    result["candidate_count"] = len(result["candidates"])
    result["not_evaluated_count"] = sum(
        row["reason"] in {"cpu_budget_exhausted", "not_evaluated"} for row in result["candidates"]
    )
    return result
