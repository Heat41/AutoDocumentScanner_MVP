"""PR5: fail-closed review policy for experimental KTP perspective candidates.

This is a decision advisory, NOT a calibrated probability or automatic crop
permission. Production and manual interaction are intentionally untouched.
"""
from __future__ import annotations

import math
import numpy as np


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _same_quad(a, b, image_shape, tolerance=0.008):
    """Compare corner sets allowing cyclic/reversed vertex order."""
    try:
        x = np.asarray(a, dtype=np.float64)
        y = np.asarray(b, dtype=np.float64)
        if x.shape != (4, 2) or y.shape != (4, 2) or not (np.isfinite(x).all() and np.isfinite(y).all()):
            return False
        diag = max(float(np.hypot(image_shape[1], image_shape[0])), 1.)
        return any(
            float(np.mean(np.linalg.norm(x - np.roll(z, n, axis=0), axis=1))) / diag <= tolerance
            for z in (y, y[::-1]) for n in range(4)
        )
    except (TypeError, ValueError, IndexError):
        return False


def assess_review_gate(report, image_shape):
    """Return an explainable decision; always require review until calibration.

    States: no_candidate, manual_review, preview_review. The last state means
    suitable for showing a suggested crop, never permission to save/crop blindly.
    Ranking scores are *not* calibrated confidence.
    """
    ranked = (report.get("v3_ranking") or {}).get("ranking") or []
    if not ranked:
        return {
            "status": "no_candidate", "action": "manual_corners",
            "candidate_source": None, "auto_crop_allowed": False,
            "confidence_calibrated": False, "reasons": ["no_valid_candidate"],
        }
    best = ranked[0]
    source = best.get("source")
    if not isinstance(source, str):
        return {"status": "no_candidate", "action": "manual_corners",
                "candidate_source": None, "auto_crop_allowed": False,
                "confidence_calibrated": False, "reasons": ["invalid_source"]}
    if source in ("stable", "adaptive"):
        item = report.get(source) or {}
    elif source.startswith("v2_") and source[3:].isdigit():
        idx = int(source[3:])
        proposals = report.get("v2") or []
        item = proposals[idx] if idx < len(proposals) else {}
    else:
        item = {}
    reasons = list(best.get("review_reasons") or [])
    evidence = item.get("boundary_evidence") or {}
    if not item.get("geometry_valid") or not evidence.get("valid"):
        reasons.append("invalid_geometry_or_boundary")
    if not evidence.get("evidence_complete"):
        reasons.append("incomplete_evidence")
    if _number(item.get("warp_quality")) is None:
        reasons.append("invalid_warp_quality")
    if (item.get("geometry") or {}).get("slight_extrapolation"):
        reasons.append("extrapolated_corner")
    if evidence.get("frame_review_needed"):
        reasons.append("possible_photo_frame")
    if len(ranked) > 1:
        second = ranked[1]
        first_score, second_score = _number(best.get("score")), _number(second.get("score"))
        if first_score is None or second_score is None or first_score - second_score < 0.08:
            first_item = (report.get(second.get("source")) or {}) if second.get("source") in ("stable", "adaptive") else {}
            if not _same_quad(item.get("corners"), first_item.get("corners"), image_shape):
                reasons.append("ambiguous_candidates")
    reasons = sorted(set(reasons))
    safe_preview = not reasons and not best.get("review_required", True)
    return {
        "status": "preview_review" if safe_preview else "manual_review",
        "action": "preview_suggested_corners" if safe_preview else "manual_corners",
        "candidate_source": source,
        "auto_crop_allowed": False,
        "confidence_calibrated": False,
        "reasons": reasons,
    }
