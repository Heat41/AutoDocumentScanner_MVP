"""Offline V3 candidate ranking, not a calibrated probability or production selector."""
from __future__ import annotations

import math


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rank_candidates(candidates):
    """Rank evaluated diagnostics without ever treating missing edges as zero.

    Each item is (source_label, metrics_from_detector_comparison).
    Returns candidates sorted by evidence score. An undecidable selection
    remains review_required, even if it ranks first.
    """
    ranked = []
    for label, item in candidates:
        if not item.get("found") or not item.get("geometry_valid"):
            continue
        boundary = item.get("boundary_evidence") or {}
        sides = boundary.get("side_support") or []
        coverage = boundary.get("side_coverage") or []
        warp = _finite(item.get("warp_quality"))
        if warp is None:
            continue
        observed = [_finite(x) for x in sides]
        observed = [x for x in observed if x is not None]
        complete = bool(boundary.get("evidence_complete")) and len(observed) == 4
        weak = min(observed) if observed else None
        support = sum(observed) / len(observed) if observed else None
        # Quality and measured boundaries compete; area never gives a bonus.
        score = 0.45 * max(0.0, min(1.0, warp))
        if support is not None:
            score += 0.35 * max(0.0, min(1.0, support))
        if weak is not None:
            score += 0.20 * max(0.0, min(1.0, weak))
        frame = bool(item.get("geometry", {}).get("frame_suspect") or
                     boundary.get("frame_suspect"))
        if frame:
            score -= 0.12
        if not complete:
            score -= 0.12
        score = round(max(0.0, score), 4)
        reasons = []
        if frame:
            reasons.append("frame_suspect")
        if not complete:
            reasons.append("incomplete_boundary")
        if weak is not None and weak < 0.15:
            reasons.append("weak_side")
        if warp < 0.25:
            reasons.append("low_warp_quality")
        ranked.append({
            "source": str(label), "score": score,
            "review_required": bool(reasons), "review_reasons": reasons,
            "boundary_complete": complete,
            "observed_sides": len(observed),
        })
    ranked.sort(key=lambda item: (-item["score"], item["source"]))
    # A ranking is not confidence. Hold all auto-selection until external
    # corner annotations establish safe thresholds and accuracy.
    return {
        "ranking": ranked,
        "suggested_for_review": ranked[0]["source"] if ranked else None,
        "auto_select": None,
        "selection_status": "offline_review_only" if ranked else "no_valid_candidates",
    }
