"""PR6 advisory gate: stricter review for distinct plausible candidate geometries.

Runs alongside (never in place of) the V3 gate, and cannot auto-crop.
"""
from __future__ import annotations


def assess_pr6_review_gate(report):
    baseline = report.get("v3_review_gate") or {}
    shadow = report.get("pr6_selection_shadow") or {}
    reasons = set(baseline.get("reasons") or [])
    warnings = set(shadow.get("warnings") or [])
    groups = shadow.get("groups") or []
    # Only independent geometry with a plausibly competitive evidence score
    # can invalidate preview. Do not treat stable/adaptive duplicates as votes.
    if len(groups) >= 2:
        first, second = groups[:2]
        if first.get("score", 0) - second.get("score", 0) < .15:
            reasons.add("competing_geometries")
        if ("different_candidate_extent" in warnings and
                second.get("score", 0) >= .30):
            reasons.add("different_candidate_extent")
    if "no_valid_candidate" in warnings:
        reasons.add("no_valid_candidate")
    reasons = sorted(reasons)
    preview = baseline.get("status") == "preview_review" and not reasons
    return {
        "status": "preview_review" if preview else "manual_review",
        "action": "preview_suggested_corners" if preview else "manual_corners",
        "candidate_source": baseline.get("candidate_source"),
        "auto_crop_allowed": False,
        "confidence_calibrated": False,
        "reasons": reasons,
        "selection_status": "pr6_shadow_review_only",
    }
