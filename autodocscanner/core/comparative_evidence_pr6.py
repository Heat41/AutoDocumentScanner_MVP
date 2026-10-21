"""PR6 comparative evidence between candidate groups (diagnostic only).

Produces explainable pairwise observations, never a winner. Side counts are
heuristics, not calibrated accuracy; nesting alone never rejects a candidate.
No annotation/ground-truth data is consumed.
"""
from __future__ import annotations


def compare_candidate_evidence(report):
    combined = (report.get("pr6_combined_candidate_shadow") or {}).get("candidates") or []
    spatial = (report.get("pr6_spatial_relationship_shadow") or {}).get("relationships") or []
    candidates = {c["source"]: c for c in combined if isinstance(c, dict) and c.get("source")}
    assessments = []
    for relation in spatial:
        a, b = relation.get("candidate_a"), relation.get("candidate_b")
        if a not in candidates or b not in candidates or a == b:
            continue
        ca, cb = candidates[a], candidates[b]
        sa, sb = ca.get("side_checks") or {}, cb.get("side_checks") or {}
        dimensions = ("supported_external", "localized_aligned", "measurable_contrast")
        deltas = {}
        for key in dimensions:
            va, vb = sa.get(key), sb.get(key)
            if not isinstance(va, int) or isinstance(va, bool) or not isinstance(vb, int) or isinstance(vb, bool):
                deltas[key] = None
            else:
                deltas[key] = va - vb

        available = [v for v in deltas.values() if v is not None]
        positive = sum(v > 0 for v in available)
        negative = sum(v < 0 for v in available)
        if len(available) != len(dimensions):
            balance = "insufficient_evidence"
        elif positive and not negative:
            balance = "candidate_a_more_supported"
        elif negative and not positive:
            balance = "candidate_b_more_supported"
        elif positive and negative:
            balance = "mixed_evidence"
        else:
            balance = "equal_side_evidence"

        nested = relation.get("nested_geometry_risk") is True
        iou = relation.get("intersection_over_union")
        near_identical = isinstance(iou, (int, float)) and not isinstance(iou, bool) and .9 <= iou <= 1.
        if near_identical:
            geometry = "near_equivalent_geometry"
        elif nested:
            geometry = "nested_geometry_ambiguous"
        else:
            geometry = "distinct_or_partial_overlap"

        assessments.append({
            "candidate_a": a, "candidate_b": b,
            "geometry_relation": geometry,
            "side_evidence_delta_a_minus_b": deltas,
            "evidence_balance": balance,
            "candidate_a_warnings": list(ca.get("warnings") or []),
            "candidate_b_warnings": list(cb.get("warnings") or []),
            "smaller_source": relation.get("smaller_source") if nested else None,
            "larger_source": relation.get("larger_source") if nested else None,
            "requires_review": True,
            "winner": None,
        })
    return {
        "pairwise_comparisons": assessments,
        "compared_pair_count": len(assessments),
        "uncompared_candidates": [name for name in candidates if not any(
            row["candidate_a"] == name or row["candidate_b"] == name for row in assessments
        )],
        "suggested_for_review": None,
        "auto_select": None,
        "auto_crop_allowed": False,
        "status": "offline_diagnostic_only",
    }
