"""PR6 spatial-aware risk observations, shadow-only and non-decisional.

Combine overlap relationships with independently collected boundary diagnostics.
Nesting does not prove internal contour: a card may be small, and the enclosing
geometry may itself be a frame. No thresholds are fitted to holdout examples.
"""
from __future__ import annotations


def assess_spatial_risk(report):
    combined = (report.get("pr6_combined_candidate_shadow") or {}).get("candidates") or []
    spatial = report.get("pr6_spatial_relationship_shadow") or {}
    rows = spatial.get("relationships") or []
    by_source = {c.get("source"): c for c in combined if c.get("source")}
    evidence = []
    for source, candidate in by_source.items():
        inside = [r for r in rows if r.get("nested_geometry_risk") and
                  r.get("smaller_source") == source]
        encloses = [r for r in rows if r.get("nested_geometry_risk") and
                    r.get("larger_source") == source]
        reasons = list(candidate.get("warnings") or [])
        if inside:
            reasons.append("nested_within_other_proposal")
        if encloses:
            reasons.append("encloses_smaller_proposal")
        # A fully measured internal contour can still be the wrong candidate.
        # Equally, its enclosing candidate may be a photo frame rather than a card.
        ambiguous = bool(inside or encloses)
        observation = "spatial_ambiguity_review" if ambiguous else (
            "insufficient_boundary_evidence" if reasons else "no_observed_risk"
        )
        evidence.append({
            "source": source,
            "observation": observation,
            "reasons": sorted(set(reasons)),
            "nested_inside_count": len(inside),
            "encloses_count": len(encloses),
            "complete_boundary_evidence": bool(candidate.get("complete_independent_evidence")),
            "requires_human_validation": ambiguous or bool(reasons),
        })
    return {
        "candidate_assessments": evidence,
        "unresolved_nested_relationships": sum(
            1 for r in rows if r.get("nested_geometry_risk")
        ),
        "missing_spatial_sources": list(spatial.get("invalid_or_missing_sources") or []),
        "suggested_for_review": None,
        "auto_select": None,
        "auto_crop_allowed": False,
        "status": "offline_diagnostic_only",
    }
