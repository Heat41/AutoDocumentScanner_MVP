"""PR6 geometric consensus and evidence interpretation (offline shadow only).

Near-equivalent quads use complete-link clustering to prevent chain merging.
Boundary offset consistency is descriptive, not an accuracy certificate.
Ground-truth annotations are never read. No candidate winner or crop decision.
"""
from __future__ import annotations

import math

from autodocscanner.core.combined_candidate_pr6 import _finite


def _offset_summary(localization):
    sides = (localization or {}).get("side_localization") or []
    if len(sides) != 4:
        return {"classification": "insufficient_measurements", "coherent_side_count": 0,
                "offsets": [], "coverage_sufficient": False}
    offsets = []
    coherent = 0
    for side in sides:
        if not isinstance(side, dict):
            offsets.append(None)
            continue
        coverage = _finite(side.get("coverage"))
        offset = _finite(side.get("median_offset_fraction"))
        spread = _finite(side.get("offset_spread"))
        if coverage is None or offset is None or spread is None or coverage < .5:
            offsets.append(None)
            continue
        offsets.append(round(offset, 4))
        if spread <= .25:
            coherent += 1
    complete = all(v is not None for v in offsets)
    if not complete:
        category = "insufficient_measurements"
    elif coherent == 4:
        category = ("aligned_consistent" if all(abs(v) <= .25 for v in offsets)
                    else "coherent_displaced_boundary")
    else:
        category = "variable_boundary_offsets"
    return {"classification": category, "coherent_side_count": coherent,
            "offsets": offsets, "coverage_sufficient": complete}


def interpret_candidate_consensus(report):
    candidates = (report.get("pr6_combined_candidate_shadow") or {}).get("candidates") or []
    relationships = (report.get("pr6_spatial_relationship_shadow") or {}).get("relationships") or []
    localized = report.get("pr6_boundary_localization_shadow") or {}
    names = [x.get("source") for x in candidates if isinstance(x, dict) and isinstance(x.get("source"), str)]
    # Full-link: every two members must be near-equivalent; not simply connected.
    equivalent = set()
    for relation in relationships:
        if not isinstance(relation, dict):
            continue
        a, b = relation.get("candidate_a"), relation.get("candidate_b")
        iou = _finite(relation.get("intersection_over_union"))
        if a in names and b in names and a != b and iou is not None and .9 <= iou <= 1:
            equivalent.add(frozenset((a, b)))
    clusters = []
    for name in names:
        placed = False
        for cluster in clusters:
            if all(frozenset((name, other)) in equivalent for other in cluster):
                cluster.append(name)
                placed = True
                break
        if not placed:
            clusters.append([name])

    annotations = []
    by_name = {candidate["source"]: candidate for candidate in candidates if isinstance(candidate, dict) and candidate.get("source")}
    for name in names:
        candidate = by_name[name]
        offset = _offset_summary(localized.get(name))
        warnings = candidate.get("warnings") or []
        annotations.append({
            "source": name,
            "offset_evidence": offset,
            "boundary_warnings": list(warnings),
            "evidence_complete": candidate.get("complete_independent_evidence") is True,
            "consensus_group": next(idx for idx, cluster in enumerate(clusters) if name in cluster),
        })
    nested_conflicts = []
    for relation in relationships:
        if not isinstance(relation, dict) or relation.get("nested_geometry_risk") is not True:
            continue
        inner, outer = relation.get("smaller_source"), relation.get("larger_source")
        if inner not in by_name or outer not in by_name:
            continue
        # Both 'inner strong' and 'outer frame-risk' are observational only.
        nested_conflicts.append({
            "inner": inner, "outer": outer,
            "inner_evidence_complete": by_name[inner].get("complete_independent_evidence") is True,
            "outer_frame_warning": "frame_or_extrapolation_risk" in
                                   (by_name[outer].get("warnings") or []),
            "interpretation": "ambiguous_nested_candidates",
            "winner": None,
        })
    return {
        "consensus_groups": [{"members": group, "member_count": len(group),
                              "is_near_equivalent": len(group) > 1}
                             for group in clusters],
        "candidate_interpretations": annotations,
        "nested_conflicts": nested_conflicts,
        "consensus_is_not_ground_truth": True,
        "suggested_for_review": None, "auto_select": None,
        "auto_crop_allowed": False, "status": "offline_diagnostic_only",
    }
