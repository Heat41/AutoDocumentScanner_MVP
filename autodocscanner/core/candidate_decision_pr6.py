"""Shadow-only PR6 candidate decision and conflict resolution.

Reconciles existing geometry consensus, boundary evidence, and review gates.
Never reads annotation/ground truth, chooses a crop, or changes production.
"""
from __future__ import annotations

from collections import defaultdict
import math


_REQUIRED = (
    "pr6_combined_candidate_shadow", "pr6_evidence_consensus_shadow",
    "pr6_spatial_relationship_shadow", "pr6_comparative_evidence_shadow",
)
_SIDE_KEYS = ("supported_external", "localized_aligned", "measurable_contrast")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _side_complete(candidate):
    checks = candidate.get("side_checks")
    return isinstance(checks, dict) and all(
        isinstance(checks.get(key), int) and not isinstance(checks[key], bool)
        and checks[key] == 4 for key in _SIDE_KEYS
    )


def _offset_agreement(a, b):
    """Compare localization offsets within the existing quarter-window scale."""
    aa = (a.get("offset_evidence") or {}).get("offsets") or []
    bb = (b.get("offset_evidence") or {}).get("offsets") or []
    if len(aa) != 4 or len(bb) != 4:
        return False
    if any(_number(x) is None for x in aa + bb):
        return False
    return max(abs(x - y) for x, y in zip(aa, bb)) <= .25


def decide_candidate_review(report):
    """Decide *review routing* only, never candidate correctness or auto-crop."""
    if not isinstance(report, dict):
        report = {}
    def section(key):
        value = report.get(key)
        return value if isinstance(value, dict) else {}

    combined = section("pr6_combined_candidate_shadow")
    consensus = section("pr6_evidence_consensus_shadow")
    spatial = section("pr6_spatial_relationship_shadow")
    comparison = section("pr6_comparative_evidence_shadow")
    baseline_gate = section("v3_review_gate")
    prior_gate = section("pr6_review_gate_shadow")
    raw_candidates = combined.get("candidates") or []
    raw_groups = consensus.get("consensus_groups") or []
    raw_interp = consensus.get("candidate_interpretations") or []

    reasons = set()
    if any(not isinstance(report.get(key), dict) or
           report[key].get("status") != "offline_diagnostic_only" for key in _REQUIRED):
        reasons.add("missing_or_invalid_shadow_evidence")
    if baseline_gate.get("status") != "preview_review" or baseline_gate.get("reasons"):
        reasons.add("baseline_review_gate_requires_manual")
    if prior_gate.get("status") != "preview_review" or prior_gate.get("reasons"):
        reasons.add("prior_review_gate_requires_manual")

    candidates = {}
    for item in raw_candidates:
        if not isinstance(item, dict) or not isinstance(item.get("source"), str) or not item["source"]:
            reasons.add("invalid_candidate_evidence")
            continue
        source = item["source"]
        if source in candidates:
            reasons.add("duplicate_candidate_source")
        candidates[source] = item

    interpretations = {}
    for item in raw_interp:
        if not isinstance(item, dict) or item.get("source") not in candidates:
            reasons.add("invalid_candidate_interpretation")
            continue
        source = item["source"]
        if source in interpretations:
            reasons.add("duplicate_candidate_interpretation")
        interpretations[source] = item

    candidate_assessments = []
    clean = set()
    for name, item in candidates.items():
        interpretation = interpretations.get(name) or {}
        offset = interpretation.get("offset_evidence") or {}
        warnings = item.get("warnings") or []
        concerns = []
        if not _side_complete(item) or item.get("complete_independent_evidence") is not True:
            concerns.append("incomplete_independent_boundary_evidence")
        if warnings:
            concerns.append("boundary_warnings_present")
        if (offset.get("classification") != "aligned_consistent" or
                offset.get("coverage_sufficient") is not True or
                offset.get("coherent_side_count") != 4):
            concerns.append("offset_alignment_unverified")
        if not interpretation:
            concerns.append("missing_candidate_interpretation")
        if not concerns:
            clean.add(name)
        candidate_assessments.append({
            "source": name,
            "evidence_state": "complete_for_review" if not concerns else "requires_manual_verification",
            "reasons": concerns,
            "offset_classification": offset.get("classification", "unknown"),
            "warnings": list(warnings),
        })

    groups = []
    seen = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict) or not isinstance(raw_group.get("members"), list):
            reasons.add("invalid_consensus_group")
            continue
        members = raw_group["members"]
        if not members or any(not isinstance(name, str) or name not in candidates for name in members):
            reasons.add("invalid_consensus_members")
            continue
        if len(set(members)) != len(members) or any(name in seen for name in members):
            reasons.add("overlapping_consensus_membership")
        seen.update(members)
        offset_disagreement = len(members) > 1 and any(
            not _offset_agreement(interpretations.get(members[i]) or {},
                                  interpretations.get(members[j]) or {})
            for i in range(len(members)) for j in range(i + 1, len(members))
        )
        groups.append({
            "group_id": len(groups), "members": list(members),
            "all_members_complete": all(name in clean for name in members),
            "boundary_offset_disagreement": bool(offset_disagreement),
            "requires_review": bool(offset_disagreement) or any(name not in clean for name in members),
        })
    if not candidates:
        reasons.add("no_valid_candidate")
    elif not groups or seen != set(candidates):
        reasons.add("incomplete_consensus_coverage")
    if len(groups) > 1:
        reasons.add("distinct_consensus_groups")
    if any(group["boundary_offset_disagreement"] for group in groups):
        reasons.add("within_group_boundary_disagreement")

    by_group = {name: group["group_id"] for group in groups for name in group["members"]}
    grouped_relations = defaultdict(list)
    grouped_comparisons = defaultdict(list)
    for pair in comparison.get("pairwise_comparisons") or []:
        if not isinstance(pair, dict):
            continue
        a, b = pair.get("candidate_a"), pair.get("candidate_b")
        if a in by_group and b in by_group and by_group[a] != by_group[b]:
            key = tuple(sorted((by_group[a], by_group[b])))
            grouped_comparisons[key].append({
                "candidate_a": a, "candidate_b": b,
                "evidence_balance": pair.get("evidence_balance", "insufficient_evidence"),
            })
    for relation in spatial.get("relationships") or []:
        if not isinstance(relation, dict):
            continue
        left, right = relation.get("candidate_a"), relation.get("candidate_b")
        if left not in by_group or right not in by_group:
            continue
        gleft, gright = by_group[left], by_group[right]
        if gleft != gright:
            grouped_relations[tuple(sorted((gleft, gright)))].append(relation)

    conflicts = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            relations = grouped_relations.get((i, j)) or []
            comparisons = grouped_comparisons.get((i, j)) or []
            nested = [rel for rel in relations if rel.get("nested_geometry_risk") is True]
            frame_ambiguity = any(
                "frame_or_extrapolation_risk" in (candidates.get(rel.get("larger_source")) or {}).get("warnings", [])
                for rel in nested
            )
            strong_inner = any(rel.get("smaller_source") in clean for rel in nested)
            conflicts.append({
                "group_a": i, "group_b": j,
                "kind": ("missing_spatial_relation" if not relations else
                         "nested_extent_conflict" if nested else "distinct_geometry_conflict"),
                "nested_observations": len(nested),
                "outer_frame_warning": frame_ambiguity,
                "well_supported_inner": strong_inner,
                "comparative_evidence": comparisons,
                "resolution": "human_four_corner_review",
                "winner": None,
            })
            if not relations:
                reasons.add("missing_spatial_relation")
            if not comparisons:
                reasons.add("missing_comparative_relation")
    if conflicts:
        reasons.add("unresolved_geometry_conflict")

    # Preview is permitted only for a single, fully corroborated group,
    # consistent offsets, and an existing PR6 gate that already permits preview.
    preview = (not reasons and len(groups) == 1 and
               groups[0]["all_members_complete"] and
               not groups[0]["boundary_offset_disagreement"])
    if not preview and not reasons and groups:
        reasons.add("candidate_evidence_requires_review")
    review_source = groups[0]["members"][0] if preview else None
    return {
        "status": "preview_review" if preview else "manual_review",
        "action": "preview_suggested_corners" if preview else "manual_corners",
        "suggested_for_review": review_source,
        "candidate_assessments": candidate_assessments,
        "consensus_assessments": groups,
        "conflicts": conflicts,
        "reasons": sorted(reasons),
        "requires_explicit_confirmation": True,
        "auto_select": None,
        "auto_crop_allowed": False,
        "confidence_calibrated": False,
        "mode": "offline_shadow_review_only",
    }
