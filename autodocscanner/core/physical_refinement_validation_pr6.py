"""Interpret physical-boundary context of PR6 refinement proposals (shadow only).

Accepted geometric refinement is NOT a verified KTP boundary. All evidence is
image-derived from earlier PR6 diagnostics. Nesting is never proof that the
smaller or larger contour is correct. Ground-truth error is evaluation-only.

This module never modifies original candidates, rankings, gates or auto-crop.
"""
from __future__ import annotations

import math


def _section(report, key):
    value = report.get(key)
    return value if isinstance(value, dict) else {}


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _four_sides(candidate, physical, localized):
    """Cross-check previously computed independent evidence, fail-closed."""
    checks = candidate.get("side_checks")
    if not isinstance(checks, dict) or any(
        type(checks.get(key)) is not int or checks[key] != 4
        for key in ("supported_external", "localized_aligned", "measurable_contrast")
    ):
        return False
    if candidate.get("complete_independent_evidence") is not True:
        return False
    if candidate.get("warnings") != []:
        return False
    if physical.get("measurable") is not True or localized.get("measurable") is not True:
        return False
    if localized.get("edge_aligned") is not True:
        return False
    physical_sides = physical.get("side_measurements")
    localized_sides = localized.get("side_localization")
    if not isinstance(physical_sides, list) or not isinstance(localized_sides, list):
        return False
    if len(physical_sides) != 4 or len(localized_sides) != 4:
        return False
    for p, l in zip(physical_sides, localized_sides):
        if not isinstance(p, dict) or not isinstance(l, dict):
            return False
        contrast, p_coverage = _finite(p.get("contrast")), _finite(p.get("coverage"))
        offset, spread = _finite(l.get("median_offset_fraction")), _finite(l.get("offset_spread"))
        l_coverage = _finite(l.get("coverage"))
        if any(v is None for v in (contrast, p_coverage, offset, spread, l_coverage)):
            return False
        if not (contrast >= 5 and .5 <= p_coverage <= 1 and
                .5 <= l_coverage <= 1 and abs(offset) <= .25 and
                0 <= spread <= .25):
            return False
    return True


def validate_physical_refinement_context(report):
    """Conservative context evaluation; never certify a physical KTP contour.

    A single well-supported refinement may be described as corroborated *for
    human review*, never auto-selected. Distinct/nested geometries, uncertain
    offsets, missing evidence or restrictive earlier gates keep manual review.
    """
    if not isinstance(report, dict):
        report = {}
    multi = _section(report, "pr6_multi_refinement_shadow")
    combined = _section(report, "pr6_combined_candidate_shadow")
    spatial = _section(report, "pr6_spatial_relationship_shadow")
    consensus = _section(report, "pr6_evidence_consensus_shadow")
    physical = _section(report, "pr6_physical_boundary_shadow")
    localization = _section(report, "pr6_boundary_localization_shadow")
    decision = _section(report, "pr6_candidate_decision_shadow")
    baseline_gate = _section(report, "v3_review_gate")
    pr6_gate = _section(report, "pr6_review_gate_shadow")

    result = {
        "candidate_assessments": [],
        "geometrically_accepted_count": 0,
        "corroborated_for_review_count": 0,
        "preview_review_eligible_count": 0,
        "manual_review_count": 0,
        "suggested_for_review": None,
        "auto_select": None,
        "auto_crop_allowed": False,
        "requires_explicit_confirmation": True,
        "physical_boundary_verified": False,
        "status": "offline_diagnostic_only",
    }
    required = (multi, combined, spatial, consensus)
    report_valid = (
        all(x.get("status") == "offline_diagnostic_only" for x in required)
        and isinstance(report.get("pr6_physical_boundary_shadow"), dict)
        and isinstance(report.get("pr6_boundary_localization_shadow"), dict)
        and decision.get("mode") == "offline_shadow_review_only"
    )
    candidates_raw = combined.get("candidates")
    refinement_raw = multi.get("candidates")
    groups_raw = consensus.get("consensus_groups")
    relations_raw = spatial.get("relationships")
    if not all(isinstance(x, list) for x in
               (candidates_raw, refinement_raw, groups_raw, relations_raw)):
        result["reason"] = "invalid_or_missing_evidence_sections"
        return result

    sources = {}
    duplicated = set()
    for row in candidates_raw:
        if not isinstance(row, dict):
            continue
        name = row.get("source")
        if not isinstance(name, str) or not name:
            continue
        if name in sources:
            duplicated.add(name)
        sources[name] = row

    membership = {}
    invalid_groups = False
    for group in groups_raw:
        if not isinstance(group, dict) or not isinstance(group.get("members"), list):
            invalid_groups = True
            continue
        members = group["members"]
        if not members or len(members) != len(set(x for x in members if isinstance(x, str))):
            invalid_groups = True
        for name in members:
            if not isinstance(name, str) or name not in sources or name in membership:
                invalid_groups = True
                continue
            membership[name] = group["members"][0]
    # Rebuild explicit group IDs for diagnostic comparison only.
    membership = {
        name: index for index, group in enumerate(groups_raw)
        if isinstance(group, dict) for name in group.get("members") or []
        if isinstance(name, str) and name in sources
    }

    relation_map = {}
    invalid_relations = False
    for relation in relations_raw:
        if not isinstance(relation, dict):
            invalid_relations = True
            continue
        a, b = relation.get("candidate_a"), relation.get("candidate_b")
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            invalid_relations = True
            continue
        key = frozenset((a, b))
        if key in relation_map:
            invalid_relations = True
        relation_map[key] = relation

    group_offset_issue = {}
    for row in decision.get("consensus_assessments") or []:
        if isinstance(row, dict) and type(row.get("group_id")) is int:
            group_offset_issue[row["group_id"]] = row.get("boundary_offset_disagreement") is True

    gate_preview = all(
        gate.get("status") == "preview_review" and not gate.get("reasons")
        for gate in (baseline_gate, pr6_gate, decision)
    )
    accepted_rows = [
        row for row in refinement_raw
        if isinstance(row, dict) and row.get("accepted_for_shadow") is True
    ]
    result["geometrically_accepted_count"] = len(accepted_rows)
    refinement_sources = set()
    for row in refinement_raw:
        if not isinstance(row, dict) or not isinstance(row.get("source"), str):
            continue
        name = row["source"]
        reasons = set()
        if name in refinement_sources:
            reasons.add("duplicate_refinement_source")
        refinement_sources.add(name)
        if not report_valid or invalid_groups or invalid_relations or duplicated:
            reasons.add("inconsistent_or_missing_shadow_context")
        if name not in sources or name not in membership:
            reasons.add("missing_candidate_or_consensus_context")
        if membership.keys() != sources.keys():
            reasons.add("incomplete_consensus_membership")
        geometry_accepted = row.get("accepted_for_shadow") is True
        if not geometry_accepted:
            result["candidate_assessments"].append({
                "source": name, "geometric_refinement_accepted": False,
                "geometric_reason": row.get("reason"),
                "boundary_context_status": "not_evaluated_refinement_declined",
                "eligible_for_preview_review": False,
                "requires_manual_review": True,
                "reasons": ["geometric_refinement_not_accepted"],
                "physical_boundary_verified": False,
            })
            result["manual_review_count"] += 1
            continue
        if row.get("reason") != "geometrically_stable_refinement" or row.get("after_corners") is None:
            reasons.add("inconsistent_geometric_acceptance")
        candidate = sources.get(name) or {}
        physical_row = physical.get(name) if isinstance(physical.get(name), dict) else {}
        localized_row = localization.get(name) if isinstance(localization.get(name), dict) else {}
        supported = _four_sides(candidate, physical_row, localized_row)
        if not supported:
            reasons.add("independent_four_side_evidence_incomplete")
        group_index = membership.get(name)
        if group_index is not None and group_offset_issue.get(group_index) is True:
            reasons.add("near_equivalent_offset_disagreement")

        spatial_roles = []
        for other in sources:
            if other == name:
                continue
            relation = relation_map.get(frozenset((name, other)))
            if relation is None:
                reasons.add("missing_pairwise_spatial_evidence")
                continue
            if group_index == membership.get(other):
                continue
            if relation.get("nested_geometry_risk") is True:
                reasons.add("unresolved_nested_contour_conflict")
                if name == relation.get("smaller_source"):
                    spatial_roles.append("inside_larger_candidate")
                elif name == relation.get("larger_source"):
                    spatial_roles.append("encloses_smaller_candidate")
                else:
                    reasons.add("invalid_nested_spatial_roles")
            else:
                reasons.add("unresolved_distinct_candidate_conflict")

        if len(accepted_rows) > 1:
            reasons.add("multiple_geometric_refinements_accepted")
        if not gate_preview:
            reasons.add("existing_review_gate_requires_manual")
        else:
            gate_sources = set(row.get("selection_group_sources") or []) | {name}
            for gate in (baseline_gate, pr6_gate):
                if gate.get("candidate_source") not in gate_sources:
                    reasons.add("review_gate_candidate_mismatch")
        if decision.get("suggested_for_review") not in (None, name):
            reasons.add("candidate_decision_disagreement")

        if "unresolved_nested_contour_conflict" in reasons:
            state = "unresolved_nested_geometry"
        elif "unresolved_distinct_candidate_conflict" in reasons:
            state = "unresolved_competing_geometry"
        elif not supported:
            state = "insufficient_physical_boundary_evidence"
        elif any(x in reasons for x in
                 ("missing_pairwise_spatial_evidence", "near_equivalent_offset_disagreement",
                  "incomplete_consensus_membership", "inconsistent_or_missing_shadow_context")):
            state = "unresolved_context_integrity"
        else:
            state = "corroborated_boundary_signals_not_verified"
        if state == "corroborated_boundary_signals_not_verified":
            result["corroborated_for_review_count"] += 1
        eligible = not reasons and state == "corroborated_boundary_signals_not_verified"
        if eligible:
            result["preview_review_eligible_count"] += 1
        else:
            result["manual_review_count"] += 1
        result["candidate_assessments"].append({
            "source": name,
            "geometric_refinement_accepted": True,
            "geometric_reason": row.get("reason"),
            "boundary_context_status": state,
            "independent_four_side_evidence": supported,
            "spatial_roles": sorted(set(spatial_roles)),
            "eligible_for_preview_review": eligible,
            "requires_manual_review": not eligible,
            "reasons": sorted(reasons),
            "physical_boundary_verified": False,
        })
    return result
