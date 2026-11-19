"""Bounded, read-only PR6 candidate diagnostics for a local visual overlay.

Contains only original-image quad coordinates and image-derived numerical
evidence. No image bytes, personal text, filenames, annotations, paths or
ground-truth errors. It is deliberately NOT a crop, ranking, or acceptance API.
"""
from __future__ import annotations

import math

import numpy as np

from autodocscanner.core.boundary_localization_pr6 import localize_candidate_boundary
from autodocscanner.core.physical_boundary_pr6 import verify_physical_boundary


MAX_DISPLAY_CANDIDATES = 40


def _balanced_sources(sources, limit):
    """Keep each family visible; earlier contour lists must not hide regions.

    A family may have many geometrically valid candidates. Display slots are
    shared round-robin across methods, preserving source order within each.
    This is *presentation only*, not a candidate ranking or selection.
    """
    families, order = {}, []
    for item in sources:
        family = item[1]
        if family not in families:
            families[family] = []
            order.append(family)
        families[family].append(item)
    selected = []
    while len(selected) < limit and any(families.values()):
        for family in order:
            if families[family] and len(selected) < limit:
                selected.append(families[family].pop(0))
    return selected


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _number(value):
    try:
        if isinstance(value, bool) or value is None:
            return None
        v = float(value)
        return round(v, 4) if math.isfinite(v) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _points(value, image_shape):
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if arr.shape != (4, 2) or not np.isfinite(arr).all():
        return None
    h, w = image_shape
    if ((arr[:, 0] < -.04 * w).any() or (arr[:, 0] > 1.04 * w).any()
            or (arr[:, 1] < -.04 * h).any() or
            (arr[:, 1] > 1.04 * h).any()):
        return None
    return [[round(float(x), 3), round(float(y), 3)] for x, y in arr]


def _codes(value):
    return [x for x in _list(value) if isinstance(x, str) and len(x) <= 120][:24]


def build_pr6_visual_diagnostics(report, image_shape, *, image=None):
    """Make a small immutable-by-convention display snapshot from PR6 report.

    Supports stable/adaptive and all post-validation V2 candidate indices,
    including provenance of optional four-independent-line and chromatic-edge
    proposals. Every source can be selected regardless of the final route.

    If source pixels are provided, independently remeasure ALL displayed
    candidates for this *visualization only*. These numbers never enter
    selection, ranking, or any review gate, and are not cached in the report.
    """
    report = _dict(report)
    try:
        h, w = [int(x) for x in image_shape[:2]]
    except (TypeError, ValueError, IndexError, OverflowError):
        h = w = 0
    if not 16 <= h <= 100000 or not 16 <= w <= 100000:
        return {
            "status": "invalid_image", "image_shape": None,
            "candidates": [], "auto_select": None, "auto_crop_allowed": False,
        }

    selection = _dict(report.get("pr6_selection_shadow"))
    combined = _dict(report.get("pr6_combined_candidate_shadow"))
    localization = _dict(report.get("pr6_boundary_localization_shadow"))
    physical = _dict(report.get("pr6_physical_boundary_shadow"))
    decision = _dict(report.get("pr6_candidate_decision_shadow"))
    gate = _dict(report.get("pr6_review_consolidation_shadow"))
    ranked = _dict(report.get("v3_ranking"))
    v2 = _list(report.get("v2"))
    origins = _list(report.get("pr6_v2_proposal_origins"))
    outer = _dict(report.get("pr6_outer_region_recovery_shadow"))
    side_recovery = _dict(report.get("pr6_side_recovery_shadow"))
    side_recovery_rows = _list(side_recovery.get("candidates"))
    paired_alignment = _dict(report.get("pr6_paired_alignment_shadow"))
    paired_rows = _list(paired_alignment.get("candidates"))
    independent = _dict(report.get("pr6_independent_sides_shadow"))
    independent_rows = _list(independent.get("candidates"))
    outer_checks = {
        row["source"]: row
        for raw in _list(outer.get("candidate_checks"))
        if (row := _dict(raw)) and isinstance(row.get("source"), str)
    }

    groups = {}
    for group in _list(selection.get("groups")):
        group = _dict(group)
        rep = group.get("representative")
        if not isinstance(rep, str):
            continue
        for name in _list(group.get("sources")):
            if isinstance(name, str):
                groups[name] = rep

    combined_by_source = {
        row["source"]: row
        for raw in _list(combined.get("candidates"))
        if (row := _dict(raw)) and isinstance(row.get("source"), str)
    }
    assessed = {
        row["source"]: row
        for raw in _list(decision.get("candidate_assessments"))
        if (row := _dict(raw)) and isinstance(row.get("source"), str)
    }
    rank_by_source = {
        row["source"]: row
        for raw in _list(ranked.get("ranking"))
        if (row := _dict(raw)) and isinstance(row.get("source"), str)
    }

    sources = [
        ("stable", "Stable", _dict(report.get("stable"))),
        ("adaptive", "Adaptive", _dict(report.get("adaptive"))),
    ]
    for index, item in enumerate(v2):
        origin = origins[index] if index < len(origins) else "contour"
        family = (
            "Four-Line" if origin == "four_line" else
            "Chroma" if origin == "chroma" else
            "Outer Region" if origin == "outer_region" else "Kontur V2"
        )
        sources.append((f"v2_{index}", family, _dict(item)))
    for index, item in enumerate(side_recovery_rows[:12]):
        item = _dict(item)
        # Read-only side-recovered hypotheses are NOT registered in the V2
        # candidate list and never influence ranking or approval gates.
        sources.append((
            f"side_recovery_{index}", "Side Recovery",
            {"corners": item.get("corners"), "geometry_valid": True},
        ))
    for index, raw in enumerate(paired_rows[:12]):
        paired = _dict(raw)
        # Post-gate previews only, never injected into PR6 candidate ranking.
        sources.append((
            f"paired_alignment_{index}", "Paired Alignment",
            {"corners": paired.get("corners"), "geometry_valid": True},
        ))
    for index, raw in enumerate(independent_rows[:12]):
        proposal = _dict(raw)
        sources.append((
            f"independent_sides_{index}", "Independent Sides",
            {"corners": proposal.get("corners"), "geometry_valid": True},
        ))

    display_sources = _balanced_sources(sources, MAX_DISPLAY_CANDIDATES)
    rows = []
    for name, family, source in display_sources:
        points = _points(source.get("corners"), (h, w))
        if points is None:
            continue

        representative = groups.get(name)
        combined_info = _dict(combined_by_source.get(representative))
        visualization_remeasured = (
            isinstance(image, np.ndarray)
            and image.shape[:2] == (h, w)
            and image.dtype == np.uint8
        )
        if visualization_remeasured:
            # Selection-group representative metrics are NOT attributable to
            # every member's geometry. Measure this exact four-corner quad.
            try:
                localization_info = localize_candidate_boundary(image, points)
                physical_info = verify_physical_boundary(image, points)
            except (ValueError, TypeError, OverflowError):
                localization_info, physical_info = {}, {}
        else:
            localization_info = (
                _dict(localization.get(name)) if representative == name else {}
            )
            physical_info = (
                _dict(physical.get(name)) if representative == name else {}
            )
        outer_sides = _dict(source.get("boundary_evidence")).get("side_support")
        local_sides = _list(localization_info.get("side_localization"))
        physical_sides = _list(physical_info.get("side_measurements"))
        sides = []
        for index in range(4):
            loc = _dict(local_sides[index]) if index < len(local_sides) else {}
            phys = _dict(physical_sides[index]) if index < len(physical_sides) else {}
            sides.append({
                "external_support": _number(
                    outer_sides[index]
                    if isinstance(outer_sides, list) and index < len(outer_sides)
                    else None
                ),
                "local_coverage": _number(loc.get("coverage")),
                "median_offset_fraction": _number(loc.get("median_offset_fraction")),
                "offset_spread": _number(loc.get("offset_spread")),
                "inner_outer_contrast": _number(phys.get("contrast")),
                "contrast_coverage": _number(phys.get("coverage")),
            })
        ranking = _dict(rank_by_source.get(name))
        assessment = _dict(assessed.get(representative))
        paired_item = (
            _dict(paired_rows[int(name.split("_")[-1])])
            if name.startswith("paired_alignment_") else None
        )
        independent_item = (
            _dict(independent_rows[int(name.split("_")[-1])])
            if name.startswith("independent_sides_") else None
        )
        rows.append({
            "source": name,
            "family": family,
            "corners": points,
            "selection_representative": representative,
            "ranking_score": _number(ranking.get("score")),
            "geometry_valid": source.get("geometry_valid") is True,
            "area_ratio": _number(source.get("area_ratio")),
            "side_evidence": sides,
            "side_checks": {
                key: combined_info.get("side_checks", {}).get(key)
                for key in ("supported_external", "localized_aligned",
                            "measurable_contrast")
            } if isinstance(combined_info.get("side_checks"), dict) else {},
            "candidate_warnings": _codes(combined_info.get("warnings")),
            "ranking_warnings": _codes(ranking.get("review_reasons")),
            "evidence_state": (
                assessment.get("evidence_state")
                if isinstance(assessment.get("evidence_state"), str) else None
            ),
            "assessment_reasons": _codes(assessment.get("reasons")),
            "physical_measurements_for_this_quad": (
                visualization_remeasured or representative == name
            ),
            "measurements_visual_only": bool(visualization_remeasured),
            "independent_sides_details": {
                "parent_source": independent_item.get("parent_source"),
                "outer_region_source": independent_item.get("outer_region_source"),
                "paired_side_indices": independent_item.get("paired_side_indices"),
                "independent_offset_fractions": independent_item.get(
                    "independent_offset_fractions"
                ),
                "anchored_side_indices": independent_item.get(
                    "anchored_side_indices", []
                ),
                "refinement_kind": independent_item.get(
                    "refinement_kind"
                ),
                "side_scores_before": independent_item.get("side_scores_before"),
                "side_scores_after": independent_item.get("side_scores_after"),
            } if independent_item is not None else None,
            "paired_alignment_details": {
                "parent_source": paired_item.get("parent_source"),
                "outer_region_source": paired_item.get("outer_region_source"),
                "paired_side_indices": paired_item.get("paired_side_indices"),
                "shift_fraction_of_pair_gap": _number(
                    paired_item.get("shift_fraction_of_pair_gap")
                ),
                "translation_original_xy": paired_item.get(
                    "translation_original_xy"
                ),
                "side_scores_before": paired_item.get("side_scores_before"),
                "side_scores_after": paired_item.get("side_scores_after"),
            } if paired_item is not None else None,
            "side_recovery_details": {
                "parent_source": (
                    _dict(side_recovery_rows[int(name.split("_")[-1])])
                    .get("parent_source")
                ),
                "changed_side_index": (
                    _dict(side_recovery_rows[int(name.split("_")[-1])])
                    .get("changed_side_index")
                ),
                "shift_fraction_of_parent_gap": _number(
                    _dict(side_recovery_rows[int(name.split("_")[-1])])
                    .get("shift_fraction_of_parent_gap")
                ),
                "side_tilt_fraction_of_parent_gap": _number(
                    _dict(side_recovery_rows[int(name.split("_")[-1])])
                    .get("side_tilt_fraction_of_parent_gap")
                ),
                "side_score_before": _number(
                    _dict(side_recovery_rows[int(name.split("_")[-1])])
                    .get("side_score_before")
                ),
                "side_score_after": _number(
                    _dict(side_recovery_rows[int(name.split("_")[-1])])
                    .get("side_score_after")
                ),
            } if name.startswith("side_recovery_") else None,
            "outer_recovery_check": {
                "perspective_feasible": (
                    _dict(outer_checks.get(name)).get("perspective_feasible")
                ),
                "all_sides_measurable": (
                    _dict(outer_checks.get(name)).get("all_sides_measurable")
                ),
                "nested_in_larger_quad": (
                    _dict(outer_checks.get(name)).get("nested_in_larger_quad")
                ),
                "chromatic_step_by_side": _list(
                    _dict(outer_checks.get(name)).get("chromatic_step_by_side")
                )[:4],
                "reason_codes": _codes(
                    _dict(outer_checks.get(name)).get("reason_codes")
                ),
                "region_context": {
                    "interior_edge_density": _number(_dict(
                        _dict(outer_checks.get(name)).get("region_context")
                    ).get("interior_edge_density")),
                    "interior_spatial_coverage": _number(_dict(
                        _dict(outer_checks.get(name)).get("region_context")
                    ).get("interior_spatial_coverage")),
                    "interior_axis_line_bands": _dict(
                        _dict(outer_checks.get(name)).get("region_context")
                    ).get("interior_axis_line_bands"),
                    "corroborating_families": _codes(_dict(
                        _dict(outer_checks.get(name)).get("region_context")
                    ).get("corroborating_families")),
                    "nested_inner_candidates": _dict(
                        _dict(outer_checks.get(name)).get("region_context")
                    ).get("nested_inner_candidates"),
                    "reason_codes": _codes(_dict(
                        _dict(outer_checks.get(name)).get("region_context")
                    ).get("reason_codes")),
                },
            } if name in outer_checks else None,
        })

    family_counts = {
        "Stable": sum(1 for row in rows if row["family"] == "Stable"),
        "Adaptive": sum(1 for row in rows if row["family"] == "Adaptive"),
        "Kontur V2": sum(1 for row in rows if row["family"] == "Kontur V2"),
        "Four-Line": sum(1 for row in rows if row["family"] == "Four-Line"),
        "Chroma": sum(1 for row in rows if row["family"] == "Chroma"),
        "Outer Region": sum(1 for row in rows if row["family"] == "Outer Region"),
        "Side Recovery": sum(1 for row in rows if row["family"] == "Side Recovery"),
        "Paired Alignment": sum(1 for row in rows if row["family"] == "Paired Alignment"),
        "Independent Sides": sum(1 for row in rows if row["family"] == "Independent Sides"),
    }
    return {
        "status": "offline_visual_diagnostic_only",
        "image_shape": [h, w],
        "candidate_count": len(rows),
        "candidate_slots": MAX_DISPLAY_CANDIDATES,
        "source_hypotheses_total": len(sources),
        "source_hypotheses_hidden": max(
            0, len(sources) - len(display_sources)
        ),
        "family_counts": family_counts,
        "generated_four_line_count": int(
            report.get("pr6_line_proposal_count") or 0
        ),
        "generated_chroma_count": int(
            report.get("pr6_chroma_proposal_count") or 0
        ),
        "generated_outer_region_count": int(
            report.get("pr6_region_proposal_count") or 0
        ),
        "generated_side_recovery_count": min(
            len(side_recovery_rows), 12
        ),
        "side_recovery_parents_examined": int(
            side_recovery.get("parents_examined") or 0
        ),
        "side_recovery_hypotheses_before_dedup": int(
            side_recovery.get("hypotheses_before_dedup") or 0
        ),
        "side_recovery_reason_codes": _codes(
            side_recovery.get("reason_codes")
        ),
        "generated_paired_alignment_count": min(len(paired_rows), 12),
        "paired_alignment_parents_examined": int(
            paired_alignment.get("parents_examined") or 0
        ),
        "paired_alignment_reason_codes": _codes(
            paired_alignment.get("reason_codes")
        ),
        "independent_sides_count": min(len(independent_rows), 12),
        "independent_sides_parents_examined": int(
            independent.get("parents_examined") or 0
        ),
        "independent_direct_region_parents_examined": int(
            independent.get("direct_outer_region_parents_examined") or 0
        ),
        "independent_direct_region_pair_checks": int(
            independent.get("direct_outer_region_pair_checks") or 0
        ),
        "independent_sides_reason_codes": _codes(
            independent.get("reason_codes")
        ),
        "independent_sides_parent_checks": [
            {
                "source": item.get("source"),
                "outer_region_source": item.get("outer_region_source"),
                "parent_family": item.get("parent_family"),
                "paired_side_indices": [
                    index for index in _list(item.get("paired_side_indices"))
                    if type(index) is int and 0 <= index < 4
                ][:2],
                "status": item.get("status"),
                "failure_reason": item.get("failure_reason"),
                "side_checks": [
                    {
                        "side_index": check.get("side_index"),
                        "score_before": _number(check.get("score_before")),
                        "required_score": _number(check.get("required_score")),
                        "strongest_score": _number(check.get("strongest_score")),
                        "strongest_offset_fraction": _number(
                            check.get("strongest_offset_fraction")
                        ),
                        "has_supported_alternative": (
                            check.get("has_supported_alternative") is True
                        ),
                        "original_side_anchor_supported": (
                            check.get("original_side_anchor_supported") is True
                        ),
                        "supported_shift_count": (
                            int(check["supported_shift_count"])
                            if type(check.get("supported_shift_count")) is int
                            and 0 <= check["supported_shift_count"] <= 5
                            else 0
                        ),
                    }
                    for raw_check in _list(item.get("side_checks"))[:2]
                    if (check := _dict(raw_check))
                ],
                "pairs_evaluated": item.get("pairs_evaluated", 0),
            }
            for raw in _list(independent.get("parent_checks"))[:40]
            if (item := _dict(raw))
        ],
        "paired_alignment_parent_checks": [
            {
                "source": item.get("source"),
                "outer_region_source": item.get("outer_region_source"),
                "status": item.get("status"),
                "failure_reason": item.get("failure_reason"),
                "paired_side_indices": [
                    index for index in _list(item.get("paired_side_indices"))
                    if isinstance(index, int) and not isinstance(index, bool)
                    and 0 <= index < 4
                ][:2],
                "attempts_evaluated": item.get("attempts_evaluated", 0),
                "best_attempt": (
                    {
                        "shift_fraction_of_pair_gap": _number(
                            _dict(item.get("best_attempt")).get(
                                "shift_fraction_of_pair_gap"
                            )
                        ),
                        "side_scores_before": [
                            _number(x) for x in _list(
                                _dict(item.get("best_attempt")).get(
                                    "side_scores_before"
                                )
                            )[:2]
                        ],
                        "side_scores_after": [
                            _number(x) for x in _list(
                                _dict(item.get("best_attempt")).get(
                                    "side_scores_after"
                                )
                            )[:2]
                        ],
                        "required_side_scores": [
                            _number(x) for x in _list(
                                _dict(item.get("best_attempt")).get(
                                    "required_side_scores"
                                )
                            )[:2]
                        ],
                        "blocking_side_index": _dict(
                            item.get("best_attempt")
                        ).get("blocking_side_index"),
                        "both_pass": _dict(item.get("best_attempt")).get(
                            "both_pass"
                        ) is True,
                    } if isinstance(item.get("best_attempt"), dict) else None
                ),
            }
            for raw in _list(paired_alignment.get("parent_checks"))[:12]
            if (item := _dict(raw))
        ],
        "rejected_after_geometry": int(
            report.get("v2_rejected_post_order") or 0
        ),
        "candidates": rows,
        "review_decision": (
            gate.get("status") if gate.get("status") in (
                "manual_review", "preview_review") else "manual_review"
        ),
        "decision_reason_codes": _codes(gate.get("reason_codes")),
        "selection_warnings": _codes(selection.get("warnings")),
        "auto_select": None,
        "auto_crop_allowed": False,
        "physical_boundary_verified": False,
        "requires_explicit_confirmation": True,
    }
