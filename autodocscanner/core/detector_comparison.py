"""Offline comparison of KTP detectors; NEVER changes production selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from autodocscanner.core.experimental_candidates import contour_candidates
from autodocscanner.core.line_quad_proposals_pr6 import line_quad_candidates
from autodocscanner.core.chroma_quad_proposals_pr6 import chroma_quad_candidates
from autodocscanner.core.outer_region_recovery_pr6 import (
    region_outer_quad_candidates, inspect_outer_perspective_hypothesis,
)
from autodocscanner.core.region_context_pr6 import inspect_region_context
from autodocscanner.core.side_boundary_recovery_pr6 import (
    recover_one_side_hypotheses, prepare_side_recovery_lab,
)
from autodocscanner.core.paired_boundary_alignment_pr6 import (
    propose_paired_alignment,
)
from autodocscanner.core.quad_validation import validate_quad
from autodocscanner.core.boundary_evidence import boundary_evidence
from autodocscanner.core.candidate_ranking_v3 import rank_candidates
from autodocscanner.core.review_gate_v3 import assess_review_gate
from autodocscanner.core.manual_handoff_v3 import prepare_manual_handoff
from autodocscanner.core.corner_refinement_v3 import refine_corners
from autodocscanner.core.candidate_selection_pr6 import compare_selection
from autodocscanner.core.pr6_review_gate import assess_pr6_review_gate
from autodocscanner.core.physical_boundary_pr6 import verify_physical_boundary
from autodocscanner.core.boundary_localization_pr6 import localize_candidate_boundary
from autodocscanner.core.combined_candidate_pr6 import compare_combined_evidence
from autodocscanner.core.spatial_relationship_pr6 import compare_spatial_relationships
from autodocscanner.core.spatial_risk_pr6 import assess_spatial_risk
from autodocscanner.core.comparative_evidence_pr6 import compare_candidate_evidence
from autodocscanner.core.evidence_consensus_pr6 import interpret_candidate_consensus
from autodocscanner.core.candidate_decision_pr6 import decide_candidate_review
from autodocscanner.core.multi_refinement_pr6 import evaluate_multi_candidate_refinement
from autodocscanner.core.physical_refinement_validation_pr6 import validate_physical_refinement_context
from autodocscanner.core.review_gate_consolidation_pr6 import consolidate_review_gate
from autodocscanner.core.manual_fallback_validation_pr6 import prepare_manual_fallback_pr6
from autodocscanner.core.perspective_engine import AutoPerspectiveEngine
from autodocscanner.core.perspective_engine_baseline import (
    AutoPerspectiveEngine as StableBaselinePerspectiveEngine,
)


def _metric(engine, image, corners, expected=None):
    if corners is None:
        return {"found": False}
    # Validate the exact corners that will be warped and reported.
    points = engine.order_points(np.asarray(corners, dtype=np.float32))
    area = abs(cv2.contourArea(points)) / float(max(image.shape[0]*image.shape[1], 1))
    warped = engine.warp(image, points)
    quality = float(engine._warp_quality(warped))
    valid, reasons, geometry = validate_quad(points, image.shape)
    data = {
        "geometry_valid": valid,
        "geometry_reasons": reasons,
        "geometry": geometry,
        "boundary_evidence": boundary_evidence(image, points),
        "found": True,
        "corners": np.round(points, 2).tolist(),
        "area_ratio": round(float(area), 4),
        "warp_quality": round(quality, 4),
    }
    if expected is not None:
        truth = engine.order_points(np.asarray(expected, dtype=np.float32))
        diagonal = max(float(np.hypot(image.shape[1], image.shape[0])), 1.)
        error = np.linalg.norm(points - truth, axis=1)
        data["mean_corner_error_px"] = round(float(np.mean(error)), 3)
        data["normalized_corner_error"] = round(float(np.mean(error)/diagonal), 5)
    return data


def _filter_valid_candidates(evaluated):
    """Filter post-order metrics, preserving only usable original-image corners."""
    valid = [item for item in evaluated if item.get("geometry_valid") is True]
    return valid, len(evaluated) - len(valid)


def compare_detectors(image, *, expected=None, max_proposals=24,
                      include_line_candidates=False,
                      include_chroma_candidates=False,
                      include_region_candidates=False):
    """Side-by-side diagnostics. Scores are NOT proof of correct KTP bounds."""
    stable = StableBaselinePerspectiveEngine()
    adaptive = AutoPerspectiveEngine()
    report = {"stable": {"found": False}, "adaptive": {"found": False}, "v2": []}
    for name, engine in (("stable", stable), ("adaptive", adaptive)):
        corners, meta = engine.detect(image)
        report[name] = _metric(engine, image, corners, expected)
        report[name]["detector_score"] = round(float((meta or {}).get("score", 0) or 0), 4)
        report[name]["source"] = (meta or {}).get("selected_source")

    proposals = contour_candidates(image, max_candidates=max_proposals)
    proposal_origins = ["contour"] * len(proposals)
    # Optional additional *physical-boundary hypotheses* from independently
    # observed Hough line segments. Default offline reports and the released
    # automatic detector remain unchanged. These are never auto-accepted.
    line_added = 0
    if include_line_candidates:
        for proposed in line_quad_candidates(image, max_candidates=6):
            if len(proposed) != 4:
                continue
            points = np.asarray(proposed, dtype=np.float32)
            if not np.isfinite(points).all():
                continue
            if any(
                float(np.linalg.norm(
                    points - np.asarray(old, dtype=np.float32), axis=1,
                ).mean()) / max(float(np.hypot(*image.shape[:2])), 1.) < .018
                for old in proposals
            ):
                continue
            proposals.append(points)
            proposal_origins.append("four_line")
            line_added += 1
    chroma_added = 0
    if include_chroma_candidates:
        # Lab a/b channels can reveal card/background discontinuity when
        # grayscale contrast is weak. These hypotheses do NOT outrank or
        # bypass the existing physical-boundary and conflict review gates.
        for proposed in chroma_quad_candidates(image, max_candidates=8):
            points = np.asarray(proposed, dtype=np.float32)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                continue
            if any(
                float(np.linalg.norm(
                    points - np.asarray(old, dtype=np.float32), axis=1,
                ).mean()) / max(float(np.hypot(*image.shape[:2])), 1.) < .018
                for old in proposals
            ):
                continue
            proposals.append(points)
            proposal_origins.append("chroma")
            chroma_added += 1
    region_added = 0
    if include_region_candidates:
        # Region-first geometry is another *hypothesis* source, never an
        # approved document crop. Preserve provenance through post-validation.
        context_seeds = [
            {"family": name, "corners": report[name]["corners"]}
            for name in ("stable", "adaptive")
            if report[name].get("geometry_valid")
        ]
        context_seeds.extend({
            "family": origin, "corners": np.asarray(candidate, dtype=np.float32)
        } for origin, candidate in zip(proposal_origins, proposals))
        for candidate in region_outer_quad_candidates(
            image, max_candidates=12,
            context_candidates=context_seeds,
            contextual_ranking=True,
        ):
            points = np.asarray(candidate, dtype=np.float32)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                continue
            if any(
                float(np.linalg.norm(
                    points - np.asarray(old, dtype=np.float32), axis=1,
                ).mean()) / max(float(np.hypot(*image.shape[:2])), 1.) < .018
                for old in proposals
            ):
                continue
            proposals.append(points)
            proposal_origins.append("outer_region")
            region_added += 1
    evaluated = [_metric(adaptive, image, quad, expected) for quad in proposals]
    # Match origin labels to post-validation v2 indices; rejected candidates
    # must not shift the display labels. These labels carry no authority.
    report["pr6_v2_proposal_origins"] = [
        label for label, item in zip(proposal_origins, evaluated)
        if item.get("geometry_valid") is True
    ]
    report["pr6_line_proposal_count"] = line_added
    report["pr6_chroma_proposal_count"] = chroma_added
    report["pr6_region_proposal_count"] = region_added
    # Filter using the actual ordered corners that are reported and warped.
    report["v2"], report["v2_rejected_post_order"] = _filter_valid_candidates(evaluated)
    report["v2_proposal_count"] = len(report["v2"])
    # Stage 1 region identification -> Stage 2 recovered outer corners ->
    # Stage 3 geometric/color-border plausibility, *read-only*. Not inserted
    # into any existing PR6 approval logic; nested rectangles remain suspect.
    if include_region_candidates:
        contextual_candidates = [
            {"family": name, "corners": report[name]["corners"]}
            for name in ("stable", "adaptive")
            if report[name].get("geometry_valid")
        ]
        contextual_candidates.extend({
            "family": origin, "corners": item["corners"]
        } for item, origin in zip(
            report["v2"], report["pr6_v2_proposal_origins"]
        ) if item.get("geometry_valid"))
        other_quads = [
            row["corners"] for row in contextual_candidates
        ]
        report["pr6_outer_region_recovery_shadow"] = {
            "status": "offline_diagnostic_only",
            "auto_select": None,
            "auto_crop_allowed": False,
            "candidate_checks": [
                {
                    "source": f"v2_{index}",
                    **inspect_outer_perspective_hypothesis(
                        image, item["corners"],
                        other_quads=[
                            other for other in other_quads
                            if not np.allclose(
                                item["corners"], other,
                                atol=1e-2, rtol=0,
                            )
                        ],
                    ),
                    "region_context": inspect_region_context(
                        image, item["corners"],
                        other_candidates=[
                            other for other in contextual_candidates
                            if not np.allclose(
                                item["corners"], other["corners"],
                                atol=1e-2, rtol=0,
                            )
                            and other["family"] != "outer_region"
                        ],
                    ),
                }
                for index, (item, origin) in enumerate(
                    zip(report["v2"], report["pr6_v2_proposal_origins"])
                ) if origin == "outer_region"
            ],
        }
    candidates = [(name, report[name]) for name in ("stable", "adaptive")]
    candidates.extend((f"v2_{index}", item) for index, item in enumerate(report["v2"]))
    report["v3_ranking"] = rank_candidates(candidates)
    report["pr6_selection_shadow"] = compare_selection(report, image.shape)
    report["v3_review_gate"] = assess_review_gate(report, image.shape)
    report["pr6_review_gate_shadow"] = assess_pr6_review_gate(report)
    report["pr6_physical_boundary_shadow"] = {
        group["representative"]: verify_physical_boundary(
            image, (report[group["representative"]] if group["representative"] in ("stable", "adaptive")
                    else report["v2"][int(group["representative"][3:])])["corners"])
        for group in report["pr6_selection_shadow"]["groups"]
    }
    report["pr6_boundary_localization_shadow"] = {
        group["representative"]: localize_candidate_boundary(
            image, (report[group["representative"]] if group["representative"] in ("stable", "adaptive")
                    else report["v2"][int(group["representative"][3:])])["corners"])
        for group in report["pr6_selection_shadow"]["groups"]
    }
    report["pr6_combined_candidate_shadow"] = compare_combined_evidence(report)
    report["pr6_spatial_relationship_shadow"] = compare_spatial_relationships(report)
    report["pr6_spatial_risk_shadow"] = assess_spatial_risk(report)
    report["pr6_comparative_evidence_shadow"] = compare_candidate_evidence(report)
    report["pr6_evidence_consensus_shadow"] = interpret_candidate_consensus(report)
    report["pr6_candidate_decision_shadow"] = decide_candidate_review(report)
    report["v3_manual_handoff"] = prepare_manual_handoff(report, image.shape)
    # PR6 shadow evaluation only: do NOT alter ranking or manual handoff.
    leader = report["v3_ranking"].get("suggested_for_review")
    selected = report.get(leader) if leader in ("stable", "adaptive") else None
    if isinstance(leader, str) and leader.startswith("v2_") and leader[3:].isdigit():
        index = int(leader[3:])
        selected = report["v2"][index] if index < len(report["v2"]) else None
    if selected and selected.get("geometry_valid"):
        before = _metric(adaptive, image, selected["corners"], expected)
        if (selected.get("geometry") or {}).get("frame_suspect"):
            proposed = np.asarray(selected["corners"], dtype=np.float32)
            diagnostic = {"accepted": False, "reason": "frame_suspect",
                          "mean_shift_px": 0.0}
        else:
            proposed, diagnostic = refine_corners(image, selected["corners"])
        attempted = _metric(adaptive, image, proposed, expected)
        # Never use ground-truth errors to decide acceptance: this gate
        # is based only on the image and detector diagnostics.
        if diagnostic["accepted"]:
            old_quality, new_quality = before["warp_quality"], attempted["warp_quality"]
            if new_quality + 0.03 < old_quality:
                diagnostic = {**diagnostic, "accepted": False,
                              "reason": "warp_quality_regression"}
        after = attempted if diagnostic["accepted"] else before
        report["v3_refinement_shadow"] = {
            "source": leader, **diagnostic,
            "before": before,
            "attempted": attempted,
            "after": after,
        }
    else:
        report["v3_refinement_shadow"] = {"accepted": False, "reason": "no_candidate"}
    # Independent PR6 refinement experiment: measure each distinct proposal,
    # not only the top-ranked candidate. Does not update gates, ranking or warp.
    report["pr6_multi_refinement_shadow"] = evaluate_multi_candidate_refinement(
        image, report, adaptive
    )
    # Read-only context interpretation. Never substitutes for review gates;
    # ground-truth annotation remains evaluation-only and is not consulted.
    report["pr6_physical_refinement_validation_shadow"] = (
        validate_physical_refinement_context(report, image)
    )
    # Final read-only PR6 review advice; pre-existing gates and handoff remain
    # unchanged, and annotations never participate in the routing decision.
    report["pr6_review_consolidation_shadow"] = consolidate_review_gate(report)
    # PR6 manual-fallback contract only: no UI, crop, save or original-image
    # mutation. The legacy V3 handoff remains unchanged for production.
    report["pr6_manual_fallback_shadow"] = prepare_manual_fallback_pr6(
        report, image.shape
    )
    # New read-only hypotheses are deliberately computed AFTER every PR6
    # decision and manual-fallback contract. A candidate with one relocated
    # side can be inspected but never participates in selection or export.
    if include_region_candidates:
        recovered = []
        parents_examined = 0
        hypotheses_before_dedup = 0
        seen_corners = [
            np.asarray(item["corners"], dtype=np.float32)
            for item in [report["stable"], report["adaptive"]] + report["v2"]
            if item.get("geometry_valid") and item.get("corners") is not None
        ]
        diagonal = max(float(np.hypot(*image.shape[:2])), 1.)
        # One local, downscaled Lab image for all candidate hypotheses.
        # Do not serialize or persist any pixels into the report.
        side_recovery_prepared = prepare_side_recovery_lab(image)
        for index, (item, origin) in enumerate(zip(
            report["v2"], report["pr6_v2_proposal_origins"]
        )):
            if origin != "outer_region" or len(recovered) >= 12:
                continue
            parents_examined += 1
            for variant in recover_one_side_hypotheses(
                image, item["corners"], max_results=1,
                prepared=side_recovery_prepared,
            ):
                hypotheses_before_dedup += 1
                quad = np.asarray(variant["corners"], dtype=np.float32)
                if any(
                    float(np.linalg.norm(quad - prior, axis=1).mean())
                    / diagonal < .018
                    for prior in seen_corners
                ):
                    continue
                seen_corners.append(quad)
                recovered.append({
                    **variant, "parent_source": f"v2_{index}",
                })
        report["pr6_side_recovery_shadow"] = {
            "status": "offline_visual_hypotheses_only",
            "auto_select": None,
            "auto_crop_allowed": False,
            "candidates": recovered,
            "candidate_count": len(recovered),
            "parents_examined": parents_examined,
            "hypotheses_before_dedup": hypotheses_before_dedup,
            "reason_codes": (
                [] if recovered else
                ["no_outer_region_parent"] if not parents_examined else
                ["all_recovered_hypotheses_overlap_previous_candidates"]
                if hypotheses_before_dedup else
                ["no_side_shift_passed_visual_evidence_and_geometry"]
            ),
        }

        # A parent may have both opposite edges translated in the SAME
        # direction (rather than just one misplaced side). Require evidence
        # gain on both, and evaluate after all PR6 gates like one-side review.
        paired = []
        paired_checks = []
        paired_hypotheses_before_dedup = 0
        for recovered_index, parent in enumerate(recovered[:12]):
            details = {}
            variants = propose_paired_alignment(
                image, parent["corners"],
                pair_side_index=parent["changed_side_index"] % 2,
                prepared=side_recovery_prepared, max_results=1,
                diagnostics=details,
            )
            paired_checks.append({
                "source": f"side_recovery_{recovered_index}",
                "outer_region_source": parent["parent_source"],
                **details,
            })
            for variant in variants:
                paired_hypotheses_before_dedup += 1
                quad = np.asarray(variant["corners"], dtype=np.float32)
                if any(
                    float(np.linalg.norm(quad - prior, axis=1).mean())
                    / diagonal < .018
                    for prior in seen_corners
                ):
                    continue
                seen_corners.append(quad)
                paired.append({
                    **variant,
                    "parent_source": f"side_recovery_{recovered_index}",
                    "outer_region_source": parent["parent_source"],
                })
                if len(paired) >= 12:
                    break
            if len(paired) >= 12:
                break
        report["pr6_paired_alignment_shadow"] = {
            "status": "offline_visual_hypotheses_only",
            "auto_select": None,
            "auto_crop_allowed": False,
            "parents_examined": len(recovered[:12]),
            "hypotheses_before_dedup": paired_hypotheses_before_dedup,
            "candidates": paired,
            "candidate_count": len(paired),
            "parent_checks": paired_checks,
            "reason_codes": (
                [] if paired else
                ["no_single_side_parent"] if not recovered else
                ["all_paired_hypotheses_overlap_previous_candidates"]
                if paired_hypotheses_before_dedup else
                ["no_opposite_sides_both_improved"]
            ),
        }
    if expected is not None:
        # Annotation may be used to evaluate accuracy only AFTER all shadow
        # decisions have been made. It never enters any acceptance condition.
        truth = adaptive.order_points(np.asarray(expected, dtype=np.float32))
        diagonal = max(float(np.hypot(image.shape[1], image.shape[0])), 1.)

        def _evaluation_error(corners):
            if corners is None:
                return None
            try:
                points = np.asarray(corners, dtype=np.float32)
                if points.shape != (4, 2) or not np.isfinite(points).all():
                    return None
                ordered = adaptive.order_points(points)
                return round(float(np.linalg.norm(ordered - truth, axis=1).mean()) / diagonal, 5)
            except (TypeError, ValueError):
                return None

        report["pr6_multi_refinement_ground_truth_evaluation"] = {
            "candidates": [
                {
                    "source": row["source"],
                    "before_normalized_corner_error": _evaluation_error(row["before_corners"]),
                    "attempted_normalized_corner_error": _evaluation_error(row["attempted_corners"]),
                    "after_normalized_corner_error": _evaluation_error(row["after_corners"]),
                    "accepted_for_shadow": row["accepted_for_shadow"],
                }
                for row in report["pr6_multi_refinement_shadow"]["candidates"]
            ],
            "status": "evaluation_only_not_for_runtime_decisions",
        }
    # Deliberately no automatic winner: the experiment requires ground truth.
    return report


def _load_expected(json_file):
    if not json_file:
        return {}
    annotations = json.loads(Path(json_file).read_text(encoding="utf-8"))
    if not isinstance(annotations, dict):
        raise ValueError("Annotation JSON must map filenames to four [x,y] points")
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Compare KTP detectors offline (no output images or identities)")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--annotations", type=Path, default=None, help="JSON filename -> four corners")
    parser.add_argument("--output", type=Path, default=Path("ktp_detector_comparison.json"))
    parser.add_argument(
        "--include-line-proposals", action="store_true",
        help="Include experimental four-line KTP boundary candidates for evaluation only",
    )
    parser.add_argument(
        "--include-chroma-proposals", action="store_true",
        help="Include experimental Lab chromatic-edge quadrilaterals for evaluation only",
    )
    parser.add_argument(
        "--include-region-proposals", action="store_true",
        help="Include experimental multiscale region-first outer boundary proposals (review only)",
    )
    args = parser.parse_args()
    annotations = _load_expected(args.annotations)
    results = {}
    for file in sorted(args.folder.iterdir()):
        if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        image = cv2.imread(str(file))
        if image is None:
            results[file.name] = {"error": "unreadable"}
            continue
        results[file.name] = compare_detectors(
            image, expected=annotations.get(file.name),
            include_line_candidates=args.include_line_proposals,
            include_chroma_candidates=args.include_chroma_proposals,
            include_region_candidates=args.include_region_proposals,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluated {len(results)} images -> {args.output}")
    print("No production detector or source files were changed.")


if __name__ == "__main__":
    main()
