"""Offline comparison of KTP detectors; NEVER changes production selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from autodocscanner.core.experimental_candidates import contour_candidates
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


def compare_detectors(image, *, expected=None, max_proposals=24):
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
    evaluated = [_metric(adaptive, image, quad, expected) for quad in proposals]
    # Filter using the actual ordered corners that are reported and warped.
    report["v2"], report["v2_rejected_post_order"] = _filter_valid_candidates(evaluated)
    report["v2_proposal_count"] = len(report["v2"])
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
        results[file.name] = compare_detectors(image, expected=annotations.get(file.name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluated {len(results)} images -> {args.output}")
    print("No production detector or source files were changed.")


if __name__ == "__main__":
    main()
