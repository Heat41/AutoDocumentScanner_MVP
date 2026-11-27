"""Local-only PR9.1 evaluation of V3 boundary diagnostics.

Consumes the existing V3 annotation format: {filename: [[x,y], ...]}.
Never uses annotations for detection or boundary decisions. No V2 calls,
file writes, image uploads, OCR, coordinates or filenames in the report.
A holdout label is supplied by the operator, not a guarantee of independence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SOURCES = ("stable", "adaptive")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Descriptive evaluation thresholds, not calibrated crop/release thresholds.
ERROR_LIMIT = .02
SEVERE_LIMIT = .10


def _quad(points, shape):
    from autodocscanner.core.quad_validation import validate_quad
    try:
        p = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return None
    h, w = shape[:2]
    if np.any(p < 0) or np.any(p[:, 0] >= w) or np.any(p[:, 1] >= h):
        return None
    valid, _, _ = validate_quad(p, shape)
    if not valid:
        return None
    center = p.mean(axis=0)
    return p[np.argsort(np.arctan2(p[:, 1]-center[1], p[:, 0]-center[0]))]


def _errors(candidate, reference, shape):
    # Cyclic shifts account for different starting vertices, without allowing
    # arbitrary crossed vertex assignments or using labels in detection.
    diagonal = math.hypot(*shape[:2])
    distances = min((np.linalg.norm(candidate-np.roll(reference, i, axis=0), axis=1)
                     for i in range(4)), key=lambda d: float(d.mean()))
    return float(distances.mean()/diagonal), float(distances.max()/diagonal)


def _default_detectors():
    from autodocscanner.core.perspective_engine import AutoPerspectiveEngine
    from autodocscanner.core.perspective_engine_baseline import (
        AutoPerspectiveEngine as StableBaselinePerspectiveEngine,
    )
    return {"stable": StableBaselinePerspectiveEngine(),
            "adaptive": AutoPerspectiveEngine()}


def evaluate_photo(image, annotation, *, detectors=None, auditor=None, localizer=None):
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or
            image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 64):
        return {"status": "unreadable_or_invalid_image"}
    reference = _quad(annotation, image.shape)
    if reference is None:
        return {"status": "missing_or_invalid_annotation"}
    if detectors is None:
        detectors = _default_detectors()
    if auditor is None:
        from autodocscanner.core.pr9_boundary_audit import audit_v3_boundaries_pr9
        auditor = audit_v3_boundaries_pr9
    if localizer is None:
        from autodocscanner.core.boundary_localization_pr6 import localize_candidate_boundary
        localizer = localize_candidate_boundary
    observations = {}
    failures = set()
    for source in SOURCES:
        try:
            # Copy preserves original pixels and prevents one detector from
            # changing the other detector's input. No ground truth passed.
            corners, _ = detectors[source].detect(image.copy())
        except (KeyError, AttributeError, cv2.error, ValueError, TypeError, ArithmeticError):
            failures.add(source)
            corners = None
        observations[source] = {"found": corners is not None, "corners": corners}
    diagnostic = auditor(image.copy(), observations)
    if (not isinstance(diagnostic, dict) or diagnostic.get("auto_crop_allowed") is not False
            or diagnostic.get("auto_select", "missing") is not None
            or diagnostic.get("suggested_for_review", "missing") is not None):
        raise ValueError("Unexpected PR9 safety contract")
    rows = {}
    for source in SOURCES:
        observation = observations[source]
        p = _quad(observation["corners"], image.shape) if observation["found"] else None
        d = diagnostic.get("candidates", {}).get(source, {})
        row = {"status": "detector_error" if source in failures else
               "missing_candidate" if not observation["found"] else
               "invalid_or_off_photo" if p is None else "measured",
               "mean_error": None, "max_error": None,
               "pr6_edge_aligned": None, "pr9_four_sides_aligned": None,
               "pr9_review_evidence": None}
        if p is not None:
            mean, worst = _errors(p, reference, image.shape)
            # PR6 localizer is a historical evidence baseline, not V2.
            old = localizer(image.copy(), observation["corners"])
            sides = d.get("side_observations", [])
            row.update(mean_error=mean, max_error=worst,
                       pr6_edge_aligned=old.get("edge_aligned") is True,
                       pr9_four_sides_aligned=len(sides) == 4 and all(
                           s.get("state") == "aligned" for s in sides),
                       pr9_review_evidence=d.get("fully_aligned_for_review") is True)
        rows[source] = row
    return {"status": "evaluated", "candidates": rows,
            "geometry_conflict": diagnostic.get("distinct_geometry_conflict") is True,
            "auto_crop_allowed": False}


def _stats(values):
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(max(values))}


def _evidence_counts(rows, field):
    good = sum(r["mean_error"] <= ERROR_LIMIT for r in rows)
    supported = [r for r in rows if r[field] is True]
    supported_good = sum(r["mean_error"] <= ERROR_LIMIT for r in supported)
    return {"supported": len(supported), "supported_within_2pct": supported_good,
            "supported_over_2pct": len(supported)-supported_good,
            "within_2pct_but_not_supported": good-supported_good,
            "support_precision_at_2pct": supported_good/len(supported) if supported else None}


def summarize(rows, dataset_role):
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    per_source = {}
    for source in SOURCES:
        all_rows = [r["candidates"][source] for r in evaluated]
        measured = [r for r in all_rows if r["status"] == "measured"]
        per_source[source] = {
            "evaluated_photos": len(evaluated), "measured_candidates": len(measured),
            "missing_or_invalid_candidates": len(all_rows)-len(measured),
            "mean_corner_error_fraction": _stats([r["mean_error"] for r in measured]),
            "worst_corner_error_fraction": _stats([r["max_error"] for r in measured]),
            "within_2pct": sum(r["mean_error"] <= ERROR_LIMIT for r in measured),
            "severe_over_10pct": sum(r["mean_error"] > SEVERE_LIMIT for r in measured),
            "evidence": {k: _evidence_counts(measured, k) for k in
                         ("pr6_edge_aligned", "pr9_four_sides_aligned", "pr9_review_evidence")},
        }
    return {"dataset_role_declared_by_operator": dataset_role, "photo_count": len(rows),
            "evaluated_photo_count": len(evaluated),
            "excluded_by_status": {k: sum(r["status"] == k for r in rows) for k in
                                   ("unreadable_or_invalid_image", "missing_or_invalid_annotation")},
            "source_metrics": per_source,
            "oracle_photos_with_candidate_within_2pct": sum(any(
                c["mean_error"] is not None and c["mean_error"] <= ERROR_LIMIT
                for c in r["candidates"].values()) for r in evaluated),
            "geometry_conflict_count": sum(r["geometry_conflict"] for r in evaluated),
            "auto_crop_allowed": False, "auto_select": None,
            "mode": "evaluation_only", "error_unit": "fraction_of_original_image_diagonal",
            "note": "Oracle and evidence metrics are retrospective; no selection or production accuracy claim.",
            "photos": [{"case_id": f"D{i:04d}", **r} for i, r in enumerate(rows, 1)]}


def evaluate_folder(folder, annotations, *, dataset_role, evaluator=None):
    if dataset_role not in ("development", "holdout") or not isinstance(annotations, dict):
        raise ValueError("Invalid dataset role or annotation format")
    evaluator = evaluate_photo if evaluator is None else evaluator
    rows = []
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error, ValueError):
            image = None
        rows.append(evaluator(image, annotations.get(path.name)))
    return summarize(rows, dataset_role)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local-only V3 PR9.1 holdout validation")
    parser.add_argument("folder", type=Path, help="Folder containing original photos")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--dataset-role", choices=("development", "holdout"), required=True)
    args = parser.parse_args(argv)
    try:
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        result = evaluate_folder(args.folder, annotations, dataset_role=args.dataset_role)
    except (OSError, ValueError, TypeError, cv2.error):
        parser.error("Invalid local dataset, annotation file, or evaluation contract")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["evaluated_photo_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
