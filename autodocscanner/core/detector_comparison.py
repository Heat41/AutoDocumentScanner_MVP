"""Offline comparison of KTP detectors; NEVER changes production selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from autodocscanner.core.experimental_candidates import contour_candidates
from autodocscanner.core.perspective_engine import AutoPerspectiveEngine
from autodocscanner.core.perspective_engine_baseline import (
    AutoPerspectiveEngine as StableBaselinePerspectiveEngine,
)


def _metric(engine, image, corners, expected=None):
    if corners is None:
        return {"found": False}
    points = engine.order_points(np.asarray(corners, dtype=np.float32))
    area = abs(cv2.contourArea(points)) / float(max(image.shape[0]*image.shape[1], 1))
    warped = engine.warp(image, points)
    quality = float(engine._warp_quality(warped))
    data = {
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
    report["v2"] = [_metric(adaptive, image, quad, expected) for quad in proposals]
    report["v2_proposal_count"] = len(proposals)
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
