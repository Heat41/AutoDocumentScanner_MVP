"""Summarize holdout annotation errors from an offline detector comparison JSON.

No images are uploaded, cropped, or saved. Only metrics are printed.
Run detector_comparison with --annotations before invoking this tool.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summarize(data):
    annotated = []
    for report in data.values():
        if not isinstance(report, dict):
            continue
        entries = {"stable": report.get("stable", {}), "adaptive": report.get("adaptive", {})}
        entries.update({f"v2_{i}": item for i, item in enumerate(report.get("v2") or [])})
        measured = {name: item.get("normalized_corner_error") for name, item in entries.items()
                    if isinstance(item, dict) and isinstance(item.get("normalized_corner_error"), (int, float))
                    and math.isfinite(item["normalized_corner_error"])}
        if not measured:
            continue
        gate = report.get("v3_review_gate") or {}
        annotated.append({"measured": measured, "selected": gate.get("candidate_source"),
                          "status": gate.get("status")})
    if not annotated:
        return {"annotated_images": 0, "message": "No corner annotations: accuracy cannot be measured."}
    selected = [r["measured"][r["selected"]] for r in annotated if r["selected"] in r["measured"]]
    baseline = [r["measured"]["stable"] for r in annotated if "stable" in r["measured"]]
    return {
        "annotated_images": len(annotated),
        "evaluated_suggestions": len(selected),
        "mean_normalized_corner_error_suggestion": round(sum(selected)/len(selected), 5) if selected else None,
        "mean_normalized_corner_error_stable": round(sum(baseline)/len(baseline), 5) if baseline else None,
        "preview_review_count": sum(r["status"] == "preview_review" for r in annotated),
        "manual_or_no_candidate_count": sum(r["status"] != "preview_review" for r in annotated),
        "note": "Scores are uncalibrated; errors require representative unseen photos and manual corner annotations.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path, help="JSON generated using --annotations")
    args = parser.parse_args()
    results = summarize(json.loads(args.comparison.read_text(encoding="utf-8")))
    print(json.dumps(results, indent=2))
    if not results["annotated_images"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
