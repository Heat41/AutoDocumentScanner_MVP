"""Compare PR6 hypotheses with human-marked original-image corners, locally.

Post-annotation evaluation ONLY: never modify selection, detector, crop or
files. Per-family minimum is an oracle for candidate recall, not accuracy
of the detector's actual choice.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from autodocscanner.core.manual_fallback_validation_pr6 import (
    validate_manual_ui_corners_pr6,
)

FAMILIES = frozenset({
    "Stable", "Adaptive", "Kontur V2", "Four-Line", "Chroma",
    "Outer Region", "Side Recovery", "Paired Alignment",
    "Independent Sides",
})


def _candidate_points(value, height, width):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    if (np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)):
        return None
    return points


def _corner_distances(candidate, reference):
    # Candidate source may start at a different cyclic vertex. Never
    # assume image rotation means a particular TL/TR ordering.
    # The output MUST be indexed by human-labelled TL/TR/BR/BL. Rotating
    # the *reference* would return errors indexed by arbitrary candidate
    # vertices, making the corner names misleading for rotated sources.
    variants = [
        np.roll(candidate, i, axis=0) for i in range(4)
    ] + [
        np.roll(candidate[::-1], i, axis=0) for i in range(4)
    ]
    aligned = min(
        variants,
        key=lambda alternative: float(
            np.linalg.norm(alternative - reference, axis=1).mean()
        ),
    )
    return np.linalg.norm(aligned - reference, axis=1), aligned - reference


def evaluate_pr6_candidates_against_manual(snapshot, corners, image_shape):
    """Anonymous accuracy diagnostics after explicitly marking four corners.

    No reference points, filenames or photo pixels in the output. The
    snapshot is not edited, and no candidate is selected or applied.
    """
    if (not isinstance(snapshot, dict)
            or snapshot.get("status") != "offline_visual_diagnostic_only"
            or snapshot.get("auto_select", "unsafe") is not None
            or snapshot.get("auto_crop_allowed") is not False):
        raise ValueError("Hasil Review PR6 tidak tersedia atau tidak aman.")
    if (not isinstance(image_shape, (tuple, list))
            or len(image_shape) < 2
            or any(type(v) is not int for v in image_shape[:2])):
        raise ValueError("Dimensi foto sumber tidak valid.")
    height, width = image_shape[:2]
    if not (16 <= height <= 100000 and 16 <= width <= 100000):
        raise ValueError("Dimensi foto sumber tidak valid.")
    if snapshot.get("image_shape") != [height, width]:
        raise ValueError("Orientasi atau ukuran foto berbeda dari Review PR6.")
    truth = validate_manual_ui_corners_pr6(corners, (height, width))
    if truth is None:
        raise ValueError("Empat sudut manual belum valid.")
    candidates = snapshot.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Daftar kandidat tidak valid.")

    reference = np.asarray(truth, np.float64)
    diagonal = math.hypot(width, height)
    rows, seen = [], set()
    for candidate in candidates[:40]:
        if not isinstance(candidate, dict):
            continue
        family, source = candidate.get("family"), candidate.get("source")
        if (family not in FAMILIES or not isinstance(source, str)
                or not 1 <= len(source) <= 40
                or any(not c.isascii() or not (c.isalnum() or c == "_")
                       for c in source)
                or source in seen):
            continue
        seen.add(source)
        points = _candidate_points(candidate.get("corners"), height, width)
        if points is None:
            continue
        distances, signed_offsets = _corner_distances(points, reference)
        rows.append({
            "source": source, "family": family,
            "mean_error_px": round(float(distances.mean()), 2),
            "max_error_px": round(float(distances.max()), 2),
            "normalized_error": round(float(distances.mean()) / diagonal, 6),
            # TL/TR/BR/BL ordered error vectors; no absolute annotated
            # positions, image pixels or reference coordinates are returned.
            "corner_errors_px": [
                round(float(value), 2) for value in distances
            ],
            "corner_offsets_xy_px": [
                [round(float(dx), 2), round(float(dy), 2)]
                for dx, dy in signed_offsets
            ],
            "worst_corner_index": int(np.argmax(distances)),
        })
    rows.sort(key=lambda row: (row["normalized_error"], row["source"]))
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    stats = []
    for family, entries in groups.items():
        errors = np.array([entry["normalized_error"] for entry in entries])
        stats.append({
            "family": family, "count": len(entries),
            "best_source": entries[0]["source"],
            "best_error": round(float(errors.min()), 6),
            "median_error": round(float(np.median(errors)), 6),
            "p90_error": round(float(np.percentile(errors, 90)), 6),
        })
    stats.sort(key=lambda item: (item["best_error"], item["family"]))
    def count(key):
        value = snapshot.get(key)
        return min(value, 10000) if type(value) is int and value >= 0 else 0

    return {
        "status": "post_manual_reference_evaluation_only",
        "auto_select": None, "auto_crop_allowed": False,
        "crop_performed": False, "file_write_performed": False,
        "oracle_only": True,
        "normalization": "original_image_diagonal",
        "candidate_count": len(rows),
        "skipped_count": min(len(candidates), 40) - len(rows),
        "undisplayed_hypothesis_count": count("source_hypotheses_hidden"),
        "generation": {
            "side_recovery": count("generated_side_recovery_count"),
            "paired_alignment": count("generated_paired_alignment_count"),
            "independent_sides": count("independent_sides_count"),
            "paired_parents_examined": count("paired_alignment_parents_examined"),
            "independent_parents_examined": count("independent_sides_parents_examined"),
        },
        "families": stats, "candidates": rows,
    }
