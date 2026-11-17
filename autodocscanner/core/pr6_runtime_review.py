"""Explicit opt-in runtime bridge from PR6 diagnostics to Auto KTP review.

This does NOT replace the production detector or grant automatic cropping.
The expensive OpenCV comparison runs on the ORIGINAL source image, without
ground-truth coordinates. A PR6 preview becomes an editable, uncommitted
four-corner proposal. All other decisions require manual corner placement.

The only entry point for saving remains the existing UI's user-confirmed
scan_with_corners, never this module.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from autodocscanner.core.detector_comparison import compare_detectors
from autodocscanner.core.manual_fallback_validation_pr6 import (
    validate_manual_ui_corners_pr6,
)


RUNTIME_FLAG = "AUTODOCSCANNER_PR6_RUNTIME_REVIEW"


def experimental_pr6_runtime_enabled(environ=None):
    env = os.environ if environ is None else environ
    return env.get(RUNTIME_FLAG) == "1"


def _original_candidate(report, source):
    if source in ("stable", "adaptive"):
        candidate = report.get(source)
    elif (isinstance(source, str) and source.startswith("v2_")
            and source[3:].isdigit() and len(source) < 24):
        choices = report.get("v2")
        index = int(source[3:])
        candidate = (
            choices[index]
            if isinstance(choices, list) and index < len(choices) else None
        )
    else:
        candidate = None
    return candidate.get("corners") if isinstance(candidate, dict) else None


def evaluate_pr6_runtime_image(image, *, analyzer=compare_detectors):
    """Return a minimal preview/manual route, never a crop authorization.

    No disk I/O, no annotations, no raw image contents in returned data.
    The analyzer is injectable for synthetic, deterministic tests.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] < 16 or image.shape[1] < 16
            or image.dtype != np.uint8):
        raise ValueError("Gambar sumber tidak dapat dianalisis.")

    # Existing injected analyzers keep their historical signature in tests.
    # Only the real PR6 runtime analyzer requests the new independent long-
    # edge proposals; no legacy or default offline detector is changed.
    report = (
        analyzer(image, expected=None, include_line_candidates=True)
        if analyzer is compare_detectors else
        analyzer(image, expected=None)
    )
    if not isinstance(report, dict):
        raise ValueError("Laporan pemeriksaan PR6 tidak valid.")
    gate = report.get("pr6_review_consolidation_shadow")
    plan = report.get("pr6_manual_fallback_shadow")
    if not isinstance(gate, dict) or not isinstance(plan, dict):
        raise ValueError("Review Gate PR6 belum tersedia.")

    # Diagnostic modules explicitly prohibit auto-selection/cropping. Treat
    # any unexpected contract changes as a mandatory manual review.
    contract_ok = (
        gate.get("mode") == "offline_shadow_review_only"
        and gate.get("auto_crop_allowed") is False
        and gate.get("auto_select", "bad") is None
        and gate.get("requires_explicit_confirmation") is True
        and gate.get("physical_boundary_verified") is False
        and plan.get("status_scope") == "offline_diagnostic_only"
        and plan.get("ui_integration_status") == "not_connected_shadow_only"
        and plan.get("auto_crop_allowed") is False
        and plan.get("auto_select", "bad") is None
        and plan.get("requires_explicit_confirmation") is True
        and plan.get("file_write_performed") is False
        and plan.get("source_image_policy") == "unmodified_original_image_required"
        and plan.get("source_image_shape") == list(image.shape[:2])
    )

    source = None
    points = None
    if (contract_ok and gate.get("status") == "preview_review"
            and gate.get("reason_codes") == []
            and gate.get("action") == "preview_suggested_corners"
            and plan.get("status") == "preview_review"
            and plan.get("action") == "preview_suggested_corners"):
        name = plan.get("candidate_source")
        if (isinstance(name, str) and name
                and gate.get("candidate_source") == name):
            validated = validate_manual_ui_corners_pr6(
                plan.get("suggested_corners"), image.shape
            )
            original = validate_manual_ui_corners_pr6(
                _original_candidate(report, name), image.shape
            )
            if (validated is not None and original is not None
                    and np.allclose(
                        np.asarray(validated, dtype=np.float64),
                        np.asarray(original, dtype=np.float64),
                        rtol=0, atol=1e-3,
                    )):
                source = name
                points = validated

    preview = points is not None
    # These are diagnostics, not runtime auto-select permissions.
    reasons = gate.get("reason_codes")
    if not isinstance(reasons, list) or any(not isinstance(x, str) for x in reasons):
        reasons = ["invalid_gate_reasons"]
    if not preview and not reasons:
        reasons = ["preview_contract_or_geometry_not_confirmed"]
    return {
        "status": "preview_review" if preview else "manual_review",
        "action": "open_four_corner_editor",
        "candidate_source": source,
        "suggested_corners": points,
        "reason_codes": list(reasons),
        "requires_explicit_confirmation": True,
        "auto_select": None,
        "auto_crop_allowed": False,
        "crop_performed": False,
        "file_write_performed": False,
        "image_shape": list(image.shape[:2]),
        "mode": "experimental_runtime_review_only",
    }


def evaluate_pr6_runtime_file(path, *, analyzer=compare_detectors):
    """Decode the source photo once for PR6 review. Never writes output."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Gambar asli tidak dapat dibaca untuk pemeriksaan PR6.")
    return evaluate_pr6_runtime_image(image, analyzer=analyzer)
