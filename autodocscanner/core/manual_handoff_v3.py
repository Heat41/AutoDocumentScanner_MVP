"""Experimental, UI-independent handoff to the existing four-corner editor.

A preview is NEVER an instruction to save/crop automatically. The calling UI
must implement a separate explicit user confirmation, and currently this module
is not connected to production UI.
"""
from __future__ import annotations

import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def _resolve_candidate(report, source):
    if source in ("stable", "adaptive"):
        return report.get(source) or {}
    if isinstance(source, str) and source.startswith("v2_") and source[3:].isdigit():
        index = int(source[3:])
        choices = report.get("v2") or []
        return choices[index] if index < len(choices) else {}
    return {}


def prepare_manual_handoff(report, image_shape):
    """Return a safe, serializable four-corner UI instruction.

    Suggested corners are provided ONLY for preview_review. For manual_review
    they remain absent: the UI should open its existing manual editor and ask
    the user to pick corners. Never mutate an image or write output here.
    """
    gate = report.get("v3_review_gate") or {}
    status = gate.get("status")
    source = gate.get("candidate_source")
    base = {
        "editor": "existing_manual_four_corners",
        "auto_crop_allowed": False,
        "requires_explicit_confirmation": True,
        "suggested_corners": None,
        "action": "manual_corners",
        "reasons": list(gate.get("reasons") or []),
    }
    if status != "preview_review" or gate.get("auto_crop_allowed") is not False:
        return base
    item = _resolve_candidate(report, source)
    corners = item.get("corners")
    try:
        points = np.asarray(corners, dtype=float)
        valid, _, _ = validate_quad(points, image_shape)
        if not valid:
            return base
        base["suggested_corners"] = points.tolist()
        base["action"] = "preview_suggested_corners"
    except (ValueError, TypeError):
        pass
    return base
