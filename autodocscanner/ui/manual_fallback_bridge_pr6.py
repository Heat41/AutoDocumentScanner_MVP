"""Opt-in, offline UI bridge for the existing Auto KTP four-corner dialog.

The bridge never runs the PR6 detector in the UI, saves a file, or applies a
perspective transform. It only enables access to the existing dialog from an
unmodified source image and validates user-confirmed original-image points.

Disabled by default. This is not production PR6 review-gate integration.
"""
from __future__ import annotations

import os

from autodocscanner.core.manual_fallback_validation_pr6 import (
    validate_manual_ui_corners_pr6,
)
from autodocscanner.documents.manual_document import initial_corners

FLAG = "AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE"


def experimental_manual_bridge_enabled(environ=None):
    """Require an explicit, exact opt-in; never enable silently."""
    values = os.environ if environ is None else environ
    return values.get(FLAG) == "1"


def manual_editor_available(*, has_source, has_detected_corners,
                            processing_active, experimental_enabled):
    """Continue to respect the legacy detector-corners gate by default."""
    return bool(
        has_source and not processing_active and
        (has_detected_corners or experimental_enabled)
    )


def fallback_starting_corners(image_size):
    """Independent four-point seed in ORIGINAL image coordinates.

    This is an editable starting hint, not a detected KTP boundary.
    """
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("Ukuran gambar asli tidak valid.")
    width, height = image_size
    if type(width) is not int or type(height) is not int:
        raise ValueError("Ukuran gambar asli tidak valid.")
    return initial_corners(width, height).tolist()


def validate_user_confirmed_corners(corners, image_size):
    """Return checked TL/TR/BR/BL points, or None. Never warp or save."""
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        return None
    width, height = image_size
    if type(width) is not int or type(height) is not int:
        return None
    return validate_manual_ui_corners_pr6(corners, (height, width))
