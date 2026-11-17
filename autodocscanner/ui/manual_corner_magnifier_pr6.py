"""Experimental four-corner magnifier for the existing Tk Auto KTP dialog.

Read-only presentation of ORIGINAL-image pixels. No image modification,
perspective correction, image-specific heuristic, storage or network calls.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw


def magnifier_image(source, point, *, size=160, zoom=4):
    """Return a source-pixel zoom around the selected ORIGINAL-image point.

    EXTENT samples fractional coordinates centered on the active corner;
    neutral padding is shown beyond source edges instead of moving the
    crosshair or silently shifting the target near a frame boundary.
    """
    if not isinstance(source, Image.Image) or source.width < 2 or source.height < 2:
        raise ValueError("Gambar sumber tidak valid.")
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise ValueError("Titik sudut tidak valid.")
    try:
        x, y = map(float, point)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Titik sudut tidak valid.") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("Titik sudut tidak valid.")
    if type(size) is not int or not (64 <= size <= 320):
        raise ValueError("Ukuran magnifier tidak valid.")
    if type(zoom) is not int or not (2 <= zoom <= 12):
        raise ValueError("Skala magnifier tidak valid.")

    # The dialog limits dragging to source-image bounds. Clamp only visual
    # sampling, never the underlying editable points or scanner coordinates.
    x = min(max(x, 0.0), source.width - 1.0)
    y = min(max(y, 0.0), source.height - 1.0)
    radius = size / (2.0 * zoom)
    # The Auto KTP editor already holds an RGB source. Avoid copying an
    # entire high-resolution photo on every mouse-motion event.
    rgb = source if source.mode == "RGB" else source.convert("RGB")
    patch = rgb.transform(
        (size, size),
        Image.Transform.EXTENT,
        (x - radius, y - radius, x + radius, y + radius),
        resample=Image.Resampling.NEAREST,
        fillcolor=(226, 232, 240),
    )
    draw = ImageDraw.Draw(patch)
    mid = size // 2
    # A crisp target reticle and short segmented lines avoid hiding the edge.
    for outline, width in (((255, 255, 255), 4), ((220, 38, 38), 2)):
        draw.line([(mid, 4), (mid, mid - 10)], fill=outline, width=width)
        draw.line([(mid, mid + 10), (mid, size - 5)], fill=outline, width=width)
        draw.line([(4, mid), (mid - 10, mid)], fill=outline, width=width)
        draw.line([(mid + 10, mid), (size - 5, mid)], fill=outline, width=width)
    draw.ellipse((mid - 5, mid - 5, mid + 5, mid + 5),
                 outline=(220, 38, 38), width=2)
    return patch


def magnifier_canvas_position(point, canvas_size, *, size=160, margin=12):
    """Put the magnifier in the opposite canvas corner from the active point.

    This avoids covering the target corner or the drag handle, even when the
    source card fills the entire camera frame.
    """
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise ValueError("Titik tidak valid.")
    try:
        px, py = float(point[0]), float(point[1])
        width, height = float(canvas_size[0]), float(canvas_size[1])
    except (ValueError, TypeError, IndexError, OverflowError):
        raise ValueError("Ukuran canvas tidak valid.") from None
    if not all(math.isfinite(v) for v in (px, py, width, height)):
        raise ValueError("Ukuran canvas tidak valid.")
    if width < size + 2 * margin or height < size + 2 * margin:
        return None
    return (
        float(margin if px >= width / 2.0 else width - size - margin),
        float(margin if py >= height / 2.0 else height - size - margin),
    )
