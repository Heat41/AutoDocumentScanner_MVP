import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image


_OUTPUT_SUFFIXES = {
    "image": ".jpg",
    "pdf": ".pdf",
}


def _normalize_output_format(output_format):
    value = str(output_format or "").strip().lower()
    if value not in _OUTPUT_SUFFIXES:
        raise ValueError("output_format harus 'image' atau 'pdf'.")
    return value


def output_suffix(output_format):
    return _OUTPUT_SUFFIXES[
        _normalize_output_format(output_format)
    ]


def build_output_path(output_dir, input_path, output_format):
    output_dir = Path(output_dir)
    input_path = Path(input_path)
    return output_dir / (
        f"{input_path.stem}_scanned"
        f"{output_suffix(output_format)}"
    )


def _to_rgb_pillow(image):
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError(
            "image harus berupa numpy array yang tidak kosong."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            "image harus menggunakan dtype uint8."
        )

    if image.ndim == 2:
        return Image.fromarray(
            image,
            mode="L",
        ).convert("RGB")

    if image.ndim == 3 and image.shape[2] == 3:
        rgb = np.ascontiguousarray(
            image[:, :, ::-1]
        )
        return Image.fromarray(
            rgb,
            mode="RGB",
        )

    raise ValueError(
        "image harus grayscale atau BGR 3-channel."
    )


def save_pdf(output_path, image):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_image = _to_rgb_pillow(image)
    temp_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp.pdf"
    )

    try:
        pdf_image.save(
            temp_path,
            format="PDF",
            resolution=72.0,
            save_all=False,
        )
        os.replace(
            temp_path,
            output_path,
        )
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass

    return output_path
