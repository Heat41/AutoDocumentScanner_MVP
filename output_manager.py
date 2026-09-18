import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from safe_output import atomic_imwrite


_OUTPUT_SUFFIXES = {
    "image": ".jpg",
    "pdf": ".pdf",
}


def _normalize_output_format(output_format):
    value = str(output_format or "").strip().lower()
    if value not in _OUTPUT_SUFFIXES:
        raise ValueError(
            "output_format harus 'image' atau 'pdf'."
        )
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
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "image harus berupa numpy array "
            "yang tidak kosong."
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

    if (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        rgb = np.ascontiguousarray(
            image[:, :, ::-1]
        )
        return Image.fromarray(
            rgb,
            mode="RGB",
        )

    raise ValueError(
        "image harus grayscale atau "
        "BGR 3-channel."
    )


def _atomic_pdf_save(
    output_path,
    first_image,
    append_images=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp_path = output_path.with_name(
        f".{output_path.stem}."
        f"{uuid4().hex}.tmp.pdf"
    )

    try:
        first_image.save(
            temp_path,
            format="PDF",
            resolution=72.0,
            save_all=bool(append_images),
            append_images=append_images or [],
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


def save_pdf(output_path, image):
    pdf_image = _to_rgb_pillow(image)
    return _atomic_pdf_save(
        output_path,
        pdf_image,
    )


def save_pdf_pages(output_path, images):
    images = list(images or [])
    if not images:
        raise ValueError(
            "Minimal satu halaman diperlukan "
            "untuk PDF."
        )

    pdf_images = [
        _to_rgb_pillow(image)
        for image in images
    ]

    return _atomic_pdf_save(
        output_path,
        pdf_images[0],
        pdf_images[1:],
    )


def save_document_images(
    output_dir,
    source_paths,
    images,
):
    source_paths = [
        Path(path)
        for path in source_paths
    ]
    images = list(images or [])

    if not images:
        raise ValueError(
            "Minimal satu halaman diperlukan."
        )

    if len(source_paths) != len(images):
        raise ValueError(
            "Jumlah source_paths harus sama "
            "dengan jumlah images."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = []
    for index, (
        source_path,
        image,
    ) in enumerate(
        zip(source_paths, images),
        start=1,
    ):
        output_path = output_dir / (
            f"page_{index:03d}_"
            f"{source_path.stem}"
            "_corrected.jpg"
        )
        outputs.append(
            atomic_imwrite(
                output_path,
                image,
            )
        )

    return outputs
