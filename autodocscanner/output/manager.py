import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from autodocscanner.output.safe import atomic_imwrite


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



def save_ktp_sheet_pdf(
    output_path,
    images,
    dpi=300,
):
    """Save KTP results to A4 portrait, max four cards per page.

    Layout is fixed at 2 columns x 2 rows. Each KTP is rendered at the
    physical ID-1 card size (85.60 x 53.98 mm) while preserving aspect ratio.
    """
    images = list(images or [])
    if not images:
        raise ValueError(
            "Minimal satu KTP diperlukan untuk PDF."
        )

    dpi = int(dpi)
    if dpi <= 0:
        raise ValueError("dpi harus lebih besar dari 0.")

    mm_per_inch = 25.4

    page_width = int(
        round(210.0 / mm_per_inch * dpi)
    )
    page_height = int(
        round(297.0 / mm_per_inch * dpi)
    )

    card_width = int(
        round(85.60 / mm_per_inch * dpi)
    )
    card_height = int(
        round(53.98 / mm_per_inch * dpi)
    )

    slot_width = page_width // 2
    slot_height = page_height // 2

    pages = []

    for page_start in range(0, len(images), 4):
        page = Image.new(
            "RGB",
            (page_width, page_height),
            "white",
        )

        for local_index, image in enumerate(
            images[page_start:page_start + 4]
        ):
            card = _to_rgb_pillow(image)

            card_ratio = (
                card.width / max(card.height, 1)
            )
            target_ratio = (
                card_width / max(card_height, 1)
            )

            if card_ratio >= target_ratio:
                render_width = card_width
                render_height = max(
                    1,
                    int(round(
                        card_width / card_ratio
                    )),
                )
            else:
                render_height = card_height
                render_width = max(
                    1,
                    int(round(
                        card_height * card_ratio
                    )),
                )

            card = card.resize(
                (render_width, render_height),
                Image.Resampling.LANCZOS,
            )

            row = local_index // 2
            column = local_index % 2

            slot_x = column * slot_width
            slot_y = row * slot_height

            x = (
                slot_x
                + (slot_width - render_width) // 2
            )
            y = (
                slot_y
                + (slot_height - render_height) // 2
            )

            page.paste(
                card,
                (x, y),
            )

        pages.append(page)

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
        pages[0].save(
            temp_path,
            format="PDF",
            resolution=float(dpi),
            save_all=len(pages) > 1,
            append_images=pages[1:],
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
