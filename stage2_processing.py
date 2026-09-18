from io import BytesIO

import numpy as np
from PIL import Image

from output_manager import build_output_path, save_pdf


def _preview_jpeg_bytes(image, quality=90):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "image preview tidak valid."
        )

    if image.ndim == 2:
        preview = Image.fromarray(
            image
        ).convert("RGB")
    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        preview = Image.fromarray(
            np.ascontiguousarray(
                image[:, :, ::-1]
            )
        )
    else:
        raise ValueError(
            "image preview harus grayscale "
            "atau BGR 3-channel."
        )

    buffer = BytesIO()
    preview.save(
        buffer,
        format="JPEG",
        quality=int(quality),
    )
    return buffer.getvalue()


def process_ktp_output(
    scanner,
    input_path,
    output_dir,
    output_mode="color",
    output_format="image",
):
    output_path = build_output_path(
        output_dir,
        input_path,
        output_format,
    )

    normalized_format = str(
        output_format
    ).strip().lower()

    if normalized_format == "image":
        image, corners = scanner.scan(
            input_path,
            output_path,
            mode="ktp",
            output_mode=output_mode,
        )
        preview_bytes = None
    else:
        image, corners = scanner.scan(
            input_path,
            None,
            mode="ktp",
            output_mode=output_mode,
        )
        save_pdf(
            output_path,
            image,
        )
        preview_bytes = (
            _preview_jpeg_bytes(image)
        )

    return {
        "image": image,
        "corners": corners,
        "output_path": output_path,
        "preview_bytes": preview_bytes,
    }
