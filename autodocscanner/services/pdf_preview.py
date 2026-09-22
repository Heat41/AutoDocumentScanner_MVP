from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

import numpy as np

from autodocscanner.output.manager import save_pdf


def build_ktp_preview_pdf(
    image,
    stem="ktp_preview",
    temp_dir=None,
):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "image preview PDF harus berupa numpy array yang tidak kosong."
        )

    root = Path(
        temp_dir
        if temp_dir is not None
        else gettempdir()
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_stem = (
        str(stem or "ktp_preview")
        .strip()
        .replace(" ", "_")
    )
    if not safe_stem:
        safe_stem = "ktp_preview"

    output_path = root / (
        f"{safe_stem}_{uuid4().hex[:8]}.pdf"
    )

    return save_pdf(
        output_path,
        image,
    )
