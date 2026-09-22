import cv2
import numpy as np

from autodocscanner.ktp.layout import (
    KTP_VALUE_BOXES,
    normalized_to_pixel_box,
)


ROI_LABELS = {
    "provinsi": "PROVINSI",
    "kabupaten_kota": "KAB/KOTA",
    "nik": "NIK",
    "nama": "NAMA",
    "ttl": "TTL",
    "jenis_kelamin": "JK",
    "golongan_darah": "GOL DARAH",
    "alamat": "ALAMAT",
    "rt_rw": "RT/RW",
    "kelurahan_desa": "KEL/DESA",
    "kecamatan": "KECAMATAN",
    "agama": "AGAMA",
    "status_perkawinan": "STATUS",
    "pekerjaan": "PEKERJAAN",
    "kewarganegaraan": "KEWARGANEGARAAN",
    "berlaku_hingga": "BERLAKU",
}


def build_roi_overlay(
    image,
    boxes=None,
):
    if (
        not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
    ):
        raise ValueError(
            "image ROI debug harus berupa numpy array yang tidak kosong."
        )

    if image.ndim == 2:
        overlay = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )
    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        overlay = image.copy()
    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):
        overlay = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR,
        )
    else:
        raise ValueError(
            "Format image ROI debug tidak didukung."
        )

    height, width = (
        overlay.shape[:2]
    )
    source_boxes = (
        KTP_VALUE_BOXES
        if boxes is None
        else boxes
    )

    for index, (
        name,
        box,
    ) in enumerate(
        source_boxes.items(),
        start=1,
    ):
        x1, y1, x2, y2 = (
            normalized_to_pixel_box(
                box,
                image_width=width,
                image_height=height,
            )
        )

        color = (
            30,
            220,
            255,
        )

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = (
            f"{index}. "
            f"{ROI_LABELS.get(name, name.upper())}"
        )

        text_y = max(
            14,
            y1 - 4,
        )

        cv2.putText(
            overlay,
            label,
            (
                x1,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (
                0,
                0,
                0,
            ),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (
                x1,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay
