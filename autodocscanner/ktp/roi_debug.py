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

        label = str(index)
        label_x = max(
            3,
            x1 - 22,
        )
        label_y = min(
            height - 4,
            max(
                14,
                int(
                    round(
                        (y1 + y2) / 2
                    )
                )
                + 5,
            ),
        )

        cv2.circle(
            overlay,
            (
                label_x + 6,
                label_y - 5,
            ),
            8,
            (
                0,
                0,
                0,
            ),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            overlay,
            (
                label_x + 6,
                label_y - 5,
            ),
            7,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (
                label_x + 2,
                label_y - 1,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay



def build_detection_overlay(
    image,
    detections,
):
    if (
        not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
    ):
        raise ValueError(
            "image detection overlay tidak boleh kosong."
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
            "Format image detection overlay tidak didukung."
        )

    height, width = (
        overlay.shape[:2]
    )

    for detection in detections or []:
        try:
            x1, y1, x2, y2 = [
                int(round(value))
                for value in detection.bbox
            ]
            confidence = float(
                detection.confidence
            )
            class_name = str(
                detection.class_name
            ).upper()
        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            continue

        x1 = max(
            0,
            min(width - 1, x1),
        )
        y1 = max(
            0,
            min(height - 1, y1),
        )
        x2 = max(
            x1 + 1,
            min(width, x2),
        )
        y2 = max(
            y1 + 1,
            min(height, y2),
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
            f"{class_name} "
            f"{confidence * 100:.0f}%"
        )

        (
            text_width,
            text_height,
        ), _baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            1,
        )

        label_x = max(
            2,
            x1 - text_width - 8,
        )
        label_y = min(
            height - 4,
            max(
                text_height + 3,
                int(
                    round(
                        (y1 + y2) / 2
                    )
                )
                + text_height // 2,
            ),
        )

        cv2.rectangle(
            overlay,
            (
                label_x - 2,
                label_y - text_height - 3,
            ),
            (
                min(
                    width - 1,
                    label_x + text_width + 3,
                ),
                label_y + 2,
            ),
            color,
            -1,
        )

        cv2.putText(
            overlay,
            label,
            (
                label_x,
                label_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (
                0,
                0,
                0,
            ),
            1,
            cv2.LINE_AA,
        )

    return overlay
