from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np


@dataclass(frozen=True)
class NormalizedBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        values = (
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        )

        if any(
            value < 0.0 or value > 1.0
            for value in values
        ):
            raise ValueError(
                "Koordinat normalized box "
                "harus berada pada rentang 0..1."
            )

        if (
            self.x1 >= self.x2
            or self.y1 >= self.y2
        ):
            raise ValueError(
                "Normalized box harus "
                "memiliki area positif."
            )


KTP_CANONICAL_WIDTH = 856
KTP_CANONICAL_HEIGHT = 540


def normalize_ktp_for_tracking(image):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "image KTP harus berupa numpy array yang tidak kosong."
        )

    return cv2.resize(
        image,
        (
            KTP_CANONICAL_WIDTH,
            KTP_CANONICAL_HEIGHT,
        ),
        interpolation=cv2.INTER_CUBIC,
    )


# Baseline layout for the canonical, landscape KTP image produced by
# AutoDocumentScanner. These boxes intentionally cover the value area
# conservatively; OCR parsing in later Stage 2D slices may trim labels/text.
KTP_FIELD_BOXES = {
    "provinsi": NormalizedBox(
        0.22, 0.035, 0.77, 0.105
    ),
    "kabupaten_kota": NormalizedBox(
        0.18, 0.095, 0.78, 0.165
    ),
    "nik": NormalizedBox(
        0.16, 0.165, 0.73, 0.270
    ),
    "nama": NormalizedBox(
        0.18, 0.270, 0.72, 0.335
    ),
    "ttl": NormalizedBox(
        0.18, 0.330, 0.73, 0.395
    ),
    "jenis_kelamin": NormalizedBox(
        0.18, 0.390, 0.50, 0.455
    ),
    "golongan_darah": NormalizedBox(
        0.51, 0.390, 0.71, 0.455
    ),
    "alamat": NormalizedBox(
        0.18, 0.450, 0.72, 0.520
    ),
    "rt_rw": NormalizedBox(
        0.18, 0.515, 0.48, 0.580
    ),
    "kelurahan_desa": NormalizedBox(
        0.18, 0.575, 0.72, 0.640
    ),
    "kecamatan": NormalizedBox(
        0.18, 0.635, 0.72, 0.700
    ),
    "agama": NormalizedBox(
        0.18, 0.695, 0.58, 0.755
    ),
    "status_perkawinan": NormalizedBox(
        0.18, 0.750, 0.72, 0.810
    ),
    "pekerjaan": NormalizedBox(
        0.18, 0.805, 0.72, 0.865
    ),
    "kewarganegaraan": NormalizedBox(
        0.18, 0.860, 0.72, 0.920
    ),
    "berlaku_hingga": NormalizedBox(
        0.18, 0.915, 0.72, 0.985
    ),
    "foto": NormalizedBox(
        0.735, 0.245, 0.955, 0.775
    ),
}

# Value-only OCR regions for the canonical corrected KTP.
# These start to the right of the printed field labels so OCR sees
# mostly the value instead of label + value + background noise.
KTP_VALUE_BOXES = {
    "provinsi": NormalizedBox(
        0.285, 0.030, 0.735, 0.068
    ),
    "kabupaten_kota": NormalizedBox(
        0.305, 0.078, 0.715, 0.118
    ),
    "nik": NormalizedBox(
        0.225, 0.162, 0.705, 0.206
    ),
    "nama": NormalizedBox(
        0.275, 0.246, 0.700, 0.286
    ),
    "ttl": NormalizedBox(
        0.275, 0.296, 0.700, 0.336
    ),
    "jenis_kelamin": NormalizedBox(
        0.275, 0.346, 0.515, 0.386
    ),
    "golongan_darah": NormalizedBox(
        0.565, 0.346, 0.695, 0.386
    ),
    "alamat": NormalizedBox(
        0.275, 0.396, 0.700, 0.436
    ),
    "rt_rw": NormalizedBox(
        0.275, 0.446, 0.515, 0.486
    ),
    "kelurahan_desa": NormalizedBox(
        0.275, 0.486, 0.700, 0.526
    ),
    "kecamatan": NormalizedBox(
        0.275, 0.526, 0.700, 0.566
    ),
    "agama": NormalizedBox(
        0.275, 0.566, 0.565, 0.606
    ),
    "status_perkawinan": NormalizedBox(
        0.275, 0.606, 0.700, 0.646
    ),
    "pekerjaan": NormalizedBox(
        0.275, 0.646, 0.710, 0.686
    ),
    "kewarganegaraan": NormalizedBox(
        0.275, 0.686, 0.565, 0.726
    ),
    "berlaku_hingga": NormalizedBox(
        0.275, 0.726, 0.650, 0.766
    ),
}


def build_anchor_aligned_value_boxes(
    image_height,
    anchors,
    image_width=None,
):
    height = max(
        int(image_height or 0),
        1,
    )
    width = max(
        int(image_width or 0),
        1,
    )
    anchors = dict(
        anchors or {}
    )
    result = dict(
        KTP_VALUE_BOXES
    )

    for name, box in (
        KTP_VALUE_BOXES.items()
    ):
        anchor = anchors.get(
            name
        )

        if anchor is None:
            continue

        try:
            _x, anchor_y, confidence = (
                anchor
            )
            confidence = float(
                confidence
            )
            anchor_y = float(
                anchor_y
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if confidence < 20.0:
            continue

        center_y = max(
            0.0,
            min(
                1.0,
                anchor_y / height,
            ),
        )
        half_height = min(
            0.022,
            max(
                0.016,
                (
                    box.y2
                    - box.y1
                )
                / 2.0,
            ),
        )

        y1 = max(
            0.0,
            center_y - half_height,
        )
        y2 = min(
            1.0,
            center_y + half_height,
        )

        if y2 - y1 < 0.020:
            continue

        x1 = box.x1
        x2 = box.x2

        if image_width is not None:
            anchor_x_normalized = max(
                0.0,
                min(
                    1.0,
                    float(_x) / width,
                ),
            )
            padding = 0.008
            dynamic_x1 = max(
                0.0,
                anchor_x_normalized
                - padding,
            )

            box_width = (
                box.x2
                - box.x1
            )
            x1 = dynamic_x1
            x2 = min(
                1.0,
                x1 + box_width,
            )

            if x2 - x1 < 0.08:
                x1 = box.x1
                x2 = box.x2

        result[name] = NormalizedBox(
            x1,
            y1,
            x2,
            y2,
        )

    if (
        "jenis_kelamin"
        in anchors
        and "golongan_darah"
        not in anchors
    ):
        gender = result[
            "jenis_kelamin"
        ]
        blood = result[
            "golongan_darah"
        ]
        center_y = (
            gender.y1
            + gender.y2
        ) / 2.0
        half_height = (
            blood.y2
            - blood.y1
        ) / 2.0
        result[
            "golongan_darah"
        ] = NormalizedBox(
            blood.x1,
            max(
                0.0,
                center_y - half_height,
            ),
            blood.x2,
            min(
                1.0,
                center_y + half_height,
            ),
        )

    return result


def normalized_to_pixel_box(
    box,
    image_width,
    image_height,
):
    if not isinstance(
        box,
        NormalizedBox,
    ):
        raise TypeError(
            "box harus berupa NormalizedBox."
        )

    width = int(image_width)
    height = int(image_height)

    if width < 1 or height < 1:
        raise ValueError(
            "Ukuran gambar harus positif."
        )

    x1 = max(
        0,
        min(
            width - 1,
            int(
                floor(
                    box.x1
                    * width
                )
            ),
        ),
    )
    y1 = max(
        0,
        min(
            height - 1,
            int(
                floor(
                    box.y1
                    * height
                )
            ),
        ),
    )
    x2 = max(
        x1 + 1,
        min(
            width,
            int(
                ceil(
                    box.x2
                    * width
                )
            ),
        ),
    )
    y2 = max(
        y1 + 1,
        min(
            height,
            int(
                ceil(
                    box.y2
                    * height
                )
            ),
        ),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def crop_normalized(
    image,
    box,
):
    if (
        not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
    ):
        raise ValueError(
            "image harus berupa numpy "
            "array yang tidak kosong."
        )

    if image.ndim not in (
        2,
        3,
    ):
        raise ValueError(
            "Format image tidak didukung."
        )

    height, width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = (
        normalized_to_pixel_box(
            box,
            image_width=width,
            image_height=height,
        )
    )

    return image[
        y1:y2,
        x1:x2,
    ].copy()
