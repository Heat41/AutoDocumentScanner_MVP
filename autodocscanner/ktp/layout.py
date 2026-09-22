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
        0.18, 0.025, 0.82, 0.075
    ),
    "kabupaten_kota": NormalizedBox(
        0.18, 0.075, 0.82, 0.125
    ),
    "nik": NormalizedBox(
        0.220, 0.150, 0.76, 0.220
    ),
    "nama": NormalizedBox(
        0.245, 0.235, 0.74, 0.305
    ),
    "ttl": NormalizedBox(
        0.245, 0.285, 0.74, 0.355
    ),
    "jenis_kelamin": NormalizedBox(
        0.245, 0.335, 0.54, 0.405
    ),
    "golongan_darah": NormalizedBox(
        0.585, 0.345, 0.73, 0.395
    ),
    "alamat": NormalizedBox(
        0.245, 0.385, 0.74, 0.455
    ),
    "rt_rw": NormalizedBox(
        0.245, 0.435, 0.54, 0.505
    ),
    "kelurahan_desa": NormalizedBox(
        0.245, 0.475, 0.74, 0.545
    ),
    "kecamatan": NormalizedBox(
        0.245, 0.515, 0.74, 0.585
    ),
    "agama": NormalizedBox(
        0.245, 0.555, 0.62, 0.625
    ),
    "status_perkawinan": NormalizedBox(
        0.245, 0.595, 0.74, 0.665
    ),
    "pekerjaan": NormalizedBox(
        0.245, 0.635, 0.76, 0.710
    ),
    "kewarganegaraan": NormalizedBox(
        0.245, 0.680, 0.62, 0.750
    ),
    "berlaku_hingga": NormalizedBox(
        0.245, 0.720, 0.70, 0.795
    ),
}


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
