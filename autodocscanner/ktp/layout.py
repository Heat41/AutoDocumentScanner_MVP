from dataclasses import dataclass
from math import ceil, floor

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
        0.18, 0.030, 0.78, 0.105
    ),
    "kabupaten_kota": NormalizedBox(
        0.16, 0.095, 0.80, 0.165
    ),
    "nik": NormalizedBox(
        0.30, 0.165, 0.73, 0.255
    ),
    "nama": NormalizedBox(
        0.30, 0.255, 0.72, 0.325
    ),
    "ttl": NormalizedBox(
        0.30, 0.315, 0.73, 0.385
    ),
    "jenis_kelamin": NormalizedBox(
        0.30, 0.375, 0.50, 0.445
    ),
    "golongan_darah": NormalizedBox(
        0.58, 0.375, 0.72, 0.445
    ),
    "alamat": NormalizedBox(
        0.30, 0.435, 0.72, 0.510
    ),
    "rt_rw": NormalizedBox(
        0.30, 0.495, 0.50, 0.565
    ),
    "kelurahan_desa": NormalizedBox(
        0.30, 0.545, 0.72, 0.615
    ),
    "kecamatan": NormalizedBox(
        0.30, 0.595, 0.72, 0.665
    ),
    "agama": NormalizedBox(
        0.30, 0.650, 0.58, 0.720
    ),
    "status_perkawinan": NormalizedBox(
        0.30, 0.700, 0.72, 0.770
    ),
    "pekerjaan": NormalizedBox(
        0.30, 0.750, 0.72, 0.825
    ),
    "kewarganegaraan": NormalizedBox(
        0.30, 0.805, 0.72, 0.875
    ),
    "berlaku_hingga": NormalizedBox(
        0.30, 0.855, 0.72, 0.930
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
