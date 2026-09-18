from dataclasses import dataclass

import numpy as np

from autodocscanner.ktp.layout import (
    KTP_FIELD_BOXES,
    crop_normalized,
    normalized_to_pixel_box,
)


@dataclass
class KtpFieldCrop:
    name: str
    image: np.ndarray
    pixel_box: tuple[int, int, int, int]


@dataclass
class KtpExtractionResult:
    source_image: np.ndarray
    fields: dict[str, KtpFieldCrop]

    @property
    def face_image(self):
        return self.fields[
            "foto"
        ].image


def _validate_source_image(image):
    if (
        not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
    ):
        raise ValueError(
            "image KTP harus berupa numpy "
            "array yang tidak kosong."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            "image KTP harus bertipe uint8."
        )

    if image.ndim == 2:
        return image

    if (
        image.ndim == 3
        and image.shape[2]
        in (1, 3, 4)
    ):
        return image

    raise ValueError(
        "Format image KTP tidak didukung."
    )


def extract_ktp_regions(image):
    image = _validate_source_image(
        image
    )

    height, width = (
        image.shape[:2]
    )

    fields = {}

    for name, box in (
        KTP_FIELD_BOXES.items()
    ):
        fields[name] = (
            KtpFieldCrop(
                name=name,
                image=(
                    crop_normalized(
                        image,
                        box,
                    )
                ),
                pixel_box=(
                    normalized_to_pixel_box(
                        box,
                        image_width=width,
                        image_height=height,
                    )
                ),
            )
        )

    return KtpExtractionResult(
        source_image=image.copy(),
        fields=fields,
    )
