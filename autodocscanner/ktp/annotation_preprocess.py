from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from autodocscanner.core.scanner import (
    AutoDocumentScanner,
)
from autodocscanner.ktp.layout import (
    KTP_CANONICAL_HEIGHT,
    KTP_CANONICAL_WIDTH,
    normalize_ktp_for_tracking,
)


@dataclass(frozen=True)
class AnnotationImageResult:
    image: np.ndarray
    source_mode: str


def _is_canonical_image(
    image,
):
    if (
        not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
    ):
        return False

    height, width = (
        image.shape[:2]
    )

    return (
        width
        == KTP_CANONICAL_WIDTH
        and height
        == KTP_CANONICAL_HEIGHT
    )


def prepare_annotation_image(
    image_path,
    scanner=None,
):
    image_path = Path(
        image_path
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"File tidak ditemukan: {image_path}"
        )

    source = cv2.imread(
        str(image_path)
    )

    if source is None:
        raise ValueError(
            f"Gambar tidak dapat dibaca: {image_path}"
        )

    if _is_canonical_image(
        source
    ):
        return AnnotationImageResult(
            image=source.copy(),
            source_mode="canonical",
        )

    scanner = (
        scanner
        if scanner is not None
        else AutoDocumentScanner()
    )

    corrected, _corners = (
        scanner.scan(
            image_path,
            output_path=None,
            mode="ktp",
            output_mode="color",
        )
    )

    canonical = (
        normalize_ktp_for_tracking(
            corrected
        )
    )

    return AnnotationImageResult(
        image=canonical,
        source_mode="auto_perspective",
    )
