"""KTP data, extraction, and storage domain."""

from autodocscanner.ktp.extraction import (
    KtpExtractionResult,
    KtpFieldCrop,
    extract_ktp_regions,
)
from autodocscanner.ktp.layout import (
    KTP_FIELD_BOXES,
    NormalizedBox,
    crop_normalized,
    normalized_to_pixel_box,
)

__all__ = (
    "KTP_FIELD_BOXES",
    "KtpExtractionResult",
    "KtpFieldCrop",
    "NormalizedBox",
    "crop_normalized",
    "extract_ktp_regions",
    "normalized_to_pixel_box",
)
