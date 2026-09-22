"""KTP data, extraction, OCR, tracking, and storage domain."""

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
from autodocscanner.ktp.ocr import (
    OcrReadResult,
    TesseractBackend,
    preprocess_document,
    preprocess_field,
    read_field_ocr,
)
from autodocscanner.ktp.parsing import (
    clean_text,
    parse_field,
    parse_gender,
    parse_ktp_document,
    parse_nik,
    parse_rt_rw,
    parse_ttl,
)
from autodocscanner.ktp.tracking import (
    KtpTrackingResult,
    TrackedField,
    extract_tracking_data,
)

__all__ = (
    "KTP_FIELD_BOXES",
    "KtpExtractionResult",
    "KtpFieldCrop",
    "KtpTrackingResult",
    "NormalizedBox",
    "OcrReadResult",
    "TesseractBackend",
    "TrackedField",
    "clean_text",
    "crop_normalized",
    "extract_ktp_regions",
    "extract_tracking_data",
    "normalized_to_pixel_box",
    "parse_field",
    "parse_gender",
    "parse_ktp_document",
    "parse_nik",
    "parse_rt_rw",
    "parse_ttl",
    "preprocess_document",
    "preprocess_field",
    "read_field_ocr",
)
