from dataclasses import dataclass

import numpy as np

from autodocscanner.ktp.extraction import (
    extract_ktp_regions,
)
from autodocscanner.ktp.ocr import (
    read_field_ocr,
)
from autodocscanner.ktp.parsing import (
    parse_field,
    parse_ktp_document,
)


CONFIDENCE_THRESHOLD = 55.0


@dataclass(frozen=True)
class TrackedField:
    name: str
    raw_text: str
    value: object
    confidence: float
    used_fallback: bool
    needs_review: bool


@dataclass
class KtpTrackingResult:
    corrected_image: np.ndarray
    face_image: np.ndarray
    fields: dict[str, TrackedField]
    identity: dict
    review_fields: list[str]

    @property
    def ready_for_review(self):
        return bool(
            self.fields
        )


def _parsed_value_complete(
    field_name,
    value,
):
    if field_name == "ttl":
        return bool(
            value.get(
                "tempat_lahir"
            )
            and value.get(
                "tanggal_lahir"
            )
        )

    if field_name == "rt_rw":
        return bool(
            value.get("rt")
            and value.get("rw")
        )

    return bool(
        str(
            value or ""
        ).strip()
    )


def _build_identity(
    fields,
):
    identity = {
        "nik": "",
        "nama": "",
        "tempat_lahir": "",
        "tanggal_lahir": "",
        "jenis_kelamin": "",
        "golongan_darah": "",
        "alamat": "",
        "rt": "",
        "rw": "",
        "kelurahan_desa": "",
        "kecamatan": "",
        "kabupaten_kota": "",
        "provinsi": "",
        "agama": "",
        "status_perkawinan": "",
        "pekerjaan": "",
        "kewarganegaraan": "",
        "berlaku_hingga": "",
    }

    for name, field in (
        fields.items()
    ):
        if name == "ttl":
            identity[
                "tempat_lahir"
            ] = field.value.get(
                "tempat_lahir",
                "",
            )
            identity[
                "tanggal_lahir"
            ] = field.value.get(
                "tanggal_lahir",
                "",
            )
            continue

        if name == "rt_rw":
            identity["rt"] = (
                field.value.get(
                    "rt",
                    "",
                )
            )
            identity["rw"] = (
                field.value.get(
                    "rw",
                    "",
                )
            )
            continue

        if name in identity:
            identity[name] = (
                field.value
            )

    return identity


def extract_tracking_data(
    corrected_image,
    backend=None,
    confidence_threshold=(
        CONFIDENCE_THRESHOLD
    ),
):
    extraction = extract_ktp_regions(
        corrected_image
    )

    fields = {}
    review_fields = []

    for name, crop in (
        extraction.fields.items()
    ):
        if name == "foto":
            continue

        ocr = read_field_ocr(
            crop.image,
            name,
            backend=backend,
            confidence_threshold=(
                confidence_threshold
            ),
        )

        parsed = parse_field(
            name,
            ocr.raw_text,
        )

        complete = (
            _parsed_value_complete(
                name,
                parsed,
            )
        )

        needs_review = (
            ocr.confidence
            < confidence_threshold
            or not complete
        )

        field = TrackedField(
            name=name,
            raw_text=ocr.raw_text,
            value=parsed,
            confidence=ocr.confidence,
            used_fallback=(
                ocr.used_fallback
            ),
            needs_review=(
                needs_review
            ),
        )
        fields[name] = field

        if needs_review:
            review_fields.append(
                name
            )

    return KtpTrackingResult(
        corrected_image=(
            extraction.source_image.copy()
        ),
        face_image=(
            extraction.face_image.copy()
        ),
        fields=fields,
        identity=_build_identity(
            fields
        ),
        review_fields=(
            review_fields
        ),
    )
