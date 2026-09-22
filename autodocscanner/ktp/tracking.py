from dataclasses import dataclass

import numpy as np

from autodocscanner.ktp.anchors import (
    extract_anchor_candidates,
)
from autodocscanner.ktp.extraction import (
    extract_ktp_regions,
)
from autodocscanner.ktp.layout import (
    KTP_VALUE_BOXES,
    crop_normalized,
)
from autodocscanner.ktp.ocr import (
    TesseractBackend,
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


def _looks_clean_text(value):
    text = str(
        value or ""
    ).strip()

    if not text:
        return False

    visible = [
        char
        for char in text
        if not char.isspace()
    ]

    if not visible:
        return False

    alnum_count = sum(
        1
        for char in visible
        if char.isalnum()
    )
    punctuation_count = sum(
        1
        for char in visible
        if not char.isalnum()
        and char not in (".", ",", "/", "-", "'")
    )

    alpha_words = [
        word
        for word in text.replace(
            "/",
            " ",
        ).replace(
            "-",
            " ",
        ).split()
        if any(
            char.isalpha()
            for char in word
        )
    ]

    return (
        alnum_count >= 3
        and (
            alnum_count
            / max(
                len(visible),
                1,
            )
        )
        >= 0.72
        and punctuation_count
        <= max(
            2,
            len(visible) // 12,
        )
        and bool(
            alpha_words
        )
    )


_ENUM_VALUES = {
    "agama": {
        "ISLAM",
        "KRISTEN",
        "KATOLIK",
        "HINDU",
        "BUDDHA",
        "KHONGHUCU",
    },
    "jenis_kelamin": {
        "LAKI-LAKI",
        "PEREMPUAN",
    },
    "kewarganegaraan": {
        "WNI",
        "WNA",
    },
}


def _enum_value_is_valid(
    field_name,
    parsed,
):
    allowed = _ENUM_VALUES.get(
        field_name
    )

    if allowed is None:
        return True

    return str(
        parsed or ""
    ).strip().upper() in allowed


def _candidate_is_valid(
    field_name,
    raw_text,
):
    parsed = parse_field(
        field_name,
        raw_text,
    )

    if not _parsed_value_complete(
        field_name,
        parsed,
    ):
        return False

    if field_name in (
        "nik",
        "ttl",
        "rt_rw",
    ):
        return True

    if not _enum_value_is_valid(
        field_name,
        parsed,
    ):
        return False

    return _looks_clean_text(
        raw_text
    )


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

    backend = (
        backend
        if backend is not None
        else TesseractBackend()
    )

    anchor_candidates = {}
    document_candidates = {}
    document_confidence = 0.0

    if hasattr(
        backend,
        "read_layout",
    ):
        try:
            anchor_candidates = (
                extract_anchor_candidates(
                    backend.read_layout(
                        corrected_image
                    )
                )
            )
        except Exception:
            anchor_candidates = {}

    if hasattr(
        backend,
        "read_document",
    ):
        try:
            (
                document_text,
                document_confidence,
            ) = backend.read_document(
                corrected_image
            )
            document_candidates = (
                parse_ktp_document(
                    document_text
                )
            )
        except Exception:
            document_candidates = {}
            document_confidence = 0.0

    fields = {}
    review_fields = []

    for name, crop in (
        extraction.fields.items()
    ):
        if name == "foto":
            continue

        value_image = crop_normalized(
            corrected_image,
            KTP_VALUE_BOXES[
                name
            ],
        )

        ocr = read_field_ocr(
            value_image,
            name,
            backend=backend,
            confidence_threshold=(
                confidence_threshold
            ),
        )

        raw_text = ocr.raw_text
        confidence = ocr.confidence
        used_fallback = (
            ocr.used_fallback
        )

        anchor_raw = ""
        anchor_confidence = 0.0

        if name in anchor_candidates:
            (
                anchor_raw,
                anchor_confidence,
            ) = anchor_candidates[
                name
            ]

        document_raw = (
            document_candidates.get(
                name,
                "",
            )
        )

        if (
            anchor_raw
            and _candidate_is_valid(
                name,
                anchor_raw,
            )
        ):
            raw_text = anchor_raw
            confidence = max(
                confidence,
                float(
                    anchor_confidence
                    or 0.0
                ),
            )
            used_fallback = False

        elif (
            document_raw
            and _candidate_is_valid(
                name,
                document_raw,
            )
            and (
                not _candidate_is_valid(
                    name,
                    raw_text,
                )
                or document_confidence
                >= confidence - 12.0
            )
        ):
            raw_text = document_raw
            confidence = max(
                confidence,
                float(
                    document_confidence
                    or 0.0
                ),
            )
            used_fallback = False

        parsed = parse_field(
            name,
            raw_text,
        )

        candidate_valid = (
            _candidate_is_valid(
                name,
                raw_text,
            )
        )

        if not candidate_valid:
            raw_text = ""
            parsed = parse_field(
                name,
                "",
            )
            confidence = 0.0
            used_fallback = False

        complete = (
            _parsed_value_complete(
                name,
                parsed,
            )
        )

        needs_review = (
            confidence
            < confidence_threshold
            or not complete
            or not candidate_valid
        )

        field = TrackedField(
            name=name,
            raw_text=raw_text,
            value=parsed,
            confidence=confidence,
            used_fallback=(
                used_fallback
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
