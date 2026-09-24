from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from autodocscanner.ktp.anchors import (
    extract_anchor_candidates,
    locate_label_anchors,
)
from autodocscanner.ktp.extraction import (
    extract_ktp_regions,
)
from autodocscanner.ktp.field_detection import (
    AutoFieldDetector,
    FieldDetection,
    best_detection_by_class,
    crop_detection,
    crop_detection_padded,
)
from autodocscanner.ktp.layout import (
    normalize_ktp_for_tracking,
)
from autodocscanner.ktp.nik_validation import (
    PROVINCE_NIK_PREFIXES,
    birth_segment_from_context,
    normalize_province_value,
    repair_nik_with_context,
)
from autodocscanner.ktp.ocr import (
    OcrReadResult,
    TesseractBackend,
    read_field_ocr_candidates,
    read_numeric_fragment,
)
from autodocscanner.ktp.parsing import (
    parse_field,
    parse_ktp_document,
)
from autodocscanner.ktp.registration import (
    register_ktp_to_template,
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
    debug_tracking_image: np.ndarray | None = None
    detections: list | None = None

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
    "golongan_darah": {
        "A",
        "B",
        "AB",
        "O",
        "-",
    },
    "status_perkawinan": {
        "BELUM KAWIN",
        "KAWIN",
        "CERAI HIDUP",
        "CERAI MATI",
    },
}


_FUZZY_ENUM_FIELDS = {
    "agama",
    "jenis_kelamin",
    "status_perkawinan",
}


def _normalize_enum_candidate(
    field_name,
    value,
):
    allowed = _ENUM_VALUES.get(
        field_name
    )

    text = str(
        value or ""
    ).strip().upper()

    if (
        allowed is None
        or text in allowed
        or field_name
        not in _FUZZY_ENUM_FIELDS
    ):
        return text

    compact = "".join(
        char
        for char in text
        if char.isalnum()
    )

    if not compact:
        return text

    best_value = text
    best_score = 0.0

    for candidate in allowed:
        candidate_compact = "".join(
            char
            for char in candidate
            if char.isalnum()
        )

        score = SequenceMatcher(
            None,
            compact,
            candidate_compact,
        ).ratio()

        if score > best_score:
            best_score = score
            best_value = candidate

    if best_score >= 0.72:
        return best_value

    return text


def _ocr_detection_for_field(
    field_name,
    detection,
    detection_map,
    image_width,
):
    if not isinstance(
        detection,
        FieldDetection,
    ):
        return detection

    extend_right_fields = {
        "provinsi",
        "kabupaten_kota",
        "nama",
        "ttl",
        "alamat",
        "kelurahan_desa",
        "kecamatan",
        "agama",
        "status_perkawinan",
        "pekerjaan",
        "kewarganegaraan",
        "berlaku_hingga",
    }

    if field_name not in extend_right_fields:
        return detection

    photo = detection_map.get(
        "foto"
    )

    safe_right = (
        int(
            photo.bbox[0]
        )
        - 8
        if isinstance(
            photo,
            FieldDetection,
        )
        else int(
            image_width
        )
    )

    x1, y1, x2, y2 = (
        detection.bbox
    )
    expanded_x2 = max(
        int(x2),
        min(
            int(image_width),
            safe_right,
        ),
    )

    return FieldDetection(
        class_name=detection.class_name,
        bbox=(
            int(x1),
            int(y1),
            expanded_x2,
            int(y2),
        ),
        confidence=detection.confidence,
        source=detection.source,
    )


def _recover_nik_from_fragments(
    nik_image,
    backend,
    province,
    birth_date,
    gender,
):
    if (
        nik_image is None
        or not isinstance(
            nik_image,
            np.ndarray,
        )
        or nik_image.size == 0
    ):
        return None

    province_text = normalize_province_value(
        province
    )
    province_name = (
        province_text
        .replace(
            "PROVINSI ",
            "",
            1,
        )
        .strip()
    )
    prefix = PROVINCE_NIK_PREFIXES.get(
        province_name,
        "",
    )
    birth_segment = (
        birth_segment_from_context(
            birth_date,
            gender,
        )
    )

    if (
        len(prefix) != 2
        or len(birth_segment) != 6
    ):
        return None

    height, width = (
        nik_image.shape[:2]
    )

    region_crop = nik_image[
        :,
        max(
            0,
            int(
                round(
                    width * 0.10
                )
            ),
        ):
        min(
            width,
            int(
                round(
                    width * 0.42
                )
            ),
        ),
    ]
    serial_crop = nik_image[
        :,
        max(
            0,
            int(
                round(
                    width * 0.72
                )
            ),
        ):
        width,
    ]

    region = read_numeric_fragment(
        region_crop,
        expected_length=4,
        backend=backend,
    )
    serial = read_numeric_fragment(
        serial_crop,
        expected_length=4,
        backend=backend,
    )

    region_digits = "".join(
        char
        for char in region.raw_text
        if char.isdigit()
    )
    serial_digits = "".join(
        char
        for char in serial.raw_text
        if char.isdigit()
    )

    if (
        len(region_digits) != 4
        or len(serial_digits) != 4
    ):
        return None

    value = (
        prefix
        + region_digits
        + birth_segment
        + serial_digits
    )

    return OcrReadResult(
        raw_text=value,
        confidence=min(
            float(region.confidence),
            float(serial.confidence),
        ),
        used_fallback=True,
    )


def _enum_value_is_valid(
    field_name,
    parsed,
):
    allowed = _ENUM_VALUES.get(
        field_name
    )

    if allowed is None:
        return True

    normalized = (
        _normalize_enum_candidate(
            field_name,
            parsed,
        )
    )

    return normalized in allowed


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

    if field_name == "berlaku_hingga":
        value = str(
            parsed or ""
        ).strip().upper()
        if value == "SEUMUR HIDUP":
            return True

    if not _enum_value_is_valid(
        field_name,
        parsed,
    ):
        return False

    if field_name in _ENUM_VALUES:
        return True

    return _looks_clean_text(
        raw_text
    )


def _nik_consensus_candidate(
    candidates,
):
    valid = []

    for candidate in candidates:
        digits = parse_field(
            "nik",
            candidate.raw_text,
        )

        if len(digits) == 16:
            valid.append(
                (
                    digits,
                    candidate,
                )
            )

    if not valid:
        return None

    if len(valid) == 1:
        return valid[0][1]

    consensus = []

    for index in range(16):
        counts = {}

        for digits, _candidate in valid:
            digit = digits[index]
            counts[digit] = (
                counts.get(
                    digit,
                    0,
                )
                + 1
            )

        highest = max(
            counts.values()
        )
        tied = [
            digit
            for digit, count in counts.items()
            if count == highest
        ]

        if len(tied) == 1:
            consensus.append(
                tied[0]
            )
            continue

        best = max(
            (
                pair
                for pair in valid
                if pair[0][index] in tied
            ),
            key=lambda pair: (
                pair[1].confidence
            ),
        )
        consensus.append(
            best[0][index]
        )

    return OcrReadResult(
        raw_text="".join(
            consensus
        ),
        confidence=max(
            candidate.confidence
            for _digits, candidate in valid
        ),
        used_fallback=any(
            candidate.used_fallback
            for _digits, candidate in valid
        ),
    )


def _best_valid_ocr_candidate(
    field_name,
    candidates,
):
    if field_name == "nik":
        consensus = _nik_consensus_candidate(
            candidates
        )

        if consensus is not None:
            return consensus

    valid = [
        candidate
        for candidate in candidates
        if _candidate_is_valid(
            field_name,
            candidate.raw_text,
        )
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda candidate: (
            candidate.confidence,
            len(
                candidate.raw_text
            ),
        ),
    )


def extract_tracking_data(
    corrected_image,
    backend=None,
    confidence_threshold=(
        CONFIDENCE_THRESHOLD
    ),
    detector=None,
):
    tracking_image = (
        normalize_ktp_for_tracking(
            corrected_image
        )
    )

    backend = (
        backend
        if backend is not None
        else TesseractBackend()
    )

    anchor_candidates = {}
    label_anchors = {}
    document_candidates = {}
    document_confidence = 0.0
    layout_words = []

    if hasattr(
        backend,
        "read_layout",
    ):
        try:
            layout_words = (
                backend.read_layout(
                    tracking_image
                )
            )

            # Auto Perspective sudah menghasilkan KTP canonical.
            # Jangan warp ulang gambar berdasarkan OCR anchor karena sedikit
            # noise pada posisi label dapat membuat seluruh kartu bergeser
            # atau terkompresi. Anchor hanya dipakai untuk mengoreksi ROI.
            label_anchors = (
                locate_label_anchors(
                    layout_words
                )
            )
            anchor_candidates = (
                extract_anchor_candidates(
                    layout_words
                )
            )
        except Exception:
            anchor_candidates = {}

    extraction = extract_ktp_regions(
        tracking_image
    )

    detector = (
        detector
        if detector is not None
        else AutoFieldDetector()
    )
    detections = detector.detect(
        tracking_image,
        anchors=label_anchors,
    )
    detection_map = (
        best_detection_by_class(
            detections
        )
    )

    if hasattr(
        backend,
        "read_document",
    ):
        try:
            (
                document_text,
                document_confidence,
            ) = backend.read_document(
                tracking_image
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
    nik_source_image = None

    for name in (
        extraction.fields
    ):
        if name == "foto":
            continue

        detection = detection_map.get(
            name
        )

        if detection is None:
            continue

        if name == "nik":
            nik_source_image = crop_detection(
                tracking_image,
                detection,
            )

        ocr_detection = _ocr_detection_for_field(
            name,
            detection,
            detection_map,
            tracking_image.shape[1],
        )

        value_image = crop_detection_padded(
            tracking_image,
            ocr_detection,
        )

        ocr_candidates = (
            read_field_ocr_candidates(
                value_image,
                name,
                backend=backend,
            )
        )

        best_ocr = (
            _best_valid_ocr_candidate(
                name,
                ocr_candidates,
            )
        )

        if best_ocr is None:
            raw_text = ""
            confidence = 0.0
            used_fallback = False
        else:
            raw_text = (
                best_ocr.raw_text
            )
            confidence = (
                best_ocr.confidence
            )
            used_fallback = (
                best_ocr.used_fallback
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

        bbox_candidate_valid = (
            _candidate_is_valid(
                name,
                raw_text,
            )
        )

        if (
            not bbox_candidate_valid
            and anchor_raw
            and _candidate_is_valid(
                name,
                anchor_raw,
            )
        ):
            raw_text = anchor_raw
            confidence = float(
                anchor_confidence
                or 0.0
            )
            used_fallback = True

        elif (
            not _candidate_is_valid(
                name,
                raw_text,
            )
            and document_raw
            and _candidate_is_valid(
                name,
                document_raw,
            )
        ):
            raw_text = document_raw
            confidence = float(
                document_confidence
                or 0.0
            )
            used_fallback = True

        parsed = parse_field(
            name,
            raw_text,
        )

        if name in _ENUM_VALUES:
            parsed = (
                _normalize_enum_candidate(
                    name,
                    parsed,
                )
            )

        if name == "provinsi":
            parsed = (
                normalize_province_value(
                    parsed
                )
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

    current_nik = fields.get(
        "nik"
    )
    ttl_for_nik = fields.get(
        "ttl"
    )
    province_for_nik = fields.get(
        "provinsi"
    )
    gender_for_nik = fields.get(
        "jenis_kelamin"
    )

    if (
        current_nik is not None
        and not str(
            current_nik.value or ""
        ).strip()
        and isinstance(
            backend,
            TesseractBackend,
        )
    ):
        ttl_value = (
            ttl_for_nik.value
            if ttl_for_nik is not None
            and isinstance(
                ttl_for_nik.value,
                dict,
            )
            else {}
        )

        recovered_nik = (
            _recover_nik_from_fragments(
                nik_source_image,
                backend,
                province=(
                    province_for_nik.value
                    if province_for_nik is not None
                    else ""
                ),
                birth_date=(
                    ttl_value.get(
                        "tanggal_lahir",
                        "",
                    )
                ),
                gender=(
                    gender_for_nik.value
                    if gender_for_nik is not None
                    else ""
                ),
            )
        )

        if recovered_nik is not None:
            fields["nik"] = TrackedField(
                name="nik",
                raw_text=recovered_nik.raw_text,
                value=recovered_nik.raw_text,
                confidence=(
                    recovered_nik.confidence
                ),
                used_fallback=True,
                needs_review=True,
            )

            if "nik" not in review_fields:
                review_fields.append(
                    "nik"
                )

    nik_field = fields.get(
        "nik"
    )
    ttl_field = fields.get(
        "ttl"
    )
    province_field = fields.get(
        "provinsi"
    )
    gender_field = fields.get(
        "jenis_kelamin"
    )

    if nik_field is not None:
        ttl_value = (
            ttl_field.value
            if ttl_field is not None
            and isinstance(
                ttl_field.value,
                dict,
            )
            else {}
        )
        nik_context = (
            repair_nik_with_context(
                nik_field.value,
                province=(
                    province_field.value
                    if province_field is not None
                    else ""
                ),
                birth_date=(
                    ttl_value.get(
                        "tanggal_lahir",
                        "",
                    )
                ),
                gender=(
                    gender_field.value
                    if gender_field is not None
                    else ""
                ),
            )
        )

        if nik_context.changed:
            fields["nik"] = TrackedField(
                name="nik",
                raw_text=nik_context.value,
                value=nik_context.value,
                confidence=nik_field.confidence,
                used_fallback=True,
                needs_review=True,
            )

            if "nik" not in review_fields:
                review_fields.append(
                    "nik"
                )

    return KtpTrackingResult(
        corrected_image=(
            corrected_image.copy()
        ),
        face_image=(
            crop_detection(
                tracking_image,
                detection_map["foto"],
            )
            if "foto" in detection_map
            else extraction.face_image.copy()
        ),
        fields=fields,
        identity=_build_identity(
            fields
        ),
        review_fields=(
            review_fields
        ),
        debug_tracking_image=(
            tracking_image.copy()
        ),
        detections=list(
            detection_map.values()
        ),
    )
