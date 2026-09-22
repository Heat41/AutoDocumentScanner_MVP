from difflib import SequenceMatcher
import re


LABEL_ALIASES = {
    "nik": (
        "NIK",
    ),
    "nama": (
        "NAMA",
    ),
    "ttl": (
        "TEMPAT/TGL LAHIR",
        "TEMPAT TGL LAHIR",
        "TEMPAT/TANGGAL LAHIR",
        "TEMPAT TANGGAL LAHIR",
    ),
    "jenis_kelamin": (
        "JENIS KELAMIN",
    ),
    "golongan_darah": (
        "GOL DARAH",
        "GOL. DARAH",
    ),
    "alamat": (
        "ALAMAT",
    ),
    "rt_rw": (
        "RT/RW",
        "RT RW",
    ),
    "kelurahan_desa": (
        "KEL/DESA",
        "KEL DESA",
        "KELURAHAN/DESA",
        "KELURAHAN DESA",
    ),
    "kecamatan": (
        "KECAMATAN",
    ),
    "agama": (
        "AGAMA",
    ),
    "status_perkawinan": (
        "STATUS PERKAWINAN",
    ),
    "pekerjaan": (
        "PEKERJAAN",
    ),
    "kewarganegaraan": (
        "KEWARGANEGARAAN",
    ),
    "berlaku_hingga": (
        "BERLAKU HINGGA",
    ),
}


def _normalize_label(value):
    text = str(
        value or ""
    ).upper()

    text = re.sub(
        r"[^A-Z0-9/]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _line_groups(words):
    groups = {}

    for item in words:
        groups.setdefault(
            item.line_key,
            [],
        ).append(
            item
        )

    result = []

    for line_words in groups.values():
        ordered = sorted(
            line_words,
            key=lambda item: item.left,
        )
        result.append(
            ordered
        )

    result.sort(
        key=lambda items: (
            min(
                item.top
                for item in items
            ),
            min(
                item.left
                for item in items
            ),
        )
    )

    return result


def _mean_confidence(words):
    if not words:
        return 0.0

    return sum(
        float(
            item.confidence
        )
        for item in words
    ) / len(words)


def _header_candidate(
    line_words,
):
    text = " ".join(
        item.text
        for item in line_words
    ).strip()

    normalized = _normalize_label(
        text
    )

    if normalized.startswith(
        "PROVINSI "
    ):
        return (
            "provinsi",
            text.upper(),
            _mean_confidence(
                line_words
            ),
        )

    if (
        normalized.startswith(
            "KOTA "
        )
        or normalized.startswith(
            "KABUPATEN "
        )
    ):
        return (
            "kabupaten_kota",
            text.upper(),
            _mean_confidence(
                line_words
            ),
        )

    return None


def _best_label_match(
    line_words,
):
    token_texts = [
        item.text
        for item in line_words
    ]

    best = None

    for field_name, aliases in (
        LABEL_ALIASES.items()
    ):
        for alias in aliases:
            alias_words = (
                _normalize_label(
                    alias
                ).split()
            )

            max_tokens = min(
                len(token_texts),
                len(alias_words) + 1,
            )

            for count in range(
                max(
                    1,
                    len(alias_words) - 1,
                ),
                max_tokens + 1,
            ):
                prefix = _normalize_label(
                    " ".join(
                        token_texts[
                            :count
                        ]
                    )
                )

                score = SequenceMatcher(
                    None,
                    prefix,
                    _normalize_label(
                        alias
                    ),
                ).ratio()

                candidate = (
                    score,
                    field_name,
                    count,
                )

                if (
                    best is None
                    or candidate[0]
                    > best[0]
                ):
                    best = candidate

    if (
        best is None
        or best[0] < 0.72
    ):
        return None

    return best


def _value_words_after_label(
    line_words,
    consumed_count,
):
    remaining = list(
        line_words[
            consumed_count:
        ]
    )

    while (
        remaining
        and _normalize_label(
            remaining[0].text
        )
        in ("",)
    ):
        remaining.pop(0)

    while (
        remaining
        and str(
            remaining[0].text
        ).strip()
        in (
            ":",
            "-",
            "=",
        )
    ):
        remaining.pop(0)

    return remaining


def extract_anchor_candidates(
    words,
):
    result = {}

    for line_words in (
        _line_groups(
            words
        )
    ):
        header = _header_candidate(
            line_words
        )

        if header is not None:
            (
                field_name,
                value,
                confidence,
            ) = header

            result.setdefault(
                field_name,
                (
                    value,
                    confidence,
                ),
            )
            continue

        match = _best_label_match(
            line_words
        )

        if match is None:
            continue

        (
            _score,
            field_name,
            consumed_count,
        ) = match

        value_words = (
            _value_words_after_label(
                line_words,
                consumed_count,
            )
        )

        if not value_words:
            continue

        value = " ".join(
            item.text
            for item in value_words
        ).strip()

        if not value:
            continue

        confidence = (
            _mean_confidence(
                value_words
            )
        )

        previous = result.get(
            field_name
        )

        if (
            previous is None
            or confidence
            > previous[1]
        ):
            result[
                field_name
            ] = (
                value,
                confidence,
            )

    return result



def locate_label_anchors(
    words,
):
    result = {}

    for line_words in (
        _line_groups(
            words
        )
    ):
        match = _best_label_match(
            line_words
        )

        if match is None:
            continue

        (
            score,
            field_name,
            consumed_count,
        ) = match

        label_words = line_words[
            :consumed_count
        ]

        if not label_words:
            continue

        left = min(
            item.left
            for item in label_words
        )
        top = min(
            item.top
            for item in label_words
        )
        right = max(
            item.right
            for item in label_words
        )
        bottom = max(
            item.bottom
            for item in label_words
        )

        confidence = (
            _mean_confidence(
                label_words
            )
            * score
        )

        previous = result.get(
            field_name
        )

        candidate = (
            (left + right) / 2.0,
            (top + bottom) / 2.0,
            float(confidence),
        )

        if (
            previous is None
            or candidate[2]
            > previous[2]
        ):
            result[
                field_name
            ] = candidate

    return result
