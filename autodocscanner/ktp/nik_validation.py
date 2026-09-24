from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
import re


PROVINCE_NIK_PREFIXES = {
    "ACEH": "11",
    "SUMATERA UTARA": "12",
    "SUMATERA BARAT": "13",
    "RIAU": "14",
    "JAMBI": "15",
    "SUMATERA SELATAN": "16",
    "BENGKULU": "17",
    "LAMPUNG": "18",
    "KEPULAUAN BANGKA BELITUNG": "19",
    "KEP BANGKA BELITUNG": "19",
    "KEPULAUAN RIAU": "21",
    "KEP RIAU": "21",
    "DKI JAKARTA": "31",
    "JAKARTA": "31",
    "JAWA BARAT": "32",
    "JAWA TENGAH": "33",
    "DI YOGYAKARTA": "34",
    "DAERAH ISTIMEWA YOGYAKARTA": "34",
    "JAWA TIMUR": "35",
    "BANTEN": "36",
    "BALI": "51",
    "NUSA TENGGARA BARAT": "52",
    "NUSA TENGGARA TIMUR": "53",
    "KALIMANTAN BARAT": "61",
    "KALIMANTAN TENGAH": "62",
    "KALIMANTAN SELATAN": "63",
    "KALIMANTAN TIMUR": "64",
    "KALIMANTAN UTARA": "65",
    "SULAWESI UTARA": "71",
    "SULAWESI TENGAH": "72",
    "SULAWESI SELATAN": "73",
    "SULAWESI TENGGARA": "74",
    "GORONTALO": "75",
    "SULAWESI BARAT": "76",
    "MALUKU": "81",
    "MALUKU UTARA": "82",
    "PAPUA": "91",
    "PAPUA BARAT": "92",
    "PAPUA SELATAN": "93",
    "PAPUA TENGAH": "94",
    "PAPUA PEGUNUNGAN": "95",
    "PAPUA BARAT DAYA": "96",
}


@dataclass(frozen=True)
class NikContextResult:
    value: str
    changed: bool
    context_valid: bool
    reason: str = ""


def _normalize_region_name(value):
    text = str(
        value or ""
    ).upper()
    text = re.sub(
        r"[^A-Z ]+",
        " ",
        text,
    )
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if text.startswith(
        "PROVINSI "
    ):
        text = text[
            len("PROVINSI "):
        ].strip()

    return text


def normalize_province_value(
    value,
    threshold=0.72,
):
    region = _normalize_region_name(
        value
    )

    if not region:
        return ""

    if region in PROVINCE_NIK_PREFIXES:
        return (
            "PROVINSI "
            + region
        )

    best_name = ""
    best_score = 0.0

    for name in PROVINCE_NIK_PREFIXES:
        score = SequenceMatcher(
            None,
            region,
            name,
        ).ratio()

        if score > best_score:
            best_score = score
            best_name = name

    if (
        best_name
        and best_score >= float(
            threshold
        )
    ):
        return (
            "PROVINSI "
            + best_name
        )

    return str(
        value or ""
    ).strip().upper()


def birth_segment_from_context(
    birth_date,
    gender,
):
    return _birth_segment(
        birth_date,
        gender,
    )


def _birth_segment(
    birth_date,
    gender,
):
    text = str(
        birth_date or ""
    ).strip()

    try:
        parsed = date.fromisoformat(
            text
        )
    except ValueError:
        return ""

    day = parsed.day
    gender_text = str(
        gender or ""
    ).upper().strip()

    if gender_text == "PEREMPUAN":
        day += 40

    return (
        f"{day:02d}"
        f"{parsed.month:02d}"
        f"{parsed.year % 100:02d}"
    )


def repair_nik_with_context(
    nik,
    province="",
    birth_date="",
    gender="",
):
    digits = "".join(
        char
        for char in str(
            nik or ""
        )
        if char.isdigit()
    )

    if len(digits) != 16:
        return NikContextResult(
            value=digits,
            changed=False,
            context_valid=False,
            reason="nik_length",
        )

    province_name = (
        _normalize_region_name(
            province
        )
    )
    expected_prefix = (
        PROVINCE_NIK_PREFIXES.get(
            province_name,
            "",
        )
    )
    expected_birth = (
        _birth_segment(
            birth_date,
            gender,
        )
    )

    birth_matches = (
        not expected_birth
        or digits[6:12]
        == expected_birth
    )

    if not birth_matches:
        return NikContextResult(
            value=digits,
            changed=False,
            context_valid=False,
            reason="birth_date_mismatch",
        )

    if (
        expected_prefix
        and digits[:2]
        != expected_prefix
    ):
        repaired = (
            expected_prefix
            + digits[2:]
        )

        return NikContextResult(
            value=repaired,
            changed=True,
            context_valid=True,
            reason="province_prefix_repaired",
        )

    return NikContextResult(
        value=digits,
        changed=False,
        context_valid=True,
        reason="context_matches",
    )
