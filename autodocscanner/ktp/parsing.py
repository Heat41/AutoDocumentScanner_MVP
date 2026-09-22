import re
from datetime import date


_WHITESPACE_RE = re.compile(r"\s+")
_DATE_RE = re.compile(
    r"(?P<day>\d{1,2})[\-/\.](?P<month>\d{1,2})[\-/\.](?P<year>\d{4})"
)

_NIK_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }
)


def clean_text(value):
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _strip_leading_label(text, labels):
    value = clean_text(text)

    for label in labels:
        pattern = re.compile(
            rf"^\s*{re.escape(label)}\s*[:\-]?\s*",
            flags=re.IGNORECASE,
        )
        value = pattern.sub("", value, count=1)

    return value.strip()


def parse_nik(value):
    text = _strip_leading_label(
        clean_text(value).upper(),
        ("NIK",),
    )
    repaired = text.translate(_NIK_TRANSLATION)
    digits = "".join(
        character
        for character in repaired
        if character.isdigit()
    )
    return digits if len(digits) == 16 else ""


def _normalize_date(day, month, year):
    try:
        parsed = date(
            int(year),
            int(month),
            int(day),
        )
    except ValueError:
        return ""

    return parsed.isoformat()


def parse_ttl(value):
    text = _strip_leading_label(
        value,
        (
            "TEMPAT/TGL LAHIR",
            "TEMPAT TGL LAHIR",
            "TEMPAT/TANGGAL LAHIR",
            "TEMPAT TANGGAL LAHIR",
        ),
    )
    match = _DATE_RE.search(text)

    if match is None:
        return {
            "tempat_lahir": text.strip(" ,-").upper(),
            "tanggal_lahir": "",
        }

    tempat = text[:match.start()].strip(" ,-").upper()

    return {
        "tempat_lahir": tempat,
        "tanggal_lahir": _normalize_date(
            match.group("day"),
            match.group("month"),
            match.group("year"),
        ),
    }


def parse_rt_rw(value):
    text = _strip_leading_label(
        value,
        ("RT/RW", "RT RW"),
    ).upper()

    digits = re.findall(r"\d{1,3}", text)

    if len(digits) < 2:
        return {
            "rt": "",
            "rw": "",
        }

    return {
        "rt": digits[0].zfill(3),
        "rw": digits[1].zfill(3),
    }


def parse_gender(value):
    text = _strip_leading_label(
        value,
        ("JENIS KELAMIN",),
    ).upper()
    compact = re.sub(r"[^A-Z]", "", text)

    if compact.startswith("LAKILAKI"):
        return "LAKI-LAKI"

    if compact.startswith("PEREMPUAN"):
        return "PEREMPUAN"

    return clean_text(text).upper()


def _generic_field(field_name, value):
    label = field_name.replace("_", " ").upper()
    text = _strip_leading_label(
        value,
        (label,),
    )
    return clean_text(text).upper()


def parse_field(field_name, value):
    if field_name == "nik":
        return parse_nik(value)

    if field_name == "ttl":
        return parse_ttl(value)

    if field_name == "rt_rw":
        return parse_rt_rw(value)

    if field_name == "jenis_kelamin":
        return parse_gender(value)

    return _generic_field(field_name, value)
