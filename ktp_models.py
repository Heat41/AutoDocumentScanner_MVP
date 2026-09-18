from dataclasses import asdict, dataclass, fields


REQUIRED_IDENTITY_FIELDS = (
    "nama",
    "tempat_lahir",
    "tanggal_lahir",
    "jenis_kelamin",
    "alamat",
    "rt",
    "rw",
    "kelurahan_desa",
    "kecamatan",
    "kabupaten_kota",
    "provinsi",
    "agama",
    "status_perkawinan",
    "pekerjaan",
    "kewarganegaraan",
)


def normalize_nik(value):
    nik = str(value or "").strip()
    if len(nik) != 16 or not nik.isdigit():
        raise ValueError(
            "NIK harus terdiri dari tepat 16 digit."
        )
    return nik


def _clean_optional(value):
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def _clean_required(value):
    if value is None:
        return ""
    return str(value).strip()


@dataclass
class KtpIdentityData:
    nama: str = ""
    tempat_lahir: str = ""
    tanggal_lahir: str = ""
    jenis_kelamin: str = ""
    alamat: str = ""
    rt: str = ""
    rw: str = ""
    kelurahan_desa: str = ""
    kecamatan: str = ""
    kabupaten_kota: str = ""
    provinsi: str = ""
    agama: str = ""
    status_perkawinan: str = ""
    pekerjaan: str = ""
    kewarganegaraan: str = ""
    golongan_darah: str | None = None
    berlaku_hingga: str | None = None
    jenis_wilayah: str | None = None

    @classmethod
    def from_mapping(cls, mapping):
        mapping = dict(mapping or {})
        kwargs = {}

        for item in fields(cls):
            value = mapping.get(item.name)

            if (
                item.name
                in REQUIRED_IDENTITY_FIELDS
            ):
                kwargs[item.name] = (
                    _clean_required(value)
                )
            else:
                kwargs[item.name] = (
                    _clean_optional(value)
                )

        return cls(**kwargs)

    def validate_for_record(self):
        missing = [
            name
            for name in REQUIRED_IDENTITY_FIELDS
            if not _clean_required(
                getattr(self, name)
            )
        ]

        if missing:
            raise ValueError(
                "Field identitas wajib belum lengkap: "
                + ", ".join(missing)
            )

    def as_db_dict(self):
        return asdict(self)
