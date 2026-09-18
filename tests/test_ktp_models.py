import unittest

from ktp_models import KtpIdentityData, normalize_nik


VALID = {
    "nama": "BUDI SANTOSO",
    "tempat_lahir": "PONTIANAK",
    "tanggal_lahir": "1990-01-02",
    "jenis_kelamin": "LAKI-LAKI",
    "alamat": "JL. CONTOH NO. 1",
    "rt": "001",
    "rw": "002",
    "kelurahan_desa": "SUNGAI BANGKONG",
    "kecamatan": "PONTIANAK KOTA",
    "kabupaten_kota": "KOTA PONTIANAK",
    "provinsi": "KALIMANTAN BARAT",
    "agama": "ISLAM",
    "status_perkawinan": "KAWIN",
    "pekerjaan": "KARYAWAN SWASTA",
    "kewarganegaraan": "WNI",
}


class TestKtpModels(unittest.TestCase):
    def test_normalize_nik_accepts_16_digits_and_trims_space(self):
        self.assertEqual(
            normalize_nik(" 6171010203900001 "),
            "6171010203900001",
        )

    def test_normalize_nik_rejects_non_digit_or_wrong_length(self):
        for value in (
            "617101020390000",
            "61710102039000012",
            "617101020390000A",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_nik(value)

    def test_required_identity_must_be_complete(self):
        data = dict(VALID)
        data["nama"] = " "
        identity = KtpIdentityData.from_mapping(data)
        with self.assertRaises(ValueError):
            identity.validate_for_record()

    def test_optional_fields_may_be_empty(self):
        identity = KtpIdentityData.from_mapping(VALID)
        identity.validate_for_record()
        self.assertIsNone(identity.golongan_darah)
        self.assertIsNone(identity.berlaku_hingga)
        self.assertIsNone(identity.jenis_wilayah)

    def test_mapping_round_trip_uses_database_field_names(self):
        data = dict(VALID)
        data.update(
            {
                "golongan_darah": "O",
                "berlaku_hingga": "SEUMUR HIDUP",
                "jenis_wilayah": "KELURAHAN",
            }
        )
        identity = KtpIdentityData.from_mapping(data)
        identity.validate_for_record()
        self.assertEqual(
            identity.as_db_dict(),
            data,
        )


if __name__ == "__main__":
    unittest.main()
