import unittest

from autodocscanner.ktp.parsing import (
    clean_text,
    parse_field,
    parse_gender,
    parse_nik,
    parse_rt_rw,
    parse_ttl,
    parse_ktp_document,
)


class TestKtpParsing(unittest.TestCase):
    def test_clean_text_normalizes_whitespace(self):
        self.assertEqual(
            clean_text("  NAMA   CONTOH  "),
            "NAMA CONTOH",
        )

    def test_parse_nik_keeps_exact_16_digits(self):
        self.assertEqual(
            parse_nik("NIK : 1234567890123456"),
            "1234567890123456",
        )

    def test_parse_nik_repairs_common_ocr_digit_confusions(self):
        self.assertEqual(
            parse_nik("I23456789O123456"),
            "1234567890123456",
        )

    def test_parse_nik_returns_empty_when_not_16_digits(self):
        self.assertEqual(
            parse_nik("123456789012345"),
            "",
        )

    def test_parse_ttl_splits_place_and_date(self):
        result = parse_ttl("KOTA CONTOH, 02-03-1990")
        self.assertEqual(result["tempat_lahir"], "KOTA CONTOH")
        self.assertEqual(result["tanggal_lahir"], "1990-03-02")

    def test_parse_rt_rw_splits_and_pads(self):
        self.assertEqual(
            parse_rt_rw("RT 1 RW 7"),
            {"rt": "001", "rw": "007"},
        )

    def test_parse_gender_normalizes_known_values(self):
        self.assertEqual(parse_gender("LAKI LAKI"), "LAKI-LAKI")
        self.assertEqual(parse_gender("PEREMPUAN"), "PEREMPUAN")

    def test_parse_field_dispatches_special_fields(self):
        self.assertEqual(
            parse_field("nik", "1234567890123456"),
            "1234567890123456",
        )

    def test_generic_field_preserves_value_that_starts_with_field_word(self):
        self.assertEqual(
            parse_field("nama", "NAMA CONTOH"),
            "NAMA CONTOH",
        )

    def test_generic_field_strips_explicit_label_separator(self):
        self.assertEqual(
            parse_field("nama", "NAMA: CONTOH"),
            "CONTOH",
        )

    def test_parse_ktp_document_extracts_labeled_lines(self):
        text = """
PROVINSI CONTOH
KOTA CONTOH
NIK : 1234567890123456
Nama : NAMA CONTOH
Tempat/Tgl Lahir : KOTA CONTOH, 02-03-1990
Jenis kelamin : LAKI-LAKI
Alamat : JALAN CONTOH
RT/RW : 001/002
Kel/Desa : DESA CONTOH
Kecamatan : KECAMATAN CONTOH
Agama : ISLAM
Status Perkawinan : KAWIN
Pekerjaan : KARYAWAN
Kewarganegaraan : WNI
Berlaku Hingga : SEUMUR HIDUP
"""
        result = parse_ktp_document(text)

        self.assertEqual(
            result["provinsi"],
            "PROVINSI CONTOH",
        )
        self.assertEqual(
            result["kabupaten_kota"],
            "KOTA CONTOH",
        )
        self.assertEqual(
            result["nama"],
            "NAMA CONTOH",
        )
        self.assertEqual(
            result["rt_rw"],
            "001/002",
        )

    def test_parse_ktp_document_ignores_unknown_lines(self):
        result = parse_ktp_document(
            "TEKS ACAK\nNama : NAMA CONTOH\n"
        )

        self.assertEqual(
            result,
            {"nama": "NAMA CONTOH"},
        )


if __name__ == "__main__":
    unittest.main()
