import unittest

from autodocscanner.ktp.nik_validation import (
    normalize_province_value,
    repair_nik_with_context,
)


class TestNikContextValidation(unittest.TestCase):
    def test_normalizes_noisy_province_name(self):
        self.assertEqual(
            normalize_province_value(
                "PROVINSI KALIMANTAN BAA I"
            ),
            "PROVINSI KALIMANTAN BARAT",
        )

    def test_repairs_province_prefix_when_birth_segment_matches(self):
        result = repair_nik_with_context(
            "5112340503800002",
            province="PROVINSI KALIMANTAN BARAT",
            birth_date="1980-03-05",
            gender="LAKI-LAKI",
        )

        self.assertTrue(
            result.changed
        )
        self.assertTrue(
            result.context_valid
        )
        self.assertEqual(
            result.value,
            "6112340503800002",
        )

    def test_does_not_repair_when_birth_segment_conflicts(self):
        result = repair_nik_with_context(
            "5112340101800002",
            province="KALIMANTAN BARAT",
            birth_date="1980-03-05",
            gender="LAKI-LAKI",
        )

        self.assertFalse(
            result.changed
        )
        self.assertFalse(
            result.context_valid
        )
        self.assertEqual(
            result.value,
            "5112340101800002",
        )

    def test_female_birth_segment_uses_day_plus_forty(self):
        result = repair_nik_with_context(
            "6112344503800002",
            province="KALIMANTAN BARAT",
            birth_date="1980-03-05",
            gender="PEREMPUAN",
        )

        self.assertFalse(
            result.changed
        )
        self.assertTrue(
            result.context_valid
        )


if __name__ == "__main__":
    unittest.main()
