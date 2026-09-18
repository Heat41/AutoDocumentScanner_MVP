import unittest

from autodocscanner.ui.safe import SafeScannerUI


class TestUIFailureSummary(unittest.TestCase):
    def test_validation_warning_is_preferred_over_generic_error(self):
        status, detail = SafeScannerUI._failure_summary(
            {
                "message": "Hasil scan gagal validasi geometri akhir.",
                "metadata": {
                    "validation": {
                        "warnings": [
                            "rasio output tidak sesuai KTP",
                            "confidence final terlalu rendah",
                        ]
                    }
                },
            }
        )

        self.assertEqual(status, "Status: Ditolak")
        self.assertIn(
            "rasio output tidak sesuai KTP",
            detail,
        )
        self.assertIn(
            "confidence final terlalu rendah",
            detail,
        )

    def test_quality_warning_is_used_when_validation_has_no_reason(self):
        _, detail = SafeScannerUI._failure_summary(
            {
                "message": "Proses gagal.",
                "metadata": {
                    "quality": {
                        "warnings": [
                            "gambar cukup blur",
                        ]
                    }
                },
            }
        )

        self.assertEqual(detail, "gambar cukup blur")

    def test_generic_message_is_fallback(self):
        status, detail = SafeScannerUI._failure_summary(
            {
                "message": "Batas KTP tidak berhasil dideteksi otomatis.",
                "metadata": {},
            }
        )

        self.assertEqual(status, "Status: Ditolak")
        self.assertEqual(
            detail,
            "Batas KTP tidak berhasil dideteksi otomatis.",
        )

    def test_failure_reason_is_limited_for_ui(self):
        _, detail = SafeScannerUI._failure_summary(
            {
                "metadata": {
                    "validation": {
                        "warnings": [
                            "alasan satu",
                            "alasan dua",
                            "alasan tiga",
                            "alasan empat",
                        ]
                    }
                }
            }
        )

        self.assertIn("alasan satu", detail)
        self.assertIn("alasan dua", detail)
        self.assertNotIn("alasan tiga", detail)
        self.assertIn("+2 lainnya", detail)


if __name__ == "__main__":
    unittest.main()
