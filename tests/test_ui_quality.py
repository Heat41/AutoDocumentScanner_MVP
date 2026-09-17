import unittest

from ui import ScannerUI


class TestUIQualitySummary(unittest.TestCase):
    def test_pass_quality_summary(self):
        label, detail = ScannerUI._quality_summary(
            {
                "quality": {
                    "status": "pass",
                    "score": 0.91,
                    "warnings": [],
                }
            }
        )

        self.assertEqual(label, "Kualitas: Baik")
        self.assertIn("0.91", detail)

    def test_warning_quality_summary_limits_warning_text(self):
        label, detail = ScannerUI._quality_summary(
            {
                "quality": {
                    "status": "warning",
                    "score": 0.64,
                    "warnings": [
                        "gambar cukup blur",
                        "kontras rendah",
                        "resolusi hasil rendah",
                    ],
                }
            }
        )

        self.assertEqual(
            label,
            "Kualitas: Perlu diperhatikan",
        )
        self.assertIn("gambar cukup blur", detail)
        self.assertIn("kontras rendah", detail)
        self.assertIn("+1 lainnya", detail)

    def test_missing_quality_is_neutral(self):
        label, detail = ScannerUI._quality_summary({})

        self.assertEqual(label, "Kualitas: -")
        self.assertEqual(detail, "")


if __name__ == "__main__":
    unittest.main()
