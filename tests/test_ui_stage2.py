import unittest
from pathlib import Path

from ui_responsive import ResponsiveScannerUI
from ui_stage2 import Stage2ScannerUI


class TestStage2UiContract(unittest.TestCase):
    def test_stage2_keeps_responsive_ui_as_base(self):
        self.assertTrue(
            issubclass(
                Stage2ScannerUI,
                ResponsiveScannerUI,
            )
        )

    def test_image_is_default_output_format(self):
        self.assertEqual(
            Stage2ScannerUI.DEFAULT_OUTPUT_FORMAT,
            "image",
        )

    def test_pdf_path_detection_is_case_insensitive(self):
        self.assertTrue(
            Stage2ScannerUI._is_pdf_path(
                Path("hasil.PDF")
            )
        )
        self.assertFalse(
            Stage2ScannerUI._is_pdf_path(
                Path("hasil.jpg")
            )
        )


if __name__ == "__main__":
    unittest.main()
