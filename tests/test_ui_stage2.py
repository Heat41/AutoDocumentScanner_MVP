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

    def test_manual_document_is_a_named_stage2_page(self):
        self.assertEqual(
            Stage2ScannerUI.MANUAL_DOCUMENT_PAGE,
            "manual_document",
        )

    def test_auto_ktp_is_a_named_stage2_page(self):
        self.assertEqual(
            Stage2ScannerUI.AUTO_KTP_PAGE,
            "auto_ktp",
        )

    def test_navigation_methods_are_exposed(self):
        self.assertTrue(
            callable(
                getattr(
                    Stage2ScannerUI,
                    "show_manual_document",
                )
            )
        )
        self.assertTrue(
            callable(
                getattr(
                    Stage2ScannerUI,
                    "show_auto_ktp",
                )
            )
        )

    def test_tracking_ktp_is_a_named_stage2_page(self):
        self.assertEqual(
            Stage2ScannerUI.TRACKING_KTP_PAGE,
            "tracking_ktp",
        )

    def test_top_navigation_items_are_ordered(self):
        self.assertEqual(
            Stage2ScannerUI.NAVIGATION_ITEMS,
            (
                ("Auto Koreksi KTP", "auto_ktp"),
                (
                    "Koreksi Dokumen Manual",
                    "manual_document",
                ),
                (
                    "Tracking Data KTP",
                    "tracking_ktp",
                ),
            ),
        )

    def test_tracking_navigation_method_is_exposed(self):
        self.assertTrue(
            callable(
                getattr(
                    Stage2ScannerUI,
                    "show_tracking_ktp",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
