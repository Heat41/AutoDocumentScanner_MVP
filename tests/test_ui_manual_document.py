import unittest
from tkinter import ttk

from ui_manual_document import ManualDocumentPage


class TestManualDocumentPageContract(unittest.TestCase):
    def test_page_is_a_ttk_frame(self):
        self.assertTrue(
            issubclass(
                ManualDocumentPage,
                ttk.Frame,
            )
        )

    def test_default_output_is_image(self):
        self.assertEqual(
            ManualDocumentPage.DEFAULT_OUTPUT_FORMAT,
            "image",
        )

    def test_page_label_marks_corrected_state(self):
        self.assertEqual(
            ManualDocumentPage.page_label(
                0,
                "scan one.jpg",
                False,
            ),
            "01  •  scan one.jpg",
        )
        self.assertEqual(
            ManualDocumentPage.page_label(
                1,
                "scan two.jpg",
                True,
            ),
            "02  ✓  scan two.jpg",
        )

    def test_supported_extensions_include_common_scan_images(self):
        for suffix in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        ):
            self.assertIn(
                suffix,
                ManualDocumentPage.SUPPORTED_EXTENSIONS,
            )


if __name__ == "__main__":
    unittest.main()
