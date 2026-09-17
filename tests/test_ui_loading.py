import unittest

from ui_final import FinalScannerUI


class TestUILoadingPopup(unittest.TestCase):
    def test_loading_detail_uses_live_processing_status(self):
        detail = FinalScannerUI._loading_detail(
            "Memproses 2/5: sample.jpg",
            5,
        )
        self.assertEqual(
            detail,
            "Memproses 2/5: sample.jpg",
        )

    def test_loading_detail_single_file(self):
        detail = FinalScannerUI._loading_detail(
            "",
            1,
        )
        self.assertEqual(
            detail,
            "Menyiapkan 1 file...",
        )

    def test_loading_detail_multiple_files(self):
        detail = FinalScannerUI._loading_detail(
            "Memproses...",
            4,
        )
        self.assertEqual(
            detail,
            "Memproses...",
        )

    def test_loading_popup_methods_exist(self):
        self.assertTrue(callable(FinalScannerUI._show_loading_popup))
        self.assertTrue(callable(FinalScannerUI._hide_loading_popup))
        self.assertTrue(callable(FinalScannerUI._watch_loading_popup))


if __name__ == "__main__":
    unittest.main()
