import unittest

from ui_final import FinalScannerUI
from ui_safe import SafeScannerUI


class TestFinalScannerUI(unittest.TestCase):
    def test_final_ui_keeps_safe_failure_layer(self):
        self.assertTrue(
            issubclass(
                FinalScannerUI,
                SafeScannerUI,
            )
        )

    def test_batch_summary_empty(self):
        self.assertEqual(
            FinalScannerUI._batch_summary(0),
            "Belum ada file",
        )

    def test_batch_summary_single(self):
        self.assertEqual(
            FinalScannerUI._batch_summary(1),
            "1 file siap diproses",
        )

    def test_batch_summary_multiple(self):
        self.assertEqual(
            FinalScannerUI._batch_summary(12),
            "12 file siap diproses",
        )


if __name__ == "__main__":
    unittest.main()
