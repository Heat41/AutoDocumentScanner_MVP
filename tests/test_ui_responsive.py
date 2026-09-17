import unittest

from ui_final import FinalScannerUI


class TestResponsiveWindowSizing(unittest.TestCase):
    def test_common_1366x768_display_keeps_footer_visible(self):
        width, height = FinalScannerUI._responsive_window_size(1366, 768)
        self.assertEqual(width, 1180)
        self.assertEqual(height, 648)

    def test_large_display_keeps_original_target_size(self):
        width, height = FinalScannerUI._responsive_window_size(1920, 1080)
        self.assertEqual(width, 1180)
        self.assertEqual(height, 760)

    def test_small_display_uses_safe_lower_bound(self):
        width, height = FinalScannerUI._responsive_window_size(960, 700)
        self.assertEqual(width, 900)
        self.assertEqual(height, 620)


if __name__ == "__main__":
    unittest.main()
