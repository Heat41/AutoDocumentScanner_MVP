import unittest

from autodocscanner.ui.final import FinalScannerUI
from autodocscanner.ui.textured import TexturedScannerUI


class TestTexturedScannerUI(unittest.TestCase):
    def test_textured_ui_keeps_final_ui_pipeline(self):
        self.assertTrue(
            issubclass(TexturedScannerUI, FinalScannerUI)
        )

    def test_texture_positions_are_regular(self):
        positions = TexturedScannerUI._texture_positions(
            50,
            step=16,
        )

        self.assertEqual(
            positions,
            [0, 16, 32, 48],
        )

    def test_texture_positions_handle_empty_width(self):
        self.assertEqual(
            TexturedScannerUI._texture_positions(0),
            [],
        )

    def test_textured_palette_has_surface_depth(self):
        self.assertNotEqual(
            TexturedScannerUI.BG,
            TexturedScannerUI.CARD,
        )
        self.assertNotEqual(
            TexturedScannerUI.CARD,
            TexturedScannerUI.SHADOW,
        )


if __name__ == "__main__":
    unittest.main()
