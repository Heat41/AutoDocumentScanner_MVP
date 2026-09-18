import unittest

from autodocscanner.documents.document_canvas import (
    canvas_to_image,
    fit_image_size,
    image_to_canvas,
)


class TestDocumentCanvasGeometry(unittest.TestCase):
    def test_fit_image_size_preserves_aspect_ratio(self):
        self.assertEqual(
            fit_image_size(
                1600,
                1000,
                800,
                600,
                padding=20,
            ),
            (760, 475),
        )

    def test_fit_image_size_handles_portrait(self):
        self.assertEqual(
            fit_image_size(
                1000,
                1600,
                800,
                600,
                padding=20,
            ),
            (350, 560),
        )

    def test_coordinate_round_trip(self):
        image_size = (1600, 1000)
        display_size = (760, 475)
        offset = (20.0, 62.5)
        original = (425.0, 710.0)

        canvas = image_to_canvas(
            original,
            image_size,
            display_size,
            offset,
        )
        restored = canvas_to_image(
            canvas,
            image_size,
            display_size,
            offset,
        )

        self.assertAlmostEqual(
            restored[0],
            original[0],
            places=5,
        )
        self.assertAlmostEqual(
            restored[1],
            original[1],
            places=5,
        )

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_image_size(
                0,
                100,
                800,
                600,
            )


if __name__ == "__main__":
    unittest.main()
