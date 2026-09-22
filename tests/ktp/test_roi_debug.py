import unittest

import numpy as np

from autodocscanner.ktp.field_detection import (
    FieldDetection,
)
from autodocscanner.ktp.roi_debug import (
    build_detection_overlay,
    build_roi_overlay,
)


class TestRoiDebugOverlay(unittest.TestCase):
    def test_overlay_keeps_image_shape_and_draws_pixels(self):
        image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )

        overlay = build_roi_overlay(
            image
        )

        self.assertEqual(
            overlay.shape,
            image.shape,
        )
        self.assertFalse(
            np.array_equal(
                overlay,
                image,
            )
        )

    def test_overlay_does_not_modify_source_image(self):
        image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )
        original = image.copy()

        build_roi_overlay(
            image
        )

        self.assertTrue(
            np.array_equal(
                image,
                original,
            )
        )

    def test_detection_overlay_draws_named_field_box(self):
        image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )

        overlay = build_detection_overlay(
            image,
            [
                FieldDetection(
                    class_name="nama",
                    bbox=(100, 120, 350, 155),
                    confidence=0.96,
                    source="test",
                )
            ],
        )

        self.assertEqual(
            overlay.shape,
            image.shape,
        )
        self.assertFalse(
            np.array_equal(
                overlay,
                image,
            )
        )


if __name__ == "__main__":
    unittest.main()
