import unittest

import numpy as np

from autodocscanner.ktp.roi_debug import (
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


if __name__ == "__main__":
    unittest.main()
