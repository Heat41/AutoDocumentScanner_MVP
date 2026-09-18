import unittest
import numpy as np

from manual_document import (
    initial_corners,
    clamp_corners,
    correct_manual_perspective,
    rotate_image_and_reset,
)


class TestManualDocumentGeometry(unittest.TestCase):
    def test_initial_corners_are_ordered_and_inset(self):
        points = initial_corners(1000, 600, margin_ratio=0.05)
        expected = np.array(
            [[50, 30], [949, 30], [949, 569], [50, 569]],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(points, expected))

    def test_clamp_corners_keeps_points_inside_image(self):
        points = np.array(
            [[-10, -20], [120, -1], [130, 90], [-5, 99]],
            dtype=np.float32,
        )
        clamped = clamp_corners(points, width=100, height=80)
        self.assertTrue(np.all(clamped[:, 0] >= 0))
        self.assertTrue(np.all(clamped[:, 0] <= 99))
        self.assertTrue(np.all(clamped[:, 1] >= 0))
        self.assertTrue(np.all(clamped[:, 1] <= 79))

    def test_manual_warp_uses_supplied_points(self):
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        image[10:90, 10:150] = (10, 20, 30)
        points = np.array(
            [[10, 10], [149, 10], [149, 89], [10, 89]],
            dtype=np.float32,
        )
        result = correct_manual_perspective(image, points)
        self.assertEqual(result.shape[:2], (79, 139))
        self.assertGreater(int(result[20, 20].sum()), 0)

    def test_rotate_right_resets_corners_for_new_shape(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        rotated, points = rotate_image_and_reset(image, "right")
        self.assertEqual(rotated.shape[:2], (100, 60))
        self.assertEqual(points.shape, (4, 2))
        self.assertTrue(np.all(points[:, 0] <= 59))
        self.assertTrue(np.all(points[:, 1] <= 99))

    def test_invalid_direction_is_rejected(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            rotate_image_and_reset(image, "flip")


if __name__ == "__main__":
    unittest.main()
