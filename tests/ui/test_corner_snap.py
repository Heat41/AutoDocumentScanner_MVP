import unittest

import cv2
import numpy as np

from autodocscanner.ui.corner_snap import snap_corner


class TestCornerSnap(unittest.TestCase):
    def test_empty_image_does_not_move_point(self):
        image = np.full((120, 160, 3), 128, dtype=np.uint8)
        point = [64.0, 70.0]
        self.assertEqual(snap_corner(image, point), point)

    def test_outside_image_does_not_move_point(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        point = [-5.0, 20.0]
        self.assertEqual(snap_corner(image, point), point)

    def test_intersection_stays_local(self):
        image = np.full((240, 240, 3), 80, dtype=np.uint8)
        cv2.rectangle(image, (60, 60), (185, 170), (245, 245, 245), -1)
        point = [64.0, 63.0]
        candidate = snap_corner(image, point, radius=16)
        self.assertLessEqual(
            np.linalg.norm(np.asarray(candidate) - np.asarray(point)),
            16.0,
        )


if __name__ == "__main__":
    unittest.main()
