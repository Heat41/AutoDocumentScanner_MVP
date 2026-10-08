import unittest

import cv2
import numpy as np

from autodocscanner.core.experimental_candidates import contour_candidates


class TestExperimentalCandidates(unittest.TestCase):
    def test_blank_image_has_no_candidates(self):
        image = np.full((600, 900, 3), 110, dtype=np.uint8)
        self.assertEqual(contour_candidates(image), [])

    def test_synthetic_rotated_card_candidate(self):
        image = np.full((800, 1000, 3), 50, dtype=np.uint8)
        rectangle = ((500, 400), (560, 354), 17)
        corners = cv2.boxPoints(rectangle).astype(np.int32)
        cv2.fillConvexPoly(image, corners, (220, 210, 180))
        cv2.polylines(image, [corners], True, (5, 5, 5), 4)

        candidates = contour_candidates(image)
        self.assertTrue(candidates, "Expected at least one card quad")
        self.assertTrue(all(quad.shape == (4, 2) for quad in candidates))

    def test_does_not_mutate_input(self):
        image = np.full((500, 850, 3), 90, dtype=np.uint8)
        original = image.copy()
        contour_candidates(image)
        self.assertTrue(np.array_equal(image, original))


if __name__ == "__main__":
    unittest.main()
