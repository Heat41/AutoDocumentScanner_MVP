import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autodocscanner.core.detector_comparison import compare_detectors


class TestDetectorComparison(unittest.TestCase):
    def test_reports_three_methods_without_changing_input(self):
        image = np.full((600, 900, 3), 85, dtype=np.uint8)
        cv2.rectangle(image, (155, 160), (740, 520), (195, 195, 195), -1)
        original = image.copy()
        report = compare_detectors(image, max_proposals=8)
        self.assertIn("stable", report)
        self.assertIn("adaptive", report)
        self.assertIn("v2", report)
        self.assertLessEqual(len(report["v2"]), 8)
        self.assertTrue(np.array_equal(image, original))

    def test_corner_error_is_reported_with_ground_truth(self):
        image = np.full((500, 820, 3), 80, dtype=np.uint8)
        quad = np.array([[110, 80], [700, 80], [700, 450], [110, 450]], dtype=np.float32)
        with patch(
            "autodocscanner.core.detector_comparison.StableBaselinePerspectiveEngine.detect",
            return_value=(quad, {"score": 0.92, "selected_source": "synthetic"}),
        ), patch(
            "autodocscanner.core.detector_comparison.AutoPerspectiveEngine.detect",
            return_value=(quad, {"score": 0.88, "selected_source": "synthetic"}),
        ):
            report = compare_detectors(image, expected=quad, max_proposals=2)
        self.assertEqual(report["stable"]["mean_corner_error_px"], 0.0)
        self.assertEqual(report["adaptive"]["normalized_corner_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
