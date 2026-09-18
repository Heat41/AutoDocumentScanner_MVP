import unittest

import cv2
import numpy as np

from autodocscanner.core.quality_check import DocumentQualityChecker


class TestDocumentQualityChecker(unittest.TestCase):
    def setUp(self):
        self.checker = DocumentQualityChecker()

    @staticmethod
    def _good_image():
        width = 760
        height = int(round(width / (85.60 / 53.98)))

        image = np.full(
            (height, width, 3),
            (205, 195, 110),
            dtype=np.uint8,
        )

        cv2.rectangle(
            image,
            (4, 4),
            (width - 5, height - 5),
            (55, 55, 55),
            3,
        )

        for y in range(70, height - 50, 34):
            cv2.line(
                image,
                (45, y),
                (460, y),
                (35, 35, 35),
                3,
            )

        cv2.rectangle(
            image,
            (560, 85),
            (700, 300),
            (65, 70, 75),
            -1,
        )

        return image

    def test_good_output_has_quality_report(self):
        report = self.checker.assess(
            self._good_image(),
            detection_metadata={
                "score": 0.82,
            },
        )

        self.assertIn(
            report["status"],
            {"pass", "warning"},
        )
        self.assertGreater(
            report["score"],
            0.55,
        )
        self.assertIn(
            "metrics",
            report,
        )

    def test_blur_is_reported_without_rejecting_image(self):
        image = self._good_image()
        blurred = cv2.GaussianBlur(
            image,
            (31, 31),
            0,
        )

        report = self.checker.assess(
            blurred,
            detection_metadata={
                "score": 0.80,
            },
        )

        self.assertTrue(
            any(
                "blur" in warning
                for warning in report["warnings"]
            )
        )
        self.assertGreaterEqual(
            report["score"],
            0.0,
        )

    def test_wrong_ratio_is_flagged(self):
        image = np.full(
            (500, 500, 3),
            130,
            dtype=np.uint8,
        )

        report = self.checker.assess(
            image,
            detection_metadata={
                "score": 0.80,
            },
        )

        self.assertTrue(
            any(
                "rasio" in warning
                for warning in report["warnings"]
            )
        )

    def test_low_detection_confidence_is_flagged(self):
        report = self.checker.assess(
            self._good_image(),
            detection_metadata={
                "score": 0.35,
            },
        )

        self.assertTrue(
            any(
                "confidence" in warning
                for warning in report["warnings"]
            )
        )


if __name__ == "__main__":
    unittest.main()
