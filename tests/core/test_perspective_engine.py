import unittest

import cv2
import numpy as np

from autodocscanner.core.perspective_engine import AutoPerspectiveEngine
from autodocscanner.core.scanner import AutoDocumentScanner


class TestAutoPerspectiveEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AutoPerspectiveEngine()

    def test_order_points(self):
        points = np.array(
            [
                [500, 300],
                [100, 100],
                [120, 320],
                [520, 90],
            ],
            dtype=np.float32,
        )

        ordered = self.engine.order_points(
            points
        )

        self.assertEqual(
            ordered.shape,
            (4, 2),
        )
        self.assertTrue(
            np.allclose(
                ordered[0],
                [100, 100],
            )
        )

    def test_synthetic_perspective_correction(self):
        canvas = np.full(
            (700, 900, 3),
            225,
            dtype=np.uint8,
        )

        source = np.full(
            (340, 540, 3),
            (205, 210, 115),
            dtype=np.uint8,
        )

        cv2.rectangle(
            source,
            (4, 4),
            (535, 335),
            (60, 60, 60),
            4,
        )

        for y in range(
            70,
            270,
            35,
        ):
            cv2.line(
                source,
                (35, y),
                (330, y),
                (40, 40, 40),
                3,
            )

        destination = np.array(
            [
                [180, 145],
                [735, 95],
                [770, 520],
                [125, 555],
            ],
            dtype=np.float32,
        )

        source_points = np.array(
            [
                [0, 0],
                [539, 0],
                [539, 339],
                [0, 339],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            source_points,
            destination,
        )

        warped = cv2.warpPerspective(
            source,
            matrix,
            (900, 700),
            borderValue=(225, 225, 225),
        )

        mask = cv2.warpPerspective(
            np.full(
                (340, 540),
                255,
                dtype=np.uint8,
            ),
            matrix,
            (900, 700),
        )

        canvas[
            mask > 0
        ] = warped[
            mask > 0
        ]

        corrected, corners, metadata = (
            self.engine.correct(
                canvas
            )
        )

        self.assertIsNotNone(
            corrected
        )
        self.assertIsNotNone(
            corners
        )
        self.assertGreater(
            metadata["candidate_count"],
            0,
        )

        h, w = corrected.shape[:2]
        ratio = max(w, h) / min(w, h)

        self.assertLess(
            abs(
                ratio
                - self.engine.target_ratio
            ),
            0.35,
        )


class TestScannerOutput(unittest.TestCase):
    def test_grayscale_mode_keeps_three_channels(self):
        image = np.zeros(
            (20, 30, 3),
            dtype=np.uint8,
        )
        image[:, :, 1] = 150

        result = AutoDocumentScanner.apply_output_mode(
            image,
            "grayscale",
        )

        self.assertEqual(
            result.shape,
            (20, 30, 3),
        )
        self.assertTrue(
            np.array_equal(
                result[:, :, 0],
                result[:, :, 1],
            )
        )


if __name__ == "__main__":
    unittest.main()
