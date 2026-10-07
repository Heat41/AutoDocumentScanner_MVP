import tempfile
import unittest
from pathlib import Path

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


class TestKtpOutputGeometry(unittest.TestCase):
    def test_normalize_ktp_ratio_is_exact(self):
        scanner = AutoDocumentScanner()
        image = np.full(
            (410, 620, 3),
            180,
            dtype=np.uint8,
        )

        result = scanner.normalize_ktp_ratio(
            image
        )

        h, w = result.shape[:2]
        ratio = w / float(h)

        self.assertLess(
            abs(
                ratio
                - scanner.KTP_ASPECT_RATIO
            ),
            0.005,
        )

    def test_normalize_ktp_ratio_rotates_portrait_input(self):
        scanner = AutoDocumentScanner()
        image = np.full(
            (620, 410, 3),
            180,
            dtype=np.uint8,
        )

        result = scanner.normalize_ktp_ratio(
            image
        )

        h, w = result.shape[:2]

        self.assertGreater(
            w,
            h,
        )
        self.assertLess(
            abs(
                (w / float(h))
                - scanner.KTP_ASPECT_RATIO
            ),
            0.005,
        )


class TestKtpSafeMargin(unittest.TestCase):
    def test_safe_margin_expands_quad_outward(self):
        scanner = AutoDocumentScanner(
            ktp_safe_margin=0.015,
        )

        corners = np.array(
            [
                [100, 100],
                [500, 100],
                [500, 350],
                [100, 350],
            ],
            dtype=np.float32,
        )

        expanded = scanner.expand_ktp_corners(
            (500, 700, 3),
            corners,
        )

        self.assertLess(
            expanded[0][0],
            corners[0][0],
        )
        self.assertLess(
            expanded[0][1],
            corners[0][1],
        )
        self.assertGreater(
            expanded[2][0],
            corners[2][0],
        )
        self.assertGreater(
            expanded[2][1],
            corners[2][1],
        )

    def test_safe_margin_is_clamped_to_image(self):
        scanner = AutoDocumentScanner(
            ktp_safe_margin=0.03,
        )

        corners = np.array(
            [
                [2, 3],
                [695, 4],
                [696, 496],
                [3, 495],
            ],
            dtype=np.float32,
        )

        expanded = scanner.expand_ktp_corners(
            (500, 700, 3),
            corners,
        )

        self.assertGreaterEqual(
            float(np.min(expanded[:, 0])),
            0.0,
        )
        self.assertGreaterEqual(
            float(np.min(expanded[:, 1])),
            0.0,
        )
        self.assertLessEqual(
            float(np.max(expanded[:, 0])),
            699.0,
        )
        self.assertLessEqual(
            float(np.max(expanded[:, 1])),
            499.0,
        )


class TestManualKtpFallback(unittest.TestCase):
    def test_scan_with_corners_uses_manual_source_and_ktp_ratio(self):
        scanner = AutoDocumentScanner()

        image = np.full(
            (500, 760, 3),
            210,
            dtype=np.uint8,
        )
        cv2.rectangle(
            image,
            (110, 120),
            (650, 460),
            (190, 205, 120),
            -1,
        )

        corners = np.array(
            [
                [110, 120],
                [650, 120],
                [650, 460],
                [110, 460],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "ktp.jpg"
            output_path = Path(temp_dir) / "out.jpg"
            cv2.imwrite(
                str(input_path),
                image,
            )

            result, used_corners = scanner.scan_with_corners(
                input_path,
                corners,
                output_path,
                mode="ktp",
                output_mode="color",
            )

            self.assertTrue(
                output_path.exists()
            )
            self.assertEqual(
                scanner.last_detection.get(
                    "selected_source"
                ),
                "manual_correction",
            )
            self.assertTrue(
                scanner.last_detection.get(
                    "manual"
                )
            )
            self.assertEqual(
                used_corners.shape,
                (4, 2),
            )

            h, w = result.shape[:2]
            self.assertLess(
                abs(
                    (w / float(h))
                    - scanner.KTP_ASPECT_RATIO
                ),
                0.005,
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
