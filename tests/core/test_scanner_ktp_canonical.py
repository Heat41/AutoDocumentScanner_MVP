import unittest

import cv2
import numpy as np

from autodocscanner.core.scanner import (
    AutoDocumentScanner,
)


class TestKtpCanonicalCanvas(unittest.TestCase):
    def test_normalize_ktp_canvas_returns_exact_856x540(self):
        image = np.zeros(
            (403, 641, 3),
            dtype=np.uint8,
        )

        result = (
            AutoDocumentScanner
            .normalize_ktp_canvas(
                image
            )
        )

        self.assertEqual(
            result.shape[:2],
            (540, 856),
        )

    @staticmethod
    def _synthetic_horizontal_card(
        angle=0.0,
    ):
        image = np.full(
            (540, 856, 3),
            245,
            dtype=np.uint8,
        )

        for y in range(
            120,
            421,
            45,
        ):
            cv2.line(
                image,
                (90, y),
                (690, y),
                (25, 25, 25),
                3,
            )

        if abs(
            float(angle)
        ) < 1e-9:
            return image

        matrix = cv2.getRotationMatrix2D(
            (428.0, 270.0),
            float(angle),
            1.0,
        )

        return cv2.warpAffine(
            image,
            matrix,
            (856, 540),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def test_residual_skew_keeps_straight_ktp_unchanged(self):
        image = self._synthetic_horizontal_card(
            angle=0.0
        )

        corrected, metadata = (
            AutoDocumentScanner
            .stabilize_ktp_residual_skew(
                image
            )
        )

        self.assertFalse(
            metadata["applied"]
        )
        self.assertEqual(
            corrected.shape[:2],
            (540, 856),
        )
        self.assertTrue(
            np.array_equal(
                corrected,
                image,
            )
        )

    def test_residual_skew_corrects_small_consistent_angle(self):
        image = self._synthetic_horizontal_card(
            angle=1.35
        )

        before = (
            AutoDocumentScanner
            .estimate_ktp_residual_skew(
                image
            )
        )

        self.assertTrue(
            before["applied"]
        )
        self.assertGreaterEqual(
            before["line_count"],
            AutoDocumentScanner
            .KTP_RESIDUAL_SKEW_MIN_LINES,
        )

        corrected, metadata = (
            AutoDocumentScanner
            .stabilize_ktp_residual_skew(
                image
            )
        )

        self.assertTrue(
            metadata["applied"]
        )
        self.assertEqual(
            corrected.shape[:2],
            (540, 856),
        )

        after = (
            AutoDocumentScanner
            .estimate_ktp_residual_skew(
                corrected
            )
        )

        self.assertFalse(
            after["applied"]
        )
        self.assertLess(
            abs(
                float(
                    after["angle"]
                )
            ),
            AutoDocumentScanner
            .KTP_RESIDUAL_SKEW_MIN_DEG,
        )

    def test_residual_skew_does_not_force_large_rotation(self):
        image = self._synthetic_horizontal_card(
            angle=4.0
        )

        metadata = (
            AutoDocumentScanner
            .estimate_ktp_residual_skew(
                image
            )
        )

        self.assertFalse(
            metadata["applied"]
        )

    def test_normalize_ktp_canvas_keeps_canonical_copy(self):
        image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )

        result = (
            AutoDocumentScanner
            .normalize_ktp_canvas(
                image
            )
        )

        self.assertEqual(
            result.shape[:2],
            (540, 856),
        )
        self.assertIsNot(
            result,
            image,
        )


if __name__ == "__main__":
    unittest.main()
