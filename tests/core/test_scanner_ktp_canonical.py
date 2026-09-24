import unittest

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
