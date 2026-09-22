import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from autodocscanner.ktp.annotation_preprocess import (
    prepare_annotation_image,
)
from autodocscanner.ktp.layout import (
    KTP_CANONICAL_HEIGHT,
    KTP_CANONICAL_WIDTH,
)


class FakeScanner:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def scan(
        self,
        image_path,
        output_path=None,
        mode="ktp",
        output_mode="color",
    ):
        self.calls.append(
            (
                Path(image_path),
                output_path,
                mode,
                output_mode,
            )
        )
        return (
            self.output.copy(),
            np.zeros(
                (4, 2),
                dtype=np.float32,
            ),
        )


class TestAnnotationPreprocess(unittest.TestCase):
    def test_raw_input_runs_scanner_then_canonical_normalization(self):
        raw = np.zeros(
            (900, 1200, 3),
            dtype=np.uint8,
        )
        corrected = np.full(
            (480, 760, 3),
            120,
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jpg"
            cv2.imwrite(
                str(path),
                raw,
            )
            scanner = FakeScanner(
                corrected
            )

            result = prepare_annotation_image(
                path,
                scanner=scanner,
            )

        self.assertEqual(
            result.image.shape[:2],
            (
                KTP_CANONICAL_HEIGHT,
                KTP_CANONICAL_WIDTH,
            ),
        )
        self.assertEqual(
            result.source_mode,
            "auto_perspective",
        )
        self.assertEqual(
            len(scanner.calls),
            1,
        )

    def test_exact_canonical_input_skips_scanner(self):
        canonical = np.zeros(
            (
                KTP_CANONICAL_HEIGHT,
                KTP_CANONICAL_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "canonical.png"
            cv2.imwrite(
                str(path),
                canonical,
            )
            scanner = FakeScanner(
                np.ones(
                    (100, 100, 3),
                    dtype=np.uint8,
                )
            )

            result = prepare_annotation_image(
                path,
                scanner=scanner,
            )

        self.assertEqual(
            result.image.shape,
            canonical.shape,
        )
        self.assertEqual(
            result.source_mode,
            "canonical",
        )
        self.assertEqual(
            scanner.calls,
            [],
        )


if __name__ == "__main__":
    unittest.main()
