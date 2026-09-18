import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from autodocscanner.output.safe import atomic_imwrite


class TestAtomicImageOutput(unittest.TestCase):
    def test_atomic_imwrite_creates_valid_image(self):
        image = np.full(
            (80, 120, 3),
            (90, 140, 190),
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "scan.jpg"

            result = atomic_imwrite(
                output,
                image,
            )

            self.assertEqual(result, output)
            self.assertTrue(output.exists())

            decoded = cv2.imread(str(output))
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, image.shape)

            temporary_files = list(
                Path(temp_dir).glob(".*.tmp.jpg")
            )
            self.assertEqual(temporary_files, [])

    def test_failed_write_preserves_previous_output(self):
        old_image = np.full(
            (50, 80, 3),
            70,
            dtype=np.uint8,
        )
        new_image = np.full(
            (50, 80, 3),
            210,
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "scan.jpg"
            self.assertTrue(
                cv2.imwrite(
                    str(output),
                    old_image,
                )
            )
            original_bytes = output.read_bytes()

            with patch(
                "autodocscanner.output.safe.cv2.imwrite",
                return_value=False,
            ):
                with self.assertRaises(RuntimeError):
                    atomic_imwrite(
                        output,
                        new_image,
                    )

            self.assertEqual(
                output.read_bytes(),
                original_bytes,
            )

            temporary_files = list(
                Path(temp_dir).glob(".*.tmp.jpg")
            )
            self.assertEqual(temporary_files, [])


if __name__ == "__main__":
    unittest.main()
