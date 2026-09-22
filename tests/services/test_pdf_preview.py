import tempfile
import unittest
from pathlib import Path

import numpy as np

from autodocscanner.services.pdf_preview import (
    build_ktp_preview_pdf,
)


class TestKtpPdfPreview(unittest.TestCase):
    def test_preview_pdf_is_created_from_corrected_image(self):
        image = np.zeros(
            (53, 86, 3),
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = build_ktp_preview_pdf(
                image,
                stem="sample",
                temp_dir=temp_dir,
            )

            self.assertTrue(result.exists())
            self.assertEqual(
                result.suffix.lower(),
                ".pdf",
            )
            self.assertTrue(
                result.read_bytes().startswith(
                    b"%PDF-"
                )
            )

    def test_preview_pdf_uses_requested_temp_directory(self):
        image = np.zeros(
            (53, 86, 3),
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = build_ktp_preview_pdf(
                image,
                temp_dir=temp_dir,
            )

            self.assertEqual(
                result.parent,
                Path(temp_dir),
            )

    def test_invalid_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                build_ktp_preview_pdf(
                    np.array(
                        [],
                        dtype=np.uint8,
                    ),
                    temp_dir=temp_dir,
                )


if __name__ == "__main__":
    unittest.main()
