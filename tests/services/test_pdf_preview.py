import tempfile
import unittest
from pathlib import Path

import numpy as np

from autodocscanner.services.pdf_preview import (
    build_ktp_preview_pdf,
    build_tracking_report_image,
    build_tracking_report_pdf,
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

    def test_tracking_report_image_contains_report_page(self):
        image = np.full(
            (540, 856, 3),
            180,
            dtype=np.uint8,
        )
        face = np.full(
            (180, 120, 3),
            120,
            dtype=np.uint8,
        )
        fields = {
            "nik": "1234567890123456",
            "nama": "NAMA CONTOH",
            "alamat": "JALAN CONTOH NOMOR 1",
        }

        report = build_tracking_report_image(
            image,
            face,
            fields,
        )

        self.assertEqual(
            report.mode,
            "RGB",
        )
        self.assertGreater(
            report.width,
            report.height // 2,
        )
        self.assertGreater(
            report.height,
            image.shape[0],
        )

    def test_tracking_report_pdf_is_created(self):
        image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )
        fields = {
            "nama": "NAMA HASIL REVIEW",
            "alamat": "",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = build_tracking_report_pdf(
                image,
                None,
                fields,
                stem="tracking",
                temp_dir=temp_dir,
            )

            self.assertTrue(result.exists())
            self.assertTrue(
                result.read_bytes().startswith(
                    b"%PDF-"
                )
            )

    def test_tracking_report_accepts_long_and_empty_values(self):
        image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )
        fields = {
            "alamat": "JALAN CONTOH " * 20,
            "pekerjaan": "",
        }

        report = build_tracking_report_image(
            image,
            None,
            fields,
        )

        self.assertGreater(
            report.size[0],
            0,
        )


if __name__ == "__main__":
    unittest.main()
