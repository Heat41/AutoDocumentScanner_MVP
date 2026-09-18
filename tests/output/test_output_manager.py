import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from autodocscanner.output.manager import (
    build_output_path,
    output_suffix,
    save_document_images,
    save_pdf,
    save_pdf_pages,
)


class TestOutputNaming(unittest.TestCase):
    def test_image_suffix_is_jpg(self):
        self.assertEqual(
            output_suffix("image"),
            ".jpg",
        )

    def test_pdf_suffix_is_pdf(self):
        self.assertEqual(
            output_suffix("pdf"),
            ".pdf",
        )

    def test_build_output_path_uses_scanned_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(
                build_output_path(
                    root,
                    root / "ktp_01.png",
                    "pdf",
                ),
                root / "ktp_01_scanned.pdf",
            )

    def test_unsupported_output_format_is_rejected(self):
        with self.assertRaises(ValueError):
            output_suffix("docx")


class TestPdfOutput(unittest.TestCase):
    def test_pdf_page_matches_image_dimensions_at_72_dpi(self):
        image = np.zeros(
            (53, 86, 3),
            dtype=np.uint8,
        )
        image[:, :, 2] = 255

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ktp.pdf"
            result = save_pdf(
                output,
                image,
            )
            data = output.read_bytes()

        self.assertEqual(
            result,
            output,
        )
        self.assertTrue(
            data.startswith(b"%PDF-")
        )

        match = re.search(
            rb"/MediaBox\s*\[\s*0\s+0\s+"
            rb"([\d.]+)\s+([\d.]+)\s*\]",
            data,
        )
        self.assertIsNotNone(match)
        self.assertAlmostEqual(
            float(match.group(1)),
            86.0,
            places=1,
        )
        self.assertAlmostEqual(
            float(match.group(2)),
            53.0,
            places=1,
        )

    def test_failed_save_does_not_remove_existing_target(self):
        existing = b"old-pdf"
        invalid_image = np.zeros(
            (10, 10, 5),
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ktp.pdf"
            output.write_bytes(existing)

            with self.assertRaises(ValueError):
                save_pdf(
                    output,
                    invalid_image,
                )

            self.assertEqual(
                output.read_bytes(),
                existing,
            )

    def test_multi_page_pdf_preserves_page_count(self):
        first = np.zeros(
            (50, 80, 3),
            dtype=np.uint8,
        )
        second = np.zeros(
            (70, 100, 3),
            dtype=np.uint8,
        )
        second[:, :, 1] = 200

        with tempfile.TemporaryDirectory() as temp_dir:
            output = (
                Path(temp_dir)
                / "document.pdf"
            )
            save_pdf_pages(
                output,
                [first, second],
            )
            data = output.read_bytes()

        self.assertTrue(
            data.startswith(b"%PDF-")
        )
        self.assertGreaterEqual(
            data.count(b"/MediaBox"),
            2,
        )

    def test_multi_page_pdf_rejects_empty_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                save_pdf_pages(
                    Path(temp_dir)
                    / "empty.pdf",
                    [],
                )


class TestDocumentImageOutput(unittest.TestCase):
    def test_saves_one_ordered_image_per_page(self):
        first = np.zeros(
            (30, 40, 3),
            dtype=np.uint8,
        )
        second = np.zeros(
            (35, 45, 3),
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = save_document_images(
                root,
                [
                    Path("cover.png"),
                    Path("page-two.jpg"),
                ],
                [first, second],
            )

            self.assertEqual(
                [
                    path.name
                    for path in paths
                ],
                [
                    "page_001_cover_corrected.jpg",
                    "page_002_page-two_corrected.jpg",
                ],
            )
            self.assertTrue(
                all(
                    path.exists()
                    for path in paths
                )
            )
            self.assertEqual(
                cv2.imread(
                    str(paths[0])
                ).shape[:2],
                (30, 40),
            )
            self.assertEqual(
                cv2.imread(
                    str(paths[1])
                ).shape[:2],
                (35, 45),
            )

    def test_image_export_rejects_mismatched_page_counts(self):
        image = np.zeros(
            (30, 40, 3),
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                save_document_images(
                    Path(temp_dir),
                    [
                        Path("one.jpg"),
                        Path("two.jpg"),
                    ],
                    [image],
                )


if __name__ == "__main__":
    unittest.main()
