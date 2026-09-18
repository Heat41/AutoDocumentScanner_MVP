import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from output_manager import build_output_path, output_suffix, save_pdf


class TestOutputNaming(unittest.TestCase):
    def test_image_suffix_is_jpg(self):
        self.assertEqual(output_suffix("image"), ".jpg")

    def test_pdf_suffix_is_pdf(self):
        self.assertEqual(output_suffix("pdf"), ".pdf")

    def test_build_output_path_uses_scanned_suffix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(
                build_output_path(root, root / "ktp_01.png", "pdf"),
                root / "ktp_01_scanned.pdf",
            )

    def test_unsupported_output_format_is_rejected(self):
        with self.assertRaises(ValueError):
            output_suffix("docx")


class TestPdfOutput(unittest.TestCase):
    def test_pdf_page_matches_image_dimensions_at_72_dpi(self):
        image = np.zeros((53, 86, 3), dtype=np.uint8)
        image[:, :, 2] = 255

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ktp.pdf"
            result = save_pdf(output, image)
            data = output.read_bytes()

        self.assertEqual(result, output)
        self.assertTrue(data.startswith(b"%PDF-"))

        match = re.search(
            rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]",
            data,
        )
        self.assertIsNotNone(match)
        self.assertAlmostEqual(float(match.group(1)), 86.0, places=1)
        self.assertAlmostEqual(float(match.group(2)), 53.0, places=1)

    def test_failed_save_does_not_remove_existing_target(self):
        existing = b"old-pdf"
        invalid_image = np.zeros((10, 10, 5), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "ktp.pdf"
            output.write_bytes(existing)

            with self.assertRaises(ValueError):
                save_pdf(output, invalid_image)

            self.assertEqual(output.read_bytes(), existing)


if __name__ == "__main__":
    unittest.main()
