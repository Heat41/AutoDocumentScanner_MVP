import tempfile
import unittest
from pathlib import Path

import numpy as np

from autodocscanner.services.ktp_output import process_ktp_output


class FakeScanner:
    def __init__(self):
        self.calls = []
        self.last_detection = {"score": 0.9}
        self.last_quality = {"status": "pass"}
        self.last_validation = {"hard_valid": True}
        self.last_corners = np.array(
            [[0, 0], [85, 0], [85, 52], [0, 52]],
            dtype=np.float32,
        )

    def scan(
        self,
        input_path,
        output_path=None,
        mode="ktp",
        output_mode="color",
    ):
        self.calls.append(
            {
                "input_path": Path(input_path),
                "output_path": (
                    None
                    if output_path is None
                    else Path(output_path)
                ),
                "mode": mode,
                "output_mode": output_mode,
            }
        )
        image = np.zeros(
            (53, 86, 3),
            dtype=np.uint8,
        )
        image[:, :, 1] = 180

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            output_path.write_bytes(
                b"jpg-output"
            )

        return image, self.last_corners.copy()


class TestStage2Processing(unittest.TestCase):
    def test_image_output_preserves_existing_scanner_write_path(self):
        scanner = FakeScanner()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = process_ktp_output(
                scanner,
                root / "ktp.png",
                root / "out",
                output_mode="grayscale",
                output_format="image",
            )

            self.assertEqual(
                result["output_path"],
                root / "out" / "ktp_scanned.jpg",
            )
            self.assertTrue(
                result["output_path"].exists()
            )

        self.assertEqual(
            scanner.calls[0]["mode"],
            "ktp",
        )
        self.assertEqual(
            scanner.calls[0]["output_mode"],
            "grayscale",
        )
        self.assertEqual(
            scanner.calls[0]["output_path"].name,
            "ktp_scanned.jpg",
        )
        self.assertIsNone(
            result["preview_bytes"]
        )

    def test_pdf_output_does_not_ask_scanner_to_write_image(self):
        scanner = FakeScanner()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = process_ktp_output(
                scanner,
                root / "ktp.png",
                root / "out",
                output_format="pdf",
            )
            pdf_bytes = (
                result["output_path"]
                .read_bytes()
            )

        self.assertIsNone(
            scanner.calls[0]["output_path"]
        )
        self.assertEqual(
            result["output_path"].suffix,
            ".pdf",
        )
        self.assertTrue(
            pdf_bytes.startswith(b"%PDF-")
        )
        self.assertTrue(
            result["preview_bytes"].startswith(
                b"\xff\xd8"
            )
        )
        self.assertTrue(
            result["preview_bytes"].endswith(
                b"\xff\xd9"
            )
        )


if __name__ == "__main__":
    unittest.main()
