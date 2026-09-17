import tempfile
import unittest
from pathlib import Path

import numpy as np

from batch_runner import BatchScanRunner


class FakeScanner:
    def __init__(self):
        self.last_detection = {}
        self.last_quality = {}
        self.last_validation = {}
        self.last_corners = None
        self.calls = []

    def scan(
        self,
        input_path,
        output_path,
        mode="ktp",
        output_mode="color",
    ):
        input_path = Path(input_path)
        output_path = Path(output_path)
        self.calls.append(input_path.name)

        self.last_corners = np.array(
            [
                [10, 10],
                [110, 10],
                [110, 70],
                [10, 70],
            ],
            dtype=np.float32,
        )

        if "fail" in input_path.stem:
            self.last_detection = {
                "score": 0.21,
                "selected_source": "synthetic",
            }
            self.last_quality = {
                "status": "review",
                "score": 0.31,
                "warnings": [
                    "confidence deteksi rendah"
                ],
            }
            self.last_validation = {
                "status": "review",
                "hard_valid": False,
                "warnings": [
                    "confidence final terlalu rendah"
                ],
            }
            self.last_detection["quality"] = dict(
                self.last_quality
            )
            self.last_detection["validation"] = dict(
                self.last_validation
            )
            raise RuntimeError(
                "hasil scan ditolak validator"
            )

        self.last_detection = {
            "score": 0.86,
            "selected_source": "synthetic",
            "candidate_count": 3,
        }
        self.last_quality = {
            "status": "pass",
            "score": 0.91,
            "warnings": [],
        }
        self.last_validation = {
            "status": "pass",
            "hard_valid": True,
            "warnings": [],
        }
        self.last_detection["quality"] = dict(
            self.last_quality
        )
        self.last_detection["validation"] = dict(
            self.last_validation
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_bytes(b"valid-output")

        image = np.zeros(
            (70, 110, 3),
            dtype=np.uint8,
        )
        return image, self.last_corners


class TestBatchScanRunner(unittest.TestCase):
    def test_failure_does_not_stop_following_files(self):
        scanner = FakeScanner()
        runner = BatchScanRunner(scanner)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = [
                root / "pass_one.jpg",
                root / "fail_middle.jpg",
                root / "pass_two.jpg",
            ]

            report = runner.run_many(
                inputs,
                root / "output",
            )

        self.assertEqual(
            scanner.calls,
            [
                "pass_one.jpg",
                "fail_middle.jpg",
                "pass_two.jpg",
            ],
        )
        self.assertEqual(report["total_count"], 3)
        self.assertEqual(report["success_count"], 2)
        self.assertEqual(report["failed_count"], 1)
        self.assertTrue(report["results"][0]["success"])
        self.assertFalse(report["results"][1]["success"])
        self.assertTrue(report["results"][2]["success"])

    def test_failed_run_never_reports_stale_output_as_current(self):
        scanner = FakeScanner()
        runner = BatchScanRunner(scanner)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "fail_card_scanned.jpg"
            output.write_bytes(b"old-output")

            result = runner.run_one(
                root / "fail_card.jpg",
                output,
            )

            self.assertFalse(result["success"])
            self.assertIsNone(result["output_path"])
            self.assertEqual(
                result["attempted_output_path"],
                output,
            )
            self.assertTrue(output.exists())
            self.assertEqual(
                output.read_bytes(),
                b"old-output",
            )

    def test_result_keeps_quality_and_validation_snapshot(self):
        scanner = FakeScanner()
        runner = BatchScanRunner(scanner)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = runner.run_one(
                root / "pass_card.jpg",
                root / "pass_card_scanned.jpg",
            )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["quality"].get("status"),
            "pass",
        )
        self.assertTrue(
            result["validation"].get("hard_valid")
        )
        self.assertEqual(
            result["metadata"].get("candidate_count"),
            3,
        )


if __name__ == "__main__":
    unittest.main()
