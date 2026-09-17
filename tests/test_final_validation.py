import unittest

import numpy as np

from final_validation import FinalScanValidator


class TestFinalScanValidator(unittest.TestCase):
    def setUp(self):
        self.validator = FinalScanValidator()

    def test_valid_perspective_quad_passes_hard_gate(self):
        corners = np.array(
            [
                [120, 95],
                [790, 125],
                [755, 520],
                [150, 555],
            ],
            dtype=np.float32,
        )

        result = self.validator.validate_corners(
            (720, 960, 3),
            corners,
        )

        self.assertTrue(result["hard_valid"])
        self.assertIn(
            result["status"],
            {"pass", "warning"},
        )

    def test_degenerate_quad_is_rejected(self):
        corners = np.array(
            [
                [100, 100],
                [300, 100],
                [500, 100],
                [700, 100],
            ],
            dtype=np.float32,
        )

        result = self.validator.validate_corners(
            (720, 960, 3),
            corners,
        )

        self.assertFalse(result["hard_valid"])
        self.assertEqual(result["status"], "review")

    def test_tiny_false_candidate_is_rejected(self):
        corners = np.array(
            [
                [450, 330],
                [485, 330],
                [485, 350],
                [450, 350],
            ],
            dtype=np.float32,
        )

        result = self.validator.validate_corners(
            (720, 960, 3),
            corners,
        )

        self.assertFalse(result["hard_valid"])
        self.assertIn(
            "area kartu terlalu kecil",
            result["warnings"],
        )

    def test_normalized_ktp_output_passes_hard_gate(self):
        image = np.full(
            (340, 539, 3),
            150,
            dtype=np.uint8,
        )

        result = self.validator.validate_output(
            image,
            detection_metadata={"score": 0.82},
            quality={"status": "pass"},
        )

        self.assertTrue(result["hard_valid"])
        self.assertEqual(result["status"], "pass")

    def test_bad_output_ratio_is_rejected(self):
        image = np.full(
            (340, 340, 3),
            150,
            dtype=np.uint8,
        )

        result = self.validator.validate_output(
            image,
            detection_metadata={"score": 0.82},
            quality={"status": "pass"},
        )

        self.assertFalse(result["hard_valid"])
        self.assertIn(
            "rasio output tidak sesuai KTP",
            result["warnings"],
        )


if __name__ == "__main__":
    unittest.main()
