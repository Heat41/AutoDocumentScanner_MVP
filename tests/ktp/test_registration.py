import unittest

import numpy as np

from autodocscanner.ktp.registration import (
    register_ktp_to_template,
)


class TestKtpRegistration(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )

    def test_registration_is_skipped_with_too_few_anchors(self):
        result = register_ktp_to_template(
            self.image,
            {
                "nama": (
                    80.0,
                    140.0,
                    90.0,
                ),
            },
        )

        self.assertFalse(
            result.applied
        )
        self.assertEqual(
            result.match_count,
            1,
        )
        self.assertTrue(
            np.array_equal(
                result.image,
                self.image,
            )
        )

    def test_registration_estimates_small_vertical_shift(self):
        anchors = {
            "nik": (
                80.0,
                120.0,
                90.0,
            ),
            "nama": (
                80.0,
                170.0,
                90.0,
            ),
            "alamat": (
                80.0,
                260.0,
                90.0,
            ),
            "kecamatan": (
                80.0,
                330.0,
                90.0,
            ),
        }

        result = register_ktp_to_template(
            self.image,
            anchors,
        )

        self.assertTrue(
            result.applied
        )
        self.assertGreaterEqual(
            result.match_count,
            3,
        )
        self.assertGreater(
            result.scale_y,
            0.90,
        )
        self.assertLess(
            result.scale_y,
            1.10,
        )

    def test_registration_rejects_extreme_transform(self):
        anchors = {
            "nik": (
                80.0,
                20.0,
                90.0,
            ),
            "nama": (
                80.0,
                400.0,
                90.0,
            ),
            "alamat": (
                80.0,
                500.0,
                90.0,
            ),
        }

        result = register_ktp_to_template(
            self.image,
            anchors,
        )

        self.assertFalse(
            result.applied
        )


if __name__ == "__main__":
    unittest.main()
