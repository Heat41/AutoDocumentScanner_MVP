import unittest

import numpy as np

from autodocscanner.ktp.layout import (
    KTP_CANONICAL_HEIGHT,
    KTP_CANONICAL_WIDTH,
    KTP_FIELD_BOXES,
    KTP_VALUE_BOXES,
    NormalizedBox,
    normalize_ktp_for_tracking,
    crop_normalized,
    normalized_to_pixel_box,
)


EXPECTED_FIELDS = {
    "provinsi",
    "kabupaten_kota",
    "nik",
    "nama",
    "ttl",
    "jenis_kelamin",
    "golongan_darah",
    "alamat",
    "rt_rw",
    "kelurahan_desa",
    "kecamatan",
    "agama",
    "status_perkawinan",
    "pekerjaan",
    "kewarganegaraan",
    "berlaku_hingga",
    "foto",
}


class TestKtpLayout(unittest.TestCase):
    def test_layout_contains_all_stage2d_fields(self):
        self.assertEqual(
            set(KTP_FIELD_BOXES),
            EXPECTED_FIELDS,
        )

    def test_value_layout_contains_all_text_fields(self):
        self.assertEqual(
            set(KTP_VALUE_BOXES),
            EXPECTED_FIELDS - {"foto"},
        )

    def test_tracking_normalization_uses_canonical_ktp_size(self):
        image = np.zeros(
            (270, 428, 3),
            dtype=np.uint8,
        )

        normalized = normalize_ktp_for_tracking(
            image
        )

        self.assertEqual(
            normalized.shape[:2],
            (
                KTP_CANONICAL_HEIGHT,
                KTP_CANONICAL_WIDTH,
            ),
        )

    def test_all_boxes_are_normalized_and_non_empty(self):
        for name, box in KTP_FIELD_BOXES.items():
            with self.subTest(name=name):
                self.assertIsInstance(
                    box,
                    NormalizedBox,
                )
                self.assertGreaterEqual(box.x1, 0.0)
                self.assertGreaterEqual(box.y1, 0.0)
                self.assertLessEqual(box.x2, 1.0)
                self.assertLessEqual(box.y2, 1.0)
                self.assertLess(box.x1, box.x2)
                self.assertLess(box.y1, box.y2)

    def test_normalized_box_rejects_invalid_coordinates(self):
        with self.assertRaises(ValueError):
            NormalizedBox(
                x1=0.8,
                y1=0.1,
                x2=0.2,
                y2=0.5,
            )

        with self.assertRaises(ValueError):
            NormalizedBox(
                x1=-0.1,
                y1=0.1,
                x2=0.5,
                y2=0.5,
            )

    def test_normalized_to_pixel_box_scales_and_clamps(self):
        box = NormalizedBox(
            0.10,
            0.20,
            0.90,
            0.80,
        )

        self.assertEqual(
            normalized_to_pixel_box(
                box,
                image_width=1000,
                image_height=500,
            ),
            (100, 100, 900, 400),
        )

    def test_crop_normalized_returns_independent_copy(self):
        image = np.zeros(
            (100, 200, 3),
            dtype=np.uint8,
        )
        image[20:80, 20:180] = (
            10,
            20,
            30,
        )

        crop = crop_normalized(
            image,
            NormalizedBox(
                0.10,
                0.20,
                0.90,
                0.80,
            ),
        )

        self.assertEqual(
            crop.shape,
            (60, 160, 3),
        )

        crop[:] = 255

        self.assertFalse(
            np.all(
                image[
                    20:80,
                    20:180,
                ]
                == 255
            )
        )

    def test_crop_rejects_empty_image(self):
        with self.assertRaises(ValueError):
            crop_normalized(
                np.array(
                    [],
                    dtype=np.uint8,
                ),
                NormalizedBox(
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ),
            )

    def test_value_rows_follow_standard_ktp_vertical_order(self):
        ordered = (
            "nik",
            "nama",
            "ttl",
            "jenis_kelamin",
            "alamat",
            "rt_rw",
            "kelurahan_desa",
            "kecamatan",
            "agama",
            "status_perkawinan",
            "pekerjaan",
            "kewarganegaraan",
            "berlaku_hingga",
        )

        centers = [
            (
                KTP_VALUE_BOXES[name].y1
                + KTP_VALUE_BOXES[name].y2
            ) / 2
            for name in ordered
        ]

        self.assertEqual(
            centers,
            sorted(centers),
        )

    def test_main_value_regions_start_to_right_of_labels(self):
        for name in (
            "nik",
            "nama",
            "ttl",
            "alamat",
            "rt_rw",
            "kelurahan_desa",
            "kecamatan",
            "pekerjaan",
        ):
            with self.subTest(name=name):
                self.assertGreaterEqual(
                    KTP_VALUE_BOXES[name].x1,
                    0.35,
                )


if __name__ == "__main__":
    unittest.main()
