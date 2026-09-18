import unittest

import numpy as np

from autodocscanner.ktp.extraction import (
    KtpExtractionResult,
    extract_ktp_regions,
)
from autodocscanner.ktp.layout import (
    KTP_FIELD_BOXES,
)


class TestKtpExtraction(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )

        # Give the image non-zero content so crops are easy to validate.
        for row in range(
            self.image.shape[0]
        ):
            self.image[
                row,
                :,
                0,
            ] = row % 255

    def test_extract_returns_all_defined_fields(self):
        result = extract_ktp_regions(
            self.image
        )

        self.assertIsInstance(
            result,
            KtpExtractionResult,
        )
        self.assertEqual(
            set(result.fields),
            set(KTP_FIELD_BOXES),
        )

    def test_each_field_has_non_empty_crop_and_pixel_box(self):
        result = extract_ktp_regions(
            self.image
        )

        for name, field in result.fields.items():
            with self.subTest(name=name):
                self.assertEqual(
                    field.name,
                    name,
                )
                self.assertGreater(
                    field.image.size,
                    0,
                )
                self.assertEqual(
                    len(field.pixel_box),
                    4,
                )
                x1, y1, x2, y2 = (
                    field.pixel_box
                )
                self.assertLess(
                    x1,
                    x2,
                )
                self.assertLess(
                    y1,
                    y2,
                )

    def test_face_image_is_foto_crop(self):
        result = extract_ktp_regions(
            self.image
        )

        self.assertTrue(
            np.array_equal(
                result.face_image,
                result.fields[
                    "foto"
                ].image,
            )
        )

    def test_source_image_is_independent_copy(self):
        result = extract_ktp_regions(
            self.image
        )

        result.source_image[:] = 255

        self.assertFalse(
            np.all(
                self.image == 255
            )
        )

    def test_field_crops_are_independent_copies(self):
        result = extract_ktp_regions(
            self.image
        )

        original = self.image.copy()
        result.fields[
            "nama"
        ].image[:] = 255

        self.assertTrue(
            np.array_equal(
                self.image,
                original,
            )
        )

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            extract_ktp_regions(
                np.array(
                    [],
                    dtype=np.uint8,
                )
            )

        with self.assertRaises(
            ValueError
        ):
            extract_ktp_regions(
                np.zeros(
                    (10, 10, 3),
                    dtype=np.float32,
                )
            )


if __name__ == "__main__":
    unittest.main()
