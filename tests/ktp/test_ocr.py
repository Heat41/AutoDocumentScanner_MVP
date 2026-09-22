import unittest

import numpy as np

from autodocscanner.ktp.ocr import (
    OcrReadResult,
    preprocess_field,
    read_field_ocr,
)


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def read(self, image, field_name):
        self.calls.append(
            (image.copy(), field_name)
        )
        return self.responses.pop(0)


class TestKtpOcr(unittest.TestCase):
    def setUp(self):
        self.image = np.full(
            (32, 120, 3),
            180,
            dtype=np.uint8,
        )

    def test_preprocess_returns_grayscale_uint8(self):
        result = preprocess_field(
            self.image,
            "nama",
        )

        self.assertEqual(result.ndim, 2)
        self.assertEqual(
            result.dtype,
            np.uint8,
        )
        self.assertGreater(
            result.shape[1],
            self.image.shape[1],
        )

    def test_fallback_preprocess_keeps_valid_image(self):
        result = preprocess_field(
            self.image,
            "alamat",
            fallback=True,
        )

        self.assertEqual(result.ndim, 2)
        self.assertGreater(result.size, 0)

    def test_primary_result_is_used_when_confident(self):
        backend = FakeBackend(
            [("NAMA CONTOH", 88.0)]
        )

        result = read_field_ocr(
            self.image,
            "nama",
            backend=backend,
            confidence_threshold=55.0,
        )

        self.assertIsInstance(
            result,
            OcrReadResult,
        )
        self.assertEqual(
            result.raw_text,
            "NAMA CONTOH",
        )
        self.assertEqual(
            result.confidence,
            88.0,
        )
        self.assertFalse(
            result.used_fallback
        )
        self.assertEqual(
            len(backend.calls),
            1,
        )

    def test_low_confidence_runs_one_fallback(self):
        backend = FakeBackend(
            [
                ("", 20.0),
                ("HASIL FALLBACK", 73.0),
            ]
        )

        result = read_field_ocr(
            self.image,
            "alamat",
            backend=backend,
            confidence_threshold=55.0,
        )

        self.assertEqual(
            result.raw_text,
            "HASIL FALLBACK",
        )
        self.assertEqual(
            result.confidence,
            73.0,
        )
        self.assertTrue(
            result.used_fallback
        )
        self.assertEqual(
            len(backend.calls),
            2,
        )

    def test_better_primary_is_kept_if_fallback_is_worse(self):
        backend = FakeBackend(
            [
                ("PRIMARY", 45.0),
                ("FALLBACK", 30.0),
            ]
        )

        result = read_field_ocr(
            self.image,
            "nama",
            backend=backend,
            confidence_threshold=55.0,
        )

        self.assertEqual(
            result.raw_text,
            "PRIMARY",
        )
        self.assertEqual(
            result.confidence,
            45.0,
        )
        self.assertFalse(
            result.used_fallback
        )


if __name__ == "__main__":
    unittest.main()
