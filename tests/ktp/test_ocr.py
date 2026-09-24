import unittest

import numpy as np

from autodocscanner.ktp.ocr import (
    OcrReadResult,
    OcrWord,
    preprocess_document,
    preprocess_field,
    read_field_ocr,
    read_field_ocr_candidates,
    read_numeric_fragment,
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

    def test_preprocess_document_returns_grayscale_uint8(self):
        image = np.full(
            (120, 200, 3),
            170,
            dtype=np.uint8,
        )

        result = preprocess_document(
            image
        )

        self.assertEqual(
            result.ndim,
            2,
        )
        self.assertEqual(
            result.dtype,
            np.uint8,
        )
        self.assertGreater(
            result.shape[1],
            image.shape[1],
        )

    def test_ocr_word_exposes_position_and_line_identity(self):
        word = OcrWord(
            text="Nama",
            confidence=90.0,
            left=10,
            top=20,
            width=40,
            height=15,
            block=1,
            paragraph=1,
            line=2,
        )

        self.assertEqual(
            word.right,
            50,
        )
        self.assertEqual(
            word.bottom,
            35,
        )
        self.assertEqual(
            word.line_key,
            (1, 1, 2),
        )

    def test_numeric_fragment_selects_consensus_window_from_noisy_reads(self):
        backend = FakeBackend(
            [
                ("11001", 90.0),
                ("1001", 86.0),
                ("10017", 80.0),
                ("1001", 88.0),
            ]
        )

        result = read_numeric_fragment(
            self.image,
            expected_length=4,
            backend=backend,
        )

        self.assertEqual(
            result.raw_text,
            "1001",
        )

    def test_multi_pass_returns_four_candidates(self):
        backend = FakeBackend(
            [
                ("PASS SATU", 40.0),
                ("PASS DUA", 70.0),
                ("PASS TIGA", 65.0),
                ("PASS EMPAT", 75.0),
            ]
        )

        results = read_field_ocr_candidates(
            self.image,
            "nama",
            backend=backend,
        )

        self.assertEqual(
            len(results),
            4,
        )
        self.assertEqual(
            results[1].raw_text,
            "PASS DUA",
        )
        self.assertEqual(
            results[3].raw_text,
            "PASS EMPAT",
        )


if __name__ == "__main__":
    unittest.main()
