import unittest

import cv2
import numpy as np

from autodocscanner.ktp.ocr import (
    OcrReadResult,
    OcrWord,
    _repair_segmented_digit_by_shape,
    preprocess_document,
    preprocess_field,
    read_field_ocr,
    read_field_ocr_candidates,
    read_name_ocr_candidates,
    read_nik_ocr_candidates,
    read_numeric_fragment,
    read_segmented_digits,
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


class ConfiguredFakeBackend(FakeBackend):
    def __init__(
        self,
        configured_responses,
    ):
        super().__init__(
            []
        )
        self.configured_responses = list(
            configured_responses
        )

    def read_configured(
        self,
        image,
        config,
        language=None,
    ):
        return self.configured_responses.pop(
            0
        )


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

    def test_specialized_name_ocr_returns_configured_candidates(self):
        backend = ConfiguredFakeBackend(
            [
                (
                    "SITI ISNAINI",
                    70.0,
                )
            ]
            * 9
        )

        results = read_name_ocr_candidates(
            self.image,
            backend=backend,
        )

        self.assertTrue(
            results
        )
        self.assertEqual(
            results[0].raw_text,
            "SITI ISNAINI",
        )

    def test_specialized_nik_ocr_keeps_numeric_candidates(self):
        backend = ConfiguredFakeBackend(
            [
                (
                    "6110014101980004",
                    74.0,
                )
            ]
            * 9
        )

        results = read_nik_ocr_candidates(
            self.image,
            backend=backend,
        )

        self.assertTrue(
            results
        )
        self.assertEqual(
            results[0].raw_text,
            "6110014101980004",
        )

    def test_shape_repair_changes_open_eight_reading_to_four(self):
        image = np.full(
            (60, 40),
            255,
            dtype=np.uint8,
        )

        cv2.line(
            image,
            (26, 8),
            (26, 50),
            0,
            5,
        )
        cv2.line(
            image,
            (8, 30),
            (31, 30),
            0,
            5,
        )
        cv2.line(
            image,
            (8, 30),
            (22, 8),
            0,
            5,
        )

        self.assertEqual(
            _repair_segmented_digit_by_shape(
                image,
                "8",
            ),
            "4",
        )

    def test_shape_repair_keeps_closed_eight(self):
        image = np.full(
            (70, 45),
            255,
            dtype=np.uint8,
        )

        cv2.circle(
            image,
            (22, 22),
            13,
            0,
            5,
        )
        cv2.circle(
            image,
            (22, 48),
            13,
            0,
            5,
        )

        self.assertEqual(
            _repair_segmented_digit_by_shape(
                image,
                "8",
            ),
            "8",
        )

    def test_segmented_digit_reader_keeps_four_serial_digits(self):
        backend = ConfiguredFakeBackend(
            [
                ("0", 80.0),
                ("0", 78.0),
                ("0", 82.0),
                ("0", 79.0),
                ("0", 84.0),
                ("0", 81.0),
                ("4", 90.0),
                ("4", 88.0),
            ]
        )

        image = np.full(
            (40, 160, 3),
            255,
            dtype=np.uint8,
        )
        cv = 20
        image[
            8:32,
            10:30,
        ] = cv
        image[
            8:32,
            50:70,
        ] = cv
        image[
            8:32,
            90:110,
        ] = cv
        image[
            8:32,
            130:150,
        ] = cv

        result = read_segmented_digits(
            image,
            digit_count=4,
            backend=backend,
        )

        self.assertEqual(
            result.raw_text,
            "0004",
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
