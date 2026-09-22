import json
import unittest

import numpy as np

from autodocscanner.ktp.paddle_ocr import (
    PaddleTextRecognitionBackend,
    TrackingOcrBackend,
)


class FakeResult:
    def __init__(self, text, score):
        self.json = {
            "res": {
                "rec_text": text,
                "rec_score": score,
            }
        }


class FakeModel:
    def __init__(self, text="NAMA CONTOH", score=0.93):
        self.text = text
        self.score = score
        self.calls = 0

    def predict(self, input, batch_size=1):
        self.calls += 1
        return [
            FakeResult(
                self.text,
                self.score,
            )
        ]


class FakeBackend:
    def __init__(self, response):
        self.response = response
        self.read_calls = 0
        self.layout_calls = 0
        self.document_calls = 0

    def read(self, image, field_name):
        self.read_calls += 1
        return self.response

    def read_layout(self, image):
        self.layout_calls += 1
        return []

    def read_document(self, image):
        self.document_calls += 1
        return ("", 0.0)


class TestPaddleTextRecognitionBackend(unittest.TestCase):
    def test_reads_text_and_converts_score_to_percent(self):
        model = FakeModel(
            text="SITI ISNAINI",
            score=0.91,
        )
        backend = PaddleTextRecognitionBackend(
            model_factory=lambda: model,
        )

        text, confidence = backend.read(
            np.zeros(
                (50, 200, 3),
                dtype=np.uint8,
            ),
            "nama",
        )

        self.assertEqual(
            text,
            "SITI ISNAINI",
        )
        self.assertAlmostEqual(
            confidence,
            91.0,
        )
        self.assertEqual(
            model.calls,
            1,
        )

    def test_model_is_reused_between_reads(self):
        model = FakeModel()
        created = []

        def factory():
            created.append(True)
            return model

        backend = PaddleTextRecognitionBackend(
            model_factory=factory,
        )

        image = np.zeros(
            (40, 160, 3),
            dtype=np.uint8,
        )

        backend.read(image, "nama")
        backend.read(image, "alamat")

        self.assertEqual(
            len(created),
            1,
        )
        self.assertEqual(
            model.calls,
            2,
        )


class TestTrackingOcrBackend(unittest.TestCase):
    def test_candidates_include_paddle_and_tesseract(self):
        paddle = FakeBackend(
            ("PADDLE", 88.0)
        )
        tesseract = FakeBackend(
            ("TESS", 72.0)
        )
        backend = TrackingOcrBackend(
            paddle_backend=paddle,
            tesseract_backend=tesseract,
        )

        candidates = backend.read_candidates(
            np.zeros(
                (40, 160, 3),
                dtype=np.uint8,
            ),
            "nama",
        )

        self.assertEqual(
            candidates,
            [
                ("PADDLE", 88.0),
                ("TESS", 72.0),
            ],
        )

    def test_layout_and_document_delegate_to_tesseract(self):
        paddle = FakeBackend(
            ("PADDLE", 88.0)
        )
        tesseract = FakeBackend(
            ("TESS", 72.0)
        )
        backend = TrackingOcrBackend(
            paddle_backend=paddle,
            tesseract_backend=tesseract,
        )
        image = np.zeros(
            (40, 160, 3),
            dtype=np.uint8,
        )

        backend.read_layout(image)
        backend.read_document(image)

        self.assertEqual(
            tesseract.layout_calls,
            1,
        )
        self.assertEqual(
            tesseract.document_calls,
            1,
        )


if __name__ == "__main__":
    unittest.main()
