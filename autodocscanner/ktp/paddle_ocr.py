import json

import cv2
import numpy as np

from autodocscanner.ktp.ocr import (
    TesseractBackend,
)


_MODEL_CACHE = {}


class PaddleTextRecognitionBackend:
    def __init__(
        self,
        model_name="PP-OCRv5_mobile_rec",
        device="cpu",
        model_factory=None,
    ):
        self.model_name = model_name
        self.device = device
        self.model_factory = model_factory
        self._model = None

    @staticmethod
    def _import_text_recognition():
        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR belum terpasang. "
                "Install requirements-paddleocr.txt."
            ) from exc

        return TextRecognition

    def _load_model(self):
        if self._model is not None:
            return self._model

        if self.model_factory is not None:
            self._model = self.model_factory()
            return self._model

        key = (
            self.model_name,
            self.device,
        )

        if key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[
                key
            ]
            return self._model

        TextRecognition = (
            self._import_text_recognition()
        )

        model = TextRecognition(
            model_name=self.model_name,
            device=self.device,
        )

        _MODEL_CACHE[key] = model
        self._model = model
        return model

    @staticmethod
    def _payload_from_result(
        result,
    ):
        payload = getattr(
            result,
            "json",
            result,
        )

        if callable(payload):
            payload = payload()

        if isinstance(
            payload,
            str,
        ):
            payload = json.loads(
                payload
            )

        if not isinstance(
            payload,
            dict,
        ):
            return {}

        nested = payload.get(
            "res",
            payload,
        )

        if isinstance(
            nested,
            dict,
        ):
            return nested

        return {}

    def is_available(self):
        try:
            self._import_text_recognition()
            return True
        except Exception:
            return False

    def read(
        self,
        image,
        field_name,
    ):
        if (
            not isinstance(
                image,
                np.ndarray,
            )
            or image.size == 0
        ):
            raise ValueError(
                "image PaddleOCR tidak boleh kosong."
            )

        model = self._load_model()

        paddle_image = image

        if image.ndim == 2:
            paddle_image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )
        elif (
            image.ndim == 3
            and image.shape[2] == 1
        ):
            paddle_image = cv2.cvtColor(
                image[:, :, 0],
                cv2.COLOR_GRAY2BGR,
            )

        try:
            results = model.predict(
                input=paddle_image,
                batch_size=1,
            )
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR gagal membaca field "
                f"{field_name}."
            ) from exc

        results = list(
            results or []
        )

        if not results:
            return ("", 0.0)

        payload = (
            self._payload_from_result(
                results[0]
            )
        )

        text = str(
            payload.get(
                "rec_text",
                "",
            )
            or ""
        ).strip()

        try:
            score = float(
                payload.get(
                    "rec_score",
                    0.0,
                )
                or 0.0
            )
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        if score <= 1.0:
            score *= 100.0

        confidence = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        return (
            text,
            confidence,
        )


class TrackingOcrBackend:
    def __init__(
        self,
        paddle_backend=None,
        tesseract_backend=None,
    ):
        self.paddle_backend = (
            paddle_backend
            if paddle_backend is not None
            else PaddleTextRecognitionBackend()
        )
        self.tesseract_backend = (
            tesseract_backend
            if tesseract_backend is not None
            else TesseractBackend()
        )

    def read_candidates(
        self,
        image,
        field_name,
    ):
        candidates = []

        try:
            paddle_result = (
                self.paddle_backend.read(
                    image,
                    field_name,
                )
            )
            if (
                paddle_result[0]
                or paddle_result[1] > 0
            ):
                candidates.append(
                    paddle_result
                )
        except Exception:
            pass

        try:
            tesseract_result = (
                self.tesseract_backend.read(
                    image,
                    field_name,
                )
            )
            if (
                tesseract_result[0]
                or tesseract_result[1] > 0
            ):
                candidates.append(
                    tesseract_result
                )
        except Exception:
            pass

        return candidates

    def read(
        self,
        image,
        field_name,
    ):
        candidates = (
            self.read_candidates(
                image,
                field_name,
            )
        )

        if not candidates:
            return ("", 0.0)

        return max(
            candidates,
            key=lambda item: (
                item[1],
                len(
                    item[0]
                ),
            ),
        )

    def read_layout(
        self,
        image,
    ):
        return (
            self.tesseract_backend
            .read_layout(
                image
            )
        )

    def read_document(
        self,
        image,
    ):
        return (
            self.tesseract_backend
            .read_document(
                image
            )
        )
