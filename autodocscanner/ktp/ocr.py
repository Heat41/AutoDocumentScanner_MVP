from dataclasses import dataclass

import cv2
import numpy as np


MULTILINE_FIELDS = {
    "alamat",
}

DIGIT_FIELDS = {
    "nik",
}

OCR_PSM_BY_FIELD = {
    "alamat": 6,
}


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    block: int
    paragraph: int
    line: int

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def line_key(self):
        return (
            self.block,
            self.paragraph,
            self.line,
        )


@dataclass(frozen=True)
class OcrReadResult:
    raw_text: str
    confidence: float
    used_fallback: bool = False


def _validate_image(image):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "image OCR harus berupa numpy array yang tidak kosong."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            "image OCR harus bertipe uint8."
        )


def _to_gray(image):
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )

    raise ValueError(
        "Format image OCR tidak didukung."
    )


def preprocess_document(
    image,
    fallback=False,
):
    _validate_image(image)
    gray = _to_gray(image)

    height, width = gray.shape[:2]
    target_width = max(
        1600,
        width * 2,
    )
    scale = (
        target_width
        / max(width, 1)
    )
    target_height = max(
        2,
        int(round(height * scale)),
    )

    enlarged = cv2.resize(
        gray,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )

    if fallback:
        return cv2.adaptiveThreshold(
            cv2.GaussianBlur(
                enlarged,
                (3, 3),
                0,
            ),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8),
    )
    return clahe.apply(enlarged)


def preprocess_field(
    image,
    field_name,
    fallback=False,
):
    _validate_image(image)
    gray = _to_gray(image)

    height, width = gray.shape[:2]
    target_height = max(
        72,
        height * (
            5
            if field_name == "nik"
            else 4
        ),
    )
    scale = (
        target_height
        / max(height, 1)
    )
    target_width = max(
        2,
        int(round(width * scale)),
    )

    enlarged = cv2.resize(
        gray,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )

    if fallback:
        blurred = cv2.GaussianBlur(
            enlarged,
            (3, 3),
            0,
        )
        return cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    return clahe.apply(enlarged)


def preprocess_field_otsu(
    image,
):
    _validate_image(image)
    gray = _to_gray(image)

    height, width = gray.shape[:2]
    target_height = max(
        48,
        height * 3,
    )
    scale = (
        target_height
        / max(height, 1)
    )
    target_width = max(
        2,
        int(round(width * scale)),
    )

    enlarged = cv2.resize(
        gray,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )
    blurred = cv2.GaussianBlur(
        enlarged,
        (3, 3),
        0,
    )

    _threshold, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )
    return binary


def preprocess_field_strong(
    image,
    field_name,
):
    _validate_image(image)
    gray = _to_gray(image)

    height, width = gray.shape[:2]
    target_height = max(
        96,
        height * (
            6
            if field_name == "nik"
            else 4
        ),
    )
    scale = (
        target_height
        / max(height, 1)
    )
    target_width = max(
        2,
        int(
            round(
                width * scale
            )
        ),
    )

    enlarged = cv2.resize(
        gray,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.6,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(
        enlarged
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0,
    )
    sharpened = cv2.addWeighted(
        enhanced,
        1.8,
        blurred,
        -0.8,
        0,
    )

    if field_name == "nik":
        _threshold, sharpened = (
            cv2.threshold(
                sharpened,
                0,
                255,
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU,
            )
        )

    return sharpened


class TesseractBackend:
    def __init__(
        self,
        language="ind+eng",
        executable_path=None,
    ):
        self.language = language
        self.executable_path = (
            str(executable_path)
            if executable_path
            else None
        )

    @staticmethod
    def _module():
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract belum terpasang. "
                "Jalankan pip install -r requirements.txt."
            ) from exc

        return pytesseract

    def _configure_executable(
        self,
        pytesseract,
    ):
        if self.executable_path:
            pytesseract.pytesseract.tesseract_cmd = (
                self.executable_path
            )

    def is_available(self):
        try:
            pytesseract = self._module()
            self._configure_executable(
                pytesseract
            )
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def _config_for_field(
        field_name,
    ):
        psm = OCR_PSM_BY_FIELD.get(
            field_name,
            7,
        )
        parts = [
            f"--psm {psm}",
        ]

        if field_name in DIGIT_FIELDS:
            parts.append(
                "-c tessedit_char_whitelist=0123456789"
            )

        return " ".join(parts)

    def read_layout(
        self,
        image,
    ):
        pytesseract = self._module()
        self._configure_executable(
            pytesseract
        )

        prepared = preprocess_document(
            image,
            fallback=False,
        )

        source_height, source_width = (
            image.shape[:2]
        )
        prepared_height, prepared_width = (
            prepared.shape[:2]
        )
        scale_x = (
            source_width
            / max(
                prepared_width,
                1,
            )
        )
        scale_y = (
            source_height
            / max(
                prepared_height,
                1,
            )
        )

        try:
            data = (
                pytesseract.image_to_data(
                    prepared,
                    lang=self.language,
                    config="--psm 6",
                    output_type=(
                        pytesseract.Output.DICT
                    ),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Tesseract layout OCR gagal dijalankan."
            ) from exc

        words = []
        count = len(
            data.get(
                "text",
                [],
            )
        )

        for index in range(count):
            text = str(
                data["text"][index]
                or ""
            ).strip()

            if not text:
                continue

            try:
                confidence = float(
                    data.get(
                        "conf",
                        [-1] * count,
                    )[index]
                )
            except (
                TypeError,
                ValueError,
            ):
                confidence = -1.0

            if confidence < 0:
                continue

            words.append(
                OcrWord(
                    text=text,
                    confidence=confidence,
                    left=int(
                        round(
                            int(
                                data.get(
                                    "left",
                                    [0] * count,
                                )[index]
                            )
                            * scale_x
                        )
                    ),
                    top=int(
                        round(
                            int(
                                data.get(
                                    "top",
                                    [0] * count,
                                )[index]
                            )
                            * scale_y
                        )
                    ),
                    width=max(
                        1,
                        int(
                            round(
                                int(
                                    data.get(
                                        "width",
                                        [0] * count,
                                    )[index]
                                )
                                * scale_x
                            )
                        ),
                    ),
                    height=max(
                        1,
                        int(
                            round(
                                int(
                                    data.get(
                                        "height",
                                        [0] * count,
                                    )[index]
                                )
                                * scale_y
                            )
                        ),
                    ),
                    block=int(
                        data.get(
                            "block_num",
                            [0] * count,
                        )[index]
                    ),
                    paragraph=int(
                        data.get(
                            "par_num",
                            [0] * count,
                        )[index]
                    ),
                    line=int(
                        data.get(
                            "line_num",
                            [0] * count,
                        )[index]
                    ),
                )
            )

        return words

    def read_document(
        self,
        image,
    ):
        pytesseract = self._module()
        self._configure_executable(
            pytesseract
        )

        prepared = preprocess_document(
            image,
            fallback=False,
        )

        try:
            data = (
                pytesseract.image_to_data(
                    prepared,
                    lang=self.language,
                    config="--psm 6",
                    output_type=(
                        pytesseract.Output.DICT
                    ),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Tesseract OCR dokumen gagal dijalankan."
            ) from exc

        lines = {}
        confidences = []

        count = len(
            data.get(
                "text",
                [],
            )
        )

        for index in range(count):
            value = str(
                data["text"][
                    index
                ]
                or ""
            ).strip()

            if not value:
                continue

            key = (
                data.get(
                    "block_num",
                    [0] * count,
                )[index],
                data.get(
                    "par_num",
                    [0] * count,
                )[index],
                data.get(
                    "line_num",
                    [0] * count,
                )[index],
            )

            lines.setdefault(
                key,
                [],
            ).append(
                value
            )

            try:
                confidence = float(
                    data.get(
                        "conf",
                        [-1] * count,
                    )[index]
                )
            except (
                TypeError,
                ValueError,
            ):
                confidence = -1

            if confidence >= 0:
                confidences.append(
                    confidence
                )

        text = "\n".join(
            " ".join(words)
            for words in lines.values()
            if words
        ).strip()

        confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        return (
            text,
            float(confidence),
        )

    def read(
        self,
        image,
        field_name,
    ):
        pytesseract = self._module()
        self._configure_executable(
            pytesseract
        )

        try:
            data = (
                pytesseract.image_to_data(
                    image,
                    lang=self.language,
                    config=self._config_for_field(
                        field_name
                    ),
                    output_type=(
                        pytesseract.Output.DICT
                    ),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Tesseract OCR gagal dijalankan. "
                "Pastikan executable Tesseract dan language data tersedia."
            ) from exc

        texts = []
        confidences = []

        for text, confidence in zip(
            data.get("text", []),
            data.get("conf", []),
        ):
            value = str(text or "").strip()
            if not value:
                continue

            texts.append(value)

            try:
                numeric_confidence = float(
                    confidence
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if numeric_confidence >= 0:
                confidences.append(
                    numeric_confidence
                )

        raw_text = " ".join(
            texts
        ).strip()

        confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        return (
            raw_text,
            float(confidence),
        )


def _read_once(
    image,
    field_name,
    backend,
    fallback,
):
    prepared = preprocess_field(
        image,
        field_name,
        fallback=fallback,
    )
    text, confidence = backend.read(
        prepared,
        field_name,
    )

    return OcrReadResult(
        raw_text=str(
            text or ""
        ).strip(),
        confidence=max(
            0.0,
            min(
                100.0,
                float(
                    confidence or 0.0
                ),
            ),
        ),
        used_fallback=fallback,
    )


def read_field_ocr(
    image,
    field_name,
    backend=None,
    confidence_threshold=55.0,
):
    backend = (
        backend
        if backend is not None
        else TesseractBackend()
    )

    primary = _read_once(
        image,
        field_name,
        backend,
        fallback=False,
    )

    if (
        primary.raw_text
        and primary.confidence
        >= confidence_threshold
    ):
        return primary

    fallback = _read_once(
        image,
        field_name,
        backend,
        fallback=True,
    )

    if (
        fallback.confidence
        > primary.confidence
        or (
            not primary.raw_text
            and bool(
                fallback.raw_text
            )
        )
    ):
        return fallback

    return primary


def read_numeric_fragment(
    image,
    expected_length,
    backend=None,
):
    backend = (
        backend
        if backend is not None
        else TesseractBackend()
    )
    expected_length = max(
        int(expected_length),
        1,
    )

    candidates = (
        read_field_ocr_candidates(
            image,
            "nik",
            backend=backend,
        )
    )

    window_scores = {}
    window_confidence = {}
    window_fallback = {}

    for candidate in candidates:
        digits = "".join(
            char
            for char in candidate.raw_text
            if char.isdigit()
        )

        if len(digits) < expected_length:
            continue

        windows = {
            digits[
                index:
                index
                + expected_length
            ]
            for index in range(
                0,
                len(digits)
                - expected_length
                + 1,
            )
        }

        for window in windows:
            score = (
                1.0
                + float(
                    candidate.confidence
                )
                / 100.0
            )
            window_scores[
                window
            ] = (
                window_scores.get(
                    window,
                    0.0,
                )
                + score
            )
            window_confidence[
                window
            ] = max(
                window_confidence.get(
                    window,
                    0.0,
                ),
                float(
                    candidate.confidence
                ),
            )
            window_fallback[
                window
            ] = (
                window_fallback.get(
                    window,
                    False,
                )
                or bool(
                    candidate.used_fallback
                )
            )

    if window_scores:
        best_window = max(
            window_scores,
            key=lambda value: (
                window_scores[value],
                window_confidence[
                    value
                ],
            ),
        )

        return OcrReadResult(
            raw_text=best_window,
            confidence=(
                window_confidence[
                    best_window
                ]
            ),
            used_fallback=(
                window_fallback[
                    best_window
                ]
            ),
        )

    return OcrReadResult(
        raw_text="",
        confidence=0.0,
        used_fallback=True,
    )

def read_field_ocr_candidates(
    image,
    field_name,
    backend=None,
):
    backend = (
        backend
        if backend is not None
        else TesseractBackend()
    )

    prepared_images = (
        preprocess_field(
            image,
            field_name,
            fallback=False,
        ),
        preprocess_field(
            image,
            field_name,
            fallback=True,
        ),
        preprocess_field_otsu(
            image,
        ),
        preprocess_field_strong(
            image,
            field_name,
        ),
    )

    results = []

    for index, prepared in enumerate(
        prepared_images
    ):
        text, confidence = backend.read(
            prepared,
            field_name,
        )
        results.append(
            OcrReadResult(
                raw_text=str(
                    text or ""
                ).strip(),
                confidence=max(
                    0.0,
                    min(
                        100.0,
                        float(
                            confidence or 0.0
                        ),
                    ),
                ),
                used_fallback=(
                    index > 0
                ),
            )
        )

    return results
