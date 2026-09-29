from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

from autodocscanner.ktp.layout import (
    KTP_FIELD_BOXES,
    build_anchor_aligned_value_boxes,
    normalized_to_pixel_box,
)


FIELD_CLASSES = (
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
)


@dataclass(frozen=True)
class FieldDetection:
    class_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    source: str

    def __post_init__(self):
        if self.class_name not in FIELD_CLASSES:
            raise ValueError(
                f"Class detector KTP tidak dikenal: {self.class_name}"
            )

        if len(self.bbox) != 4:
            raise ValueError(
                "bbox harus berisi x1, y1, x2, y2."
            )


def _clamp_bbox(
    bbox,
    width,
    height,
):
    x1, y1, x2, y2 = [
        int(round(value))
        for value in bbox
    ]

    x1 = max(
        0,
        min(width - 1, x1),
    )
    y1 = max(
        0,
        min(height - 1, y1),
    )
    x2 = max(
        x1 + 1,
        min(width, x2),
    )
    y2 = max(
        y1 + 1,
        min(height, y2),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def crop_detection(
    image,
    detection,
):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "image detector harus berupa numpy array yang tidak kosong."
        )

    if not isinstance(
        detection,
        FieldDetection,
    ):
        raise TypeError(
            "detection harus berupa FieldDetection."
        )

    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(
        detection.bbox,
        width,
        height,
    )

    return image[
        y1:y2,
        x1:x2,
    ].copy()


def crop_detection_padded(
    image,
    detection,
    pad_x_ratio=0.018,
    pad_y_ratio=0.12,
):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "image detector harus berupa numpy array yang tidak kosong."
        )

    if not isinstance(
        detection,
        FieldDetection,
    ):
        raise TypeError(
            "detection harus berupa FieldDetection."
        )

    height, width = image.shape[:2]
    x1, y1, x2, y2 = detection.bbox

    box_width = max(
        int(x2) - int(x1),
        1,
    )
    box_height = max(
        int(y2) - int(y1),
        1,
    )

    pad_x = max(
        2,
        int(
            round(
                box_width
                * float(pad_x_ratio)
            )
        ),
    )
    pad_y = max(
        1,
        int(
            round(
                box_height
                * float(pad_y_ratio)
            )
        ),
    )

    padded = _clamp_bbox(
        (
            x1 - pad_x,
            y1 - pad_y,
            x2 + pad_x,
            y2 + pad_y,
        ),
        width,
        height,
    )

    px1, py1, px2, py2 = padded

    return image[
        py1:py2,
        px1:px2,
    ].copy()


def best_detection_by_class(
    detections,
):
    result = {}

    for item in detections or []:
        if not isinstance(
            item,
            FieldDetection,
        ):
            continue

        previous = result.get(
            item.class_name
        )

        if (
            previous is None
            or item.confidence
            > previous.confidence
        ):
            result[
                item.class_name
            ] = item

    return result


def _runtime_root():
    frozen_root = getattr(
        sys,
        "_MEIPASS",
        None,
    )

    if frozen_root:
        return Path(
            frozen_root
        )

    return Path(
        __file__
    ).resolve().parents[2]


def _robust_median(
    values,
):
    data = np.asarray(
        list(values),
        dtype=np.float32,
    )

    if data.size == 0:
        return 0.0

    median = float(
        np.median(
            data
        )
    )
    deviation = np.abs(
        data - median
    )
    mad = float(
        np.median(
            deviation
        )
    )

    if mad <= 1e-6:
        return median

    keep = deviation <= (
        3.5
        * mad
    )
    filtered = data[
        keep
    ]

    if filtered.size == 0:
        return median

    return float(
        np.median(
            filtered
        )
    )


def _estimate_saved_template_transform(
    saved_boxes,
    anchors,
    image_width,
    image_height,
):
    samples = []

    for name, anchor in dict(
        anchors or {}
    ).items():
        bbox = saved_boxes.get(
            name
        )

        if bbox is None:
            continue

        try:
            anchor_x, anchor_y, confidence = (
                anchor
            )
            anchor_x = float(
                anchor_x
            )
            anchor_y = float(
                anchor_y
            )
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if confidence < 35.0:
            continue

        x1, y1, _x2, y2 = bbox
        samples.append(
            (
                float(
                    x1
                ),
                float(
                    (
                        y1 + y2
                    )
                    / 2.0
                ),
                anchor_x,
                anchor_y,
                confidence,
            )
        )

    if len(samples) < 3:
        return {
            "applied": False,
            "scale_y": 1.0,
            "shift_x": 0.0,
            "shift_y": 0.0,
            "sample_count": len(
                samples
            ),
        }

    y_scales = []

    for index, first in enumerate(
        samples
    ):
        for second in samples[
            index + 1:
        ]:
            template_delta = (
                second[1]
                - first[1]
            )
            detected_delta = (
                second[3]
                - first[3]
            )

            if abs(
                template_delta
            ) < 40.0:
                continue

            if (
                template_delta
                * detected_delta
                <= 0.0
            ):
                continue

            scale = (
                detected_delta
                / template_delta
            )

            if 0.90 <= scale <= 1.10:
                y_scales.append(
                    scale
                )

    scale_y = (
        _robust_median(
            y_scales
        )
        if y_scales
        else 1.0
    )
    scale_y = max(
        0.94,
        min(
            1.06,
            scale_y,
        ),
    )

    shift_x = _robust_median(
        anchor_x - template_x
        for (
            template_x,
            _template_y,
            anchor_x,
            _anchor_y,
            _confidence,
        ) in samples
    )
    shift_y = _robust_median(
        anchor_y
        - scale_y
        * template_y
        for (
            _template_x,
            template_y,
            _anchor_x,
            anchor_y,
            _confidence,
        ) in samples
    )

    shift_x = max(
        -0.035
        * float(
            image_width
        ),
        min(
            0.035
            * float(
                image_width
            ),
            shift_x,
        ),
    )
    shift_y = max(
        -0.045
        * float(
            image_height
        ),
        min(
            0.045
            * float(
                image_height
            ),
            shift_y,
        ),
    )

    return {
        "applied": True,
        "scale_y": float(
            scale_y
        ),
        "shift_x": float(
            shift_x
        ),
        "shift_y": float(
            shift_y
        ),
        "sample_count": len(
            samples
        ),
    }


def _local_anchor_bbox(
    class_name,
    bbox,
    anchors,
    image_width,
    image_height,
):
    anchor = dict(
        anchors or {}
    ).get(
        class_name
    )

    if anchor is None:
        return None

    try:
        anchor_x, anchor_y, confidence = (
            anchor
        )
        anchor_x = float(
            anchor_x
        )
        anchor_y = float(
            anchor_y
        )
        confidence = float(
            confidence
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if confidence < 50.0:
        return None

    x1, y1, x2, y2 = [
        float(
            value
        )
        for value in bbox
    ]

    template_width = max(
        x2 - x1,
        1.0,
    )
    template_height = max(
        y2 - y1,
        1.0,
    )

    value_bbox = getattr(
        anchor,
        "value_bbox",
        None,
    )

    if (
        value_bbox is not None
        and len(
            value_bbox
        ) == 4
    ):
        (
            value_x1,
            value_y1,
            value_x2,
            value_y2,
        ) = [
            float(
                value
            )
            for value in value_bbox
        ]

        value_width = max(
            value_x2 - value_x1,
            1.0,
        )
        value_height = max(
            value_y2 - value_y1,
            1.0,
        )

        pad_left = max(
            2.0,
            min(
                5.0,
                value_height * 0.22,
            ),
        )
        pad_right = max(
            4.0,
            min(
                10.0,
                value_height * 0.45,
            ),
        )
        pad_y = max(
            2.0,
            min(
                5.0,
                value_height * 0.28,
            ),
        )

        local_x1 = (
            value_x1
            - pad_left
        )
        local_x2 = (
            value_x2
            + pad_right
        )
        local_y1 = (
            value_y1
            - pad_y
        )
        local_y2 = (
            value_y2
            + pad_y
        )

        # Overlay mengikuti value aktual, tetapi jangan sampai menjadi
        # sangat kecil akibat Tesseract hanya menangkap satu karakter.
        min_width = min(
            template_width,
            max(
                24.0,
                template_width * 0.28,
            ),
        )
        current_width = (
            local_x2
            - local_x1
        )

        if current_width < min_width:
            local_x2 = (
                local_x1
                + min_width
            )

        # Jangan melampaui lebar template terlalu jauh.
        max_width = (
            template_width
            * 1.05
        )

        if (
            local_x2
            - local_x1
            > max_width
        ):
            local_x2 = (
                local_x1
                + max_width
            )

        # Tinggi overlay dibuat rapat ke glyph, tetapi tetap cukup aman.
        min_height = max(
            10.0,
            min(
                template_height,
                template_height * 0.55,
            ),
        )

        if (
            local_y2
            - local_y1
            < min_height
        ):
            center_y = (
                local_y1
                + local_y2
            ) / 2.0
            local_y1 = (
                center_y
                - min_height / 2.0
            )
            local_y2 = (
                center_y
                + min_height / 2.0
            )

    else:
        # Fallback kompatibel untuk anchor lama/tanpa word bounds.
        left_padding = max(
            2.0,
            min(
                8.0,
                template_width * 0.025,
            ),
        )

        local_x1 = (
            anchor_x
            - left_padding
        )
        local_x2 = (
            local_x1
            + template_width
        )
        local_y1 = (
            anchor_y
            - template_height / 2.0
        )
        local_y2 = (
            local_y1
            + template_height
        )

    # Simpan ukuran tight box sebelum posisi dibatasi.
    tight_width = max(
        local_x2 - local_x1,
        1.0,
    )
    tight_height = max(
        local_y2 - local_y1,
        1.0,
    )

    max_x_shift = (
        0.12
        * float(
            image_width
        )
    )
    max_y_shift = (
        0.10
        * float(
            image_height
        )
    )

    local_x1 = max(
        x1 - max_x_shift,
        min(
            x1 + max_x_shift,
            local_x1,
        ),
    )
    local_y1 = max(
        y1 - max_y_shift,
        min(
            y1 + max_y_shift,
            local_y1,
        ),
    )

    # Pertahankan ukuran tight box setelah clamp posisi.
    local_x2 = (
        local_x1
        + tight_width
    )
    local_y2 = (
        local_y1
        + tight_height
    )

    return _clamp_bbox(
        (
            local_x1,
            local_y1,
            local_x2,
            local_y2,
        ),
        image_width,
        image_height,
    )

def _align_saved_bbox(
    class_name,
    bbox,
    anchors,
    transform,
    image_width,
    image_height,
):
    x1, y1, x2, y2 = [
        float(
            value
        )
        for value in bbox
    ]

    if transform.get(
        "applied"
    ):
        scale_y = float(
            transform.get(
                "scale_y",
                1.0,
            )
        )
        shift_x = float(
            transform.get(
                "shift_x",
                0.0,
            )
        )
        shift_y = float(
            transform.get(
                "shift_y",
                0.0,
            )
        )

        x1 += shift_x
        x2 += shift_x
        y1 = (
            y1
            * scale_y
            + shift_y
        )
        y2 = (
            y2
            * scale_y
            + shift_y
        )

    anchor = dict(
        anchors or {}
    ).get(
        class_name
    )

    if anchor is not None:
        try:
            anchor_x, anchor_y, confidence = (
                anchor
            )
            anchor_x = float(
                anchor_x
            )
            anchor_y = float(
                anchor_y
            )
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            anchor = None

    if (
        anchor is not None
        and confidence >= 50.0
    ):
        box_width = max(
            x2 - x1,
            1.0,
        )
        box_height = max(
            y2 - y1,
            1.0,
        )

        current_center_y = (
            y1 + y2
        ) / 2.0

        residual_x = (
            anchor_x
            - x1
        )
        residual_y = (
            anchor_y
            - current_center_y
        )

        residual_x = max(
            -0.012
            * float(
                image_width
            ),
            min(
                0.012
                * float(
                    image_width
                ),
                residual_x,
            ),
        )
        residual_y = max(
            -0.014
            * float(
                image_height
            ),
            min(
                0.014
                * float(
                    image_height
                ),
                residual_y,
            ),
        )

        x1 += residual_x
        x2 = (
            x1
            + box_width
        )
        y1 += residual_y
        y2 = (
            y1
            + box_height
        )

    return _clamp_bbox(
        (
            x1,
            y1,
            x2,
            y2,
        ),
        image_width,
        image_height,
    )


class TemplateFieldDetector:
    DEFAULT_TEMPLATE_RELATIVE = (
        Path("dataset")
        / "ktp_fields"
        / "templates"
        / "default.txt"
    )

    def __init__(
        self,
        template_path=None,
    ):
        self.template_path = (
            Path(
                template_path
            )
            if template_path is not None
            else (
                _runtime_root()
                / self.DEFAULT_TEMPLATE_RELATIVE
            )
        )

    def _saved_template_boxes(
        self,
        image_width,
        image_height,
    ):
        if not self.template_path.is_file():
            return {}

        result = {}

        try:
            lines = self.template_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError:
            return {}

        for raw_line in lines:
            parts = raw_line.strip().split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(
                    parts[0]
                )
                center_x, center_y, box_width, box_height = [
                    float(value)
                    for value in parts[1:]
                ]
            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                class_id < 0
                or class_id >= len(FIELD_CLASSES)
            ):
                continue

            width = float(
                image_width
            )
            height = float(
                image_height
            )

            pixel_width = (
                box_width
                * width
            )
            pixel_height = (
                box_height
                * height
            )
            pixel_center_x = (
                center_x
                * width
            )
            pixel_center_y = (
                center_y
                * height
            )

            bbox = _clamp_bbox(
                (
                    pixel_center_x
                    - pixel_width / 2.0,
                    pixel_center_y
                    - pixel_height / 2.0,
                    pixel_center_x
                    + pixel_width / 2.0,
                    pixel_center_y
                    + pixel_height / 2.0,
                ),
                image_width,
                image_height,
            )

            result[
                FIELD_CLASSES[
                    class_id
                ]
            ] = bbox

        return result


    def detect(
        self,
        image,
        anchors=None,
    ):
        if (
            not isinstance(image, np.ndarray)
            or image.size == 0
        ):
            raise ValueError(
                "image detector tidak boleh kosong."
            )

        height, width = (
            image.shape[:2]
        )

        saved_boxes = (
            self._saved_template_boxes(
                image_width=width,
                image_height=height,
            )
        )

        saved_transform = (
            _estimate_saved_template_transform(
                saved_boxes,
                anchors,
                image_width=width,
                image_height=height,
            )
        )

        text_boxes = (
            build_anchor_aligned_value_boxes(
                image_height=height,
                image_width=width,
                anchors=anchors,
            )
        )

        detections = []

        for class_name in FIELD_CLASSES:
            saved_bbox = saved_boxes.get(
                class_name
            )

            if saved_bbox is not None:
                local_bbox = (
                    _local_anchor_bbox(
                        class_name,
                        saved_bbox,
                        anchors,
                        image_width=width,
                        image_height=height,
                    )
                )

                if local_bbox is not None:
                    detections.append(
                        FieldDetection(
                            class_name=class_name,
                            bbox=local_bbox,
                            confidence=0.97,
                            source="annotation_template_local",
                        )
                    )
                    continue

                aligned_bbox = (
                    _align_saved_bbox(
                        class_name,
                        saved_bbox,
                        anchors,
                        saved_transform,
                        image_width=width,
                        image_height=height,
                    )
                )

                detections.append(
                    FieldDetection(
                        class_name=class_name,
                        bbox=aligned_bbox,
                        confidence=0.95,
                        source=(
                            "annotation_template_aligned"
                            if saved_transform.get(
                                "applied"
                            )
                            else "annotation_template"
                        ),
                    )
                )
                continue

            if class_name == "foto":
                box = KTP_FIELD_BOXES[
                    "foto"
                ]
                confidence = 0.35
            else:
                box = text_boxes[
                    class_name
                ]
                anchor = (
                    dict(
                        anchors or {}
                    ).get(
                        class_name
                    )
                )

                if anchor is None:
                    confidence = 0.35
                else:
                    try:
                        confidence = max(
                            0.35,
                            min(
                                0.99,
                                float(
                                    anchor[2]
                                )
                                / 100.0,
                            ),
                        )
                    except (
                        TypeError,
                        ValueError,
                        IndexError,
                    ):
                        confidence = 0.35

            bbox = (
                normalized_to_pixel_box(
                    box,
                    image_width=width,
                    image_height=height,
                )
            )

            detections.append(
                FieldDetection(
                    class_name=class_name,
                    bbox=bbox,
                    confidence=confidence,
                    source="template",
                )
            )

        return detections


class OnnxFieldDetector:
    def __init__(
        self,
        model_path,
        input_size=640,
        confidence_threshold=0.35,
        nms_threshold=0.45,
    ):
        self.model_path = Path(
            model_path
        )
        self.input_size = int(
            input_size
        )
        self.confidence_threshold = float(
            confidence_threshold
        )
        self.nms_threshold = float(
            nms_threshold
        )
        self._net = None

    def is_available(self):
        return self.model_path.is_file()

    def _load_net(self):
        if self._net is None:
            if not self.is_available():
                raise FileNotFoundError(
                    f"Model KTP field detector tidak ditemukan: {self.model_path}"
                )

            self._net = (
                cv2.dnn.readNetFromONNX(
                    str(
                        self.model_path
                    )
                )
            )

        return self._net

    @staticmethod
    def _normalize_output(
        output,
    ):
        data = np.asarray(
            output
        )

        while data.ndim > 2:
            data = data[0]

        if data.ndim != 2:
            raise RuntimeError(
                "Output ONNX field detector tidak didukung."
            )

        expected_columns = (
            4
            + len(
                FIELD_CLASSES
            )
        )

        if (
            data.shape[0]
            == expected_columns
            and data.shape[1]
            != expected_columns
        ):
            data = data.T

        if data.shape[1] < expected_columns:
            raise RuntimeError(
                "Jumlah output class ONNX tidak sesuai FIELD_CLASSES."
            )

        return data

    def detect(
        self,
        image,
        anchors=None,
    ):
        if (
            not isinstance(image, np.ndarray)
            or image.size == 0
        ):
            raise ValueError(
                "image detector tidak boleh kosong."
            )

        height, width = (
            image.shape[:2]
        )

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(
                self.input_size,
                self.input_size,
            ),
            swapRB=True,
            crop=False,
        )

        net = self._load_net()
        net.setInput(blob)
        raw = net.forward()
        rows = self._normalize_output(
            raw
        )

        boxes = []
        scores = []
        class_ids = []

        scale_x = (
            width
            / float(
                self.input_size
            )
        )
        scale_y = (
            height
            / float(
                self.input_size
            )
        )

        for row in rows:
            class_scores = row[
                4:
                4 + len(
                    FIELD_CLASSES
                )
            ]

            class_id = int(
                np.argmax(
                    class_scores
                )
            )
            confidence = float(
                class_scores[
                    class_id
                ]
            )

            if (
                confidence
                < self.confidence_threshold
            ):
                continue

            cx, cy, box_width, box_height = [
                float(value)
                for value in row[:4]
            ]

            left = (
                cx
                - box_width / 2.0
            )
            top = (
                cy
                - box_height / 2.0
            )

            boxes.append(
                [
                    int(
                        round(
                            left
                            * scale_x
                        )
                    ),
                    int(
                        round(
                            top
                            * scale_y
                        )
                    ),
                    int(
                        round(
                            box_width
                            * scale_x
                        )
                    ),
                    int(
                        round(
                            box_height
                            * scale_y
                        )
                    ),
                ]
            )
            scores.append(
                confidence
            )
            class_ids.append(
                class_id
            )

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence_threshold,
            self.nms_threshold,
        )

        if indices is None:
            return []

        flat_indices = np.array(
            indices
        ).reshape(-1)

        detections = []

        for index in flat_indices:
            left, top, box_width, box_height = (
                boxes[
                    int(index)
                ]
            )
            bbox = _clamp_bbox(
                (
                    left,
                    top,
                    left + box_width,
                    top + box_height,
                ),
                width,
                height,
            )

            detections.append(
                FieldDetection(
                    class_name=FIELD_CLASSES[
                        class_ids[
                            int(index)
                        ]
                    ],
                    bbox=bbox,
                    confidence=float(
                        scores[
                            int(index)
                        ]
                    ),
                    source="onnx",
                )
            )

        return list(
            best_detection_by_class(
                detections
            ).values()
        )


class AutoFieldDetector:
    DEFAULT_MODEL_RELATIVE = (
        Path("models")
        / "ktp_field_detector"
        / "ktp_fields.onnx"
    )

    def __init__(
        self,
        model_path=None,
        template_detector=None,
    ):
        self.model_path = (
            Path(
                model_path
            )
            if model_path is not None
            else (
                _runtime_root()
                / self.DEFAULT_MODEL_RELATIVE
            )
        )
        self.template_detector = (
            template_detector
            if template_detector is not None
            else TemplateFieldDetector()
        )
        self.onnx_detector = (
            OnnxFieldDetector(
                self.model_path
            )
        )

    @property
    def using_model(self):
        return (
            self.onnx_detector
            .is_available()
        )

    def detect(
        self,
        image,
        anchors=None,
    ):
        if self.using_model:
            try:
                detections = (
                    self.onnx_detector
                    .detect(
                        image,
                        anchors=anchors,
                    )
                )
            except Exception:
                detections = []

            if detections:
                by_class = (
                    best_detection_by_class(
                        detections
                    )
                )

                fallback = (
                    self.template_detector
                    .detect(
                        image,
                        anchors=anchors,
                    )
                )

                for item in fallback:
                    by_class.setdefault(
                        item.class_name,
                        item,
                    )

                return list(
                    by_class.values()
                )

        return (
            self.template_detector
            .detect(
                image,
                anchors=anchors,
            )
        )
