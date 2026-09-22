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


class TemplateFieldDetector:
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

        text_boxes = (
            build_anchor_aligned_value_boxes(
                image_height=height,
                anchors=anchors,
            )
        )

        detections = []

        for class_name in FIELD_CLASSES:
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
