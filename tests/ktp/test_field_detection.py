import tempfile
import unittest
from pathlib import Path

import numpy as np

from autodocscanner.ktp.field_detection import (
    FIELD_CLASSES,
    AutoFieldDetector,
    FieldDetection,
    TemplateFieldDetector,
    best_detection_by_class,
    crop_detection,
)


class TestTemplateFieldDetector(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )

    def test_emits_all_tracking_classes(self):
        detections = TemplateFieldDetector().detect(
            self.image,
            anchors={},
        )

        self.assertEqual(
            {item.class_name for item in detections},
            set(FIELD_CLASSES),
        )
        self.assertTrue(
            all(item.source == "template" for item in detections)
        )

    def test_saved_annotation_template_overrides_hardcoded_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_path = (
                Path(tmp)
                / "default.txt"
            )
            template_path.write_text(
                "3 0.350000 0.300000 0.400000 0.080000\n",
                encoding="utf-8",
            )

            detector = TemplateFieldDetector(
                template_path=template_path
            )
            detections = best_detection_by_class(
                detector.detect(
                    self.image,
                    anchors={
                        "nama": (
                            700.0,
                            500.0,
                            99.0,
                        ),
                    },
                )
            )

        nama = detections["nama"]

        self.assertEqual(
            nama.source,
            "annotation_template",
        )
        self.assertEqual(
            nama.confidence,
            0.95,
        )
        self.assertEqual(
            nama.bbox,
            (
                128,
                140,
                471,
                184,
            ),
        )

    def test_anchor_changes_detected_row(self):
        detector = TemplateFieldDetector()

        baseline = best_detection_by_class(
            detector.detect(
                self.image,
                anchors={},
            )
        )["nama"]

        anchored = best_detection_by_class(
            detector.detect(
                self.image,
                anchors={
                    "nama": (
                        80.0,
                        210.0,
                        95.0,
                    ),
                },
            )
        )["nama"]

        self.assertNotEqual(
            baseline.bbox[1],
            anchored.bbox[1],
        )


class TestAutoFieldDetector(unittest.TestCase):
    def test_missing_onnx_uses_template_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = AutoFieldDetector(
                model_path=(
                    Path(tmp)
                    / "missing.onnx"
                )
            )

            detections = detector.detect(
                np.zeros(
                    (540, 856, 3),
                    dtype=np.uint8,
                ),
                anchors={},
            )

        self.assertTrue(detections)
        self.assertTrue(
            all(
                item.source == "template"
                for item in detections
            )
        )


class TestDetectionHelpers(unittest.TestCase):
    def test_best_detection_keeps_highest_confidence_per_class(self):
        items = [
            FieldDetection(
                "nama",
                (10, 10, 50, 30),
                0.40,
                "template",
            ),
            FieldDetection(
                "nama",
                (12, 12, 60, 34),
                0.91,
                "onnx",
            ),
        ]

        result = best_detection_by_class(
            items
        )

        self.assertEqual(
            result["nama"].source,
            "onnx",
        )

    def test_crop_detection_clamps_to_image(self):
        image = np.zeros(
            (100, 200, 3),
            dtype=np.uint8,
        )
        detection = FieldDetection(
            "nama",
            (-10, -5, 220, 120),
            0.8,
            "test",
        )

        crop = crop_detection(
            image,
            detection,
        )

        self.assertEqual(
            crop.shape,
            image.shape,
        )


if __name__ == "__main__":
    unittest.main()
