import tempfile
import unittest
from pathlib import Path

import numpy as np

from autodocscanner.ktp.anchors import ValueAnchor
from autodocscanner.ktp.field_detection import (
    FIELD_CLASSES,
    AutoFieldDetector,
    FieldDetection,
    TemplateFieldDetector,
    best_detection_by_class,
    crop_detection,
    crop_detection_padded,
)


class TestTemplateFieldDetector(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros(
            (540, 856, 3),
            dtype=np.uint8,
        )

    def test_emits_all_tracking_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = TemplateFieldDetector(
                template_path=(
                    Path(tmp)
                    / "missing-template.txt"
                )
            )
            detections = detector.detect(
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
                    anchors={},
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

    def test_saved_template_uses_global_anchor_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_path = (
                Path(tmp)
                / "default.txt"
            )
            template_path.write_text(
                "\n".join(
                    [
                        "2 0.350000 0.200000 0.300000 0.040000",
                        "3 0.350000 0.300000 0.300000 0.040000",
                        "7 0.350000 0.500000 0.300000 0.040000",
                        "16 0.820000 0.500000 0.180000 0.300000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            detector = TemplateFieldDetector(
                template_path=template_path
            )

            width = self.image.shape[1]
            height = self.image.shape[0]

            anchors = {
                "nik": (
                    0.20 * width + 10.0,
                    0.20 * height + 8.0,
                    95.0,
                ),
                "nama": (
                    0.20 * width + 10.0,
                    0.30 * height + 8.0,
                    94.0,
                ),
                "alamat": (
                    0.20 * width + 10.0,
                    0.50 * height + 8.0,
                    93.0,
                ),
            }

            detections = best_detection_by_class(
                detector.detect(
                    self.image,
                    anchors=anchors,
                )
            )

        self.assertEqual(
            detections["nik"].source,
            "annotation_template_local",
        )
        self.assertGreater(
            detections["nik"].bbox[0],
            int(
                round(
                    0.20 * width
                )
            ),
        )
        self.assertGreater(
            detections["foto"].bbox[0],
            int(
                round(
                    0.73 * width
                )
            ),
        )

    def test_local_anchor_value_bbox_tightens_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_path = (
                Path(tmp)
                / "default.txt"
            )
            template_path.write_text(
                "12 0.420000 0.600000 0.360000 0.050000\n",
                encoding="utf-8",
            )

            detector = TemplateFieldDetector(
                template_path=template_path
            )

            baseline = best_detection_by_class(
                detector.detect(
                    self.image,
                    anchors={},
                )
            )["agama"]

            anchored = best_detection_by_class(
                detector.detect(
                    self.image,
                    anchors={
                        "agama": ValueAnchor(
                            180.0,
                            250.0,
                            95.0,
                            value_bbox=(
                                180.0,
                                242.0,
                                245.0,
                                258.0,
                            ),
                        )
                    },
                )
            )["agama"]

        baseline_width = (
            baseline.bbox[2]
            - baseline.bbox[0]
        )
        anchored_width = (
            anchored.bbox[2]
            - anchored.bbox[0]
        )
        baseline_height = (
            baseline.bbox[3]
            - baseline.bbox[1]
        )
        anchored_height = (
            anchored.bbox[3]
            - anchored.bbox[1]
        )

        self.assertEqual(
            anchored.source,
            "annotation_template_local",
        )
        self.assertLess(
            anchored_width,
            baseline_width,
        )
        self.assertLessEqual(
            anchored_height,
            baseline_height,
        )
        self.assertLessEqual(
            anchored.bbox[0],
            180,
        )
        self.assertGreaterEqual(
            anchored.bbox[2],
            245,
        )

    def test_saved_template_local_anchor_moves_only_target_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_path = (
                Path(tmp)
                / "default.txt"
            )
            template_path.write_text(
                "\n".join(
                    [
                        "2 0.350000 0.200000 0.300000 0.040000",
                        "3 0.350000 0.300000 0.300000 0.040000",
                        "7 0.350000 0.500000 0.300000 0.040000",
                        "16 0.820000 0.500000 0.180000 0.300000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            detector = TemplateFieldDetector(
                template_path=template_path
            )

            baseline = best_detection_by_class(
                detector.detect(
                    self.image,
                    anchors={},
                )
            )

            anchored = best_detection_by_class(
                detector.detect(
                    self.image,
                    anchors={
                        "nama": (
                            360.0,
                            190.0,
                            95.0,
                        ),
                    },
                )
            )

        self.assertEqual(
            anchored["nama"].source,
            "annotation_template_local",
        )
        self.assertNotEqual(
            anchored["nama"].bbox,
            baseline["nama"].bbox,
        )
        self.assertEqual(
            anchored["nik"].bbox,
            baseline["nik"].bbox,
        )
        self.assertEqual(
            (
                anchored["nama"].bbox[2]
                - anchored["nama"].bbox[0]
            ),
            (
                baseline["nama"].bbox[2]
                - baseline["nama"].bbox[0]
            ),
        )
        self.assertEqual(
            (
                anchored["nama"].bbox[3]
                - anchored["nama"].bbox[1]
            ),
            (
                baseline["nama"].bbox[3]
                - baseline["nama"].bbox[1]
            ),
        )

    def test_weak_local_anchor_keeps_saved_template_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_path = (
                Path(tmp)
                / "default.txt"
            )
            template_path.write_text(
                "3 0.350000 0.300000 0.300000 0.040000\n",
                encoding="utf-8",
            )

            detector = TemplateFieldDetector(
                template_path=template_path
            )

            baseline = best_detection_by_class(
                detector.detect(
                    self.image,
                    anchors={},
                )
            )["nama"]

            weak = best_detection_by_class(
                detector.detect(
                    self.image,
                    anchors={
                        "nama": (
                            400.0,
                            230.0,
                            25.0,
                        ),
                    },
                )
            )["nama"]

        self.assertEqual(
            weak.source,
            "annotation_template",
        )
        self.assertEqual(
            weak.bbox,
            baseline.bbox,
        )

    def test_anchor_changes_detected_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            detector = TemplateFieldDetector(
                template_path=(
                    Path(tmp)
                    / "missing-template.txt"
                )
            )

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
                ),
                template_detector=(
                    TemplateFieldDetector(
                        template_path=(
                            Path(tmp)
                            / "missing-template.txt"
                        )
                    )
                ),
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

    def test_padded_crop_expands_ocr_region(self):
        image = np.zeros(
            (100, 200, 3),
            dtype=np.uint8,
        )
        detection = FieldDetection(
            "nama",
            (50, 40, 150, 60),
            0.95,
            "annotation_template",
        )

        normal = crop_detection(
            image,
            detection,
        )
        padded = crop_detection_padded(
            image,
            detection,
        )

        self.assertGreater(
            padded.shape[0],
            normal.shape[0],
        )
        self.assertGreater(
            padded.shape[1],
            normal.shape[1],
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
