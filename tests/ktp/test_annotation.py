import tempfile
import unittest
from pathlib import Path

from autodocscanner.ktp.annotation import (
    Annotation,
    denormalize_yolo_bbox,
    normalize_bbox_to_yolo,
    read_yolo_annotations,
    write_yolo_annotations,
)


class TestKtpAnnotation(unittest.TestCase):
    def test_bbox_round_trip(self):
        bbox = (
            100,
            50,
            300,
            150,
        )

        yolo = normalize_bbox_to_yolo(
            bbox,
            image_width=400,
            image_height=200,
        )
        restored = denormalize_yolo_bbox(
            yolo,
            image_width=400,
            image_height=200,
        )

        self.assertEqual(
            restored,
            bbox,
        )

    def test_write_and_read_yolo_annotations(self):
        items = [
            Annotation(
                class_name="nama",
                bbox=(100, 50, 300, 150),
            ),
            Annotation(
                class_name="foto",
                bbox=(310, 30, 390, 180),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "sample.txt"
            )

            write_yolo_annotations(
                path,
                items,
                image_width=400,
                image_height=200,
            )

            loaded = read_yolo_annotations(
                path,
                image_width=400,
                image_height=200,
            )

        self.assertEqual(
            loaded,
            items,
        )


if __name__ == "__main__":
    unittest.main()
