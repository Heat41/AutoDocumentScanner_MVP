import unittest

from autodocscanner.ktp.layout import (
    KTP_VALUE_BOXES,
    build_anchor_aligned_value_boxes,
)


class TestAnchorAlignedValueBoxes(unittest.TestCase):
    def test_detected_label_repositions_row_center(self):
        boxes = build_anchor_aligned_value_boxes(
            image_height=540,
            anchors={
                "nama": (
                    80.0,
                    180.0,
                    90.0,
                ),
            },
        )

        center = (
            boxes["nama"].y1
            + boxes["nama"].y2
        ) / 2.0

        self.assertAlmostEqual(
            center,
            180.0 / 540.0,
            places=3,
        )

    def test_missing_anchor_keeps_template_box(self):
        boxes = build_anchor_aligned_value_boxes(
            image_height=540,
            anchors={},
        )

        self.assertEqual(
            boxes["alamat"],
            KTP_VALUE_BOXES["alamat"],
        )

    def test_gender_anchor_can_align_blood_group_same_row(self):
        boxes = build_anchor_aligned_value_boxes(
            image_height=540,
            anchors={
                "jenis_kelamin": (
                    70.0,
                    210.0,
                    88.0,
                ),
            },
        )

        gender_center = (
            boxes["jenis_kelamin"].y1
            + boxes["jenis_kelamin"].y2
        ) / 2.0
        blood_center = (
            boxes["golongan_darah"].y1
            + boxes["golongan_darah"].y2
        ) / 2.0

        self.assertAlmostEqual(
            gender_center,
            blood_center,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
