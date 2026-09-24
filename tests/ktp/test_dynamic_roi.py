import unittest

from autodocscanner.ktp.layout import (
    KTP_VALUE_BOXES,
    build_anchor_aligned_value_boxes,
)


class TestAnchorAlignedValueBoxes(unittest.TestCase):
    def test_detected_label_repositions_row_center_with_bounded_shift(self):
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

        template = KTP_VALUE_BOXES[
            "nama"
        ]
        template_center = (
            template.y1
            + template.y2
        ) / 2.0
        center = (
            boxes["nama"].y1
            + boxes["nama"].y2
        ) / 2.0

        self.assertGreater(
            center,
            template_center,
        )
        self.assertAlmostEqual(
            center
            - template_center,
            0.018,
            places=3,
        )

    def test_anchor_shift_is_bounded_and_preserves_box_size(self):
        template = KTP_VALUE_BOXES[
            "nama"
        ]

        boxes = build_anchor_aligned_value_boxes(
            image_height=540,
            image_width=856,
            anchors={
                "nama": (
                    500.0,
                    400.0,
                    95.0,
                ),
            },
        )

        actual = boxes["nama"]

        self.assertAlmostEqual(
            actual.x2 - actual.x1,
            template.x2 - template.x1,
            places=6,
        )
        self.assertAlmostEqual(
            actual.y2 - actual.y1,
            template.y2 - template.y1,
            places=6,
        )

        template_center_y = (
            template.y1
            + template.y2
        ) / 2.0
        actual_center_y = (
            actual.y1
            + actual.y2
        ) / 2.0

        self.assertLessEqual(
            abs(
                actual_center_y
                - template_center_y
            ),
            0.018001,
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
