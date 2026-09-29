import unittest

from autodocscanner.ktp.anchors import (
    locate_label_anchors,
)
from autodocscanner.ktp.ocr import OcrWord


def w(text, left, top, line):
    return OcrWord(
        text=text,
        confidence=92.0,
        left=left,
        top=top,
        width=max(20, len(text) * 8),
        height=18,
        block=1,
        paragraph=1,
        line=line,
    )


class TestKtpLabelAnchors(unittest.TestCase):
    def test_locates_label_center(self):
        words = [
            w("Nama", 30, 120, 1),
            w(":", 80, 120, 1),
            w("CONTOH", 100, 120, 1),
        ]

        anchors = locate_label_anchors(
            words
        )

        self.assertIn(
            "nama",
            anchors,
        )
        x, y, confidence = anchors[
            "nama"
        ]

        self.assertGreater(
            x,
            30,
        )
        self.assertGreater(
            y,
            120,
        )
        self.assertGreater(
            confidence,
            0,
        )

    def test_anchor_keeps_tuple_compatibility_and_value_bbox(self):
        label = OcrWord(
            text="Agama",
            confidence=96.0,
            left=30,
            top=120,
            width=45,
            height=16,
            block=1,
            paragraph=1,
            line=1,
        )
        value = OcrWord(
            text="KRISTEN",
            confidence=92.0,
            left=105,
            top=122,
            width=58,
            height=17,
            block=1,
            paragraph=1,
            line=1,
        )

        anchors = locate_label_anchors(
            [
                label,
                value,
            ]
        )

        anchor = anchors["agama"]

        self.assertEqual(
            len(anchor),
            3,
        )
        x, y, confidence = anchor
        self.assertEqual(
            x,
            105.0,
        )
        self.assertGreater(
            confidence,
            0.0,
        )
        self.assertEqual(
            tuple(
                anchor.value_bbox
            ),
            (
                105.0,
                122.0,
                163.0,
                139.0,
            ),
        )

    def test_anchor_uses_value_row_center_when_value_is_offset(self):
        label = OcrWord(
            text="Nama",
            confidence=95.0,
            left=30,
            top=100,
            width=40,
            height=16,
            block=1,
            paragraph=1,
            line=1,
        )
        separator = OcrWord(
            text=":",
            confidence=95.0,
            left=75,
            top=100,
            width=8,
            height=16,
            block=1,
            paragraph=1,
            line=1,
        )
        value = OcrWord(
            text="CONTOH",
            confidence=90.0,
            left=100,
            top=106,
            width=70,
            height=20,
            block=1,
            paragraph=1,
            line=1,
        )

        anchors = locate_label_anchors(
            [
                label,
                separator,
                value,
            ]
        )

        x, y, confidence = anchors[
            "nama"
        ]

        self.assertEqual(
            x,
            100.0,
        )
        self.assertAlmostEqual(
            y,
            116.0,
            places=3,
        )
        self.assertGreater(
            confidence,
            0.0,
        )

    def test_locates_multiple_standard_labels(self):
        words = [
            w("NIK", 20, 80, 1),
            w("1234", 100, 80, 1),
            w("Alamat", 20, 200, 2),
            w("JALAN", 100, 200, 2),
            w("Kecamatan", 20, 280, 3),
            w("CONTOH", 120, 280, 3),
        ]

        anchors = locate_label_anchors(
            words
        )

        self.assertEqual(
            set(anchors),
            {
                "nik",
                "alamat",
                "kecamatan",
            },
        )


if __name__ == "__main__":
    unittest.main()
