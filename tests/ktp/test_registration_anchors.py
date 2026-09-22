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
