import unittest

from autodocscanner.ktp.anchors import (
    extract_anchor_candidates,
)
from autodocscanner.ktp.ocr import OcrWord


def word(
    text,
    left,
    line,
    confidence=90.0,
):
    return OcrWord(
        text=text,
        confidence=confidence,
        left=left,
        top=line * 30,
        width=max(20, len(text) * 8),
        height=20,
        block=1,
        paragraph=1,
        line=line,
    )


class TestKtpAnchors(unittest.TestCase):
    def test_extracts_values_after_labels(self):
        words = [
            word("NIK", 10, 1),
            word(":", 70, 1),
            word("1234567890123456", 90, 1),
            word("Nama", 10, 2),
            word(":", 70, 2),
            word("NAMA", 90, 2),
            word("CONTOH", 150, 2),
            word("RT/RW", 10, 3),
            word(":", 90, 3),
            word("001/002", 120, 3),
        ]

        result = extract_anchor_candidates(
            words
        )

        self.assertEqual(
            result["nik"][0],
            "1234567890123456",
        )
        self.assertEqual(
            result["nama"][0],
            "NAMA CONTOH",
        )
        self.assertEqual(
            result["rt_rw"][0],
            "001/002",
        )

    def test_extracts_header_lines(self):
        words = [
            word("PROVINSI", 10, 1),
            word("CONTOH", 100, 1),
            word("KOTA", 10, 2),
            word("CONTOH", 80, 2),
        ]

        result = extract_anchor_candidates(
            words
        )

        self.assertEqual(
            result["provinsi"][0],
            "PROVINSI CONTOH",
        )
        self.assertEqual(
            result["kabupaten_kota"][0],
            "KOTA CONTOH",
        )

    def test_minor_label_noise_is_tolerated(self):
        words = [
            word("Kecarnatan", 10, 1),
            word(":", 100, 1),
            word("CONTOH", 130, 1),
        ]

        result = extract_anchor_candidates(
            words
        )

        self.assertEqual(
            result["kecamatan"][0],
            "CONTOH",
        )

    def test_label_without_value_is_ignored(self):
        result = extract_anchor_candidates(
            [word("Nama", 10, 1)]
        )

        self.assertNotIn(
            "nama",
            result,
        )


if __name__ == "__main__":
    unittest.main()
