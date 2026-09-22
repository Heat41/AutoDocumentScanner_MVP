import unittest

import numpy as np

from autodocscanner.ktp.tracking import (
    KtpTrackingResult,
    extract_tracking_data,
)


RESPONSES = {
    "provinsi": ("PROVINSI CONTOH", 90.0),
    "kabupaten_kota": ("KOTA CONTOH", 90.0),
    "nik": ("1234567890123456", 95.0),
    "nama": ("NAMA CONTOH", 91.0),
    "ttl": ("KOTA CONTOH, 02-03-1990", 89.0),
    "jenis_kelamin": ("LAKI LAKI", 88.0),
    "golongan_darah": ("O", 80.0),
    "alamat": ("JALAN CONTOH 1", 84.0),
    "rt_rw": ("001/002", 92.0),
    "kelurahan_desa": ("DESA CONTOH", 86.0),
    "kecamatan": ("KECAMATAN CONTOH", 87.0),
    "agama": ("ISLAM", 90.0),
    "status_perkawinan": ("KAWIN", 90.0),
    "pekerjaan": ("KARYAWAN", 88.0),
    "kewarganegaraan": ("WNI", 92.0),
    "berlaku_hingga": ("SEUMUR HIDUP", 93.0),
}


class MappingBackend:
    def __init__(self, responses=None):
        self.responses = dict(
            responses or RESPONSES
        )
        self.calls = []

    def read(self, image, field_name):
        self.calls.append(field_name)
        return self.responses[
            field_name
        ]


class TestKtpTracking(unittest.TestCase):
    def setUp(self):
        self.image = np.full(
            (540, 856, 3),
            170,
            dtype=np.uint8,
        )

    def test_tracking_extracts_structured_identity(self):
        result = extract_tracking_data(
            self.image,
            backend=MappingBackend(),
        )

        self.assertIsInstance(
            result,
            KtpTrackingResult,
        )
        self.assertEqual(
            result.identity["nik"],
            "1234567890123456",
        )
        self.assertEqual(
            result.identity["tempat_lahir"],
            "KOTA CONTOH",
        )
        self.assertEqual(
            result.identity["tanggal_lahir"],
            "1990-03-02",
        )
        self.assertEqual(
            result.identity["rt"],
            "001",
        )
        self.assertEqual(
            result.identity["rw"],
            "002",
        )
        self.assertEqual(
            result.identity["jenis_kelamin"],
            "LAKI-LAKI",
        )

    def test_foto_is_not_sent_to_ocr(self):
        backend = MappingBackend()
        result = extract_tracking_data(
            self.image,
            backend=backend,
        )

        self.assertNotIn(
            "foto",
            backend.calls,
        )
        self.assertGreater(
            result.face_image.size,
            0,
        )

    def test_low_confidence_field_requires_review(self):
        responses = dict(RESPONSES)
        responses["nama"] = (
            "NAMA BURAM",
            20.0,
        )

        class LowConfidenceBackend:
            def __init__(self):
                self.count = {}

            def read(
                self,
                image,
                field_name,
            ):
                count = self.count.get(
                    field_name,
                    0,
                )
                self.count[
                    field_name
                ] = count + 1

                if field_name == "nama":
                    return (
                        "NAMA BURAM",
                        20.0,
                    )

                return RESPONSES[
                    field_name
                ]

        result = extract_tracking_data(
            self.image,
            backend=LowConfidenceBackend(),
        )

        self.assertTrue(
            result.fields[
                "nama"
            ].needs_review
        )
        self.assertIn(
            "nama",
            result.review_fields,
        )

    def test_missing_required_parsed_value_requires_review(self):
        responses = dict(RESPONSES)
        responses["nik"] = (
            "12345",
            90.0,
        )

        result = extract_tracking_data(
            self.image,
            backend=MappingBackend(
                responses
            ),
        )

        self.assertEqual(
            result.identity["nik"],
            "",
        )
        self.assertTrue(
            result.fields[
                "nik"
            ].needs_review
        )

    def test_corrected_image_is_independent_copy(self):
        result = extract_tracking_data(
            self.image,
            backend=MappingBackend(),
        )
        result.corrected_image[:] = 255

        self.assertFalse(
            np.all(
                self.image == 255
            )
        )


if __name__ == "__main__":
    unittest.main()
