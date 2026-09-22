import unittest

import numpy as np

from autodocscanner.ktp.ocr import OcrWord
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
        self.assertEqual(
            result.identity["golongan_darah"],
            "O",
        )
        self.assertEqual(
            result.identity["status_perkawinan"],
            "KAWIN",
        )
        self.assertEqual(
            result.identity["berlaku_hingga"],
            "SEUMUR HIDUP",
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

    def test_debug_tracking_image_matches_canonical_working_image(self):
        result = extract_tracking_data(
            self.image,
            backend=MappingBackend(),
        )

        self.assertIsNotNone(
            result.debug_tracking_image
        )
        self.assertEqual(
            result.debug_tracking_image.shape[:2],
            (540, 856),
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

    def test_full_document_candidate_can_replace_bad_bbox_candidate(self):
        class HybridBackend:
            def read_document(
                self,
                image,
            ):
                return (
                    "PROVINSI CONTOH\n"
                    "KOTA CONTOH\n"
                    "NIK : 1234567890123456\n"
                    "Nama : NAMA DOKUMEN\n"
                    "Tempat/Tgl Lahir : KOTA CONTOH, 02-03-1990\n"
                    "Jenis kelamin : LAKI-LAKI\n"
                    "Alamat : JALAN DOKUMEN\n"
                    "RT/RW : 001/002\n"
                    "Kel/Desa : DESA CONTOH\n"
                    "Kecamatan : KECAMATAN CONTOH\n"
                    "Agama : ISLAM\n"
                    "Status Perkawinan : KAWIN\n"
                    "Pekerjaan : KARYAWAN\n"
                    "Kewarganegaraan : WNI\n"
                    "Berlaku Hingga : SEUMUR HIDUP",
                    78.0,
                )

            def read(
                self,
                image,
                field_name,
            ):
                if field_name == "nama":
                    return (
                        "%%%___",
                        88.0,
                    )
                return RESPONSES[
                    field_name
                ]

        result = extract_tracking_data(
            self.image,
            backend=HybridBackend(),
        )

        self.assertEqual(
            result.identity["nama"],
            "NAMA DOKUMEN",
        )
        self.assertEqual(
            result.identity["alamat"],
            "JALAN DOKUMEN",
        )

    def test_label_anchor_candidate_has_priority(self):
        class AnchorBackend:
            def read_layout(
                self,
                image,
            ):
                return [
                    OcrWord(
                        text="Nama",
                        confidence=92.0,
                        left=10,
                        top=30,
                        width=45,
                        height=20,
                        block=1,
                        paragraph=1,
                        line=1,
                    ),
                    OcrWord(
                        text=":",
                        confidence=90.0,
                        left=60,
                        top=30,
                        width=10,
                        height=20,
                        block=1,
                        paragraph=1,
                        line=1,
                    ),
                    OcrWord(
                        text="NAMA",
                        confidence=94.0,
                        left=80,
                        top=30,
                        width=50,
                        height=20,
                        block=1,
                        paragraph=1,
                        line=1,
                    ),
                    OcrWord(
                        text="ANCHOR",
                        confidence=94.0,
                        left=140,
                        top=30,
                        width=70,
                        height=20,
                        block=1,
                        paragraph=1,
                        line=1,
                    ),
                ]

            def read(
                self,
                image,
                field_name,
            ):
                return RESPONSES[
                    field_name
                ]

        result = extract_tracking_data(
            self.image,
            backend=AnchorBackend(),
        )

        self.assertEqual(
            result.identity["nama"],
            "NAMA ANCHOR",
        )

    def test_symbol_heavy_garbage_is_rejected_instead_of_autofilled(self):
        responses = dict(RESPONSES)
        responses["alamat"] = (
            "Q__ / A GE ET AE DE ST EE ____",
            88.0,
        )

        result = extract_tracking_data(
            self.image,
            backend=MappingBackend(
                responses
            ),
        )

        self.assertEqual(
            result.identity["alamat"],
            "",
        )
        self.assertIn(
            "alamat",
            result.review_fields,
        )

    def test_invalid_enum_like_values_are_rejected(self):
        responses = dict(RESPONSES)
        responses["agama"] = (
            "E;; %___",
            91.0,
        )
        responses["kewarganegaraan"] = (
            "R Y | N B SE",
            90.0,
        )

        result = extract_tracking_data(
            self.image,
            backend=MappingBackend(
                responses
            ),
        )

        self.assertEqual(
            result.identity["agama"],
            "",
        )
        self.assertEqual(
            result.identity["kewarganegaraan"],
            "",
        )


if __name__ == "__main__":
    unittest.main()
