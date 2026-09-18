import tempfile
import unittest
from pathlib import Path

import numpy as np

from autodocscanner.ktp.storage import KtpStorage


VALID = {
    "nama": "BUDI SANTOSO",
    "tempat_lahir": "PONTIANAK",
    "tanggal_lahir": "1990-01-02",
    "jenis_kelamin": "LAKI-LAKI",
    "alamat": "JL. CONTOH NO. 1",
    "rt": "001",
    "rw": "002",
    "kelurahan_desa": "SUNGAI BANGKONG",
    "kecamatan": "PONTIANAK KOTA",
    "kabupaten_kota": "KOTA PONTIANAK",
    "provinsi": "KALIMANTAN BARAT",
    "agama": "ISLAM",
    "status_perkawinan": "KAWIN",
    "pekerjaan": "KARYAWAN SWASTA",
    "kewarganegaraan": "WNI",
}


class TestKtpStorage(unittest.TestCase):
    def setUp(self):
        self.temp = (
            tempfile.TemporaryDirectory()
        )
        self.data_dir = (
            Path(self.temp.name)
            / "data"
        )
        self.storage = KtpStorage(
            self.data_dir
        )
        self.storage.initialize()

        self.ktp = np.zeros(
            (60, 96, 3),
            dtype=np.uint8,
        )
        self.ktp[:, :, 1] = 150

        self.face = np.zeros(
            (30, 20, 3),
            dtype=np.uint8,
        )
        self.face[:, :, 2] = 180

    def tearDown(self):
        self.temp.cleanup()

    def test_new_nik_records_revision_one_and_media(self):
        result = (
            self.storage.record(
                "6171010203900001",
                VALID,
                self.ktp,
                self.face,
            )
        )

        self.assertEqual(
            result[
                "revision_number"
            ],
            1,
        )
        self.assertEqual(
            result["status"],
            "ACTIVE",
        )
        self.assertEqual(
            result["revision_id"],
            result["id"],
        )

        self.assertTrue(
            self.storage.media.resolve(
                result[
                    "ktp_image_path"
                ]
            ).exists()
        )
        self.assertTrue(
            self.storage.media.resolve(
                result[
                    "face_image_path"
                ]
            ).exists()
        )
        self.assertNotIn(
            "6171010203900001",
            result[
                "ktp_image_path"
            ],
        )

    def test_same_nik_creates_revision_two_with_stable_record_id(self):
        first = (
            self.storage.record(
                "6171010203900001",
                VALID,
                self.ktp,
            )
        )

        changed = {
            **VALID,
            "alamat": (
                "JL. BARU NO. 2"
            ),
        }

        second = (
            self.storage.record(
                "6171010203900001",
                changed,
                self.ktp,
            )
        )

        self.assertEqual(
            second["record_id"],
            first["record_id"],
        )
        self.assertEqual(
            second[
                "revision_number"
            ],
            2,
        )
        self.assertNotEqual(
            second["revision_id"],
            first["revision_id"],
        )

        revisions = (
            self.storage
            .list_revisions(
                "6171010203900001"
            )
        )

        self.assertEqual(
            len(revisions),
            2,
        )
        self.assertEqual(
            revisions[0]["status"],
            "SUPERSEDED",
        )
        self.assertEqual(
            revisions[1]["alamat"],
            "JL. BARU NO. 2",
        )
        self.assertTrue(
            self.storage.media.resolve(
                revisions[0][
                    "ktp_image_path"
                ]
            ).exists()
        )

    def test_face_is_optional(self):
        result = (
            self.storage.record(
                "6171010203900001",
                VALID,
                self.ktp,
                face_image=None,
            )
        )

        self.assertIsNone(
            result[
                "face_image_path"
            ]
        )

    def test_invalid_identity_writes_no_media_or_database_record(self):
        invalid = dict(VALID)
        invalid["nama"] = ""

        with self.assertRaises(
            ValueError
        ):
            self.storage.record(
                "6171010203900001",
                invalid,
                self.ktp,
            )

        self.assertIsNone(
            self.storage.get_current(
                "6171010203900001"
            )
        )

        media_root = (
            self.data_dir
            / "media"
            / "ktp"
        )
        self.assertFalse(
            media_root.exists()
        )

    def test_reopen_preserves_current_revision(self):
        first = (
            self.storage.record(
                "6171010203900001",
                VALID,
                self.ktp,
            )
        )

        reopened = KtpStorage(
            self.data_dir
        )
        reopened.initialize()

        current = (
            reopened.get_current(
                "6171010203900001"
            )
        )

        self.assertEqual(
            current["record_id"],
            first["record_id"],
        )
        self.assertEqual(
            current[
                "revision_number"
            ],
            1,
        )
        self.assertEqual(
            current["nama"],
            "BUDI SANTOSO",
        )


if __name__ == "__main__":
    unittest.main()
