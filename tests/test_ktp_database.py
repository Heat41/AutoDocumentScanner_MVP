import tempfile
import unittest
from pathlib import Path

from ktp_database import KtpDatabase, SCHEMA_VERSION
from ktp_models import KtpIdentityData


VALID = KtpIdentityData.from_mapping(
    {
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
)


class TestKtpDatabase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = (
            Path(self.temp.name)
            / "autoscanner.db"
        )
        self.db = KtpDatabase(
            self.db_path
        )
        self.db.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def test_schema_version_is_recorded(self):
        self.assertEqual(
            self.db.schema_version(),
            SCHEMA_VERSION,
        )

    def test_new_nik_creates_record_and_revision_one(self):
        result = self.db.record_revision(
            record_id="record-1",
            revision_id="rev-1",
            nik="6171010203900001",
            identity=VALID,
            ktp_image_path=(
                "record-1/rev-1/ktp.jpg"
            ),
        )

        self.assertEqual(
            result["record_id"],
            "record-1",
        )
        self.assertEqual(
            result["revision_number"],
            1,
        )
        self.assertEqual(
            result["status"],
            "ACTIVE",
        )

        current = (
            self.db.get_current_revision(
                "6171010203900001"
            )
        )
        self.assertEqual(
            current["id"],
            "rev-1",
        )

    def test_same_nik_reuses_primary_record_and_increments_revision(self):
        self.db.record_revision(
            record_id="record-1",
            revision_id="rev-1",
            nik="6171010203900001",
            identity=VALID,
            ktp_image_path=(
                "record-1/rev-1/ktp.jpg"
            ),
        )

        changed = KtpIdentityData.from_mapping(
            {
                **VALID.as_db_dict(),
                "alamat": "JL. BARU NO. 2",
            }
        )

        result = self.db.record_revision(
            record_id=(
                "record-should-not-be-used"
            ),
            revision_id="rev-2",
            nik="6171010203900001",
            identity=changed,
            ktp_image_path=(
                "record-1/rev-2/ktp.jpg"
            ),
        )

        self.assertEqual(
            result["record_id"],
            "record-1",
        )
        self.assertEqual(
            result["revision_number"],
            2,
        )

        revisions = (
            self.db.list_revisions(
                "6171010203900001"
            )
        )

        self.assertEqual(
            [
                revision["revision_number"]
                for revision in revisions
            ],
            [1, 2],
        )
        self.assertEqual(
            revisions[0]["status"],
            "SUPERSEDED",
        )
        self.assertEqual(
            revisions[1]["status"],
            "ACTIVE",
        )
        self.assertEqual(
            revisions[1]["alamat"],
            "JL. BARU NO. 2",
        )

    def test_one_nik_has_only_one_primary_record(self):
        self.db.record_revision(
            "record-1",
            "rev-1",
            "6171010203900001",
            VALID,
            "record-1/rev-1/ktp.jpg",
        )
        self.db.record_revision(
            "record-2",
            "rev-2",
            "6171010203900001",
            VALID,
            "record-1/rev-2/ktp.jpg",
        )

        record = self.db.get_record_by_nik(
            "6171010203900001"
        )
        self.assertEqual(
            record["id"],
            "record-1",
        )
        self.assertEqual(
            record[
                "current_revision_number"
            ],
            2,
        )

    def test_face_image_path_may_be_null(self):
        result = self.db.record_revision(
            "record-1",
            "rev-1",
            "6171010203900001",
            VALID,
            "record-1/rev-1/ktp.jpg",
            face_image_path=None,
        )
        self.assertIsNone(
            result["face_image_path"]
        )

    def test_database_persists_after_reopen(self):
        self.db.record_revision(
            "record-1",
            "rev-1",
            "6171010203900001",
            VALID,
            "record-1/rev-1/ktp.jpg",
        )

        reopened = KtpDatabase(
            self.db_path
        )
        reopened.initialize()

        current = (
            reopened.get_current_revision(
                "6171010203900001"
            )
        )
        self.assertEqual(
            current["revision_number"],
            1,
        )
        self.assertEqual(
            current["nama"],
            "BUDI SANTOSO",
        )


if __name__ == "__main__":
    unittest.main()
