import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from ktp_media import KtpMediaStore


class TestKtpMediaStore(unittest.TestCase):
    def setUp(self):
        self.temp = (
            tempfile.TemporaryDirectory()
        )
        self.root = (
            Path(self.temp.name)
            / "media"
            / "ktp"
        )
        self.store = KtpMediaStore(
            self.root
        )
        self.record_id = str(
            uuid4()
        )
        self.revision_id = str(
            uuid4()
        )

        self.ktp = np.zeros(
            (60, 96, 3),
            dtype=np.uint8,
        )
        self.ktp[:, :, 1] = 180

        self.face = np.zeros(
            (40, 30, 3),
            dtype=np.uint8,
        )
        self.face[:, :, 2] = 200

    def tearDown(self):
        self.temp.cleanup()

    def test_save_revision_writes_required_ktp_and_optional_face(self):
        result = (
            self.store.save_revision(
                self.record_id,
                self.revision_id,
                self.ktp,
                self.face,
            )
        )

        self.assertEqual(
            result[
                "ktp_image_path"
            ],
            (
                f"{self.record_id}/"
                f"{self.revision_id}/"
                "ktp.jpg"
            ),
        )
        self.assertEqual(
            result[
                "face_image_path"
            ],
            (
                f"{self.record_id}/"
                f"{self.revision_id}/"
                "face.jpg"
            ),
        )

        self.assertIsNotNone(
            cv2.imread(
                str(
                    self.store.resolve(
                        result[
                            "ktp_image_path"
                        ]
                    )
                )
            )
        )
        self.assertIsNotNone(
            cv2.imread(
                str(
                    self.store.resolve(
                        result[
                            "face_image_path"
                        ]
                    )
                )
            )
        )

    def test_face_is_optional(self):
        result = (
            self.store.save_revision(
                self.record_id,
                self.revision_id,
                self.ktp,
            )
        )

        self.assertIsNone(
            result[
                "face_image_path"
            ]
        )
        self.assertTrue(
            self.store.resolve(
                result[
                    "ktp_image_path"
                ]
            ).exists()
        )

    def test_invalid_ktp_image_is_rejected_without_revision_directory(self):
        with self.assertRaises(
            ValueError
        ):
            self.store.save_revision(
                self.record_id,
                self.revision_id,
                np.array(
                    [],
                    dtype=np.uint8,
                ),
            )

        revision_dir = (
            self.root
            / self.record_id
            / self.revision_id
        )
        self.assertFalse(
            revision_dir.exists()
        )

    def test_ids_must_be_uuid_values(self):
        with self.assertRaises(
            ValueError
        ):
            self.store.save_revision(
                "6171010203900001",
                self.revision_id,
                self.ktp,
            )

    def test_resolve_rejects_path_traversal(self):
        with self.assertRaises(
            ValueError
        ):
            self.store.resolve(
                "../outside.jpg"
            )

    def test_cleanup_revision_removes_only_requested_revision(self):
        other_revision = str(
            uuid4()
        )

        self.store.save_revision(
            self.record_id,
            self.revision_id,
            self.ktp,
        )
        self.store.save_revision(
            self.record_id,
            other_revision,
            self.ktp,
        )

        self.store.cleanup_revision(
            self.record_id,
            self.revision_id,
        )

        self.assertFalse(
            (
                self.root
                / self.record_id
                / self.revision_id
            ).exists()
        )
        self.assertTrue(
            (
                self.root
                / self.record_id
                / other_revision
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
