import unittest
from pathlib import Path

import numpy as np

from autodocscanner.services.ktp_tracking import (
    correct_tracking_input,
    process_tracking_input,
    track_corrected_ktp,
)


class FakeScanner:
    def __init__(self):
        self.calls = []

    def scan(
        self,
        image_path,
        output_path=None,
        mode="ktp",
        output_mode="color",
    ):
        self.calls.append(
            {
                "image_path": Path(image_path),
                "output_path": output_path,
                "mode": mode,
                "output_mode": output_mode,
            }
        )
        image = np.full(
            (540, 856, 3),
            150,
            dtype=np.uint8,
        )
        corners = np.array(
            [
                [0, 0],
                [855, 0],
                [855, 539],
                [0, 539],
            ],
            dtype=np.float32,
        )
        return image, corners


class FakeBackend:
    def read(
        self,
        image,
        field_name,
    ):
        values = {
            "provinsi": "PROVINSI CONTOH",
            "kabupaten_kota": "KOTA CONTOH",
            "nik": "1234567890123456",
            "nama": "NAMA CONTOH",
            "ttl": "KOTA CONTOH, 02-03-1990",
            "jenis_kelamin": "LAKI LAKI",
            "golongan_darah": "O",
            "alamat": "JALAN CONTOH",
            "rt_rw": "001/002",
            "kelurahan_desa": "DESA CONTOH",
            "kecamatan": "KECAMATAN CONTOH",
            "agama": "ISLAM",
            "status_perkawinan": "KAWIN",
            "pekerjaan": "KARYAWAN",
            "kewarganegaraan": "WNI",
            "berlaku_hingga": "SEUMUR HIDUP",
        }
        return values[field_name], 90.0


class TestKtpTrackingService(unittest.TestCase):
    def test_process_uses_locked_ktp_scanner_without_output_write(self):
        scanner = FakeScanner()

        result = process_tracking_input(
            scanner,
            "sample.jpg",
            backend=FakeBackend(),
        )

        self.assertEqual(
            len(scanner.calls),
            1,
        )
        call = scanner.calls[0]
        self.assertIsNone(
            call["output_path"]
        )
        self.assertEqual(
            call["mode"],
            "ktp",
        )
        self.assertEqual(
            call["output_mode"],
            "color",
        )
        self.assertEqual(
            result["corrected_image"].shape,
            (540, 856, 3),
        )
        self.assertEqual(
            result["tracking"].identity["nama"],
            "NAMA CONTOH",
        )

    def test_result_preserves_corners(self):
        result = process_tracking_input(
            FakeScanner(),
            "sample.jpg",
            backend=FakeBackend(),
        )

        self.assertEqual(
            result["corners"].shape,
            (4, 2),
        )

    def test_correction_stage_returns_corrected_image_only(self):
        result = correct_tracking_input(
            FakeScanner(),
            "sample.jpg",
        )

        self.assertIn(
            "corrected_image",
            result,
        )
        self.assertIn(
            "corners",
            result,
        )
        self.assertNotIn(
            "tracking",
            result,
        )

    def test_tracking_stage_accepts_corrected_image(self):
        corrected = np.full(
            (540, 856, 3),
            150,
            dtype=np.uint8,
        )

        tracking = track_corrected_ktp(
            corrected,
            backend=FakeBackend(),
        )

        self.assertEqual(
            tracking.corrected_image.shape,
            corrected.shape,
        )
        self.assertEqual(
            tracking.identity["nama"],
            "NAMA CONTOH",
        )


if __name__ == "__main__":
    unittest.main()
