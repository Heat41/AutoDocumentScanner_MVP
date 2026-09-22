import unittest
from tkinter import ttk

from autodocscanner.ui.tracking_ktp import TrackingKtpPage


class TestTrackingKtpPageContract(unittest.TestCase):
    def test_tracking_page_is_a_ttk_frame(self):
        self.assertTrue(
            issubclass(
                TrackingKtpPage,
                ttk.Frame,
            )
        )

    def test_page_title_is_locked(self):
        self.assertEqual(
            TrackingKtpPage.PAGE_TITLE,
            "Tracking Data KTP",
        )

    def test_review_sections_are_defined(self):
        self.assertEqual(
            TrackingKtpPage.SECTIONS,
            (
                "Input KTP",
                "Preview KTP",
                "Review Tracking",
            ),
        )

    def test_review_fields_include_structured_identity(self):
        self.assertIn(
            "nik",
            TrackingKtpPage.REVIEW_FIELDS,
        )
        self.assertIn(
            "tanggal_lahir",
            TrackingKtpPage.REVIEW_FIELDS,
        )
        self.assertIn(
            "kabupaten_kota",
            TrackingKtpPage.REVIEW_FIELDS,
        )

    def test_review_status_text_for_clean_result(self):
        self.assertEqual(
            TrackingKtpPage.review_status_text([]),
            "Hasil OCR siap direview.",
        )

    def test_review_status_text_lists_attention_count(self):
        self.assertEqual(
            TrackingKtpPage.review_status_text(
                ["nama", "alamat"]
            ),
            "2 field perlu diperiksa.",
        )

    def test_review_value_mapping_keeps_known_identity_fields(self):
        class FakeTracking:
            identity = {
                "nik": "1234567890123456",
                "nama": "NAMA CONTOH",
                "tempat_lahir": "KOTA CONTOH",
                "tanggal_lahir": "1990-03-02",
            }

        result = (
            TrackingKtpPage
            .review_value_mapping(
                FakeTracking()
            )
        )

        self.assertEqual(
            result["nama"],
            "NAMA CONTOH",
        )
        self.assertEqual(
            result["tanggal_lahir"],
            "1990-03-02",
        )

    def test_primary_columns_prioritize_review_panel(self):
        self.assertEqual(
            TrackingKtpPage.COLUMN_WEIGHTS,
            (1, 3, 4),
        )

    def test_review_groups_are_defined(self):
        self.assertEqual(
            tuple(
                TrackingKtpPage.REVIEW_GROUPS
            ),
            (
                "Identitas Utama",
                "Alamat",
                "Data Lainnya",
            ),
        )

    def test_field_status_text_is_quiet_for_valid_values(self):
        self.assertEqual(
            TrackingKtpPage.field_status_text(
                value="NAMA CONTOH",
                needs_review=False,
            ),
            "",
        )

    def test_field_status_text_marks_review_and_empty_values(self):
        self.assertEqual(
            TrackingKtpPage.field_status_text(
                value="NAMA BURAM",
                needs_review=True,
            ),
            "Periksa",
        )
        self.assertEqual(
            TrackingKtpPage.field_status_text(
                value="",
                needs_review=True,
            ),
            "Kosong",
        )

    def test_pdf_preview_method_is_exposed(self):
        self.assertTrue(
            callable(
                getattr(
                    TrackingKtpPage,
                    "preview_pdf",
                )
            )
        )

    def test_roi_calibration_toggle_is_exposed(self):
        self.assertTrue(
            callable(
                getattr(
                    TrackingKtpPage,
                    "toggle_roi_overlay",
                )
            )
        )

    def test_detector_sample_export_method_is_exposed(self):
        self.assertTrue(
            callable(
                getattr(
                    TrackingKtpPage,
                    "save_detector_sample",
                )
            )
        )

    def test_current_review_values_method_is_exposed(self):
        self.assertTrue(
            callable(
                getattr(
                    TrackingKtpPage,
                    "current_review_values",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
