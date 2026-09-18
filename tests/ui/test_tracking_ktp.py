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


if __name__ == "__main__":
    unittest.main()
