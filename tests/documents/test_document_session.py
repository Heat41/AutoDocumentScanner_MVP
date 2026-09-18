import unittest
from pathlib import Path
import numpy as np

from autodocscanner.documents.document_session import ManualDocumentSession


class TestManualDocumentSession(unittest.TestCase):
    def setUp(self):
        self.session = ManualDocumentSession()
        self.image = np.zeros(
            (100, 160, 3),
            dtype=np.uint8,
        )
        self.image[10:90, 10:150] = (
            20,
            40,
            60,
        )

    def test_add_image_creates_page_with_default_corners(self):
        index = self.session.add_image(
            Path("page1.jpg"),
            self.image,
        )
        self.assertEqual(index, 0)
        self.assertEqual(
            len(self.session.pages),
            1,
        )
        self.assertEqual(
            self.session.pages[0].corners.shape,
            (4, 2),
        )
        self.assertIsNone(
            self.session.pages[0].corrected_image
        )

    def test_set_corners_invalidates_corrected_result(self):
        index = self.session.add_image(
            "page1.jpg",
            self.image,
        )
        self.session.apply_correction(index)
        self.assertIsNotNone(
            self.session.pages[index].corrected_image
        )

        corners = (
            self.session.pages[index]
            .corners.copy()
        )
        corners[0] += (5, 5)
        self.session.set_corners(
            index,
            corners,
        )

        self.assertIsNone(
            self.session.pages[index].corrected_image
        )

    def test_rotate_resets_corners_and_invalidates_result(self):
        index = self.session.add_image(
            "page1.jpg",
            self.image,
        )
        self.session.apply_correction(index)
        self.session.rotate(
            index,
            "right",
        )

        page = self.session.pages[index]
        self.assertEqual(
            page.original_image.shape[:2],
            (160, 100),
        )
        self.assertIsNone(
            page.corrected_image
        )
        self.assertTrue(
            np.all(page.corners[:, 0] <= 99)
        )
        self.assertTrue(
            np.all(page.corners[:, 1] <= 159)
        )

    def test_all_corrected_requires_every_page(self):
        first = self.session.add_image(
            "page1.jpg",
            self.image,
        )
        second = self.session.add_image(
            "page2.jpg",
            self.image,
        )
        self.session.apply_correction(first)

        self.assertFalse(
            self.session.all_corrected()
        )
        self.session.apply_correction(second)
        self.assertTrue(
            self.session.all_corrected()
        )

    def test_corrected_images_preserves_page_order(self):
        first = self.session.add_image(
            "a.jpg",
            self.image,
        )
        second_image = self.image.copy()
        second_image[:] = (
            100,
            80,
            60,
        )
        second = self.session.add_image(
            "b.jpg",
            second_image,
        )
        first_result = (
            self.session.apply_correction(
                first
            )
        )
        second_result = (
            self.session.apply_correction(
                second
            )
        )

        images = (
            self.session.corrected_images()
        )
        self.assertEqual(
            len(images),
            2,
        )
        self.assertTrue(
            np.array_equal(
                images[0],
                first_result,
            )
        )
        self.assertTrue(
            np.array_equal(
                images[1],
                second_result,
            )
        )

    def test_corrected_images_rejects_incomplete_session(self):
        self.session.add_image(
            "page1.jpg",
            self.image,
        )
        with self.assertRaises(RuntimeError):
            self.session.corrected_images()


if __name__ == "__main__":
    unittest.main()
