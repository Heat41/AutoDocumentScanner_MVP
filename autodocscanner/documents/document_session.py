from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autodocscanner.documents.manual_document import (
    clamp_corners,
    correct_manual_perspective,
    initial_corners,
    rotate_image_and_reset,
)


@dataclass
class DocumentPage:
    source_path: Path
    original_image: np.ndarray
    corners: np.ndarray
    corrected_image: np.ndarray | None = None


class ManualDocumentSession:
    def __init__(self):
        self.pages = []

    def _page(self, index):
        try:
            return self.pages[int(index)]
        except (
            IndexError,
            ValueError,
            TypeError,
        ):
            raise IndexError(
                "Index halaman tidak valid."
            ) from None

    def add_image(self, path, image):
        if (
            not isinstance(image, np.ndarray)
            or image.size == 0
        ):
            raise ValueError(
                "Gambar halaman tidak valid."
            )

        image = image.copy()
        height, width = image.shape[:2]
        page = DocumentPage(
            source_path=Path(path),
            original_image=image,
            corners=initial_corners(
                width,
                height,
            ),
        )
        self.pages.append(page)
        return len(self.pages) - 1

    def set_corners(self, index, corners):
        page = self._page(index)
        height, width = (
            page.original_image.shape[:2]
        )
        page.corners = clamp_corners(
            corners,
            width,
            height,
        )
        page.corrected_image = None

    def apply_correction(self, index):
        page = self._page(index)
        page.corrected_image = (
            correct_manual_perspective(
                page.original_image,
                page.corners,
            )
        )
        return page.corrected_image

    def rotate(self, index, direction):
        page = self._page(index)
        image, corners = (
            rotate_image_and_reset(
                page.original_image,
                direction,
            )
        )
        page.original_image = image
        page.corners = corners
        page.corrected_image = None

    def reset_corners(self, index):
        page = self._page(index)
        height, width = (
            page.original_image.shape[:2]
        )
        page.corners = initial_corners(
            width,
            height,
        )
        page.corrected_image = None

    def all_corrected(self):
        return (
            bool(self.pages)
            and all(
                page.corrected_image
                is not None
                for page in self.pages
            )
        )

    def corrected_images(self):
        if not self.all_corrected():
            raise RuntimeError(
                "Semua halaman harus dikoreksi "
                "sebelum diekspor."
            )

        return [
            page.corrected_image
            for page in self.pages
        ]
