import tempfile
import unittest
from pathlib import Path

from autodocscanner.ktp.annotation import (
    Annotation,
)
from autodocscanner.ktp.annotation_template import (
    load_annotation_seed,
    save_annotation_template_if_missing,
)


class TestAnnotationTemplate(unittest.TestCase):
    def setUp(self):
        self.annotations = [
            Annotation(
                class_name="nama",
                bbox=(100, 100, 300, 140),
            ),
            Annotation(
                class_name="nik",
                bbox=(120, 60, 420, 95),
            ),
        ]

    def test_first_save_creates_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_path = (
                Path(tmp)
                / "default.txt"
            )

            created = (
                save_annotation_template_if_missing(
                    template_path,
                    self.annotations,
                    image_width=856,
                    image_height=540,
                )
            )

            self.assertTrue(created)
            self.assertTrue(
                template_path.is_file()
            )

    def test_existing_template_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_path = (
                Path(tmp)
                / "default.txt"
            )

            save_annotation_template_if_missing(
                template_path,
                self.annotations,
                image_width=856,
                image_height=540,
            )

            original = (
                template_path.read_text(
                    encoding="utf-8"
                )
            )

            created = (
                save_annotation_template_if_missing(
                    template_path,
                    [
                        Annotation(
                            class_name="foto",
                            bbox=(
                                600,
                                100,
                                800,
                                400,
                            ),
                        )
                    ],
                    image_width=856,
                    image_height=540,
                )
            )

            self.assertFalse(created)
            self.assertEqual(
                template_path.read_text(
                    encoding="utf-8"
                ),
                original,
            )

    def test_own_label_has_priority_over_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            own = root / "sample.txt"
            template = root / "default.txt"

            save_annotation_template_if_missing(
                template,
                self.annotations,
                image_width=856,
                image_height=540,
            )

            own.write_text(
                "3 0.500000 0.500000 0.200000 0.100000\n",
                encoding="utf-8",
            )

            result = load_annotation_seed(
                own_label_path=own,
                template_label_path=template,
                image_width=856,
                image_height=540,
            )

            self.assertEqual(
                result.source,
                "image",
            )
            self.assertEqual(
                len(result.annotations),
                1,
            )
            self.assertEqual(
                result.annotations[0].class_name,
                "nama",
            )

    def test_template_seeds_unlabeled_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            own = root / "missing.txt"
            template = root / "default.txt"

            save_annotation_template_if_missing(
                template,
                self.annotations,
                image_width=856,
                image_height=540,
            )

            result = load_annotation_seed(
                own_label_path=own,
                template_label_path=template,
                image_width=856,
                image_height=540,
            )

            self.assertEqual(
                result.source,
                "template",
            )
            self.assertEqual(
                result.annotations,
                self.annotations,
            )


if __name__ == "__main__":
    unittest.main()
