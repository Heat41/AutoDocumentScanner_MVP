import unittest

from autodocscanner.ktp.field_detection import (
    FIELD_CLASSES,
)
from autodocscanner.tools.ktp_annotator import (
    CANONICAL_DIR,
    DATASET_ROOT,
    KtpFieldAnnotator,
)


class TestKtpAnnotatorContract(unittest.TestCase):
    def test_dataset_is_local_ktp_fields_folder(self):
        self.assertEqual(
            DATASET_ROOT.parts[-2:],
            (
                "dataset",
                "ktp_fields",
            ),
        )

    def test_detector_classes_are_available_for_annotation(self):
        self.assertIn(
            "nik",
            FIELD_CLASSES,
        )
        self.assertIn(
            "nama",
            FIELD_CLASSES,
        )
        self.assertIn(
            "foto",
            FIELD_CLASSES,
        )

    def test_canonical_output_is_inside_local_dataset(self):
        self.assertEqual(
            CANONICAL_DIR.parts[-3:],
            (
                "dataset",
                "ktp_fields",
                "canonical",
            ),
        )

    def test_annotator_exposes_raw_to_canonical_preparation(self):
        self.assertTrue(
            callable(
                getattr(
                    KtpFieldAnnotator,
                    "_prepare_canonical",
                )
            )
        )

    def test_canonical_stem_is_not_duplicated(self):
        self.assertEqual(
            KtpFieldAnnotator._canonical_stem(
                "sample.png"
            ),
            "sample_canonical",
        )
        self.assertEqual(
            KtpFieldAnnotator._canonical_stem(
                "sample_canonical.png"
            ),
            "sample_canonical",
        )
        self.assertEqual(
            KtpFieldAnnotator._canonical_stem(
                "sample_canonical_canonical.png"
            ),
            "sample_canonical",
        )


if __name__ == "__main__":
    unittest.main()
