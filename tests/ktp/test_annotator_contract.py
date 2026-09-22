import unittest

from autodocscanner.ktp.field_detection import (
    FIELD_CLASSES,
)
from autodocscanner.tools.ktp_annotator import (
    DATASET_ROOT,
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


if __name__ == "__main__":
    unittest.main()
