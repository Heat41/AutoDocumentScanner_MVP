import unittest
from pathlib import Path

from autodocscanner.core.scanner import AutoDocumentScanner
from autodocscanner.documents.document_session import ManualDocumentSession
from autodocscanner.ktp.storage import KtpStorage
from autodocscanner.output.manager import save_pdf
from autodocscanner.services.batch import BatchScanRunner
from autodocscanner.ui.stage2 import Stage2ScannerUI


class TestPackageStructure(unittest.TestCase):
    def test_public_modules_import_from_domain_packages(self):
        self.assertIsNotNone(AutoDocumentScanner)
        self.assertIsNotNone(ManualDocumentSession)
        self.assertIsNotNone(KtpStorage)
        self.assertTrue(callable(save_pdf))
        self.assertIsNotNone(BatchScanRunner)
        self.assertIsNotNone(Stage2ScannerUI)

    def test_legacy_root_source_modules_are_removed(self):
        root = Path(__file__).resolve().parents[1]
        legacy_files = (
            "scanner.py",
            "perspective_engine.py",
            "robustness_engine.py",
            "quality_check.py",
            "final_validation.py",
            "safe_output.py",
            "output_manager.py",
            "batch_runner.py",
            "stage2_processing.py",
            "manual_document.py",
            "document_session.py",
            "document_canvas.py",
            "ktp_models.py",
            "ktp_database.py",
            "ktp_media.py",
            "ktp_storage.py",
            "ui.py",
            "ui_safe.py",
            "ui_final.py",
            "ui_textured.py",
            "ui_responsive.py",
            "ui_stage2.py",
            "ui_manual_document.py",
            "ui_tracking_ktp.py",
            "branding.py",
        )

        existing = [
            name
            for name in legacy_files
            if (root / name).exists()
        ]

        self.assertEqual(existing, [])

    def test_domain_directories_exist(self):
        root = Path(__file__).resolve().parents[1]
        package_root = root / "autodocscanner"

        for name in (
            "core",
            "documents",
            "ktp",
            "output",
            "services",
            "ui",
            "support",
        ):
            with self.subTest(name=name):
                self.assertTrue(
                    (package_root / name).is_dir()
                )


if __name__ == "__main__":
    unittest.main()
