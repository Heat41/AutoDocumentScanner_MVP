import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from branding import ensure_brand_assets, source_root


class TestBranding(unittest.TestCase):
    def test_brand_assets_are_generated_from_tracked_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = root / "assets"
            assets.mkdir(parents=True, exist_ok=True)

            shutil.copyfile(
                source_root() / "assets" / "logo_source.b64",
                assets / "logo_source.b64",
            )

            paths = ensure_brand_assets(root)

            self.assertTrue(paths["png"].exists())
            self.assertTrue(paths["ico"].exists())

            with Image.open(paths["png"]) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (192, 192))

            with Image.open(paths["ico"]) as image:
                self.assertEqual(image.format, "ICO")

    def test_packaging_configs_reference_official_logo(self):
        root = source_root()

        spec = (root / "AutoDocumentScanner.spec").read_text(
            encoding="utf-8"
        )
        installer = (
            root / "installer" / "AutoDocumentScanner.iss.in"
        ).read_text(encoding="utf-8")

        self.assertIn('icon="assets/logo.ico"', spec)
        self.assertIn('("assets/logo.png", "assets")', spec)
        self.assertIn("SetupIconFile=logo.ico", installer)


if __name__ == "__main__":
    unittest.main()
