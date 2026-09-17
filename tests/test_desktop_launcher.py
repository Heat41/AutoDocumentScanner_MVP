import unittest
from pathlib import Path

from desktop_launcher import application_root


class TestDesktopLauncher(unittest.TestCase):
    def test_source_mode_uses_source_file_parent(self):
        source_file = Path("project") / "desktop_launcher.py"
        root = application_root(
            frozen=False,
            source_file=source_file,
        )

        self.assertEqual(
            root,
            source_file.resolve().parent,
        )

    def test_frozen_mode_uses_executable_parent(self):
        executable = Path("dist") / "AutoDocumentScanner" / "AutoDocumentScanner.exe"
        root = application_root(
            frozen=True,
            executable=executable,
        )

        self.assertEqual(
            root,
            executable.resolve().parent,
        )


if __name__ == "__main__":
    unittest.main()
