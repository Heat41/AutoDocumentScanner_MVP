import os
import sys
from pathlib import Path


def application_root(frozen=None, executable=None, source_file=None):
    """Return folder that should own runtime input/output paths.

    Source mode points to the repository folder. PyInstaller mode points to
    the folder containing the packaged executable, so double-click execution
    does not depend on an arbitrary Windows working directory.
    """
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))

    if frozen:
        executable = executable or sys.executable
        return Path(executable).resolve().parent

    source_file = source_file or __file__
    return Path(source_file).resolve().parent


def prepare_runtime_directory():
    root = application_root()
    os.chdir(root)

    # Keep runtime folders local to the application package. The scanner UI
    # may still let the user choose any other output folder explicitly.
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)

    return root


def main():
    prepare_runtime_directory()

    # Import only after the runtime directory is prepared. The responsive
    # shell changes layout only; the locked scanner pipeline stays unchanged.
    from autodocscanner.ui.stage2 import main as run_ui

    run_ui()


if __name__ == "__main__":
    main()
