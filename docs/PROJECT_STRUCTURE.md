# AutoDocumentScanner Project Structure

Production Python code lives under `autodocscanner/` and is grouped by domain.

```
autodocscanner/
├── core/        # KTP scanner, perspective detection, quality and validation
├── documents/   # Manual document perspective correction and page sessions
├── ktp/         # KTP identity models, SQLite repository and media storage
├── output/      # Atomic image/PDF output helpers
├── services/    # Application workflows that coordinate multiple domains
├── ui/          # Tkinter UI hierarchy and Stage 2 pages
└── support/     # Branding and other support utilities
```

Repository root is reserved for application entry points and operational files:

- `app.py` — CLI/source entry point.
- `desktop_launcher.py` — packaged Windows application entry point.
- `AutoDocumentScanner.spec` — PyInstaller configuration.
- `build_windows.ps1`, `release_windows.ps1`, `build_installer.ps1` — Windows build/release pipeline.
- `requirements*.txt`, `VERSION.txt`, `DEPLOYMENT.md` — project metadata and deployment configuration.

Tests mirror the source domains under `tests/`:

```
tests/
├── core/
├── documents/
├── ktp/
├── output/
├── services/
├── ui/
└── support/
```

## Dependency direction

The intended dependency direction is:

```
core/output
    ↑
documents / ktp
    ↑
services
    ↑
ui
    ↑
app / desktop_launcher
```

UI modules must not contain scanner/storage algorithms. Core and storage modules must not import Tkinter UI modules.
