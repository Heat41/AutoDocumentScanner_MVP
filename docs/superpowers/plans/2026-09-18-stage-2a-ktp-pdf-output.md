# Stage 2A KTP PDF Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PDF output for corrected KTP images without changing the existing auto-perspective scanner behavior.

**Architecture:** The locked scanner and existing UI hierarchy remain untouched. `output_manager.py` owns output naming and atomic PDF serialization, `stage2_processing.py` routes one corrected KTP result to the existing JPG path or the new PDF path, and `ui_stage2.py` wraps `ResponsiveScannerUI` to add the Gambar/PDF choice. Entry points launch the wrapper while the original scanner/UI layers remain available for rollback.

**Tech Stack:** Python 3, OpenCV, NumPy, Pillow, Tkinter, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Do not change automatic KTP corner detection or perspective-correction behavior.
- Default visual mode remains color.
- Default output format remains image.
- PDF must not place the KTP on A4.
- PDF output uses the corrected image dimensions/aspect ratio without cropping or A4 padding.
- Existing JPG-output path remains available and continues to be written by `scanner.scan`.
- Stage 2A must not implement document-manual-correction, OCR tracking, database, or WP Client API work.

---

### Task 1: Output Manager

**Files:**
- Create: `output_manager.py`
- Create: `tests/test_output_manager.py`

**Interfaces:**
- `output_suffix(output_format: str) -> str`
- `build_output_path(output_dir, input_path, output_format: str) -> Path`
- `save_pdf(output_path, image) -> Path`

- [x] Write failing output-manager tests.
- [x] Verify RED: module missing.
- [x] Implement atomic single-page PDF output using Pillow at 72 DPI.
- [x] Verify GREEN: six output-manager tests pass in isolated TDD sandbox.

### Task 2: Stage 2 Output Processing

**Files:**
- Create: `stage2_processing.py`
- Create: `tests/test_stage2_processing.py`

**Interfaces:**
- `process_ktp_output(scanner, input_path, output_dir, output_mode="color", output_format="image") -> dict`

**Behavior:**
- Image: preserve existing `scanner.scan(..., output_path=...jpg)` path.
- PDF: call `scanner.scan(..., output_path=None)`, atomically save PDF, and generate compressed in-memory JPEG preview bytes.
- Always forward `mode="ktp"` and the selected color/grayscale mode.

- [x] Write failing processing tests.
- [x] Verify RED: module missing.
- [x] Implement routing.
- [x] Verify GREEN together with output-manager tests.

### Task 3: Isolated Stage 2 UI Wrapper

**Files:**
- Create: `ui_stage2.py`
- Create: `tests/test_ui_stage2.py`
- Modify: `app.py`
- Modify: `desktop_launcher.py`

**Architecture:**
- `Stage2ScannerUI(ResponsiveScannerUI)` keeps all old UI/scanner files unchanged.
- Add separate `Gambar` / `PDF` controls.
- Capture the format on the main thread before processing starts.
- Reuse the existing safe failure state.
- Render PDF results from compressed in-memory preview bytes instead of requiring a PDF renderer.

- [x] Write failing wrapper smoke tests.
- [x] Verify RED: wrapper module missing.
- [x] Implement wrapper.
- [x] Verify 11 Stage 2A tests + Python compile in isolated sandbox.
- [x] Point source and packaged desktop entry points to `ui_stage2`.

### Task 4: Regression and Packaging Guard

**Files inspected:**
- `AutoDocumentScanner.spec`
- `build_windows.ps1`
- `release_windows.ps1`
- `build_installer.ps1`

**Findings:**
- PyInstaller starts from `desktop_launcher.py` with `hiddenimports=[]`; no manual module list needs updating.
- `build_windows.ps1` runs `python -m unittest discover -s tests -v` before PyInstaller.
- Release and installer scripts package the already-built runtime and require no Stage 2A source-module changes.

- [x] Confirm diff leaves `scanner.py`, `perspective_engine.py`, `ui.py`, `ui_safe.py`, `ui_final.py`, and `ui_responsive.py` untouched.
- [x] Run Stage 2A isolated tests: 11 tests, 0 failures.
- [ ] Run full repository regression on Windows project checkout.
- [ ] Run `build_windows.ps1`.
- [ ] UAT generated JPG and PDF in packaged executable.

## Windows verification commands

```powershell
cd D:\AutoDocumentScanner
git fetch origin
git switch feature/stage2a-pdf-output
git pull origin feature/stage2a-pdf-output

python -m unittest discover -s tests -v
python .\app.py --ui
```

For packaging verification after UAT:

```powershell
.\build_windows.ps1
```
