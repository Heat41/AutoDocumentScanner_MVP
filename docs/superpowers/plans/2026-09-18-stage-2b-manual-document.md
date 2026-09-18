# Stage 2B Manual Document Perspective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate Manual Document Correction page where the user drags four corners, applies perspective correction page-by-page, and exports corrected pages as images or one multi-page PDF.

**Architecture:** Keep the locked KTP pipeline unchanged. Add a generic manual perspective module that reuses the existing perspective warp geometry, a page/session model for multi-page state, and a new Tkinter page mounted by `ui_stage2.py`. The Stage 2 shell switches between the existing Auto KTP page and the new Manual Document page without modifying the legacy UI classes.

**Tech Stack:** Python 3, OpenCV, NumPy, Pillow, Tkinter, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Do not modify `scanner.py`, `perspective_engine.py`, `ui.py`, `ui_safe.py`, `ui_final.py`, or `ui_responsive.py`.
- Manual document mode must not use automatic corner detection.
- Four corner points must appear automatically near the image corners.
- All four points must be draggable and clamped to image bounds.
- Perspective correction must use the same geometric warp behavior as the existing engine.
- Multi-page work is page-by-page.
- Image export writes one corrected image per page.
- PDF export writes all corrected pages into one PDF in page order.
- Existing Auto KTP Gambar/PDF behavior must remain unchanged.

---

### Task 1: Manual perspective core

**Files:**
- Create: `manual_document.py`
- Create: `tests/test_manual_document.py`

**Interfaces:**
- `initial_corners(width: int, height: int, margin_ratio: float = 0.06) -> np.ndarray`
- `clamp_corners(points, width: int, height: int) -> np.ndarray`
- `correct_manual_perspective(image, points) -> np.ndarray`
- `rotate_image_and_reset(image, direction: str) -> tuple[np.ndarray, np.ndarray]`

**TDD:**
- [x] Write tests for default point order/placement, bounds clamping, warp output, and 90-degree rotation.
- [x] Run `python -m unittest tests.test_manual_document -v` and verify RED because the module is missing.
- [x] Implement the minimum code using `AutoPerspectiveEngine.warp`.
- [x] Verify GREEN.

---

### Task 2: Multi-page document session

**Files:**
- Create: `document_session.py`
- Create: `tests/test_document_session.py`

**Interfaces:**
- `DocumentPage` dataclass with source path, original image, corners, corrected image.
- `ManualDocumentSession.add_image(path, image) -> int`
- `ManualDocumentSession.set_corners(index, corners) -> None`
- `ManualDocumentSession.apply_correction(index) -> np.ndarray`
- `ManualDocumentSession.rotate(index, direction) -> None`
- `ManualDocumentSession.reset_corners(index) -> None`
- `ManualDocumentSession.all_corrected() -> bool`
- `ManualDocumentSession.corrected_images() -> list[np.ndarray]`

**TDD:**
- [x] Write state-transition tests.
- [x] Verify RED.
- [x] Implement minimum state model.
- [ ] Verify GREEN.

---

### Task 3: Multi-page output support

**Files:**
- Modify: `output_manager.py`
- Modify: `tests/test_output_manager.py`

**Interfaces:**
- Add `save_pdf_pages(output_path, images) -> Path`
- Add `save_document_images(output_dir, source_paths, images) -> list[Path]`

**Behavior:**
- PDF output is atomic.
- Page order is preserved.
- No A4 canvas is introduced.
- Image output uses `*_corrected.jpg` naming and existing atomic image write semantics.

**TDD:**
- [x] Add failing two-page PDF and multi-image output tests.
- [ ] Verify RED.
- [x] Implement minimum output functions.
- [ ] Verify GREEN.

---

### Task 4: Canvas coordinate helpers

**Files:**
- Create: `document_canvas.py`
- Create: `tests/test_document_canvas.py`

**Interfaces:**
- `fit_image_size(image_width, image_height, canvas_width, canvas_height, padding=24) -> tuple[int, int]`
- `image_to_canvas(point, image_size, display_size, offset) -> tuple[float, float]`
- `canvas_to_image(point, image_size, display_size, offset) -> tuple[float, float]`

**TDD:**
- [x] Test fit behavior and round-trip coordinate transforms.
- [ ] Verify RED.
- [x] Implement helpers.
- [ ] Verify GREEN.

---

### Task 5: Manual Document page and Stage 2 navigation

**Files:**
- Create: `ui_manual_document.py`
- Modify: `ui_stage2.py`
- Create: `tests/test_ui_manual_document.py`
- Modify: `tests/test_ui_stage2.py`

**Architecture:**
- `ManualDocumentPage(ttk.Frame)` is a distinct page.
- `Stage2ScannerUI` keeps the existing KTP root frame as the Auto KTP page.
- Stage 2 adds a button in the existing sidebar to open Manual Document.
- Page switching uses `pack_forget()` / `pack()` on page frames; no legacy UI classes are edited.
- Manual page includes file selection, page list, canvas, draggable TL/TR/BR/BL handles, Reset, Rotate Left/Right, Preview/Apply, output choice, and Export.
- PDF export is enabled only when all pages have corrected results.

**TDD:**
- [x] Add headless smoke tests for page inheritance, navigation constants, and helper contracts.
- [ ] Verify RED.
- [x] Implement page and navigation wrapper.
- [x] Verify GREEN for Stage 2 tests.

---

### Task 6: Regression and packaging guard

**Verification on Windows checkout:**
- [ ] `python -m unittest discover -s tests -v`
- [ ] `python .\app.py --ui`
- [ ] UAT Auto KTP image output.
- [ ] UAT Auto KTP PDF output.
- [ ] UAT Manual Document single-page image output.
- [ ] UAT Manual Document multi-page combined PDF.
- [ ] `.\build_windows.ps1` after UAT.

**Diff guard:**
- [x] Confirm locked legacy scanner/UI files remain unchanged.


## Implementation checkpoint — 2026-09-18

- Isolated headless Stage 2B verification: 34 tests, 0 failures.
- Branch: `feature/stage2b-manual-document`.
- Full Windows repository regression and packaged UAT remain intentionally pending on the project PC.
