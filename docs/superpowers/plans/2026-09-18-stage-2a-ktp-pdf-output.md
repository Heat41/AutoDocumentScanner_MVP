# Stage 2A KTP PDF Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PDF output for corrected KTP images without changing the existing auto-perspective scanner behavior.

**Architecture:** Keep `scanner.py` responsible for producing the corrected image and existing image output. Introduce a focused `output_manager.py` for format naming and PDF serialization. The UI gains a separate output-format choice (Image/PDF), while color/grayscale remains independent. PDF serialization converts BGR to RGB and writes a single-page PDF at 72 DPI so the PDF media box matches the corrected image pixel dimensions without A4 padding.

**Tech Stack:** Python 3, OpenCV, NumPy, Pillow, Tkinter, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Do not change automatic KTP corner detection or perspective-correction behavior.
- Default visual output remains color.
- PDF must not place the KTP on A4.
- PDF output must preserve corrected KTP image dimensions/aspect ratio without resampling.
- Existing image-output behavior must remain available.
- Stage 2A must not implement document-manual-correction, OCR tracking, database, or WP Client API work.

---

### Task 1: Output Manager contract and PDF writer

**Files:**
- Create: `output_manager.py`
- Create: `tests/test_output_manager.py`

**Interfaces:**
- Produces: `output_suffix(output_format: str) -> str`
- Produces: `build_output_path(output_dir, input_path, output_format: str) -> Path`
- Produces: `save_pdf(output_path, image) -> Path`

- [ ] **Step 1: Write failing tests**

Test exact naming for image/PDF, rejection of unsupported formats, PDF file creation, and PDF page dimensions matching image width/height at 72 DPI.

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_output_manager -v`

Expected: FAIL because `output_manager` does not exist.

- [ ] **Step 3: Implement minimal output manager**

Use Pillow only (already in requirements). Convert OpenCV BGR arrays to RGB; grayscale arrays to RGB safely. Save via a temporary file in the target directory and `os.replace` to preserve atomic-write semantics. Use `resolution=72.0` so a W x H image yields a W x H point PDF page.

- [ ] **Step 4: Run output-manager tests and verify GREEN**

Run: `python -m unittest tests.test_output_manager -v`

Expected: PASS.

- [ ] **Step 5: Run existing output/scanner regression tests**

Run:
`python -m unittest tests.test_perspective_engine tests.test_batch_runner -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add atomic PDF output manager`

---

### Task 2: UI output format selection

**Files:**
- Modify: `ui.py`
- Modify: `ui_final.py` only if presentation layer requires placement adjustment
- Create: `tests/test_ui_output_format.py`

**Interfaces:**
- Consumes: `build_output_path(...)`, `save_pdf(...)`
- UI state: `self.output_format` values `image` or `pdf`

- [ ] **Step 1: Write failing UI-independent tests**

Test a small pure helper or static contract used by the UI to map `image` and `pdf` to destination paths. Do not instantiate Tk in headless tests.

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_ui_output_format -v`

Expected: FAIL because the UI format contract is not wired.

- [ ] **Step 3: Add format selector**

Add a distinct output-format variable with default `image`. Keep the existing `output_mode` color/grayscale variable unchanged. Present `Gambar` and `PDF` as a separate choice.

- [ ] **Step 4: Wire processing**

For image: retain current `scanner.scan(..., output_path=jpg)` path.

For PDF: call `scanner.scan(..., output_path=None)`, then `save_pdf(pdf_path, result)`. Store the generated PDF path separately from preview requirements; retain an in-memory/image preview or a temporary preview strategy rather than asking Pillow/Tk to render PDF.

- [ ] **Step 5: Verify tests**

Run:
`python -m unittest tests.test_ui_output_format tests.test_output_manager tests.test_perspective_engine tests.test_batch_runner tests.test_ui_responsive -v`

Expected: PASS.

- [ ] **Step 6: Commit**

Commit message: `feat: add KTP image or PDF output choice`

---

### Task 3: Full regression and packaging guard

**Files:**
- Modify tests only if an existing packaging test requires explicit inclusion of `output_manager.py`.
- Modify packaging spec/script only if it enumerates source modules explicitly.

**Interfaces:**
- No new runtime behavior.

- [ ] **Step 1: Run full test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 2: Verify imports**

Run: `python -c "import app, scanner, output_manager, ui_responsive"`

Expected: exits 0.

- [ ] **Step 3: Build/package smoke check**

Use the repository's existing build scripts/spec without changing installer behavior except including the new module when necessary.

- [ ] **Step 4: Commit any packaging-only change**

Commit message: `build: include Stage 2A output manager`
