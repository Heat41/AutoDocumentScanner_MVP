# Stage 2D Tracking UI Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Connect the existing Tracking Data KTP page to the locked KTP scanner and Stage 2D OCR tracking service so a user can select one KTP image, process it, and review editable OCR results.

**Architecture:** Add a headless service that coordinates `AutoDocumentScanner.scan(..., output_path=None)` with `extract_tracking_data()`. Keep Tkinter responsible only for file selection, worker-thread orchestration, previews, editable review fields, and visible review warnings. No Stage 2C database write occurs in this slice.

**Tech Stack:** Python 3.12, Tkinter/ttk, OpenCV, Pillow, Tesseract OCR, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Work directly on the green `main` baseline.
- Do not modify scanner or perspective-engine algorithms.
- Tracking KTP processes one selected KTP image at a time.
- Use the existing KTP auto-perspective pipeline before OCR.
- OCR processing runs off the Tk main thread.
- Review fields are editable.
- Fields flagged by `review_fields` must be visibly marked.
- Show corrected KTP preview and face crop.
- The Rekam Data KTP button must remain disabled in this slice.
- Processing failure must leave the page recoverable for another attempt.

## Review Focus

- Unreadable/unsupported image: show a failure status and re-enable processing.
- Tesseract unavailable: show an explicit OCR/runtime failure without crashing the app.
- OCR returns incomplete identity: populate what is available and mark review fields.
- Reprocessing another image: previous review values and warnings must be replaced, not mixed.
- Navigating away during/after tracking processing: Stage 2 top navigation must remain functional.

---

### Task 1 — Headless Tracking Input Service
**Files:**
- Create: `autodocscanner/services/ktp_tracking.py`
- Create: `tests/services/test_ktp_tracking.py`

**Interfaces:**
- `process_tracking_input(scanner, input_path, backend=None) -> dict`
- returns `corrected_image`, `corners`, `tracking`.

### Task 2 — Tracking Review Helpers
**Files:**
- Modify: `autodocscanner/ui/tracking_ktp.py`
- Modify: `tests/ui/test_tracking_ktp.py`

**Interfaces:**
- `REVIEW_FIELDS`
- `FIELD_LABELS`
- `review_value_mapping(tracking_result)`
- `review_status_text(review_fields)`

### Task 3 — Functional Tracking Page
**Files:**
- Modify: `autodocscanner/ui/tracking_ktp.py`
- Modify: `autodocscanner/ui/stage2.py`

**Behavior:**
- Pass the existing scanner instance into `TrackingKtpPage`.
- Select one image.
- Process in a worker thread.
- Render corrected preview and face preview.
- Populate editable identity fields.
- Mark OCR fields requiring review.
- Keep Rekam disabled.

### Task 4 — Regression checkpoint
- Full unit regression.
- UI smoke: select image, process, switch between all three top navigation pages, process a second image.


## Implementation checkpoint — 2026-09-22

Completed on `main`:
- Headless Tracking KTP processing service.
- Tracking page uses the existing shared scanner instance.
- Single-image KTP selection.
- Worker-thread processing for auto perspective + OCR.
- Corrected KTP preview.
- Face crop preview.
- Editable structured identity review fields.
- Visible review markers for OCR fields requiring attention.
- Reprocessing replaces prior review state.
- Rekam Data KTP remains disabled pending Stage 2F.

Audit:
- Scanner and perspective engine files were not changed.
- Stage 2C storage is not written automatically.
- Full regression and Windows UI smoke are pending.
