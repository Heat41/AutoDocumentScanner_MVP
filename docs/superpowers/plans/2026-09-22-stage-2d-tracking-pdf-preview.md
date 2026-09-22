# Stage 2D Tracking KTP PDF Preview Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Add a Preview PDF action to Tracking Data KTP using the corrected KTP image only.

**Architecture:** Reuse `autodocscanner.output.manager.save_pdf` so PDF dimensions remain identical to the corrected KTP image at 72 DPI. A small service writes the preview to the system temporary directory and the Tracking UI opens it with the Windows default PDF viewer.

**Tech Stack:** Python 3.12, pathlib, tempfile, os.startfile on Windows, Pillow PDF output, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Corrected KTP image only.
- One-page PDF.
- No A4 padding or resizing.
- No OCR/tracking text in the PDF.
- Preview button disabled until Tracking has completed successfully.
- Do not change scanner, perspective, OCR, parsing, or storage behavior.
- Reuse existing `save_pdf`; do not duplicate PDF-writing logic.

## Review Focus

- Preview requested before processing: button remains disabled.
- Temporary PDF generation failure: show an error without losing Tracking result.
- Repeated preview: safely replace/create a fresh preview file.
- PDF viewer launch failure: PDF remains generated and error is shown.
- Windows path containing spaces/non-ASCII characters: use Path/string safely.

---

### Task 1 — PDF preview service
**Files:**
- Create: `autodocscanner/services/pdf_preview.py`
- Create: `tests/services/test_pdf_preview.py`

**Interfaces:**
- `build_ktp_preview_pdf(image, stem="ktp_preview", temp_dir=None) -> Path`

### Task 2 — Tracking UI button
**Files:**
- Modify: `autodocscanner/ui/tracking_ktp.py`
- Modify: `tests/ui/test_tracking_ktp.py`

**Behavior:**
- Add `Preview PDF` button near Tracking actions.
- Disabled before successful processing.
- Enabled after `_finish_success`.
- Calls preview service with `tracking.corrected_image`.
- Opens PDF with default Windows viewer.
- Reset disables it when a new KTP is selected.
