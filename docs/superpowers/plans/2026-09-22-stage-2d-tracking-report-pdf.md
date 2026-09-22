# Stage 2D Tracking Report PDF Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Change Tracking Data KTP Preview PDF from a KTP-only page into a structured tracking report containing the corrected KTP, face crop, and the latest values shown in the editable review fields.

**Architecture:** Keep the existing KTP-only PDF output untouched. Add a dedicated Tracking Report renderer under services that composes a report page with Pillow and saves it as PDF. The Tracking UI collects the current StringVar values at preview time, so OCR auto-fill remains unchanged and user corrections are reflected in the report.

**Tech Stack:** Python 3.12, Pillow, NumPy, pathlib, tempfile, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Do not change scanner, perspective, OCR, parsing, tracking, or storage behavior.
- OCR continues to auto-fill the right-side review fields.
- PDF uses the current right-side values, including manual edits.
- Tracking report PDF is separate from the existing KTP-size PDF output.
- Include corrected KTP image and face crop.
- Include all review fields in readable groups.
- Preview remains temporary and opens in the Windows default PDF viewer.
- Rekam Data KTP remains disabled.

## Review Focus

- Manual edits after OCR must appear in the next PDF preview.
- Empty values must render as `-`, not break layout.
- Long address/job text must wrap rather than overflow.
- Missing/empty face crop must not break report generation.
- Repeated preview must generate a fresh valid PDF.

---

### Task 1 — Tracking report renderer
**Files:**
- Replace/extend: `autodocscanner/services/pdf_preview.py`
- Modify: `tests/services/test_pdf_preview.py`

**Interfaces:**
- `build_tracking_report_image(corrected_image, face_image, fields) -> PIL.Image`
- `build_tracking_report_pdf(corrected_image, face_image, fields, stem="tracking_ktp", temp_dir=None) -> Path`

### Task 2 — UI uses current editable values
**Files:**
- Modify: `autodocscanner/ui/tracking_ktp.py`
- Modify: `tests/ui/test_tracking_ktp.py`

**Behavior:**
- Add `current_review_values()`.
- Preview PDF reads all current StringVar values.
- OCR auto-fill behavior remains unchanged.
- Preview PDF sends current values + corrected KTP + face crop to report service.

### Task 3 — Regression checkpoint
- Run full regression with `python -m unittest discover -s tests -q`.
- UI smoke: process KTP, edit one field manually, Preview PDF, verify edited value appears.
