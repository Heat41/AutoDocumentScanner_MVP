# Stage 2D Hybrid KTP OCR Accuracy Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Improve automatic Tracking KTP field population by combining full-card label-aware OCR with the existing per-field bbox OCR fallback.

**Architecture:** Add a document OCR pass that reads the corrected KTP as a whole and parses labeled KTP lines into field candidates. For each field, compare the full-card candidate with the existing bbox candidate and prefer the more structurally valid result; keep bbox OCR as fallback. Existing UI auto-fill and PDF report behavior remain unchanged.

**Tech Stack:** Python 3.12, OpenCV, NumPy, pytesseract/Tesseract, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Do not change scanner or perspective correction.
- Keep existing normalized bbox OCR as fallback.
- Full-card OCR runs on the corrected KTP image.
- Use label-aware parsing for standard Indonesian KTP labels.
- NIK must still validate to exactly 16 digits.
- TTL and RT/RW candidates must pass their existing structural parsers.
- Unit tests must not require a local Tesseract installation.
- Existing fake backends that implement only `read()` must continue to work.

## Review Focus

- Shifted KTP text: full-card OCR can still populate fields.
- Full-card OCR misses a label: bbox fallback remains available.
- Full-card OCR returns malformed NIK/TTL/RT-RW: invalid candidate must not replace a better bbox result.
- Header province/city lines without colon are recognized.
- OCR garbage with many symbols must not outrank a clean field candidate.

---

### Task 1 — Full-card OCR
**Files:**
- Modify: `autodocscanner/ktp/ocr.py`
- Modify: `tests/ktp/test_ocr.py`

**Interfaces:**
- `preprocess_document(image, fallback=False)`
- `TesseractBackend.read_document(image) -> tuple[str, float]`

### Task 2 — Label-aware KTP document parser
**Files:**
- Modify: `autodocscanner/ktp/parsing.py`
- Create/modify: `tests/ktp/test_parsing.py`

**Interfaces:**
- `parse_ktp_document(text) -> dict[str, str]`

### Task 3 — Hybrid candidate selection
**Files:**
- Modify: `autodocscanner/ktp/tracking.py`
- Modify: `tests/ktp/test_tracking.py`

**Behavior:**
- if backend supports `read_document`, get document candidates;
- OCR bbox candidates as before;
- choose per field using parsed completeness and cleanliness;
- preserve current API and review behavior.

### Task 4 — Regression
- Full test: `python -m unittest discover -s tests -q`.
- UI smoke using the same problematic KTP examples.
