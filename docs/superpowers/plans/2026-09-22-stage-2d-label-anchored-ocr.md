# Stage 2D Label-Anchored Tracking OCR Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Make Tracking KTP read text fields by locating KTP labels in the corrected image and extracting values spatially to the right of each label.

**Architecture:** Tesseract exposes word-level OCR boxes for the corrected KTP. A label-anchor parser groups words into lines, matches standard KTP labels, and builds field candidates from words positioned after each label. Existing full-card parsing and static bbox OCR remain fallback layers only. Photo crop remains unchanged.

**Tech Stack:** Python 3.12, OpenCV, NumPy, pytesseract/Tesseract, difflib, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Only Tracking OCR internals change.
- Do not modify Auto Koreksi KTP page, manual document page, scanner geometry, or perspective engine.
- OCR input is the corrected KTP image.
- Photo crop stays on the existing bbox because it is already working.
- Label-anchor candidate has priority when structurally valid.
- Full-card text parsing is second fallback.
- Static field bbox OCR is last fallback.
- Existing fake backends without layout support must keep working.

## Review Focus

- Label and value on same line.
- Label OCR with minor punctuation/spelling noise.
- Header province/kabupaten/kota lines without colon.
- Values containing spaces and punctuation.
- Missing label must fall back safely without filling random garbage.

### Task 1 — Word layout OCR
- Add `OcrWord` dataclass and `TesseractBackend.read_layout(image)`.

### Task 2 — Label-anchor candidate extraction
- Add `autodocscanner/ktp/anchors.py`.
- Group OCR words by line and match KTP label aliases.
- Return raw candidate strings per field.

### Task 3 — Hybrid priority update
- Tracking selection priority: anchored label candidate -> full-card candidate -> bbox candidate.
- Reject symbol-heavy garbage.

### Task 4 — Regression
- Run `python -m unittest discover -s tests -q`.
- Smoke test several different KTP images from Tracking page.
