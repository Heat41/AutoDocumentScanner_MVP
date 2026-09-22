# Stage 2D KTP OCR and Parsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Convert Stage 2D KTP field crops into reviewable structured text using CPU-only Tesseract OCR with field-specific parsing and confidence/fallback handling.

**Architecture:** Keep OCR outside the locked scanner pipeline. `autodocscanner.ktp.ocr` owns preprocessing and a lazy Tesseract backend, `autodocscanner.ktp.parsing` normalizes field text, and `autodocscanner.ktp.tracking` coordinates extraction + OCR + parsing into a review contract. OCR backend injection keeps unit tests independent from the local Tesseract installation.

**Tech Stack:** Python 3.12, OpenCV, NumPy, pytesseract, Tesseract OCR, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Existing scanner and perspective engine remain unchanged.
- CPU-only operation.
- Tesseract executable is an external runtime dependency; Python imports must be lazy and errors must be explicit.
- Unit tests must not require an installed Tesseract binary.
- OCR must operate on Stage 2D field crops, not the original uncorrected photo.
- `foto` is never sent to OCR.
- Low-confidence/empty primary OCR receives one fallback preprocessing pass.
- OCR output is review data, never silently trusted as final recorded identity.
- Stage 2C storage is not written automatically in this slice.

---

### Task 1 — Field parsers
**Files:**
- Create: `autodocscanner/ktp/parsing.py`
- Create: `tests/ktp/test_parsing.py`

**Interfaces:**
- `clean_text(value) -> str`
- `parse_nik(value) -> str`
- `parse_ttl(value) -> dict`
- `parse_rt_rw(value) -> dict`
- `parse_gender(value) -> str`
- `parse_field(field_name, value) -> object`

### Task 2 — OCR preprocessing and backend
**Files:**
- Create: `autodocscanner/ktp/ocr.py`
- Create: `tests/ktp/test_ocr.py`
- Modify: `requirements.txt`

**Interfaces:**
- `OcrReadResult`
- `preprocess_field(image, field_name, fallback=False) -> np.ndarray`
- `TesseractBackend.is_available() -> bool`
- `TesseractBackend.read(image, field_name) -> tuple[str, float]`
- `read_field_ocr(image, field_name, backend, confidence_threshold=55.0) -> OcrReadResult`

### Task 3 — Tracking extraction service
**Files:**
- Create: `autodocscanner/ktp/tracking.py`
- Create: `tests/ktp/test_tracking.py`

**Interfaces:**
- `TrackedField`
- `KtpTrackingResult`
- `extract_tracking_data(corrected_image, backend=None) -> KtpTrackingResult`

**Behavior:**
- extract all normalized regions;
- skip OCR for `foto`;
- OCR + parse all textual fields;
- split TTL into `tempat_lahir` and `tanggal_lahir`;
- split RT/RW into `rt` and `rw`;
- expose corrected KTP and face crop;
- mark low-confidence fields for mandatory review.

### Task 4 — Public package exports and regression
**Files:**
- Modify: `autodocscanner/ktp/__init__.py`
- Full repository regression.
