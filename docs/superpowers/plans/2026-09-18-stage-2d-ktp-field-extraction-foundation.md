# Stage 2D KTP Field Extraction Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Build the normalized KTP field-box and crop foundation that Stage 2D OCR and review UI will consume.

**Architecture:** Represent every KTP field area as normalized coordinates in the canonical corrected KTP image space. A layout module converts normalized boxes to pixel boxes, crops individual fields and the portrait area, and returns a deterministic extraction-result contract. No OCR engine is introduced in this first Stage 2D slice.

**Tech Stack:** Python 3, dataclasses, NumPy, OpenCV, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Work from the current green `main` baseline.
- Do not modify perspective detection or scanner geometry logic.
- Corrected KTP image is assumed landscape and normalized by the existing scanner.
- Bounding boxes are stored as normalized `(x1, y1, x2, y2)` values in range 0..1.
- Crop functions must clamp safely to image boundaries.
- Portrait crop is a first-class field named `foto`.
- OCR, master-wilayah matching, and final Review/Rekam actions are not implemented in this slice.
- Existing Stage 2A/2B/2C behavior must remain unchanged.

### Task 1 — Normalized KTP layout
**Files:**
- Create: `autodocscanner/ktp/layout.py`
- Create: `tests/ktp/test_layout.py`

**Interfaces:**
- `NormalizedBox`
- `KTP_FIELD_BOXES`
- `normalized_to_pixel_box(box, image_width, image_height)`
- `crop_normalized(image, box)`

**Field keys:**
`provinsi`, `kabupaten_kota`, `nik`, `nama`, `ttl`, `jenis_kelamin`, `golongan_darah`, `alamat`, `rt_rw`, `kelurahan_desa`, `kecamatan`, `agama`, `status_perkawinan`, `pekerjaan`, `kewarganegaraan`, `berlaku_hingga`, `foto`.

### Task 2 — Crop/extraction contract
**Files:**
- Create: `autodocscanner/ktp/extraction.py`
- Create: `tests/ktp/test_extraction.py`

**Interfaces:**
- `KtpFieldCrop`
- `KtpExtractionResult`
- `extract_ktp_regions(image) -> KtpExtractionResult`

**Behavior:**
- Validate non-empty uint8 input.
- Produce one crop for every field key.
- Return independent image copies.
- Expose `face_image` as the `foto` crop.
- Preserve the corrected KTP image copy as `source_image`.

### Task 3 — Package exports and regression guard
**Files:**
- Modify: `autodocscanner/ktp/__init__.py`
- Add/modify relevant tests.

**Behavior:**
- Export layout/extraction public interfaces.
- Full repository regression must remain green.


## Implementation checkpoint — 2026-09-18

Completed on `main`:
- Normalized 17-field KTP layout.
- Safe normalized-to-pixel conversion.
- Independent field crops.
- Portrait crop exposed as `face_image`.
- Stable extraction result contract.
- Public KTP package exports.

Audit:
- No scanner, perspective-engine, storage-schema, or UI files changed in this slice.
- Full Windows regression is pending.
