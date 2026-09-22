# Stage 2D Strict Tracking Pipeline Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Enforce the Tracking KTP sequence so OCR can only receive the corrected KTP image produced by Auto Perspective.

**Architecture:** Split the current tracking service into two explicit stages: perspective correction and OCR tracking. The UI worker calls them sequentially and updates status between stages. Raw selected photos are never passed into OCR.

**Tech Stack:** Python 3.12, Tkinter, NumPy, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Raw KTP photo may only enter the perspective-correction stage.
- OCR/tracking accepts only the corrected image object.
- Existing scanner algorithm remains unchanged.
- Existing tracking result and PDF report behavior remain unchanged.
- UI must clearly show which stage is running.
- Before processing, preview area must not imply that the raw image is the corrected result.

### Task 1 — Split tracking service
- Add `correct_tracking_input(scanner, input_path)`.
- Add `track_corrected_ktp(corrected_image, backend=None)`.
- Keep `process_tracking_input` as an ordered wrapper.

### Task 2 — UI stage sequencing
- UI worker runs correction first, then OCR.
- Status changes between Auto Perspective and Tracking/OCR.
- Corrected preview appears after correction succeeds.
- Raw selection is shown only as selected filename, not as corrected preview.

### Task 3 — Regression
- Verify OCR service receives scanner output object.
- Run `python -m unittest discover -s tests -q`.
