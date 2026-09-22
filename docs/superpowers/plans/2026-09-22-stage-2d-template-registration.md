# Stage 2D KTP Template Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Stabilize Tracking OCR by aligning each corrected KTP to a canonical template grid before coordinate ROI OCR.

**Architecture:** Use OCR-detected KTP labels as registration anchors. Estimate a small horizontal offset plus vertical scale/offset from detected label centers to expected canonical label positions, warp only the Tracking working image, then run the existing ROI OCR on the registered image. If insufficient anchors are found, keep the normalized image unchanged.

**Tech Stack:** Python 3.12, OpenCV, NumPy, pytesseract, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints
- Tracking page only; do not alter Auto Koreksi KTP or manual-document workflows.
- Scanner/perspective engine remains unchanged.
- Registration operates after Auto Perspective and canonical resize.
- Use detected labels as anchors; no external template image file required.
- Require multiple anchors before applying registration.
- Clamp transform so OCR mistakes cannot heavily distort the KTP.
- Photo crop and report output continue to work.

## Tasks
1. Rescale OCR layout word coordinates back to source-image coordinates.
2. Expose detected label anchor centers.
3. Add registration module with constrained transform.
4. Run Tracking ROI OCR on registered image.
5. Add regression tests and full quiet regression.
