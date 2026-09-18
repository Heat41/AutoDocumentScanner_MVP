# Stage 2D Tracking KTP Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Add a third top-navigation page named Tracking Data KTP, ready to host the Stage 2D extraction/review workflow.

**Architecture:** Keep the existing Auto KTP and Manual Document pages untouched. Add a separate `TrackingKtpPage` and extend `Stage2ScannerUI` navigation routing so all three pages share the same top navigation bar.

**Tech Stack:** Python 3, Tkinter/ttk, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Do not modify legacy scanner/perspective/UI classes.
- Tracking Data KTP is a separate page.
- OCR/extraction logic is not implemented in this shell task.
- The page must be ready for later input, preview, review, and Rekam Data KTP controls.
- Existing Stage 2A/2B behavior remains unchanged.

### Task 1 — Tracking page shell
- Create `ui_tracking_ktp.py`.
- Create `tests/test_ui_tracking_ktp.py`.
- Expose `TrackingKtpPage(ttk.Frame)`.

### Task 2 — Top-navigation integration
- Modify `ui_stage2.py`.
- Modify `tests/test_ui_stage2.py`.
- Add `TRACKING_KTP_PAGE = "tracking_ktp"`.
- Append `("Tracking Data KTP", TRACKING_KTP_PAGE)` to `NAVIGATION_ITEMS`.
- Add `show_tracking_ktp()`.
- Ensure only one page is visible at a time.

### Task 3 — Regression
- Run full Windows regression in IDE.
- Smoke test all three top-navigation buttons.
