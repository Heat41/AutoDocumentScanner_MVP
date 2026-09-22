# Stage 2D Tracking UI Layout Refinement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Improve the Tracking Data KTP page so review work is clearer, less cramped, and responsive across common Windows desktop sizes.

**Architecture:** Keep all OCR/scanner/service behavior unchanged. Refactor only the Tkinter layout and presentation of `TrackingKtpPage`: compact controls on the left, visual preview in the center, and a wider grouped review panel on the right with clearer status indicators and responsive resizing.

**Tech Stack:** Python 3.12, Tkinter/ttk, Pillow, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Work directly on green `main`.
- Do not modify scanner, perspective, OCR, parsing, tracking-service, or storage behavior.
- Preserve existing Tracking KTP workflow and editable review fields.
- Rekam Data KTP remains disabled.
- Keep top Stage 2 navigation unchanged.
- UI must remain usable on smaller desktop sizes and expand sensibly on larger windows.

## Review Focus

- Narrow window: controls and review form remain usable without overlapping.
- Tall field list: vertical scrolling remains accessible.
- Long OCR values: entry widgets stretch with the review panel.
- Reprocessing: status indicators update cleanly without stale markers.
- Face preview: remains visible without consuming excessive vertical space.

---

### Task 1 — Responsive three-column layout
**Files:**
- Modify: `autodocscanner/ui/tracking_ktp.py`
- Modify: `tests/ui/test_tracking_ktp.py`

**Behavior:**
- Left control pane compact.
- Center preview pane medium width.
- Right review pane receives the largest expansion weight.
- Use grid geometry for primary page layout.

### Task 2 — Grouped review form
**Files:**
- Modify: `autodocscanner/ui/tracking_ktp.py`

**Groups:**
- Identitas Utama
- Alamat
- Data Lainnya

### Task 3 — Clearer field statuses
**Files:**
- Modify: `autodocscanner/ui/tracking_ktp.py`
- Modify: `tests/ui/test_tracking_ktp.py`

**Status text:**
- empty for normal fields,
- `Periksa` for low-confidence fields,
- `Kosong` for empty parsed values.

### Task 4 — Preview polish and final regression
- Improve corrected KTP and face preview sizing.
- Keep worker/process behavior unchanged.
- Full regression and UI smoke.
