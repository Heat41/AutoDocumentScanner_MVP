# PR6 Visual Candidate Debugging — experimental, read-only

## Purpose

Expose the *existing PR6 diagnostic evidence* on the original KTP photo rather
than showing only `manual_review` or producing additional JSON reports.
This is a **visual explanation of candidate proposals**, not an automatic
crop engine or evidence that any colored line is the true card boundary.

This feature is opt-in on branch
`experiment/ktp-dynamic-perspective-v3` and must not affect `main`.

## Run on Windows

```powershell
cd D:\AutoDocumentScanner
git switch experiment/ktp-dynamic-perspective-v3
git pull --ff-only origin experiment/ktp-dynamic-perspective-v3

$env:AUTODOCSCANNER_PR6_RUNTIME_REVIEW = "1"
python app.py --ui
```

1. Select one or more local photos.
2. Click **Review PR6** (in the sidebar).
3. After review, a read-only **Visual Candidate Debugging PR6** window
   opens for the currently selected photo when a report is available.
4. Choose **Stable**, **Adaptive**, **Kontur V2**, or **Four-Line** in the
   candidate selector to compare different original-image four-corner
   hypotheses, or **Outer Region** for region-first recovered outer
   corners. Individual candidate availability depends on the input.
5. Read per-side external support, edge localization, inside/outside
   contrast, candidate warnings, and **reason codes** that blocked preview.
6. Close the debug window. Use **Koreksi Manual** with magnifier if needed.
   No corners from the debug window are applied automatically.
7. To reopen it for the active reviewed photo, click **Lihat Kandidat**.

A photo can have many candidate quadrilaterals but no trusted candidate.
The overlay does not override PR6's `manual_review` decision.

## Line colors

- **Green:** this candidate side met *the existing combined-evidence display
  bands* for external gradient support, edge alignment and contrast. It is
  **not** proof that it is the outer physical KTP boundary.
- **Orange:** measurements exist but are weak or poorly aligned.
- **Gray:** evidence on that precise candidate side is missing or incomplete.

The **Outer Region** family additionally shows numerical perspective
feasibility, chromatic inside/outside side changes, and a possible-internal-
rectangle warning if the candidate is nested in a larger proposed shape.
These are review-only diagnostics, **not** a verified card identity or
permission to crop.

The metrics for a selection-group representative **are not silently copied**
to a different source geometry in the same group. Missing evidence remains
unknown, not zero.

## What remains unchanged

- The **Proses Otomatis** button remains the previous production-like
  OpenCV pipeline; Visual Debugging does **not** change its detector choice.
- Review PR6 still has `auto_select = null` and
  `auto_crop_allowed = false`. It never saves or warps an image.
- The dialog provides no apply, save, export or networking actions.
- The photo is read locally in memory and EXIF-oriented to match the
  scanner's original-image coordinates. If dimensions mismatch, the viewer
  refuses to overlay potentially misleading geometry.
- Four-point manual editor, magnifier, SnipSheet, JPG/PDF export, source files,
  and previous outputs are not modified.

## Testing

```powershell
python -m pytest tests/ktp/test_pr6_visual_diagnostics.py -v
python -m pytest tests/ui/test_pr6_candidate_debug.py -v
python -m pytest tests/ui/test_pr6_runtime_review_flow.py -v
python -m pytest -q
```

The branch CI also runs real Tkinter dialog tests under Xvfb, including
candidate switching, no-candidate behavior, original-image dimension
mismatch and no file writes.

## Next measurement-driven step

Collect fresh, unseen KTP photos with independent **original-image** human
annotations of TL/TR/BR/BL corners (stored locally only). Use Visual
Debugging to inspect failures, then compare actual normalized corner errors
and severe-failure rates for Stable/Adaptive/V2/Four-Line against the stable
baseline. Keep failed candidates in the dataset; do not hardcode thresholds
to the example shown in the app or silently enable auto-crop.
