# PR6 Stage 3 — Experimental Auto KTP Manual UI Bridge

## Intent / scope

The experimental branch `experiment/ktp-dynamic-perspective-v3` can expose
the existing Auto KTP four-corner dialog even when the automatic detector
returned **no corners**. This is strictly opt-in, does **not** connect
`pr6_review_consolidation_shadow` to live scanner decisions, and does not
change the production `main` branch.

There is no AI/ML, no new image-dependent threshold tuned to a sample, no
network access, and no uploaded real KTP data.

## Enable for a development session (PowerShell)

```powershell
cd D:\AutoDocumentScanner
git switch experiment/ktp-dynamic-perspective-v3
git pull --ff-only origin experiment/ktp-dynamic-perspective-v3

$env:AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE = "1"
python app.py
```

To disable in the current PowerShell session:

```powershell
Remove-Item Env:AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE -ErrorAction SilentlyContinue
```

The flag must equal **exactly** `1`; all other values and missing values
preserve the legacy UI (manual button remains disabled without detector corners).

## Flow

1. Select a KTP photo. When enabled, the **Koreksi Manual** button is available
   even if Auto KTP detected no corners or was never run.
2. The editor opens the **original source image**. In experiment mode the
   preview is EXIF-transposed so coordinates match OpenCV's `imread`.
3. With no detector geometry, the four editable points start inset from the
   image edges using the general manual-document corner seed. They **are not
   evidence of detected KTP boundaries**; the user must position them.
4. **Batal** only closes the dialog; it never invokes the scanner, mutates
   existing results or writes a file.
5. **Terapkan Koreksi** checks that processing is idle, decoder dimensions
   still match the source preview, and all four user coordinates are finite,
   ordered TL/TR/BR/BL, non-crossing and inside the image.
6. After the explicit click and checks, the existing
   `AutoDocumentScanner.scan_with_corners(..., mode="ktp")` pathway performs
   the perspective correction and final validation/output. Errors leave
   the dialog open and do not update the success mappings.
7. A successful call clears any earlier failure state for that source,
   records the manually confirmed corners and output, and refreshes the UI.

The manual dialog still has no dependency on ground-truth annotations,
diagnostic corner error or personal names.

## What this does NOT claim

- No automatic acceptance or crop from PR6 shadow reports.
- No assertion that an inset seed is a correct physical KTP contour.
- No production integration or change to `main`.
- No claim of passing full Windows desktop, packaged/installer, SnipSheet, or
  JPG/PDF end-to-end testing until those tests are actually run.
- No standalone UI run has been performed by adding these tests alone.

## Test plan

```powershell
python -m pytest tests/ui/test_manual_fallback_bridge_pr6.py -v
python -m pytest tests/ktp/test_manual_fallback_validation_pr6.py -v
python -m pytest -q
```

Additional Windows manual QA: use a blank/non-detectable synthetic photo,
a portrait photo, a landscape photo, and an EXIF-rotated synthetic photo.
Check button state with flag off/on, no-candidate editor entry, movable
corners, incorrect/crossed coordinates, Reset Titik, Batal, explicit Apply,
failed scan recovery, switching files, light/dark themes, and installed-app
behavior. Check existing manual document page and SnipSheet remain unchanged.
Keep real KTP photos and diagnostics local.

## Subsequent decision

After Windows UI + regression tests and fresh unseen image testing, review
whether to integrate a PR6 review decision into a user-visible app. The
runtime detector and `main` must remain untouched until explicit approval.
