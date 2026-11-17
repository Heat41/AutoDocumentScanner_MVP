# PR6 — Experimental Runtime Review Integration (Auto KTP)

## Scope

This stage connects existing classical CV PR6 diagnostics to the actual
`Proses Otomatis` UI action **on the experimental branch only**. It is a
deliberately conservative integration of already-implemented CV evidence:
stable vs adaptive vs V2 contour proposals, physical-side localization,
geometric nesting conflicts, candidate consensus, review gates, and
refinement context.

**This is not an automatic crop/acceptance feature.** Current PR6 shadow
results do not establish calibrated confidence or physical-boundary
correctness. The stage replaces an unsafe automatic write with a **review
decision and a proposed corner geometry** when safe enough for user review.

No custom thresholds, filenames, private KTP images, annotations or
ground-truth evaluation are used to decide a runtime route.

## Enable in PowerShell (test only)

```powershell
cd D:\AutoDocumentScanner
git switch experiment/ktp-dynamic-perspective-v3
git pull --ff-only origin experiment/ktp-dynamic-perspective-v3

$env:AUTODOCSCANNER_PR6_RUNTIME_REVIEW = "1"
python app.py --ui
```

This runtime flag also enables the manual four-corner editor and its 4×
magnifier; the separate `AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE` flag is no
longer necessary when runtime PR6 review is enabled.

Disable and return to the original production-like processing behavior:

```powershell
Remove-Item Env:AUTODOCSCANNER_PR6_RUNTIME_REVIEW -ErrorAction SilentlyContinue
Remove-Item Env:AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE -ErrorAction SilentlyContinue
python app.py --ui
```

Do not enable this flag for production or a packaged release without
approval and successful Windows regression tests.

## What happens when Proses Otomatis is pressed

1. Open each ORIGINAL source photo with OpenCV locally.
2. Call `compare_detectors(image, expected=None)`: do not supply or read
   ground truth or personal annotations.
3. Inspect `pr6_review_consolidation_shadow` and
   `pr6_manual_fallback_shadow` and independently verify a proposed
   four-corner geometry against the same original candidate.
4. If PR6 suggests a sufficiently supported candidate for preview, expose
   it only as an editable *suggestion* in **Koreksi Manual**.
5. If PR6 flags conflicts or incomplete physical-boundary evidence,
   reject any stale cached detector quad and open manual entry with a
   neutral four-corner starting frame.
6. Do not call `process_ktp_output`, `scanner.scan` or auto-save while in
   PR6 runtime review mode. `auto_select = null`,
   `auto_crop_allowed = false`, and explicit confirmation remains mandatory.
7. **Batal** never saves anything. **Terapkan Koreksi** in the existing UI
   validates the four corners, then uses the existing
   `scanner.scan_with_corners` (normal quality, JPG/PDF and KTP-processing
   paths). Old output mappings are not reused as fresh outputs.
8. If analyzing one photo fails, fail closed and keep the manual editor
   available. Do not fall back to an unreviewed legacy auto-crop.

The UI distinguishes **Menunggu konfirmasi** from **Ditolak** and keeps
previous successful output files unmodified until the user explicitly
confirms a replacement. It does not upload or persist PR6 diagnostic JSON.

## CLI behavior

With `AUTODOCSCANNER_PR6_RUNTIME_REVIEW=1`, plain `python app.py`
prints PR6 review results for local input images and performs **no crop or
file writes**. To use the editor and confirm a crop, launch
`python app.py --ui`.

Without the flag, the existing CLI and UI auto-processing paths are
unchanged.

## Tests

```powershell
python -m pytest tests/ktp/test_pr6_runtime_review.py -v
python -m pytest tests/ui/test_pr6_runtime_review_flow.py -v
python -m pytest tests/ui/test_manual_fallback_bridge_pr6.py -v
python -m pytest -q
```

Windows QA must cover multiple unseen photo orientations/backgrounds,
corner drag/magnifier, four-corner cross/invalid input, cancellation,
failed decoder/image error, detecting no KTP, switching selected source,
re-running auto processing, stale cached output, original-image EXIF
orientation, JPG/PDF export, SnipSheet, manual document, and packaged app.

## Important limitations

- **No proven accuracy increase yet.** The existing five development
  photos repeatedly required manual review in previous diagnostic
  checkpoints. They are not an unseen holdout and must not drive new
  hardcoded rules.
- Some apparently good detections are conservatively rejected because
  boundary evidence is incomplete. Future changes should improve
  measurable candidate selection, not relax safety only to produce crops.
- This is **review integration** rather than automatic crop deployment.
  Automated crop may only be enabled after appropriate fresh holdout
  comparison, failure-rate measurement, regression testing and explicit
  approval.
- No commits to `main`, no changes to production detector selection,
  no network/OpenCV.js dependency. This uses Python/OpenCV locally.

Keep real KTP photos, private filenames, annotations and diagnostic JSON
outside version control.
