# PR6 — Experimental Runtime Review Integration (Auto KTP)

## Scope

This stage exposes existing classical CV PR6 diagnostics as a **separate
`Review PR6` UI action on the experimental branch only**. The ordinary
`Proses Otomatis` button retains the legacy scanner path and is never
silently redirected to no-crop review. It is a deliberately conservative
integration of already-implemented CV evidence:
stable vs adaptive vs V2 contour proposals, physical-side localization,
geometric nesting conflicts, candidate consensus, review gates, and
refinement context.

**This is not an automatic crop/acceptance feature.** Current PR6 shadow
results do not establish calibrated confidence or physical-boundary
correctness. The `Review PR6` action supplies a **review decision and a proposed
corner geometry** when safe enough for user review. This is a distinct
experimental workflow; it does not overwrite the legacy automatic button.

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

This runtime flag exposes a **Review PR6** button next to the file
controls (**Kosongkan**) without crowding the responsive bottom toolbar,
and enables the manual four-corner editor with its 4× magnifier. The `AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE` flag
is not needed when runtime PR6 review is enabled.

**Use `Proses Otomatis` to run the existing automatic scanner. Use
`Review PR6` to analyze the physical KTP boundary without automatically
writing a crop.** Existing automatic results are not evidence that PR6
approved the crop; the legacy detector can still be inaccurate.

Disable and return to the original production-like processing behavior:

```powershell
Remove-Item Env:AUTODOCSCANNER_PR6_RUNTIME_REVIEW -ErrorAction SilentlyContinue
Remove-Item Env:AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE -ErrorAction SilentlyContinue
python app.py --ui
```

Do not enable this flag for production or a packaged release without
approval and successful Windows regression tests.

## What happens when Review PR6 is pressed

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

Plain `python app.py` always retains the existing batch automatic
processing workflow, whether the experiment flag is set or not.

For a deliberate read-only PR6 diagnostic in the CLI, enable the flag and
run `python app.py --pr6-review`; this prints decisions but performs
**no crop or output writes**. To use the review editor, launch
`python app.py --ui` and click **Review PR6**, followed by **Koreksi Manual**
for confirmation or adjustment.

With the runtime flag absent, the extra **Review PR6** button is not shown.
The separate manual magnifier can still be enabled by
`AUTODOCSCANNER_PR6_MANUAL_UI_BRIDGE=1`.

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
- The PR6 runtime experiment does not yet replace the old automatic
  detector: it provides separate diagnostics and user review while preserving
  `Proses Otomatis` for comparison.
- This is **review integration** rather than automatic crop deployment.
  Automated crop may only be enabled after appropriate fresh holdout
  comparison, failure-rate measurement, regression testing and explicit
  approval.
- No commits to `main`, no changes to production detector selection,
  no network/OpenCV.js dependency. This uses Python/OpenCV locally.

Keep real KTP photos, private filenames, annotations and diagnostic JSON
outside version control.
