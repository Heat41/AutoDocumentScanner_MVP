# PR6 — Four Independent-Line Candidate Experiment

## Why this exists

The current automatic KTP crop can contain background and retain a tilted
physical card even when the legacy image-quality check reports `pass`.
A high-quality warp is **not** evidence that the chosen quadrilateral follows
the actual physical KTP border.

The original experimental V2 proposals use closed contours. This additional
OpenCV-only generator proposes quads by intersecting **four separately visible
straight border segments**, including when the border is not one closed
contour. It is not tied to any document identity, sample background, fixed
position, or KTP-specific pixel colors.

**There is no guarantee that the four lines belong to the KTP**: fabric,
table edges, text, and internal card markings can also generate segments.
All outputs are only hypotheses and must pass the existing PR6 physical
evidence, spatial conflict, and review gates. No automatic crop/save was
enabled.

## Scope and default behavior

- `autodocscanner/core/line_quad_proposals_pr6.py` generates up to six
  candidate quads from Canny/HoughLinesP and four-side segment overlap.
- The new generator is **opt-in** in
  `compare_detectors(image, include_line_candidates=True)`.
- Existing `compare_detectors(image)` behavior remains contour-only.
- `Review PR6` requests these extra proposals in its runtime candidate
  comparison, still requiring user confirmation and never saving a crop.
- **Proses Otomatis** remains on the existing detector until measured accuracy
  and safe candidate selection justify a separate explicitly approved
  integration. This stage alone does **not** fix the tilted automatic output.

## Evaluate locally on different KTP photos

Keep all KTP photos, annotations and reports **outside version control**.

Use separate, new image folders for genuine *unseen holdout* evaluation.
Annotate the four ORIGINAL-image physical card corners in TL, TR, BR, BL order.
Do not annotate from a predicted crop.

```powershell
python -m autodocscanner.core.detector_comparison "D:\AutoDocumentScanner\test-data-holdout" --annotations "D:\AutoDocumentScanner\holdout_annotations.json" --output "D:\AutoDocumentScanner\ktp_baseline_eval.json"

python -m autodocscanner.core.detector_comparison "D:\AutoDocumentScanner\test-data-holdout" --annotations "D:\AutoDocumentScanner\holdout_annotations.json" --include-line-proposals --output "D:\AutoDocumentScanner\ktp_four_line_eval.json"
```

Both reports are read-only. They compare the original detector, adaptive
detector and additional candidate options; no crop or changes to the input
photos occur. Accuracy is measurable only when genuine human annotations
are supplied.

Compare per-image mean, median and p90 normalized corner errors; number
of severe errors, count of candidate-free photos, PR6 preview/manual rates
and any regressions relative to the existing stable detector. Use distinct
development and holdout roles.

Holdout should cover card rotation, projective distortion, close-up/distant
images, horizontal/vertical photo orientation, shadows, reflections, clutter,
textured surfaces, low resolution, torn/partially occluded borders, and
different capture devices.

## Known limitations / verification

- Synthetic rotated-card tests show successful geometric hypotheses across
  multiple backgrounds and angles, but OpenCV versions differ in Hough line
  results. One challenging synthetic fixture was not accurately recovered on
  the first CI run. That case remains in the recall evaluation.
- A wrong-looking candidate can be proposed; ranking score or long-edge
  overlap is **not calibrated accuracy or a verified physical boundary**.
- Missing/cut-off physical corners may be impossible to recover reliably.
  Keep manual confirmation and magnifier as the safe fallback.
- CI success means the code and synthetic tests passed; it does **not**
  imply reliable automatic crop for arbitrary real-world KTP photographs.
- Main/released v1.1.1, SnipSheet and PDF/JPG flows are not modified.

For opt-in preview of PR6 proposals (without auto-crop):

```powershell
$env:AUTODOCSCANNER_PR6_RUNTIME_REVIEW = "1"
python app.py --ui
```

Click **Review PR6** (not Proses Otomatis) and then **Koreksi Manual** to
inspect or adjust suggested corners before a manually confirmed crop.
