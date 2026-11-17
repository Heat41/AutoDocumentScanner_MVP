# PR6 — Chroma Boundary Candidate Experiment (OpenCV / Lab)

## Observed failure mode

Visual Candidate Debugging showed Stable and Adaptive sometimes agreeing on a
large background quadrilateral, with the available contour V2 alternatives
covering unrelated objects. Their agreement is **not independent evidence**
that the visible physical KTP card has been found. The existing selector
correctly retains `manual_review` in uncertain cases, but conservatism alone
cannot recover a missing candidate.

The original contour generator works mostly from grayscale Canny edges. A
visible card border can be chromatically distinct yet weak in luminance,
especially under textured, low-contrast backgrounds. The long-line generator
also misses short/broken edges.

## Implemented experimental change

`autodocscanner/core/chroma_quad_proposals_pr6.py` proposes four-corner
geometries from the OpenCV Lab **a** and **b** chromaticity channels, using
Canny, two scale-relative morphological closing passes, convex geometry and
quadrilateral prefilters. No trained models, fixed colors, sample positions,
private data or photo-specific coordinates are used.

The feature is **not enabled** for `Proses Otomatis` or the production
scanner. It runs only when `Review PR6` is explicitly used (or the separate
`--include-chroma-proposals` offline comparison flag is provided).

The review pipeline now collects:
- existing Stable and Adaptive candidates
- grayscale contour V2 candidates
- independent Four-Line proposals (when present)
- new Chroma proposals (when present)

All sources are validated and deduplicated geometrically; no candidate
is authorized for automatic cropping. The four-side evidence, conflicts and
manual fallback gates remain in effect.

## Visual review and diagnostics

Under `Review PR6`, the **Lihat Kandidat** dialog identifies the proposal
origin as `Kontur V2`, `Four-Line` or `Chroma` and displays counts for
each family. A family count of zero means **no valid displayed proposal
from that family**, not that the physical card is absent.

Where the original image is available, edge localization and inside/outside
contrast are computed for **each displayed candidate's exact corners**,
instead of showing only the chosen selection-group representative's values.
These measurements are visual-only; ranking, gating, preview decisions and
crop permissions are unchanged.

## Proper evaluation

Use unseen local KTP photos with independent TL/TR/BR/BL labels on the
original images. Do **not** commit photos, labels or reports.

```powershell
python -m autodocscanner.core.detector_comparison "D:\AutoDocumentScanner\test-data-holdout" --annotations "D:\AutoDocumentScanner\holdout_annotations.json" --output "D:\AutoDocumentScanner\ktp_baseline_eval.json"

python -m autodocscanner.core.detector_comparison "D:\AutoDocumentScanner\test-data-holdout" --annotations "D:\AutoDocumentScanner\holdout_annotations.json" --include-line-proposals --include-chroma-proposals --output "D:\AutoDocumentScanner\ktp_extended_eval.json"
```

Compare normalized corner-error median and p90, severe errors, geometrically
distinct candidate recall, and review/manual rates. Synthetic success
demonstrates only a hypothesis, never measured KTP recognition accuracy.

## Safety and remaining limitations

Colored backgrounds may generate spurious Chroma rectangles; this is expected
and must remain observable in Visual Debugging. The generator does not
identify what an object is, distinguish text cards from other colored
rectangles, or guarantee that a contour is the physical card. It is not
permission to bypass PR6 manual review.

No changes to released `main`, automated scan selection, JPG/PDF export,
SnipSheet, or manual four-corner editor/magnifier.

Review Windows GUI behavior separately from Linux CI before any production
decision.
