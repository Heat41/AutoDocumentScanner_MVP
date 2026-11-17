# PR6 — Outer Region Recovery and Perspective Plausibility (experimental)

## Objective and failure mode

PR6 Visual Candidate Debugging revealed two recurring possibilities:
- Stable/Adaptive can select a large rectangle including the background.
- V2 can lock onto a smaller, sharply bounded **internal rectangle**, such
  as a portrait window on a document, rather than the physical outer card.

Four clean sides and a well-conditioned homography **do not identify a KTP**.
If the correct outer-border candidate was never generated, a ranking or
Review Gate change alone cannot recover it.

This experimental stage adds three related **offline** capabilities without
discarding the previous PR6 analysis.

### 1. Card Region Identification

`region_outer_quad_candidates` builds candidate **regions**, not individual
edges, by inspecting locally smoothed Lab L/a/b and HSV saturation channels.
It uses image-derived Otsu/quantile thresholds, both polarities, simple
morphology and external connected contours.

It is not restricted to blue documents or a particular card placement.
Relative image-size thresholds allow distant and close-up candidates. It
is still a color/region hypothesis: fabric, advertisements and other
rectangles may be detected as well.

### 2. Outer Corner Recovery

The convex hull of each region is approximated as a quadrilateral; a
minimum-area rectangle is a fallback **only** for compact, approximately
rectangular regions. Candidate deduplication and size-band sampling retain
multiple plausible sizes so a small card is not always displaced by a large
background contour.

The result is original-image TL/TR/BR/BL coordinate hypotheses, **not** an
automatically selected or corrected card.

### 3. Perspective and Outer-Side Verification

`inspect_outer_perspective_hypothesis` checks:
- geometrical validity and numerical feasibility of homography,
- four-sided Lab color transitions just inside/outside the proposed card,
- nesting of a small candidate inside a larger candidate.

This is **plausibility only**, not physical card identification. A portrait
box can pass geometric and chromatic tests. A nested candidate is marked
with `possible_internal_rectangle`, not automatically rejected as an
impossible card, because close-up photos and unrelated overlapping shapes
can be ambiguous.

## Integration boundary

Only `Review PR6` enables this new candidate source by default.

- New source label in **Lihat Kandidat**: **Outer Region**.
- Existing source labels remain Stable, Adaptive, Kontur V2, Four-Line, Chroma.
- Output fields: `pr6_region_proposal_count` and, when enabled,
  `pr6_outer_region_recovery_shadow` with per-candidate checks.
- `auto_select = null`, `auto_crop_allowed = false`,
  `status = manual_review_only` for the plausibility inspector.
- PR6 selection and decision gates are **unchanged**. No region candidate
  bypasses them, and the legacy `Proses Otomatis` detector is unchanged.
- No machine learning, web API, private photos, annotations, OCR,
  or user-specific rules are used.

## Compare against the prior pipeline on your local holdout

```powershell
cd D:\AutoDocumentScanner
git switch experiment/ktp-dynamic-perspective-v3
git pull --ff-only origin experiment/ktp-dynamic-perspective-v3

python -m autodocscanner.core.detector_comparison "D:\AutoDocumentScanner\test-data-holdout" --annotations "D:\AutoDocumentScanner\holdout_annotations.json" --output "D:\AutoDocumentScanner\baseline_pr6.json"

python -m autodocscanner.core.detector_comparison "D:\AutoDocumentScanner\test-data-holdout" --annotations "D:\AutoDocumentScanner\holdout_annotations.json" --include-line-proposals --include-chroma-proposals --include-region-proposals --output "D:\AutoDocumentScanner\outer_region_pr6.json"
```

Human annotations must mark **original-image physical card corners**, and
the new holdout should not have been used to tune the algorithm. The JSON
files and personal photos stay outside GitHub.

Report candidate recall near those corners, normalized mean/median/p90
corner errors, severe failures, preview/manual rates, and regressions
relative to the stable pipeline. If a holdout image is used to fix a bug,
treat it as development material and measure generalization again on
fresh unseen holdout data.

## Current verified status

- Synthetic KTP-like region tests include different color families,
  rotation, perspective, distant cards, grayscale cards, textured
  backgrounds and invalid/no-document inputs.
- PR6 candidate-selection safeguards and manual four-corner fallback
  remain enabled.
- Linux CI includes synthetic logic tests and actual Tkinter GUI tests.
  **Passing synthetic tests does not establish real KTP crop accuracy.**
- Test Windows GUI, installed application, export JPG/PDF and SnipSheet
  before considering any production integration.

The next decision is based on real photo evaluation, not visual similarity
to an individual debugging screenshot.
