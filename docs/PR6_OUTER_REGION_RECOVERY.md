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


## Follow-up: Outer Region selects a separate background object

A field screenshot showed the first Outer Region overlay on an **object
outside the card**. The previous generator sorted most region hypotheses by
compactness and area. Those two measures are not KTP identification, and a
clean external rectangle can look better than the card.

The experimental follow-up adds:

1. `region_context_pr6.py` with **interior spatial edge coverage**,
   repeated straight interior strokes, intersections with candidate
   geometries from other methods, and nested smaller regions.
2. Optional **contextual ranking** of Outer Region proposals using those
   observations. This is used by explicit `Review PR6` only; the old
   automatic processing flow is not redirected. The original shape-based
   generator remains available when contextual ranking is disabled.
3. Visual warnings for `sparse_internal_structure` and
   `no_independent_geometric_corroboration`, with precise values for the
   inspected candidate. Other methods can share the same underlying image
   features, so their agreement is **not independent proof**.
4. **Balanced visual display slots** across Stable, Adaptive, Contour V2,
   Four-Line, Chroma and Outer Region. Many earlier contours can no longer
   occupy all 40 visible slots and hide an outer-region alternative.
5. A synthetic distractor case where a separate colored rectangle is
   present elsewhere in the frame. A *different* case where the distractor
   actually occludes the physical card border remains ambiguous and must
   not be interpreted as reliable automatic recovery.

**Important:** these additions help inspect and prioritize *candidate
hypotheses*, not definitively label KTP versus other objects. Backgrounds
with texture or documents with little internal print can confuse the cues.
A case where the actual card boundary is absent from the generated
candidates cannot be corrected by ranking. Do not declare real-photo
accuracy improved without original-image annotations and fresh unseen
holdout results.

## Windows debugging loop

```powershell
$env:AUTODOCSCANNER_PR6_RUNTIME_REVIEW = "1"
python app.py --ui
```

Choose a photo, click **Review PR6**, select **Outer Region** candidates
in **Lihat Kandidat**, and inspect region-context warnings in the
right-hand detail pane. Do not mistake the initially displayed candidate
for a selected or verified crop. None of these candidates are automatically
applied to the scanner or saved.

To make objective progress, compare normalized physical-card corner errors
and severe false-crop rates on a local set of independently annotated, new
photos. Keep real KTP images and JSON outside the repository.
