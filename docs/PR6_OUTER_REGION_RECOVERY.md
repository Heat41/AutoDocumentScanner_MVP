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


## Experimental Side Recovery — correcting a misaligned single side

A visual-review example showed Outer Region hypotheses whose **bottom, left
and right edges were near a physical card, but the top edge remained on the
background**. No global contour-ranking score could move that missing side.

`autodocscanner/core/side_boundary_recovery_pr6.py` provides an additional
**review-only, one-side-at-a-time hypothesis**:

1. Start with a valid Outer Region quadrilateral in original-image pixels.
2. For each side, scan parallel offsets over a fraction of the quadrilateral
   width/height. Measure robust, directionally coherent transitions across
   that side in **Lab color space**; do not search for any fixed card color.
3. If a displaced line has much stronger continuous evidence than the
   existing line, intersect it with the two adjacent original side lines
   to produce two new corners. The opposite corners remain unchanged.
4. Require valid convex geometry and bounded area change. Never assume the
   biggest gradient is necessarily a KTP boundary.

The generator adds a separate **Side Recovery** family to the Visual Candidate
Debugging list, identifying its `v2_...` parent, the side changed and the
before/after edge-evidence values. These evidence values are **not confidence
percentages**, accuracy measurements or a verified document boundary.

Crucially, the hypotheses are appended **after all existing PR6 decisions
and manual-fallback calculations have completed**. They are not candidates
for PR6 auto-select, do not modify its Review Gate, and are never used for
automatic perspective correction or image export.

### Limitations and evaluation

This can help when an initial quadrilateral is already near the card on
most sides, but cannot reliably solve entirely wrong backgrounds,
self-occluded edges, complex partial borders, or an image without a
discernible physical document. A wrongly aligned internal photo box can
also have strong edge transitions. If no better line is found, **Side
Recovery may legitimately show zero proposals**.

Synthetic tests cover multiple rotations, color/gray cards, background
texture, correctly aligned candidates (which must not move), uniform
images, invalid geometry, original-image immutability and unchanged
PR6 manual-review routing. These tests are not a substitute for a fresh
unseen, manually annotated set of real KTP photos.

To inspect locally, enable `AUTODOCSCANNER_PR6_RUNTIME_REVIEW=1`, click
**Review PR6**, then select **Side Recovery** in Visual Candidate Debugging.
Compare it to its originating **Outer Region** geometry; do not save the
recovered quadrilateral automatically.
