# AutoDocumentScanner V3 — Development Agent Rules

> Scope: all work in this repository, especially dynamic KTP perspective detection and PR6.
> Status: binding project engineering rules; revise only with explicit user approval.

## 1. Locked product objective (NON-NEGOTIABLE)

Build a **general-purpose KTP perspective detector** that works on **unseen, real-world KTP photographs**, not just the current sample photos. The handful of sample images are **diagnostic fixtures only**, not the definition of success.

The detector must handle variations in card placement and capture conditions: portrait/landscape photo orientation, rotated or tilted KTP, perspective distortion, different backgrounds, lighting and shadows, camera distance, image resolution, and imperfect boundaries. Do not promise perfect automatic detection when card corners are occluded or cut off. Provide safe human review/manual fallback instead.

## 2. Algorithm restrictions

- Use classical computer vision / OpenCV and explicit geometry. **No AI/ML models, YOLO, learned weights, network inference, or external services** unless the user explicitly changes this requirement.
- **No overfitting**: never use file names, individual KTP identities, hard-coded per-photo coordinates, fixed screenshot placements, specific sample background colors, or rules tuned to make a named example pass.
- Every change must be explainable as a broadly applicable, image-derived rule. Avoid brittle absolute pixel thresholds where resolution-scaled thresholds are appropriate.
- Handle arbitrary image rotations and perspective without distorting aspect ratios in annotation/preview UI; annotation coordinates must map back to the original image.

## 3. Production safety / interaction

- Keep the current production Auto Perspective detector working. PR6 algorithms remain **offline/shadow experiments** until validated and explicitly approved for integration.
- Preserve existing **manual four-corner perspective editing**, JPG/PDF export, SnipSheet, and other production features.
- No automatic crop/save when candidate reliability is uncertain. Review Gate should fail closed: preview proposed corners only when safe, otherwise open manual four-corner review; require explicit user confirmation.
- A high edge score, warp quality, or heuristic ranking is **not calibrated accuracy** and cannot alone justify silently selecting/cropping.
- Never overwrite the production detector or merge experimental work into `main` without explicit approval.

## 4. PR6 — Candidate Selection Improvement

- Primary focus: **select the correct physical KTP contour**, rather than only nudging an already incorrect quadrilateral.
- Compare stable, adaptive, and V2 proposals using general geometry, multi-side external-boundary evidence, card-shaped plausibility, frame-suspect detection, and safeguards against internal contours.
- Candidate refinement can be evaluated as an additional independent option but must be rejected if its evidence or image quality regresses.
- Keep human ground-truth coordinates strictly for **evaluation**, not runtime scoring, candidate ranking, or heuristics.
- Document ambiguous/no-candidate cases explicitly; never fabricate confidence or use the wrong bounding box just to produce an output.

## 5. Evaluation discipline / generalization

- Maintain separate data roles:
  - **Development examples**: used to debug and understand failure modes.
  - **Unseen holdout set**: new images not used to set thresholds or adapt logic; used for unbiased evaluation.
- Collect human-labeled original-image physical KTP corners; do not label from detector output. Keep original-photo names and image-coordinate mapping intact.
- Compare mean/median/p90 normalized corner error, per-image regressions, acceptance vs. manual-review rates, and severe-failure counts against the stable baseline.
- Do not declare success merely because the current five examples improve, unit tests pass, or edge/warp metrics rise. Test different lighting, orientations, backgrounds, scales, and real photo conditions; expand holdout before promoting to production.
- If a holdout reveals a failure and is then used to tune a rule, treat that image as development data and evaluate the revised algorithm on **fresh unseen holdout images**.
- Unit regression tests are required for new geometry, filtering, fallback, and rotation changes. Report actual measured results; never claim local tests passed without execution.

## 6. Privacy and repository hygiene

- KTP photos, real personal data, human corner annotations, and diagnostic JSON stay **local**; never commit or upload them to GitHub. Maintain `.gitignore` entries for test image/holdout directories and annotation/output files.
- Do not add real KTP data to tests or fixtures. Use synthetic images and anonymized geometry for automated tests.
- Work on `experiment/ktp-dynamic-perspective-v3`; commit small, reviewable changes with tests. Do not modify `main` unless specifically asked.

## 7. Required agent checklist before each PR6 implementation

1. State the general failure mode and why a proposed change applies to unseen KTP photos.
2. Confirm the change introduces **no sample-specific assumptions or use of ground truth at inference**.
3. Keep the experiment isolated from production and fail safely to manual review.
4. Add regression tests covering improvement intent and negative/failure cases.
5. Evaluate on development fixtures, then **new unseen** images; report per-image and aggregate errors relative to Stable.
6. Clearly distinguish **implemented**, **tested**, and **validated for production**.

**Acceptance principle:** A change is not finished because the example photos look correct. It is finished only when it demonstrates safer, more accurate behavior on representative unseen KTP photos without breaking established production workflows.
