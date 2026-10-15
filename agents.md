# AutoDocumentScanner — System-Wide Agent Rules

> Scope: **ALL AutoDocumentScanner work**, across every feature, module, UI, workflow, build, test, integration, and future release. PR6 is only one workstream.
> Status: binding project engineering rules; revise only with explicit user approval.

## 1. Locked system-wide product objective (NON-NEGOTIABLE)

Build and maintain a **general-purpose, reliable, offline-first document scanning application** that works on diverse real-world user inputs, **not only the supplied examples**. Images and documents provided for testing are diagnostic examples, never hardcoded targets. This principle applies equally to automatic/manual document perspective correction, KTP scanning, SnipSheet, image processing, exports, future tracking/OCR and UI workflows.

Preserve original input fidelity, predictable user control, privacy, stability, and usability. No new feature may silently break an established flow. Features under development remain isolated until properly validated and approved for production.

### KTP-specific objective

Build a **general-purpose KTP perspective detector** that works on **unseen, real-world KTP photographs**, not just the current sample photos. The handful of sample images are **diagnostic fixtures only**, not the definition of success.

The detector must handle variations in card placement and capture conditions: portrait/landscape photo orientation, rotated or tilted KTP, perspective distortion, different backgrounds, lighting and shadows, camera distance, image resolution, and imperfect boundaries. Do not promise perfect automatic detection when card corners are occluded or cut off. Provide safe human review/manual fallback instead.

## 2. Cross-system engineering rules

- **Generalize by design** across input formats, document sizes, orientations, lighting, resolutions, capture devices, Windows installations, and normal error conditions; do not engineer solely for demonstration files.
- **No sample-specific patches** in any module: no filenames, user identities, coordinates, backgrounds, or exact sample outputs as runtime shortcuts.
- Keep processing **local/offline by default**, including images and personal information, unless integration is explicitly approved.
- **Preserve existing interfaces and workflows**; avoid unnecessary rewrites. Prefer small reversible changes with clear ownership, contracts, and meaningful error handling.
- Keep dependencies purposeful and lightweight; follow the existing Windows/Python/PySide6/OpenCV architecture where applicable, and avoid breaking packaging or installer behavior.
- Keep input images unmodified unless the user explicitly confirms a transformation; maintain correct coordinate mapping between previews, zoom, rotation, detection, and exported output.
- Make failure states visible and recoverable; do not silently save wrong crops, reorder pages, stretch cards, or drop data.

## 3. Algorithm restrictions — KTP perspective

- Use classical computer vision / OpenCV and explicit geometry. **No AI/ML models, YOLO, learned weights, network inference, or external services** unless the user explicitly changes this requirement.
- **No overfitting**: never use file names, individual KTP identities, hard-coded per-photo coordinates, fixed screenshot placements, specific sample background colors, or rules tuned to make a named example pass.
- Every change must be explainable as a broadly applicable, image-derived rule. Avoid brittle absolute pixel thresholds where resolution-scaled thresholds are appropriate.
- Handle arbitrary image rotations and perspective without distorting aspect ratios in annotation/preview UI; annotation coordinates must map back to the original image.

## 4. Production safety / interaction

- Keep the current production Auto Perspective detector working. PR6 algorithms remain **offline/shadow experiments** until validated and explicitly approved for integration.
- Preserve existing **manual four-corner perspective editing**, JPG/PDF export, SnipSheet, and other production features.
- No automatic crop/save when candidate reliability is uncertain. Review Gate should fail closed: preview proposed corners only when safe, otherwise open manual four-corner review; require explicit user confirmation.
- A high edge score, warp quality, or heuristic ranking is **not calibrated accuracy** and cannot alone justify silently selecting/cropping.
- Never overwrite the production detector or merge experimental work into `main` without explicit approval.

## 5. Document processing, UI, and exports

- Preserve and regression-test **Auto KTP, Manual Document/four-corner perspective, SnipSheet, preview/navigation, and JPG/PDF saving** whenever related code changes.
- Manual perspective must allow user-controlled corrections without degrading the source image; maintain accurate rotation and aspect ratio in portrait and landscape photos.
- Generated PDFs must preserve intended paper/card dimensions, page sequence, margins, and multi-page behavior; no clipped or stretched KTP cards.
- UI additions must respect existing light/dark themes, responsive panel/sidebar behavior, meaningful loading/error feedback, and maintainable PySide6 signal/state flow.
- Maintain clear boundaries between **publicly available features**, **Coming Soon features**, and **Supervisor-only/future-integrated Tracking KTP**. Do not unlock restricted features or invent authentication/session flows without explicit authorization.
- Treat future OCR/Tracking KTP as advisory extraction requiring verification of critical fields (especially NIK and name); never silently persist uncertain identity data.
- Keep local build, packaging, Inno Setup installer, and installed-app resource paths functioning; test both development and packaged contexts when changing file/resource handling.

## 6. PR6 — Candidate Selection Improvement (workstream-specific)

- Primary focus: **select the correct physical KTP contour**, rather than only nudging an already incorrect quadrilateral.
- Compare stable, adaptive, and V2 proposals using general geometry, multi-side external-boundary evidence, card-shaped plausibility, frame-suspect detection, and safeguards against internal contours.
- Candidate refinement can be evaluated as an additional independent option but must be rejected if its evidence or image quality regresses.
- Keep human ground-truth coordinates strictly for **evaluation**, not runtime scoring, candidate ranking, or heuristics.
- Document ambiguous/no-candidate cases explicitly; never fabricate confidence or use the wrong bounding box just to produce an output.

## 7. Evaluation discipline / generalization — ALL modules

- Maintain separate data roles:
  - **Development examples**: used to debug and understand failure modes.
  - **Unseen holdout set**: new images not used to set thresholds or adapt logic; used for unbiased evaluation.
- Collect human-labeled original-image physical KTP corners; do not label from detector output. Keep original-photo names and image-coordinate mapping intact.
- Compare mean/median/p90 normalized corner error, per-image regressions, acceptance vs. manual-review rates, and severe-failure counts against the stable baseline.
- Do not declare success merely because the current five examples improve, unit tests pass, or edge/warp metrics rise. Test different lighting, orientations, backgrounds, scales, and real photo conditions; expand holdout before promoting to production.
- If a holdout reveals a failure and is then used to tune a rule, treat that image as development data and evaluate the revised algorithm on **fresh unseen holdout images**.
- Unit regression tests are required for new geometry, filtering, fallback, and rotation changes. Report actual measured results; never claim local tests passed without execution.

## 8. Privacy and repository hygiene

- KTP photos, real personal data, human corner annotations, and diagnostic JSON stay **local**; never commit or upload them to GitHub. Maintain `.gitignore` entries for test image/holdout directories and annotation/output files.
- Do not add real KTP data to tests or fixtures. Use synthetic images and anonymized geometry for automated tests.
- Work on `experiment/ktp-dynamic-perspective-v3`; commit small, reviewable changes with tests. Do not modify `main` unless specifically asked.

## 9. Required agent checklist before ANY implementation

1. State the user-facing goal and underlying failure mode; explain why the approach generalizes to unseen inputs and the relevant subsystem.
2. Confirm there are **no sample-specific assumptions**, private data leaks, or evaluation ground truth used at runtime.
3. Assess impacted existing workflows; isolate experiments from production, preserve fallbacks, and avoid unauthorized feature/role changes.
4. Add focused regression tests covering normal inputs, diverse orientations/formats, negative/failure cases, and integration boundaries as appropriate.
5. Validate on representative varied inputs, then fresh unseen data when algorithm accuracy matters; compare against the relevant previous stable behavior.
6. Confirm UI/export/installed-app behavior where relevant; clearly distinguish **implemented**, **tested**, and **validated for production**.

**System-wide acceptance principle:** A feature is not finished merely because example files look correct or unit tests pass. It must generalize to representative new inputs and must not regress established document, KTP, UI, export, security, or installed-app workflows. Any PR-specific instruction is subordinate to these repository-wide rules.
