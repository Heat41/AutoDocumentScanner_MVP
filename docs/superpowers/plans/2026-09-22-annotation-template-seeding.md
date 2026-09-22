# Stage 2D Annotation Template Seeding Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Reuse the first corrected KTP annotation as an automatic starting template for later KTP samples.

**Architecture:** Store a local canonical YOLO template under the ignored dataset folder. When a canonical KTP has its own saved label, load that label. Otherwise seed the annotator from the template. Saving the first non-empty annotation automatically creates the template; later samples do not overwrite it unless explicitly requested in a future change.

**Tech Stack:** Python 3.12, Tkinter, YOLO text labels, unittest.

**Spec:** Stage 2D Full KTP Field Detector plan.

## Global Constraints
- Dataset and annotation template remain local under `dataset/` and never enter Git.
- Existing per-image labels always take priority over the template.
- Missing/empty fields may be deleted from the seeded boxes before saving.
- Canonical size remains 856x540.
- Do not alter locked scanner/perspective behavior.

## Tasks
1. Add reusable template load/save service.
2. Auto-create template from first saved non-empty annotation.
3. Seed new unlabeled KTPs from template.
4. Surface status text indicating whether boxes came from image labels or template.
5. Add regression tests.
