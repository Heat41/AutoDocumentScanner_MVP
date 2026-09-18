# Repository Package Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Reorganize AutoDocumentScanner into domain-based Python packages without changing runtime behavior.

**Architecture:** Keep only application entry points and build/release files at repository root. Move production modules into `autodocscanner/` subpackages by responsibility, update imports and PyInstaller/build references, and categorize tests to mirror the package layout. This is a structural refactor only: no algorithm, UI behavior, storage schema, or output behavior changes.

**Tech Stack:** Python 3, Tkinter, OpenCV, Pillow, sqlite3, unittest, PyInstaller.

**Base branch:** `feature/stage2d-tracking-ktp`

## Global Constraints

- No scanner algorithm changes.
- No perspective-detection changes.
- No Stage 2A/2B/2C/2D behavior changes.
- Root entry points remain `app.py` and `desktop_launcher.py`.
- Existing Windows commands remain valid:
  - `python .\app.py --ui`
  - `python -m unittest discover -s tests -v`
  - `.\build_windows.ps1`
- Runtime directories `input/`, `output/`, and `data/` remain unchanged.
- PyInstaller packaging must continue to launch the Stage 2 UI.

## Target Source Layout

```
autodocscanner/
  core/
    scanner.py
    perspective_engine.py
    robustness_engine.py
    quality_check.py
    final_validation.py
  documents/
    manual_document.py
    document_session.py
    document_canvas.py
  ktp/
    models.py
    database.py
    media.py
    storage.py
  output/
    manager.py
    safe.py
  services/
    batch.py
    ktp_output.py
  ui/
    base.py
    safe.py
    final.py
    textured.py
    responsive.py
    stage2.py
    manual_document.py
    tracking_ktp.py
  support/
    branding.py
```

## Task 1 — Create package skeleton
- Add `autodocscanner/__init__.py`.
- Add `__init__.py` to every subpackage.
- Add import smoke test for package boundaries.

## Task 2 — Move core/output/services modules
- Move scanner and perspective/validation modules to `autodocscanner/core/`.
- Move output manager/safe output to `autodocscanner/output/`.
- Move batch runner and Stage 2 KTP output routing to `autodocscanner/services/`.
- Update internal imports.

## Task 3 — Move document and KTP storage modules
- Move manual document modules to `autodocscanner/documents/`.
- Move KTP storage modules to `autodocscanner/ktp/`.
- Update cross-package imports.

## Task 4 — Move UI and branding modules
- Move UI hierarchy to `autodocscanner/ui/`.
- Move branding helper to `autodocscanner/support/branding.py`.
- Update all UI imports and entry points.
- Update build script branding invocation.

## Task 5 — Categorize tests
- Move tests into domain folders matching source layout.
- Add `__init__.py` where required for unittest discovery.
- Update test imports to package paths.

## Task 6 — Packaging and regression guard
- Update `AutoDocumentScanner.spec` if module/data path references require it.
- Verify `desktop_launcher.py` imports `autodocscanner.ui.stage2`.
- Verify `app.py --ui` imports `autodocscanner.ui.stage2`.
- Run full regression.
- Run import smoke checks.
- Build Windows package only after regression is green.
