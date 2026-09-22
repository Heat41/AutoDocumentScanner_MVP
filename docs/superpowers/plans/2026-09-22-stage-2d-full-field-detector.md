# Stage 2D Full KTP Field Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Replace direct ROI-driven Tracking with a detector contract where every KTP field is represented by a class, bounding box, and detector confidence, then OCR reads the detected value boxes.

**Architecture:** Introduce a detector abstraction with a local ONNX implementation for future trained models and a template/dynamic fallback so the current app remains usable before training. Tracking consumes detector results as the single source of crop coordinates. Debug overlay renders object-detection-style labels such as `NAMA 95%` and `NIK 98%`.

**Tech Stack:** Python 3.12, OpenCV DNN/ONNX, NumPy, Tesseract, unittest.

**Spec:** Current AutoDocumentScanner Stage 2D Tracking requirements in project context.

## Global Constraints
- Work only on Tracking KTP internals/UI.
- Keep Auto Perspective and the locked KTP scanner unchanged.
- CPU-only and offline.
- Do not add cloud/runtime model download.
- ONNX model path is local under `models/ktp_field_detector/ktp_fields.onnx`.
- Before a trained ONNX model exists, use the current label-aligned template boxes as fallback detections.
- OCR must crop from detector bounding boxes, not directly from static ROI tables.
- Face detection/crop remains available.
- Existing PDF/report and review UI remain functional.

## Review Focus
- Missing ONNX file must transparently fall back without crashing.
- Duplicate class detections must keep the highest-confidence result.
- Invalid/out-of-bounds boxes must be clamped before cropping.
- Detector classes must map exactly to Tracking field names.
- Debug overlay must render the actual detections used by OCR.

---

### Task 1: Detector contract and fallback detector

**Files:**
- Create: `autodocscanner/ktp/field_detection.py`
- Test: `tests/ktp/test_field_detection.py`

**Interfaces:**
- Produces `FieldDetection(class_name, bbox, confidence, source)`.
- Produces `TemplateFieldDetector.detect(image, anchors=None)`.
- Produces `OnnxFieldDetector.detect(image, anchors=None)`.
- Produces `AutoFieldDetector.detect(image, anchors=None)`.

- [ ] Define field classes and normalized/pixel box conversion.
- [ ] Add template fallback detections using anchor-aligned boxes.
- [ ] Add local ONNX loader and inference normalization.
- [ ] Prefer highest-confidence detection per class.
- [ ] Add tests that require no model file.

### Task 2: Tracking consumes detections

**Files:**
- Modify: `autodocscanner/ktp/tracking.py`
- Test: `tests/ktp/test_tracking.py`

**Interfaces:**
- `extract_tracking_data(..., detector=None)` accepts injectable detector.
- `KtpTrackingResult.detections` exposes actual detector results.

- [ ] Detect fields after normalization/registration.
- [ ] Crop OCR input from each detection bbox.
- [ ] Preserve OCR/parser/review behavior.
- [ ] Use photo detection for face crop when valid; otherwise preserve existing face fallback.

### Task 3: Object-detection-style overlay

**Files:**
- Modify: `autodocscanner/ktp/roi_debug.py`
- Modify: `autodocscanner/ui/tracking_ktp.py`
- Test: `tests/ktp/test_roi_debug.py`

**Interfaces:**
- `build_detection_overlay(image, detections)`.
- UI toggle shows `[CLASS confidence]` labels over the exact boxes used by Tracking.

- [ ] Render class name + confidence above bounding box.
- [ ] UI consumes `tracking.detections`.
- [ ] Keep normal preview unchanged when debug is off.

### Task 4: Local-model packaging contract

**Files:**
- Create: `models/ktp_field_detector/README.md`
- Create: `models/ktp_field_detector/classes.txt`

- [ ] Lock class order for future training/export.
- [ ] Document expected local ONNX filename and CPU/offline runtime.
- [ ] Do not add a fake model file.

### Task 5: Regression/UAT

- [ ] Run `python -m unittest discover -s tests -q`.
- [ ] UAT Tracking with existing Melawi and Singkawang samples.
- [ ] Verify overlay displays named detections and OCR crops follow those boxes.
