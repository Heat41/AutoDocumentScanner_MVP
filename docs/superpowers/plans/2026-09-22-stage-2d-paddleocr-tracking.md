# Stage 2D PaddleOCR-First Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** Replace Tesseract as the primary per-field text recognizer in Tracking KTP with CPU-only PaddleOCR TextRecognition while retaining Tesseract for registration/layout and fallback.

**Architecture:** Add a lazy Paddle text-recognition backend using PP-OCRv5 mobile recognition on ROI crops. Wrap it with a hybrid backend: `read()` prefers Paddle and falls back to Tesseract, while `read_layout()` and `read_document()` continue using Tesseract so the existing registration and fallback flow remains intact. Paddle dependencies stay in a separate experimental requirements file until UAT proves accuracy and packaging viability.

**Tech Stack:** Python 3.12, PaddlePaddle CPU 3.3.1, PaddleOCR 3.7.0, PP-OCRv5 mobile recognition, Tesseract fallback, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints
- Tracking KTP only.
- CPU-only.
- Existing Auto Perspective, manual-document, storage, UI navigation, and PDF report behavior remain unchanged.
- Paddle imports and model initialization are lazy.
- Unit tests must not download or load Paddle models.
- Tesseract remains available as fallback.
- Paddle model is instantiated once and reused across all fields.
- Initial Paddle validation is optional dependency installation; installer bundling follows only after UAT.

## Tasks
1. Add optional Paddle dependency file.
2. Add lazy Paddle ROI recognition backend.
3. Add Paddle-first/Tesseract-fallback hybrid backend.
4. Make Tracking default to hybrid backend.
5. Add unit tests that use fake Paddle models and no network/model download.
6. Run quiet regression and then real Windows UAT.
