# AutoDocumentScanner Stage 2 — Locked Design

**Date:** 2026-09-18
**Status:** DESIGN LOCKED

## Baseline rule

The existing Auto Koreksi KTP flow remains unchanged in behavior. Stage 2 adds new features as separate modules/pages and reuses existing engines where appropriate.

## Locked feature set

1. **Auto Koreksi KTP**
   - Keep automatic perspective correction behavior.
   - Add output choice: image or PDF.
   - PDF page follows the corrected KTP image dimensions/aspect ratio; no A4 canvas.

2. **Koreksi Dokumen Manual**
   - Separate page.
   - Four draggable corner points appear near image corners by default.
   - Manual perspective correction uses the same geometric warp principle as the KTP engine.
   - Multi-page documents are corrected page-by-page.
   - Export as separate images or one combined PDF.

3. **Tracking Data KTP**
   - Separate page from Auto Koreksi KTP.
   - Choose KTP image -> process -> internal auto perspective -> normalized bounding-box OCR -> master-region validation -> review -> Rekam Data KTP.
   - Review is mandatory before recording.
   - Face photo is tracked when available but is optional.
   - Required identity fields must be complete.

4. **Master Wilayah Indonesia**
   - Entire Indonesia.
   - Hierarchy: Province -> Regency/City -> District -> Village/Subdistrict -> administrative type.
   - Bounding boxes read text; master data validates administrative meaning.
   - Manual review remains fallback.

5. **Storage and revision**
   - SQLite for structured data.
   - Media files stored separately.
   - One NIK = one active primary record.
   - Reprocessing the same NIK creates a new revision, preserving previous revisions.

6. **WP Client**
   - Future API integration to office website's Rekam WP Client.
   - OCR/tracking/storage remain independent of API adapter and field mapper.

## Page separation

- Dashboard
- Auto Koreksi KTP
- Koreksi Dokumen Manual
- Tracking Data KTP
- Data KTP
- Detail/Revisi KTP
- Master Wilayah
- WP Client (future API stage)

## Implementation stages

- Stage 2A: Output Manager / KTP PDF
- Stage 2B: Manual Document Perspective
- Stage 2C: KTP Data Storage
- Stage 2D: Bounding Box & OCR
- Stage 2E: Master Wilayah Nasional
- Stage 2F: Review & Rekam Data KTP
- Stage 2G: WP Client API integration
