# PR6 — Manual Fallback Validation (shadow-only)

## Scope

This work validates a proposed *review-to-manual-editor contract*, not the running
Auto KTP user interface. It must not overwrite the production detector, save a
crop, mutate a source photo, or change JPG/PDF/SnipSheet workflows.

`pr6_manual_fallback_shadow` is appended after
`pr6_review_consolidation_shadow` in the offline detector comparison report.

## Review behavior

- `manual_review`: discard any legacy V3 preview corners; request four points
  in original-image coordinates from the user. Keep `suggested_corners = null`.
- `preview_review`: only carry original detector corners when the PR6 gate,
  existing handoff and original source agree; require explicit user review.
- `cancel`: discard pending review state, do not crop or write.
- `confirm`: validate all four finite, ordered, convex points against the
  original image's dimensions. Return a **data-only** manual-apply validation.
  The existing UI must independently receive the user's action before calling
  the production scanner.
- Invalid/mismatched data fails closed; no annotation/ground truth affects
  the decision. `auto_select = null` and `auto_crop_allowed = false`.

## Confirmed existing Auto KTP UI gap — not fixed by this step

The current `autodocscanner/ui/final.py` disables the manual correction button
when `_active_file_has_corners()` is false, and `open_manual_correction`
returns early when the selected file has no detector corners. Thus **the Auto
KTP UI cannot currently enter its four-corner editor directly after a total
detection failure**. The standalone Manual Document page is a separate path.

Also, the current Auto KTP manual dialog initializes from detected corners;
its **Batal** button closes the dialog, and **Terapkan Koreksi** calls
`scan_with_corners`, writing the result. The new PR6 decision has not been
connected to that dialog, so a legacy V3 preview might still appear in the UI
until integration is expressly approved and tested.

The shadow report exposes `ui_gap_codes` and
`ui_integration_status = not_connected_shadow_only` rather than claiming
the UI has already been repaired.

## Required next checks before UI integration

1. Enable a manual four-corner entry path from the unmodified original KTP
   image even when detection returned no valid corners.
2. Keep user edits in a temporary dialog state; cancel must leave the source
   and previous output untouched.
3. On user-initiated confirmation only, validate corner order, coordinate
   mapping and image dimensions before executing the existing scanner.
4. Check portrait/landscape image mapping, reset/cancel, failed scan recovery,
   JPG/PDF export, SnipSheet and packaged Windows behavior.
5. Expand holdout evaluations using unseen photos; do not tune runtime
   decisions to the prior five diagnostic examples.

## Validation

Synthetic tests: `python -m pytest tests/ktp/test_manual_fallback_validation_pr6.py -v`

Full suite: `python -m pytest -q`

Offline report: `python -m autodocscanner.core.detector_comparison <photos> --annotations <local.json> --output <local.json>`

Keep all real KTP photos, annotations, logs containing personal information
and diagnostic JSONs local; never commit them.
