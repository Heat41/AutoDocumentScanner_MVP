"""PR6 manual fallback comparison report integration (synthetic blank image)."""

import numpy as np


def test_no_detected_candidate_still_produces_manual_fallback_plan(monkeypatch):
    from autodocscanner.core import detector_comparison as comparison

    image = np.full((320, 480, 3), 100, dtype=np.uint8)

    class EmptyDetector:
        def detect(self, source):
            return None, {"score": 0, "selected_source": "none"}

    monkeypatch.setattr(comparison, "StableBaselinePerspectiveEngine", EmptyDetector)
    monkeypatch.setattr(comparison, "AutoPerspectiveEngine", EmptyDetector)
    monkeypatch.setattr(comparison, "contour_candidates", lambda *args, **kwargs: [])

    result = comparison.compare_detectors(image)
    plan = result["pr6_manual_fallback_shadow"]
    assert plan["status"] == "manual_review"
    assert plan["suggested_corners"] is None
    assert plan["candidate_source"] is None
    assert plan["editor"] == "existing_manual_four_corners"
    assert plan["auto_select"] is None
    assert plan["auto_crop_allowed"] is False
    assert plan["file_write_performed"] is False
    assert plan["ui_integration_status"] == "not_connected_shadow_only"
    assert "auto_ktp_manual_editor_requires_existing_detector_corners" in plan["ui_gap_codes"]
    assert result["v3_manual_handoff"]["action"] == "manual_corners"
