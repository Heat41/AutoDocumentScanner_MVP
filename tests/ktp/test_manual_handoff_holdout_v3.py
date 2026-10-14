"""Safe offline handoff and holdout evaluation regression tests."""
from autodocscanner.core.manual_handoff_v3 import prepare_manual_handoff
from autodocscanner.core.holdout_evaluation_v3 import summarize


def test_manual_review_has_no_suggested_corners():
    report = {"v3_review_gate": {
        "status": "manual_review", "candidate_source": "stable",
        "auto_crop_allowed": False, "reasons": ["weak_side"]},
        "stable": {"corners": [[20, 20], [380, 20], [380, 280], [20, 280]]}}
    outcome = prepare_manual_handoff(report, (300, 400))
    assert outcome["action"] == "manual_corners"
    assert outcome["suggested_corners"] is None
    assert outcome["auto_crop_allowed"] is False


def test_preview_valid_corners_require_explicit_confirmation():
    quad = [[20, 20], [380, 20], [380, 280], [20, 280]]
    report = {"v3_review_gate": {
        "status": "preview_review", "candidate_source": "stable",
        "auto_crop_allowed": False}, "stable": {"corners": quad}}
    outcome = prepare_manual_handoff(report, (300, 400))
    assert outcome["action"] == "preview_suggested_corners"
    assert outcome["suggested_corners"] == quad
    assert outcome["requires_explicit_confirmation"]
    assert not outcome["auto_crop_allowed"]


def test_preview_invalid_geometry_fails_closed():
    report = {"v3_review_gate": {
        "status": "preview_review", "candidate_source": "stable",
        "auto_crop_allowed": False},
        "stable": {"corners": [[20, 20], [380, 20], [380, 20], [20, 280]]}}
    outcome = prepare_manual_handoff(report, (300, 400))
    assert outcome["action"] == "manual_corners"
    assert outcome["suggested_corners"] is None


def test_explicit_auto_crop_true_is_not_trusted():
    report = {"v3_review_gate": {
        "status": "preview_review", "candidate_source": "stable",
        "auto_crop_allowed": True},
        "stable": {"corners": [[20, 20], [380, 20], [380, 280], [20, 280]]}}
    assert prepare_manual_handoff(report, (300, 400))["suggested_corners"] is None


def test_holdout_without_annotations_reports_no_accuracy():
    result = summarize({"image": {"stable": {"found": True}, "v3_review_gate": {
        "candidate_source": "stable", "status": "preview_review"}}})
    assert result["annotated_images"] == 0


def test_holdout_comparison_uses_explicit_errors_only():
    result = summarize({"image": {
        "stable": {"normalized_corner_error": 0.06},
        "adaptive": {"normalized_corner_error": 0.025},
        "v3_review_gate": {"candidate_source": "adaptive", "status": "preview_review"}}})
    assert result["annotated_images"] == 1
    assert result["mean_normalized_corner_error_suggestion"] == 0.025
    assert result["mean_normalized_corner_error_stable"] == 0.06
