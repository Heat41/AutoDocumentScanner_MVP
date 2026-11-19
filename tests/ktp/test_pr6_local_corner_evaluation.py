"""Synthetic original-image PR6 reference errors; no KTP or private files."""
from copy import deepcopy
import numpy as np
import pytest

from autodocscanner.core.pr6_local_corner_evaluation import (
    evaluate_pr6_candidates_against_manual as evaluate,
)

SIZE = (400, 600)
TRUTH = [[140, 90], [460, 100], [450, 300], [145, 290]]


def _snapshot():
    return {
        "status": "offline_visual_diagnostic_only",
        "image_shape": [400, 600], "auto_select": None,
        "auto_crop_allowed": False,
        "candidates": [
            {"source": "stable", "family": "Stable", "corners": [
                [90, 40], [510, 40], [500, 355], [90, 355],
            ]},
            {"source": "side_recovery_3", "family": "Side Recovery",
             "corners": deepcopy(TRUTH)},
            {"source": "independent_sides_0", "family": "Independent Sides",
             "corners": (np.asarray(TRUTH)+[4, 0]).tolist()},
        ],
    }


def test_exact_and_near_candidates_measured_without_changing_snapshot():
    snap = _snapshot()
    original = deepcopy(snap)
    result = evaluate(snap, TRUTH, SIZE)
    assert result["candidate_count"] == 3
    assert result["candidates"][0]["source"] == "side_recovery_3"
    assert result["candidates"][0]["mean_error_px"] == 0
    assert result["candidates"][1]["mean_error_px"] == 4
    assert result["oracle_only"] and result["auto_select"] is None
    assert result["file_write_performed"] is False
    assert "manual_corners" not in result and "path" not in result
    assert original == snap


def test_rotated_start_and_reversed_points_evaluate_without_false_error():
    for points in (
        np.roll(TRUTH, 1, axis=0).tolist(),
        np.asarray(TRUTH)[::-1].tolist(),
    ):
        snap = _snapshot()
        snap["candidates"][1]["corners"] = points
        result = evaluate(snap, TRUTH, SIZE)
        assert result["candidates"][0]["mean_error_px"] == 0


def test_normalized_error_uses_original_image_diagonal():
    snap = _snapshot()
    snap["candidates"] = [{
        "source": "v2_1", "family": "Kontur V2",
        "corners": (np.asarray(TRUTH)+[12, 0]).tolist(),
    }]
    row = evaluate(snap, TRUTH, SIZE)["candidates"][0]
    assert row["mean_error_px"] == 12
    assert abs(row["normalized_error"]-12/np.hypot(*SIZE)) < 1e-6


def test_family_stats_are_oracle_only_and_bounded():
    result = evaluate(_snapshot(), TRUTH, SIZE)
    families = {r["family"]: r for r in result["families"]}
    assert families["Side Recovery"]["best_error"] == 0
    assert families["Stable"]["median_error"] > 0.03
    assert families["Independent Sides"]["p90_error"] > 0
    assert result["auto_crop_allowed"] is False


@pytest.mark.parametrize("invalid", [
    None, {}, {"status": "other"},
    {"status": "offline_visual_diagnostic_only",
     "auto_crop_allowed": True, "auto_select": None,
     "image_shape": [400, 600], "candidates": []},
])
def test_invalid_snapshot_rejected(invalid):
    with pytest.raises(ValueError):
        evaluate(invalid, TRUTH, SIZE)


@pytest.mark.parametrize("corners", [
    [[1, 1]]*4,
    [[float("nan"), 2]]*4,
    [[-1, 30], [400, 30], [400, 200], [10, 200]],
    [TRUTH[i] for i in [0, 2, 1, 3]],
])
def test_invalid_reference_rejected(corners):
    with pytest.raises(ValueError):
        evaluate(_snapshot(), corners, SIZE)


def test_exif_orientation_mismatch_rejected():
    with pytest.raises(ValueError, match="Orientasi"):
        evaluate(_snapshot(), TRUTH, (600, 400))


def test_missing_bad_candidates_not_invented():
    snap = _snapshot()
    snap["candidates"] = [
        {"source": "v2_1", "family": "Kontur V2",
         "corners": [[-30, 0]]*4},
        {"source": "v2_2", "family": "Kontur V2",
         "corners": [[float("nan"), 0]]*4},
    ]
    result = evaluate(snap, TRUTH, SIZE)
    assert result["candidate_count"] == 0
    assert result["skipped_count"] == 2
    assert result["families"] == []


def test_no_candidates_reported_as_no_data_not_an_accuracy_claim():
    snap = _snapshot()
    snap["candidates"] = []
    result = evaluate(snap, TRUTH, SIZE)
    assert result["candidate_count"] == 0
    assert result["candidates"] == []
