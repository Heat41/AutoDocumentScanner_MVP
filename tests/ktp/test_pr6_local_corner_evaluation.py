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



def test_corner_error_vectors_follow_manual_labels_after_cyclic_rotation():
    snap = _snapshot()
    # One displaced physical top-left, the rest exactly aligned. The source
    # can begin at a different corner without changing the named error.
    modified = np.asarray(TRUTH, np.float64).copy()
    modified[0] += [15, -20]
    snap["candidates"] = [{
        "source": "v2_3", "family": "Outer Region",
        "corners": np.roll(modified, 2, axis=0).tolist(),
    }]
    row = evaluate(snap, TRUTH, SIZE)["candidates"][0]
    assert row["corner_errors_px"] == [25.0, 0.0, 0.0, 0.0]
    assert row["corner_offsets_xy_px"] == [
        [15.0, -20.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    ]
    assert row["worst_corner_index"] == 0


@pytest.mark.parametrize("angle", [0, 35, 93, 174])
def test_corner_offsets_and_errors_rotation_invariant(angle):
    center = np.asarray(TRUTH, np.float64).mean(axis=0)
    theta = np.deg2rad(angle)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    # Rotate around image centre, then fit into a large original frame.
    shifted = (np.asarray(TRUTH, np.float64) - center) @ rotation.T
    shifted += np.asarray([700, 700])
    wrong = shifted.copy()
    wrong[2] += [8, 6]
    # Valid manual TL/TR/BR/BL order requires clockwise polygon order,
    # independently of the photo's portrait/landscape rotation.
    snap = {
        "status": "offline_visual_diagnostic_only",
        "image_shape": [1500, 1500], "auto_crop_allowed": False,
        "auto_select": None,
        "candidates": [{
            "source": "side_recovery_0", "family": "Side Recovery",
            "corners": np.roll(wrong, 1, axis=0).tolist(),
        }],
    }
    row = evaluate(snap, shifted.tolist(), (1500, 1500))["candidates"][0]
    assert row["corner_errors_px"] == [0., 0., 10., 0.]
    assert row["corner_offsets_xy_px"][2] == [8., 6.]
    assert row["worst_corner_index"] == 2


def test_generation_counts_distinguish_absent_from_hidden_proposals():
    snap = _snapshot()
    snap.update({
        "generated_side_recovery_count": 3,
        "generated_paired_alignment_count": 0,
        "independent_sides_count": 2,
        "paired_alignment_parents_examined": 3,
        "independent_sides_parents_examined": 3,
        "source_hypotheses_hidden": 12,
    })
    evaluated = evaluate(snap, TRUTH, SIZE)
    assert evaluated["generation"] == {
        "side_recovery": 3, "paired_alignment": 0,
        "independent_sides": 2, "paired_parents_examined": 3,
        "independent_parents_examined": 3,
    }
    assert evaluated["undisplayed_hypothesis_count"] == 12
    assert evaluated["auto_select"] is None
    assert evaluated["auto_crop_allowed"] is False


def test_no_absolute_annotation_coordinates_enter_evaluation_report():
    report = evaluate(_snapshot(), TRUTH, SIZE)
    assert "truth" not in report
    assert "manual_corners" not in report
    assert "ground_truth" not in report
    assert "raw_photo" not in report
    for row in report["candidates"]:
        assert "corners" not in row
        assert "reference_corners" not in row
        assert len(row["corner_offsets_xy_px"]) == 4



def test_off_frame_independent_sides_is_diagnosed_and_measured_separately():
    snapshot = _snapshot()
    snapshot["candidates"] = [{
        "source": "independent_sides_0",
        "family": "Independent Sides",
        "corners": [[-6, 90], [460, 100], [450, 300], [145, 290]],
    }]
    original = deepcopy(snapshot)
    outcome = evaluate(snapshot, TRUTH, SIZE)
    assert outcome["candidate_count"] == 0
    assert outcome["skipped_count"] == 1
    assert outcome["families"] == []
    assert outcome["candidates"] == []
    assert outcome["skipped_details"] == [{
        "source": "independent_sides_0",
        "family": "Independent Sides",
        "reason": "corner_outside_original_photo",
    }]
    measured = outcome["outside_frame_evaluations"]
    assert len(measured) == 1
    assert measured[0]["source"] == "independent_sides_0"
    assert measured[0]["corner_errors_px"] == [146., 0., 0., 0.]
    assert measured[0]["mean_error_px"] == 36.5
    assert measured[0]["max_error_px"] == 146.
    assert outcome["auto_select"] is None
    assert outcome["auto_crop_allowed"] is False
    assert outcome["crop_performed"] is False
    assert outcome["file_write_performed"] is False
    assert snapshot == original


def test_two_off_frame_independent_sides_are_not_silent_or_ranked():
    snapshot = _snapshot()
    snapshot["candidates"] = [
        {"source": "v2_3", "family": "Outer Region",
         "corners": deepcopy(TRUTH)},
        {"source": "independent_sides_0", "family": "Independent Sides",
         "corners": [[-6, 90], [460, 100], [450, 300], [145, 290]]},
        {"source": "independent_sides_1", "family": "Independent Sides",
         "corners": [[140, -8], [460, 100], [450, 300], [145, 290]]},
    ]
    measured = evaluate(snapshot, TRUTH, SIZE)
    assert measured["candidate_count"] == 1
    assert measured["skipped_count"] == 2
    assert [x["source"] for x in measured["outside_frame_evaluations"]] == [
        "independent_sides_1", "independent_sides_0",
    ]
    assert measured["families"][0]["family"] == "Outer Region"
    assert all(row["family"] == "Independent Sides"
               for row in measured["outside_frame_evaluations"])
    assert measured["auto_select"] is None


def test_far_outside_visual_tolerance_still_skipped_without_geometry_measurement():
    snapshot = _snapshot()
    snapshot["candidates"] = [
        {"source": "independent_sides_0", "family": "Independent Sides",
         "corners": [[-55, 90], [460, 100], [450, 300], [145, 290]]},
    ]
    result = evaluate(snapshot, TRUTH, SIZE)
    assert result["candidate_count"] == 0
    assert result["skipped_count"] == 1
    assert result["outside_frame_evaluations"] == []
    assert result["skipped_details"][0]["reason"] == (
        "outside_visual_diagnostic_extent"
    )
