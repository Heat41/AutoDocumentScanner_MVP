"""Synthetic tests for PR6 region-context evidence; NEVER identity/crop scoring."""
import cv2
import numpy as np
import pytest

from autodocscanner.core.region_context_pr6 import inspect_region_context


def _scene():
    image = np.full((460, 800, 3), (126, 126, 126), np.uint8)
    card = [[90, 85], [405, 85], [405, 295], [90, 295]]
    distractor = [[500, 85], [760, 85], [760, 280], [500, 280]]
    cv2.fillConvexPoly(image, np.asarray(card, np.int32), (150, 175, 193))
    cv2.fillConvexPoly(image, np.asarray(distractor, np.int32), (200, 80, 110))
    for i in range(7):
        y = 111 + i * 24
        cv2.line(image, (110, y), (288, y), (30, 30, 30), 2)
    cv2.rectangle(image, (324, 116), (382, 263), (30, 60, 70), 2)
    return image, card, distractor


def test_structured_document_like_area_has_more_internal_cues_than_blank_object():
    image, card, distractor = _scene()
    before = image.copy()
    card_context = inspect_region_context(image, card)
    blank_context = inspect_region_context(image, distractor)
    assert card_context["interior_edge_density"] > blank_context["interior_edge_density"]
    assert card_context["interior_spatial_coverage"] > blank_context["interior_spatial_coverage"]
    assert "sparse_internal_structure" in blank_context["reason_codes"]
    assert card_context["auto_crop_allowed"] is False
    assert blank_context["auto_crop_allowed"] is False
    assert np.array_equal(image, before)


def test_matching_other_method_is_context_not_proof_of_correct_card():
    image, card, distractor = _scene()
    result = inspect_region_context(
        image, card,
        other_candidates=[
            {"family": "stable", "corners": card},
            {"family": "chroma", "corners": card},
            {"family": "outer_region", "corners": distractor},
        ]
    )
    assert result["corroborating_families"] == ["chroma", "stable"]
    assert "no_independent_geometric_corroboration" not in result["reason_codes"]
    assert result["auto_select"] is None
    assert result["auto_crop_allowed"] is False


def test_blank_rectangle_risk_never_asserts_it_is_not_a_ktp():
    image, card, distractor = _scene()
    result = inspect_region_context(image, distractor)
    assert result["status"] == "offline_diagnostic_only"
    assert "sparse_internal_structure" in result["reason_codes"]
    assert "no_independent_geometric_corroboration" in result["reason_codes"]
    assert "wrong_object" not in result["reason_codes"]


def test_nested_smaller_regions_are_observed_without_auto_acceptance():
    image, card, distractor = _scene()
    small = [[320, 115], [380, 115], [380, 260], [320, 260]]
    result = inspect_region_context(
        image, card,
        other_candidates=[{"family": "contour", "corners": small}],
    )
    assert result["nested_inner_candidates"] == 1
    assert result["auto_crop_allowed"] is False


@pytest.mark.parametrize("input_image,input_points", [
    (None, [[0, 0]] * 4),
    (np.zeros((20, 20, 3), dtype=np.uint8), [[0, 0]] * 4),
    (np.zeros((150, 300), dtype=np.uint8), [[0, 0]] * 4),
    (np.zeros((150, 300, 3), dtype=np.float32), [[0, 0]] * 4),
    (np.zeros((150, 300, 3), dtype=np.uint8), [[0, 0]] * 4),
])
def test_invalid_input_or_corners_fail_closed(input_image, input_points):
    result = inspect_region_context(input_image, input_points)
    assert result["auto_select"] is None
    assert result["auto_crop_allowed"] is False
    assert result["reason_codes"][0] in ("invalid_image", "invalid_geometry")


def test_missing_or_spurious_other_candidates_do_not_raise():
    image, card, distractor = _scene()
    result = inspect_region_context(
        image, card,
        other_candidates=[
            None, {}, {"family": "bad", "corners": []},
            {"family": "bad", "corners": [[float("nan"), 0]] * 4},
        ]
    )
    assert result["corroborating_families"] == []
    assert "no_independent_geometric_corroboration" in result["reason_codes"]


def test_region_inspector_never_writes_or_returns_sensitive_image_data(tmp_path):
    image, card, distractor = _scene()
    before = set(tmp_path.iterdir())
    result = inspect_region_context(image, card)
    assert set(tmp_path.iterdir()) == before
    assert "image" not in result and "path" not in result
    assert result["auto_crop_allowed"] is False
