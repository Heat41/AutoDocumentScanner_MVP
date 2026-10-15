"""Synthetic, non-identity PR6 physical boundary observations."""
import numpy as np
from autodocscanner.core.physical_boundary_pr6 import verify_physical_boundary


def test_uniform_image_does_not_claim_detected_card():
    img = np.full((360, 560, 3), 120, dtype=np.uint8)
    quad = [[80, 70], [480, 70], [480, 290], [80, 290]]
    result = verify_physical_boundary(img, quad)
    assert result["median_polarity_consistency"] is None
    assert result["measurable"] is False


def test_rectangular_card_has_measurable_inside_outside_transition():
    img = np.full((360, 560, 3), 30, dtype=np.uint8)
    img[70:290, 80:480] = 220
    quad = [[80, 70], [479, 70], [479, 289], [80, 289]]
    result = verify_physical_boundary(img, quad)
    assert result["measurable"] is True
    assert result["median_polarity_consistency"] >= .95


def test_cropped_card_edge_fails_closed_when_unobservable():
    img = np.full((360, 560, 3), 50, dtype=np.uint8)
    quad = [[0, 0], [559, 0], [559, 359], [0, 359]]
    result = verify_physical_boundary(img, quad)
    assert result["measurable"] is False


def test_invalid_quad_does_not_crash_or_score():
    img = np.full((360, 560, 3), 50, dtype=np.uint8)
    result = verify_physical_boundary(img, [[1, 1]] * 4)
    assert result["measurable"] is False
    assert result["reason"] == "invalid_geometry"
