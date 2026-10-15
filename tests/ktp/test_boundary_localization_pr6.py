"""PR6 localization regression tests; no real documents or personal data."""
import numpy as np
from autodocscanner.core.boundary_localization_pr6 import localize_candidate_boundary


def test_synthetic_rectangle_measured():
    image = np.full((400, 640, 3), 25, dtype=np.uint8)
    image[70:330, 90:550] = 220
    points = [[90,70],[549,70],[549,329],[90,329]]
    result = localize_candidate_boundary(image, points)
    assert result["measurable"]
    assert result["auto_select"] is None
    assert len(result["side_localization"]) == 4


def test_uniform_background_has_no_credible_gradients():
    image = np.full((400,640,3), 127, dtype=np.uint8)
    points = [[90,70],[549,70],[549,329],[90,329]]
    result = localize_candidate_boundary(image, points)
    assert not result["measurable"]
    assert result["reason"] == "insufficient_gradient_coverage"


def test_invalid_geometry_is_rejected():
    image = np.zeros((400,640,3), dtype=np.uint8)
    result = localize_candidate_boundary(image, [[1,1]]*4)
    assert not result["measurable"]
    assert result["reason"] == "invalid_geometry"


def test_rotated_quad_never_autoselects():
    image = np.zeros((400,640,3), dtype=np.uint8)
    points = [[160,30],[580,120],[470,350],[60,260]]
    result = localize_candidate_boundary(image, points)
    assert result["auto_select"] is None
