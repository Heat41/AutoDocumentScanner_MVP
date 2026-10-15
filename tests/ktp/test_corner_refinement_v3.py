"""PR6 refinement must be cautious and never corrupt source corners."""
import cv2
import numpy as np

from autodocscanner.core.corner_refinement_v3 import refine_corners


def test_blank_image_preserves_corners():
    image = np.full((400, 600, 3), 120, dtype=np.uint8)
    quad = np.array([[80, 80], [520, 80], [520, 320], [80, 320]], np.float32)
    candidate, meta = refine_corners(image, quad)
    assert not meta["accepted"]
    np.testing.assert_array_equal(candidate, quad)


def test_invalid_quad_cannot_refine():
    image = np.zeros((400, 600, 3), np.uint8)
    quad = np.array([[80, 80], [520, 80], [520, 80], [80, 320]], np.float32)
    candidate, meta = refine_corners(image, quad)
    assert meta["reason"] == "invalid_input"
    np.testing.assert_array_equal(candidate, quad)


def test_nearby_rectangle_refinement_cannot_jump_far():
    image = np.full((400, 600, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (80, 70), (520, 330), (220, 220, 220), -1)
    quad = np.array([[83, 73], [517, 73], [517, 327], [83, 327]], np.float32)
    candidate, meta = refine_corners(image, quad)
    assert candidate.shape == (4, 2)
    assert np.isfinite(candidate).all()
    assert np.max(np.linalg.norm(candidate - quad, axis=1)) <= .025 * np.hypot(400, 600) + .01


def test_out_of_frame_candidate_refuses_refinement():
    image = np.zeros((400, 600, 3), np.uint8)
    quad = np.array([[-100, 80], [520, 80], [520, 320], [80, 320]], np.float32)
    candidate, meta = refine_corners(image, quad)
    assert not meta["accepted"]
    np.testing.assert_array_equal(candidate, quad)
