"""Synthetic, non-identity tests for physical boundary diagnostics."""
import cv2
import numpy as np

from autodocscanner.core.boundary_evidence import boundary_evidence


def test_invalid_quad_has_no_boundary_evidence():
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    data = boundary_evidence(img, [[10, 10], [400, 20], [400, 20], [10, 230]])
    assert not data["valid"]
    assert data["score"] == 0.0


def test_detects_four_contrasting_physical_sides():
    img = np.full((350, 550, 3), 35, dtype=np.uint8)
    cv2.rectangle(img, (70, 60), (475, 290), (230, 230, 230), thickness=-1)
    quad = [[70, 60], [475, 60], [475, 290], [70, 290]]
    data = boundary_evidence(img, quad)
    assert data["valid"]
    assert len(data["side_support"]) == 4
    assert all(0 <= v <= 1 for v in data["side_support"])
    assert data["score"] > 0.1


def test_uniform_image_cannot_prove_card_boundary():
    img = np.full((350, 550, 3), 125, dtype=np.uint8)
    quad = [[70, 60], [475, 60], [475, 290], [70, 290]]
    data = boundary_evidence(img, quad)
    assert data["valid"]
    assert data["score"] < 0.01

def test_edge_outside_frame_is_unknown_not_zero():
    image = np.full((250, 400, 3), 125, dtype=np.uint8)
    quad = [[1, 1], [398, 1], [398, 248], [1, 248]]
    data = boundary_evidence(image, quad)
    assert data["valid"]
    assert data["frame_suspect"]
    assert not data["evidence_complete"]
    assert all(side is None for side in data["side_support"])
    assert data["score"] is None
    assert data["frame_review_needed"]


def test_genuine_near_frame_card_is_flagged_not_rejected():
    image = np.full((250, 400, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (3, 3), (396, 246), (230, 230, 230), -1)
    data = boundary_evidence(image, [[3, 3], [396, 3], [396, 246], [3, 246]])
    assert data["valid"]
    assert data["frame_suspect"]
    assert len(data["side_support"]) == 4
