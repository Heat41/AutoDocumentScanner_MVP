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
