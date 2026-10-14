"""Offline tests for conservative four-corner validation."""
import numpy as np

from autodocscanner.core.quad_validation import validate_quad


def test_valid_trapezoid():
    quad = [[40, 30], [350, 45], [320, 220], [50, 210]]
    ok, reasons, data = validate_quad(quad, (260, 400, 3))
    assert ok, reasons
    assert data["area_ratio"] > 0.1


def test_duplicate_corner_rejected():
    quad = [[20, 20], [370, 20], [370, 220], [370, 220]]
    ok, reasons, _ = validate_quad(quad, (260, 400, 3))
    assert not ok
    assert "short_or_duplicate_side" in reasons


def test_nan_rejected():
    ok, reasons, _ = validate_quad([[0, 0], [100, 0], [100, np.nan], [0, 100]], (200, 200))
    assert not ok
    assert "invalid_points" in reasons


def test_too_small_rejected():
    ok, reasons, _ = validate_quad([[20, 20], [40, 20], [40, 40], [20, 40]], (1000, 1000))
    assert not ok
    assert "area_too_small" in reasons


def test_outside_image_rejected():
    ok, reasons, _ = validate_quad([[-100, 20], [350, 20], [350, 200], [20, 200]], (260, 400))
    assert not ok
    assert "outside_image" in reasons


def test_large_card_near_frame_flagged_not_rejected():
    quad = [[2, 2], [397, 2], [397, 257], [2, 257]]
    ok, reasons, data = validate_quad(quad, (260, 400))
    assert ok, reasons
    assert data["frame_suspect"]


def test_invalid_shape_rejected():
    ok, reasons, _ = validate_quad([[1, 1], [2, 2], [3, 3]], (260, 400))
    assert not ok
    assert "invalid_points" in reasons


def test_slight_corner_extrapolation_allowed():
    quad = [[-3, 25], [370, 20], [380, 220], [2, 218]]
    ok, reasons, metrics = validate_quad(quad, (260, 400))
    assert ok, reasons
    assert metrics["slight_extrapolation"]


def test_two_near_frame_vertices_are_suspicious_at_large_area():
    quad = [[1, 4], [148, 9], [148, 190], [1, 187]]
    ok, reasons, metrics = validate_quad(quad, (194, 150))
    assert ok, reasons
    assert metrics["frame_suspect"]
