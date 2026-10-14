"""Holdout annotations are human-provided, never detector-generated."""
import json
import pytest

from autodocscanner.core.holdout_annotation_tool_v3 import (normalize_corners, save_annotation, original_to_rotated, rotated_to_original, preview_geometry)


def test_order_independent_annotation():
    data = normalize_corners([[370, 250], [20, 20], [20, 250], [370, 20]], (300, 400))
    assert data == [[20.0, 20.0], [370.0, 20.0], [370.0, 250.0], [20.0, 250.0]]


def test_duplicate_annotation_refused():
    with pytest.raises(ValueError):
        normalize_corners([[20, 20], [370, 20], [370, 20], [20, 250]], (300, 400))


def test_save_annotations_merges_existing(tmp_path):
    output = tmp_path / "annotations.json"
    quad = [[20, 20], [370, 20], [370, 250], [20, 250]]
    save_annotation(output, "first.jpg", quad, (300, 400))
    save_annotation(output, "second.jpg", quad, (300, 400))
    result = json.loads(output.read_text(encoding="utf-8"))
    assert set(result) == {"first.jpg", "second.jpg"}


def test_preview_rotation_roundtrip_keeps_original_pixel_coordinates():
    shape = (300, 500, 3)
    corners = [[10, 20], [480, 20], [480, 270], [10, 270]]
    for turns in range(4):
        for original in corners:
            rotated = original_to_rotated(original, shape, turns)
            restored = rotated_to_original(rotated, shape, turns)
            assert restored == [float(v) for v in original]


def test_preview_fit_preserves_aspect_ratio_and_bounds():
    shape = (1800, 900, 3)
    for turns in range(4):
        left, top, w, h = preview_geometry(shape, turns, 960, 680)
        expected_ratio = (1800 / 900) if turns % 2 else (900 / 1800)
        assert abs(w / h - expected_ratio) < 0.005
        assert 0 <= left and 0 <= top
        assert left + w <= 960
        assert top + h <= 680


def test_clicks_unchanged_after_preview_rotation_and_annotation_save(tmp_path):
    shape = (300, 500, 3)
    original = [[10, 20], [480, 20], [480, 270], [10, 270]]
    rotated = [original_to_rotated(p, shape, 1) for p in original]
    restored = [rotated_to_original(p, shape, 1) for p in rotated]
    path = tmp_path / "corners.json"
    saved = save_annotation(path, "photo.jpg", restored, shape)
    assert saved == normalize_corners(original, shape)
