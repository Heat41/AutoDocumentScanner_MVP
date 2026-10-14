"""Holdout annotations are human-provided, never detector-generated."""
import json
import pytest

from autodocscanner.core.holdout_annotation_tool_v3 import normalize_corners, save_annotation


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
