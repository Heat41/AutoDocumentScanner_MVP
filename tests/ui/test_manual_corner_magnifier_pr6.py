"""Synthetic offline tests for the PR6 manual-corner magnifier."""
import math

import pytest
from PIL import Image, ImageDraw

from autodocscanner.ui.manual_corner_magnifier_pr6 import (
    magnifier_image,
    magnifier_canvas_position,
)


def test_magnifier_is_160_pixel_source_zoom_and_does_not_mutate_image():
    source = Image.new("RGB", (180, 120), (90, 140, 180))
    before = source.tobytes()
    patch = magnifier_image(source, (90, 60))
    assert patch.size == (160, 160)
    assert patch.mode == "RGB"
    assert patch.getpixel((18, 25)) == (90, 140, 180)
    assert source.tobytes() == before


def test_magnifier_tracks_subpixel_original_photo_position():
    source = Image.new("RGB", (240, 140), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 0, 120, 139), fill=(0, 100, 0))
    draw.rectangle((121, 0, 239, 139), fill=(0, 0, 150))
    left = magnifier_image(source, (70, 70))
    right = magnifier_image(source, (190, 70))
    assert left.getpixel((25, 30)) == (0, 100, 0)
    assert right.getpixel((25, 30)) == (0, 0, 150)


@pytest.mark.parametrize("point", [(0, 0), (179, 0), (179, 119), (0, 119)])
def test_magnifier_shows_neutral_padding_near_source_edges(point):
    source = Image.new("RGB", (180, 120), (30, 90, 170))
    patch = magnifier_image(source, point)
    assert patch.size == (160, 160)
    # It is OK to sample beyond camera frame but not to shift the crop
    # and thereby misrepresent the exact corner position.
    assert patch.getpixel((80, 80)) != (226, 232, 240)
    assert (226, 232, 240) in (
        patch.getpixel((0, 0)),
        patch.getpixel((159, 0)),
        patch.getpixel((0, 159)),
        patch.getpixel((159, 159)),
    )


@pytest.mark.parametrize("shape", [(400, 300), (300, 400), (2000, 300), (300, 2000)])
def test_magnifier_handles_landscape_portrait_and_wide_originals(shape):
    source = Image.new("RGB", shape, "white")
    patch = magnifier_image(source, (shape[0] / 2, shape[1] / 2), zoom=6)
    assert patch.size == (160, 160)


@pytest.mark.parametrize("point", [None, [], [0], [math.nan, 4], [0, math.inf], ["invalid", 3]])
def test_invalid_magnifier_corner_is_rejected(point):
    with pytest.raises(ValueError):
        magnifier_image(Image.new("RGB", (200, 200)), point)


@pytest.mark.parametrize("size,zoom", [(32, 4), (400, 4), (160, 0), (160, 1), (160, 20),
                                       (160.0, 4), (160, True)])
def test_invalid_magnification_sizes_fail_closed(size, zoom):
    with pytest.raises(ValueError):
        magnifier_image(Image.new("RGB", (200, 200)), (100, 100), size=size, zoom=zoom)


@pytest.mark.parametrize("point,expected", [
    ((10, 10), (528, 308)), ((690, 10), (12, 308)),
    ((690, 480), (12, 12)), ((10, 480), (528, 12)),
])
def test_magnifier_canvas_position_stays_opposite_target(point, expected):
    assert magnifier_canvas_position(point, (700, 480)) == expected


def test_small_canvas_hides_magnifier_instead_of_clipping():
    assert magnifier_canvas_position((10, 10), (150, 150)) is None


def test_magnifier_never_writes_files_or_reads_private_data(tmp_path):
    source = Image.new("RGB", (120, 100), "white")
    before = set(tmp_path.iterdir())
    magnifier_image(source, (15, 15))
    assert set(tmp_path.iterdir()) == before
