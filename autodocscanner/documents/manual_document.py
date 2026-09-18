import cv2
import numpy as np

from autodocscanner.core.perspective_engine import AutoPerspectiveEngine


def _validate_image_size(width, height):
    width = int(width)
    height = int(height)
    if width < 2 or height < 2:
        raise ValueError("Ukuran gambar minimal 2x2.")
    return width, height


def initial_corners(width, height, margin_ratio=0.06):
    width, height = _validate_image_size(width, height)
    margin_ratio = float(margin_ratio)
    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError(
            "margin_ratio harus antara 0 dan kurang dari 0.5."
        )

    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))

    left = min(max(margin_x, 0), width - 1)
    top = min(max(margin_y, 0), height - 1)
    right = max(left, width - 1 - margin_x)
    bottom = max(top, height - 1 - margin_y)

    return np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float32,
    )


def clamp_corners(points, width, height):
    width, height = _validate_image_size(
        width,
        height,
    )
    points = (
        np.asarray(
            points,
            dtype=np.float32,
        )
        .reshape(4, 2)
        .copy()
    )
    points[:, 0] = np.clip(
        points[:, 0],
        0,
        width - 1,
    )
    points[:, 1] = np.clip(
        points[:, 1],
        0,
        height - 1,
    )
    return points


def correct_manual_perspective(image, points):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "Gambar dokumen tidak valid."
        )

    height, width = image.shape[:2]
    points = clamp_corners(
        points,
        width,
        height,
    )

    engine = AutoPerspectiveEngine()
    ordered = engine.order_points(points)

    area = abs(
        cv2.contourArea(
            ordered.astype(np.float32)
        )
    )
    if area < 4.0:
        raise ValueError(
            "Empat titik dokumen menghasilkan area "
            "yang terlalu kecil."
        )

    return engine.warp(
        image,
        ordered,
    )


def rotate_image_and_reset(image, direction):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "Gambar dokumen tidak valid."
        )

    direction = str(
        direction or ""
    ).strip().lower()

    if direction == "left":
        rotated = cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )
    elif direction == "right":
        rotated = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )
    else:
        raise ValueError(
            "direction harus 'left' atau 'right'."
        )

    height, width = rotated.shape[:2]
    return (
        rotated,
        initial_corners(
            width,
            height,
        ),
    )
