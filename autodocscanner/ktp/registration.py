from dataclasses import dataclass

import cv2
import numpy as np

from autodocscanner.ktp.layout import (
    KTP_VALUE_BOXES,
)


MIN_REGISTRATION_MATCHES = 3
MIN_SCALE_Y = 0.92
MAX_SCALE_Y = 1.08
MAX_OFFSET_RATIO = 0.10
MAX_MEDIAN_RESIDUAL_RATIO = 0.045


@dataclass(frozen=True)
class KtpRegistrationResult:
    image: np.ndarray
    applied: bool
    match_count: int
    scale_y: float
    offset_y: float
    median_residual: float


def _expected_y(
    field_name,
    height,
):
    box = KTP_VALUE_BOXES[
        field_name
    ]
    return (
        (
            box.y1
            + box.y2
        )
        / 2.0
        * height
    )


def register_ktp_to_template(
    image,
    anchors,
):
    if (
        not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
    ):
        raise ValueError(
            "image registration harus berupa numpy array yang tidak kosong."
        )

    height, width = (
        image.shape[:2]
    )

    pairs = []

    for field_name, anchor in (
        dict(
            anchors or {}
        ).items()
    ):
        if (
            field_name
            not in KTP_VALUE_BOXES
        ):
            continue

        try:
            _x, detected_y, confidence = (
                anchor
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if float(
            confidence
        ) < 20.0:
            continue

        pairs.append(
            (
                float(
                    detected_y
                ),
                _expected_y(
                    field_name,
                    height,
                ),
            )
        )

    if (
        len(pairs)
        < MIN_REGISTRATION_MATCHES
    ):
        return KtpRegistrationResult(
            image=image.copy(),
            applied=False,
            match_count=len(
                pairs
            ),
            scale_y=1.0,
            offset_y=0.0,
            median_residual=0.0,
        )

    source_y = np.array(
        [
            pair[0]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    target_y = np.array(
        [
            pair[1]
            for pair in pairs
        ],
        dtype=np.float64,
    )

    matrix = np.column_stack(
        (
            source_y,
            np.ones_like(
                source_y
            ),
        )
    )

    scale_y, offset_y = (
        np.linalg.lstsq(
            matrix,
            target_y,
            rcond=None,
        )[0]
    )

    predicted = (
        scale_y
        * source_y
        + offset_y
    )
    residuals = np.abs(
        predicted
        - target_y
    )
    median_residual = float(
        np.median(
            residuals
        )
    )

    if (
        scale_y
        < MIN_SCALE_Y
        or scale_y
        > MAX_SCALE_Y
        or abs(
            offset_y
        )
        > (
            height
            * MAX_OFFSET_RATIO
        )
        or median_residual
        > (
            height
            * MAX_MEDIAN_RESIDUAL_RATIO
        )
    ):
        return KtpRegistrationResult(
            image=image.copy(),
            applied=False,
            match_count=len(
                pairs
            ),
            scale_y=float(
                scale_y
            ),
            offset_y=float(
                offset_y
            ),
            median_residual=(
                median_residual
            ),
        )

    transform = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                float(
                    scale_y
                ),
                float(
                    offset_y
                ),
            ],
        ],
        dtype=np.float32,
    )

    registered = cv2.warpAffine(
        image,
        transform,
        (
            width,
            height,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=(
            cv2.BORDER_REPLICATE
        ),
    )

    return KtpRegistrationResult(
        image=registered,
        applied=True,
        match_count=len(
            pairs
        ),
        scale_y=float(
            scale_y
        ),
        offset_y=float(
            offset_y
        ),
        median_residual=(
            median_residual
        ),
    )
