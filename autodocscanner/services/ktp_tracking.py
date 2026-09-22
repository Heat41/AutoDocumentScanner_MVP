from pathlib import Path

from autodocscanner.ktp.tracking import (
    extract_tracking_data,
)


def correct_tracking_input(
    scanner,
    input_path,
):
    input_path = Path(
        input_path
    )

    corrected_image, corners = (
        scanner.scan(
            input_path,
            output_path=None,
            mode="ktp",
            output_mode="color",
        )
    )

    return {
        "corrected_image": (
            corrected_image
        ),
        "corners": corners,
    }


def track_corrected_ktp(
    corrected_image,
    backend=None,
):
    return extract_tracking_data(
        corrected_image,
        backend=backend,
    )


def process_tracking_input(
    scanner,
    input_path,
    backend=None,
):
    corrected = correct_tracking_input(
        scanner,
        input_path,
    )

    tracking = track_corrected_ktp(
        corrected[
            "corrected_image"
        ],
        backend=backend,
    )

    return {
        "corrected_image": (
            corrected[
                "corrected_image"
            ]
        ),
        "corners": (
            corrected[
                "corners"
            ]
        ),
        "tracking": tracking,
    }
