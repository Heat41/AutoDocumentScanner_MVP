from pathlib import Path

from autodocscanner.ktp.tracking import (
    extract_tracking_data,
)


def process_tracking_input(
    scanner,
    input_path,
    backend=None,
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

    tracking = extract_tracking_data(
        corrected_image,
        backend=backend,
    )

    return {
        "corrected_image": (
            corrected_image
        ),
        "corners": corners,
        "tracking": tracking,
    }



def tracking_process_entry(
    corrected_image,
    result_queue,
):
    """
    Entry point process terisolasi untuk OCR/Tracking KTP.

    Native crash pada OpenCV/Tesseract hanya akan menghentikan process ini,
    bukan proses Tkinter utama.
    """
    try:
        tracking = extract_tracking_data(
            corrected_image
        )
        result_queue.put(
            (
                "success",
                tracking,
            )
        )
    except BaseException as exc:
        result_queue.put(
            (
                "failure",
                f"{type(exc).__name__}: {exc}",
            )
        )
