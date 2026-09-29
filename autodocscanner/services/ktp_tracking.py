from pathlib import Path
import os

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

    Worker memakai CPU budget konservatif agar aplikasi tetap nyaman
    dipakai offline pada laptop CPU-only.
    """
    # Tesseract memakai OpenMP. Batasi satu thread supaya satu proses OCR
    # tidak mengambil seluruh core CPU laptop.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    try:
        import cv2

        cv2.setNumThreads(1)

        try:
            cv2.ocl.setUseOpenCL(
                False
            )
        except Exception:
            pass
    except Exception:
        pass

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
