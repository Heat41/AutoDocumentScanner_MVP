import os
from pathlib import Path
from uuid import uuid4

import cv2


def atomic_imwrite(output_path, image):
    """Simpan gambar secara atomik tanpa meninggalkan file target setengah jadi.

    Gambar ditulis lebih dulu ke file sementara pada folder yang sama. Target
    baru diganti setelah ``cv2.imwrite`` sukses. Jika proses gagal, file target
    lama (bila ada) tetap utuh dan file sementara dibersihkan.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = output_path.suffix or ".jpg"
    temp_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp{suffix}"
    )

    try:
        if not cv2.imwrite(
            str(temp_path),
            image,
        ):
            raise RuntimeError(
                f"Gagal menulis file sementara: {output_path}"
            )

        os.replace(
            temp_path,
            output_path,
        )

    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass

    return output_path
