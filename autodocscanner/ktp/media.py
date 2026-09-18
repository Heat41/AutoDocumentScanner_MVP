import shutil
from pathlib import Path
from uuid import UUID

import numpy as np

from autodocscanner.output.safe import atomic_imwrite


def _canonical_uuid(
    value,
    name,
):
    try:
        parsed = UUID(
            str(value)
        )
    except (
        ValueError,
        TypeError,
        AttributeError,
    ):
        raise ValueError(
            f"{name} harus berupa "
            "UUID valid."
        ) from None

    return str(parsed)


def _validate_image(
    image,
    name,
):
    if (
        not isinstance(
            image,
            np.ndarray,
        )
        or image.size == 0
        or image.dtype != np.uint8
    ):
        raise ValueError(
            f"{name} harus berupa "
            "numpy uint8 yang tidak kosong."
        )

    if image.ndim == 2:
        return

    if (
        image.ndim == 3
        and image.shape[2]
        in (1, 3, 4)
    ):
        return

    raise ValueError(
        f"{name} harus berupa image "
        "grayscale/BGR/BGRA."
    )


class KtpMediaStore:
    def __init__(
        self,
        media_root,
    ):
        self.media_root = Path(
            media_root
        )

    def resolve(
        self,
        relative_path,
    ):
        relative = Path(
            str(
                relative_path
                or ""
            )
        )

        if relative.is_absolute():
            raise ValueError(
                "Path media harus relatif."
            )

        candidate = (
            self.media_root
            / relative
        ).resolve()

        root = (
            self.media_root
            .resolve()
        )

        try:
            candidate.relative_to(
                root
            )
        except ValueError:
            raise ValueError(
                "Path media keluar dari "
                "root yang diizinkan."
            ) from None

        return candidate

    def save_revision(
        self,
        record_id,
        revision_id,
        ktp_image,
        face_image=None,
    ):
        record_id = _canonical_uuid(
            record_id,
            "record_id",
        )
        revision_id = _canonical_uuid(
            revision_id,
            "revision_id",
        )

        _validate_image(
            ktp_image,
            "ktp_image",
        )

        if face_image is not None:
            _validate_image(
                face_image,
                "face_image",
            )

        record_dir = (
            self.media_root
            / record_id
        )
        revision_dir = (
            record_dir
            / revision_id
        )

        record_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        revision_dir.mkdir(
            exist_ok=False,
        )

        try:
            ktp_path = (
                revision_dir
                / "ktp.jpg"
            )
            atomic_imwrite(
                ktp_path,
                ktp_image,
            )

            face_path = None

            if face_image is not None:
                face_path = (
                    revision_dir
                    / "face.jpg"
                )
                atomic_imwrite(
                    face_path,
                    face_image,
                )

        except Exception:
            shutil.rmtree(
                revision_dir,
                ignore_errors=True,
            )
            raise

        return {
            "ktp_image_path": (
                Path(record_id)
                / revision_id
                / "ktp.jpg"
            ).as_posix(),
            "face_image_path": (
                (
                    Path(record_id)
                    / revision_id
                    / "face.jpg"
                ).as_posix()
                if face_path is not None
                else None
            ),
        }

    def cleanup_revision(
        self,
        record_id,
        revision_id,
    ):
        record_id = _canonical_uuid(
            record_id,
            "record_id",
        )
        revision_id = _canonical_uuid(
            revision_id,
            "revision_id",
        )

        revision_dir = (
            self.media_root
            / record_id
            / revision_id
        )

        shutil.rmtree(
            revision_dir,
            ignore_errors=True,
        )

        record_dir = (
            self.media_root
            / record_id
        )

        try:
            record_dir.rmdir()
        except OSError:
            pass
