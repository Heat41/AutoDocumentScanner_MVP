from pathlib import Path
from uuid import uuid4

from ktp_database import KtpDatabase
from ktp_media import KtpMediaStore
from ktp_models import (
    KtpIdentityData,
    normalize_nik,
)


class KtpStorage:
    def __init__(
        self,
        data_dir="data",
    ):
        self.data_dir = Path(
            data_dir
        )

        self.database = KtpDatabase(
            self.data_dir
            / "autoscanner.db"
        )
        self.media = KtpMediaStore(
            self.data_dir
            / "media"
            / "ktp"
        )

    def initialize(self):
        self.database.initialize()

    @staticmethod
    def _identity(identity):
        if isinstance(
            identity,
            KtpIdentityData,
        ):
            result = identity
        else:
            result = (
                KtpIdentityData
                .from_mapping(
                    identity
                )
            )

        result.validate_for_record()
        return result

    def _enrich(
        self,
        nik,
        revision,
    ):
        if revision is None:
            return None

        record = (
            self.database
            .get_record_by_nik(
                nik
            )
        )

        result = dict(
            revision
        )
        result["record_id"] = (
            record["id"]
        )
        result["revision_id"] = (
            result["id"]
        )
        return result

    def record(
        self,
        nik,
        identity,
        ktp_image,
        face_image=None,
    ):
        nik = normalize_nik(
            nik
        )
        identity = self._identity(
            identity
        )
        self.initialize()

        existing = (
            self.database
            .get_record_by_nik(
                nik
            )
        )

        record_id = (
            existing["id"]
            if existing is not None
            else str(
                uuid4()
            )
        )
        revision_id = str(
            uuid4()
        )

        media_paths = (
            self.media
            .save_revision(
                record_id,
                revision_id,
                ktp_image,
                face_image,
            )
        )

        try:
            result = (
                self.database
                .record_revision(
                    record_id=(
                        record_id
                    ),
                    revision_id=(
                        revision_id
                    ),
                    nik=nik,
                    identity=identity,
                    ktp_image_path=(
                        media_paths[
                            "ktp_image_path"
                        ]
                    ),
                    face_image_path=(
                        media_paths[
                            "face_image_path"
                        ]
                    ),
                )
            )

        except Exception:
            self.media.cleanup_revision(
                record_id,
                revision_id,
            )
            raise

        result = dict(
            result
        )
        result["revision_id"] = (
            result["id"]
        )
        return result

    def get_current(
        self,
        nik,
    ):
        nik = normalize_nik(
            nik
        )
        self.initialize()

        return self._enrich(
            nik,
            self.database
            .get_current_revision(
                nik
            ),
        )

    def list_revisions(
        self,
        nik,
    ):
        nik = normalize_nik(
            nik
        )
        self.initialize()

        record = (
            self.database
            .get_record_by_nik(
                nik
            )
        )

        if record is None:
            return []

        revisions = (
            self.database
            .list_revisions(
                nik
            )
        )

        result = []

        for revision in revisions:
            item = dict(
                revision
            )
            item["record_id"] = (
                record["id"]
            )
            item["revision_id"] = (
                item["id"]
            )
            result.append(
                item
            )

        return result
