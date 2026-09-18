import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ktp_models import (
    KtpIdentityData,
    normalize_nik,
)


SCHEMA_VERSION = 1

IDENTITY_COLUMNS = (
    "nama",
    "tempat_lahir",
    "tanggal_lahir",
    "jenis_kelamin",
    "alamat",
    "rt",
    "rw",
    "kelurahan_desa",
    "kecamatan",
    "kabupaten_kota",
    "provinsi",
    "agama",
    "status_perkawinan",
    "pekerjaan",
    "kewarganegaraan",
    "golongan_darah",
    "berlaku_hingga",
    "jenis_wilayah",
)


def _utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


class KtpDatabase:
    def __init__(self, db_path):
        self.db_path = Path(db_path)

    def _connect(self):
        connection = sqlite3.connect(
            self.db_path
        )
        connection.row_factory = (
            sqlite3.Row
        )
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )
        connection.execute(
            "PRAGMA busy_timeout = 5000"
        )
        return connection

    def initialize(self):
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ktp_records (
                    id TEXT PRIMARY KEY,
                    nik TEXT UNIQUE NOT NULL,
                    current_revision_id TEXT,
                    current_revision_number INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ktp_revisions (
                    id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    revision_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    nama TEXT NOT NULL,
                    tempat_lahir TEXT NOT NULL,
                    tanggal_lahir TEXT NOT NULL,
                    jenis_kelamin TEXT NOT NULL,
                    alamat TEXT NOT NULL,
                    rt TEXT NOT NULL,
                    rw TEXT NOT NULL,
                    kelurahan_desa TEXT NOT NULL,
                    kecamatan TEXT NOT NULL,
                    kabupaten_kota TEXT NOT NULL,
                    provinsi TEXT NOT NULL,
                    agama TEXT NOT NULL,
                    status_perkawinan TEXT NOT NULL,
                    pekerjaan TEXT NOT NULL,
                    kewarganegaraan TEXT NOT NULL,
                    golongan_darah TEXT,
                    berlaku_hingga TEXT,
                    jenis_wilayah TEXT,
                    ktp_image_path TEXT NOT NULL,
                    face_image_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(record_id)
                        REFERENCES ktp_records(id)
                        ON DELETE CASCADE,
                    UNIQUE(
                        record_id,
                        revision_number
                    )
                );

                CREATE INDEX IF NOT EXISTS
                    idx_ktp_revisions_record
                ON ktp_revisions(
                    record_id,
                    revision_number
                );
                """
            )

            existing = connection.execute(
                """
                SELECT value
                FROM schema_meta
                WHERE key = 'schema_version'
                """
            ).fetchone()

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO schema_meta(
                        key,
                        value
                    )
                    VALUES(
                        'schema_version',
                        ?
                    )
                    """,
                    (
                        str(
                            SCHEMA_VERSION
                        ),
                    ),
                )
            elif (
                int(existing["value"])
                != SCHEMA_VERSION
            ):
                raise RuntimeError(
                    "Versi schema "
                    "tidak didukung: "
                    f"{existing['value']}."
                )

    def schema_version(self):
        self.initialize()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT value
                FROM schema_meta
                WHERE key = 'schema_version'
                """
            ).fetchone()

        return int(
            row["value"]
        )

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
                .from_mapping(identity)
            )

        result.validate_for_record()
        return result

    def get_record_by_nik(self, nik):
        nik = normalize_nik(nik)
        self.initialize()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM ktp_records
                WHERE nik = ?
                """,
                (nik,),
            ).fetchone()

        return (
            dict(row)
            if row
            else None
        )

    def get_current_revision(
        self,
        nik,
    ):
        nik = normalize_nik(nik)
        self.initialize()

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT rev.*
                FROM ktp_records record
                JOIN ktp_revisions rev
                  ON rev.id =
                     record.current_revision_id
                WHERE record.nik = ?
                """,
                (nik,),
            ).fetchone()

        return (
            dict(row)
            if row
            else None
        )

    def list_revisions(self, nik):
        nik = normalize_nik(nik)
        self.initialize()

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT rev.*
                FROM ktp_records record
                JOIN ktp_revisions rev
                  ON rev.record_id =
                     record.id
                WHERE record.nik = ?
                ORDER BY
                    rev.revision_number ASC
                """,
                (nik,),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def record_revision(
        self,
        record_id,
        revision_id,
        nik,
        identity,
        ktp_image_path,
        face_image_path=None,
    ):
        nik = normalize_nik(nik)
        identity = self._identity(
            identity
        )

        ktp_image_path = str(
            ktp_image_path or ""
        ).strip()

        if not ktp_image_path:
            raise ValueError(
                "Path gambar KTP wajib diisi."
            )

        face_image_path = (
            str(
                face_image_path
            ).strip()
            if face_image_path
            else None
        )

        record_id = str(
            record_id or ""
        ).strip()
        revision_id = str(
            revision_id or ""
        ).strip()

        if (
            not record_id
            or not revision_id
        ):
            raise ValueError(
                "record_id dan "
                "revision_id wajib diisi."
            )

        self.initialize()
        now = _utc_now()
        connection = self._connect()

        try:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            existing = connection.execute(
                """
                SELECT *
                FROM ktp_records
                WHERE nik = ?
                """,
                (nik,),
            ).fetchone()

            if existing is None:
                active_record_id = (
                    record_id
                )

                connection.execute(
                    """
                    INSERT INTO ktp_records(
                        id,
                        nik,
                        current_revision_id,
                        current_revision_number,
                        created_at,
                        updated_at
                    )
                    VALUES(
                        ?,
                        ?,
                        NULL,
                        0,
                        ?,
                        ?
                    )
                    """,
                    (
                        active_record_id,
                        nik,
                        now,
                        now,
                    ),
                )
                revision_number = 1

            else:
                active_record_id = (
                    existing["id"]
                )
                revision_number = (
                    int(
                        existing[
                            "current_revision_number"
                        ]
                    )
                    + 1
                )

                connection.execute(
                    """
                    UPDATE ktp_revisions
                    SET status =
                        'SUPERSEDED'
                    WHERE record_id = ?
                      AND status =
                          'ACTIVE'
                    """,
                    (
                        active_record_id,
                    ),
                )

            values = (
                identity.as_db_dict()
            )
            columns = ", ".join(
                IDENTITY_COLUMNS
            )
            placeholders = ", ".join(
                "?"
                for _ in IDENTITY_COLUMNS
            )

            connection.execute(
                f"""
                INSERT INTO ktp_revisions(
                    id,
                    record_id,
                    revision_number,
                    status,
                    {columns},
                    ktp_image_path,
                    face_image_path,
                    created_at
                )
                VALUES(
                    ?,
                    ?,
                    ?,
                    'ACTIVE',
                    {placeholders},
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    revision_id,
                    active_record_id,
                    revision_number,
                    *(
                        values[name]
                        for name
                        in IDENTITY_COLUMNS
                    ),
                    ktp_image_path,
                    face_image_path,
                    now,
                ),
            )

            connection.execute(
                """
                UPDATE ktp_records
                SET current_revision_id = ?,
                    current_revision_number = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    revision_id,
                    revision_number,
                    now,
                    active_record_id,
                ),
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        result = (
            self.get_current_revision(
                nik
            )
        )
        result["record_id"] = (
            active_record_id
        )
        return result
