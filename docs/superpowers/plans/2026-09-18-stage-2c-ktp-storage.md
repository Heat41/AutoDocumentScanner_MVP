# Stage 2C KTP Data Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable local KTP storage foundation using SQLite plus revision-scoped media folders, enforcing one primary record per NIK while preserving every recorded revision.

**Architecture:** Store structured identity/revision metadata in `data/autoscanner.db` and corrected KTP/optional face images under `data/media/ktp/<record_uuid>/<revision_uuid>/`. The database owns the one-NIK invariant and current-revision pointer; a service layer coordinates media writes with revision commits and removes newly written media if the database write fails. UI, OCR, master wilayah, and WP Client remain outside Stage 2C.

**Tech Stack:** Python 3 standard-library `sqlite3`, `dataclasses`, `uuid`, OpenCV/NumPy, unittest.

**Spec:** `docs/superpowers/specs/2026-09-18-autodocument-scanner-stage-2-design.md`

## Global Constraints

- Existing KTP perspective engine and Stage 2A/2B behavior must remain unchanged.
- One NIK maps to exactly one primary KTP record.
- Recording the same NIK again creates the next revision; it never creates a second primary record.
- Previous revisions remain queryable.
- Corrected/full KTP image is required for a recorded revision.
- Face crop is optional and must not block recording.
- Required identity fields must be complete at the final storage boundary.
- SQLite stores relative media paths rather than image BLOBs.
- Media directory names use opaque internal UUIDs, not literal NIK values.
- No OCR, master-wilayah matching, review UI, or WP Client API work is implemented in Stage 2C.

---

### Task 1: KTP identity model and validation

**Files:**
- Create: `ktp_models.py`
- Create: `tests/test_ktp_models.py`

**Interfaces:**
- `normalize_nik(value: str) -> str`
- `KtpIdentityData.from_mapping(mapping) -> KtpIdentityData`
- `KtpIdentityData.validate_for_record() -> None`
- `KtpIdentityData.as_db_dict() -> dict`

**Required fields:**
`nama`, `tempat_lahir`, `tanggal_lahir`, `jenis_kelamin`, `alamat`, `rt`, `rw`, `kelurahan_desa`, `kecamatan`, `kabupaten_kota`, `provinsi`, `agama`, `status_perkawinan`, `pekerjaan`, `kewarganegaraan`.

**Optional fields:**
`golongan_darah`, `berlaku_hingga`, `jenis_wilayah`.

- [ ] Write failing validation tests for 16-digit NIK, required identity completeness, optional fields, and mapping round-trip.
- [ ] Run `python -m unittest tests.test_ktp_models -v` and verify RED because the module is missing.
- [ ] Implement the dataclass and validation functions.
- [ ] Verify GREEN.
- [ ] Commit.

---

### Task 2: SQLite schema and repository

**Files:**
- Create: `ktp_database.py`
- Create: `tests/test_ktp_database.py`

**Schema:**
- `schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)`
- `ktp_records(id TEXT PRIMARY KEY, nik TEXT UNIQUE NOT NULL, current_revision_id TEXT, current_revision_number INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)`
- `ktp_revisions(id TEXT PRIMARY KEY, record_id TEXT NOT NULL REFERENCES ktp_records(id), revision_number INTEGER NOT NULL, status TEXT NOT NULL, all identity columns, ktp_image_path TEXT NOT NULL, face_image_path TEXT, created_at TEXT NOT NULL, UNIQUE(record_id, revision_number))`

**Interfaces:**
- `KtpDatabase(db_path).initialize() -> None`
- `KtpDatabase.get_record_by_nik(nik) -> dict | None`
- `KtpDatabase.get_current_revision(nik) -> dict | None`
- `KtpDatabase.list_revisions(nik) -> list[dict]`
- `KtpDatabase.record_revision(record_id, revision_id, nik, identity, ktp_image_path, face_image_path=None) -> dict`

**Transaction behavior:**
- For a new NIK, insert the primary record and revision 1 in one transaction.
- For an existing NIK, reuse its existing record ID regardless of the caller-provided new-record ID.
- Mark the previous active revision `SUPERSEDED`.
- Insert the next revision as `ACTIVE`.
- Update the primary record's current revision pointer and number.
- Roll back all DB changes on error.

- [ ] Write failing tests for schema version, new record, second revision, one-NIK invariant, current revision, revision ordering, and persistence after reopening.
- [ ] Verify RED.
- [ ] Implement schema initialization and repository methods.
- [ ] Verify GREEN.
- [ ] Commit.

---

### Task 3: Revision media store

**Files:**
- Create: `ktp_media.py`
- Create: `tests/test_ktp_media.py`

**Interfaces:**
- `KtpMediaStore(media_root).save_revision(record_id, revision_id, ktp_image, face_image=None) -> dict`
- `KtpMediaStore.cleanup_revision(record_id, revision_id) -> None`
- `KtpMediaStore.resolve(relative_path) -> Path`

**Behavior:**
- Required KTP image saved as `<record_uuid>/<revision_uuid>/ktp.jpg`.
- Optional face image saved as `face.jpg`.
- Return relative POSIX paths suitable for SQLite.
- Use atomic image writes.
- Reject invalid/empty KTP image.
- Never use NIK in folder names.

- [ ] Write failing tests for required KTP media, optional face media, relative paths, UUID-only folder inputs, resolve, and cleanup.
- [ ] Verify RED.
- [ ] Implement media store.
- [ ] Verify GREEN.
- [ ] Commit.

---

### Task 4: KTP storage service

**Files:**
- Create: `ktp_storage.py`
- Create: `tests/test_ktp_storage.py`

**Interfaces:**
- `KtpStorage(data_dir="data")`
- `KtpStorage.initialize() -> None`
- `KtpStorage.record(nik, identity, ktp_image, face_image=None) -> dict`
- `KtpStorage.get_current(nik) -> dict | None`
- `KtpStorage.list_revisions(nik) -> list[dict]`

**Behavior:**
- Normalize/validate NIK and identity before writing media.
- Reuse existing record UUID for repeat NIK; otherwise generate a new UUID.
- Generate a fresh revision UUID for every recording.
- Save media first under UUID paths.
- Commit DB revision using returned relative media paths.
- If DB commit fails, remove only the newly-created revision media directory.
- Return record ID, revision ID, revision number, current data, and media paths.

- [ ] Write failing end-to-end tests for new NIK, repeat NIK revision increment, stable record ID, changed identity, optional face, media existence, and reopen persistence.
- [ ] Verify RED.
- [ ] Implement orchestrator.
- [ ] Verify GREEN.
- [ ] Commit.

---

### Task 5: Runtime data-path guard and regression

**Files:**
- Modify: `.gitignore`
- Modify: `desktop_launcher.py` only if runtime directory creation needs `data/`.
- Create/modify tests only for runtime directory behavior if needed.

**Behavior:**
- Local runtime database/media are never committed.
- Packaged/source runtime can create `data/` beside the application root.
- Existing `input/` and `output/` behavior remains intact.

- [ ] Add `data/` to `.gitignore`.
- [ ] Ensure desktop runtime setup creates `data/` without changing existing input/output handling.
- [ ] Run full project regression on Windows: `python -m unittest discover -s tests -v`.
- [ ] Verify Stage 2A KTP image/PDF and Stage 2B manual-document smoke behavior still open normally.
- [ ] Commit runtime guard changes.
