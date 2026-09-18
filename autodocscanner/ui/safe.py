from pathlib import Path
from tkinter import messagebox

from autodocscanner.ui.base import ScannerUI


class SafeScannerUI(ScannerUI):
    """UI layer dengan failure state yang eksplisit.

    Hasil lama di folder output tidak dianggap sebagai hasil run terbaru
    apabila proses untuk file tersebut gagal. Dengan begitu preview tidak
    menampilkan output stale yang dapat disalahartikan sebagai hasil sukses.
    """

    def __init__(self):
        self._failure_by_file = {}
        super().__init__()

    @staticmethod
    def _failure_summary(failure):
        failure = failure or {}
        message = str(
            failure.get("message")
            or "Proses otomatis tidak menghasilkan output yang valid."
        ).strip()

        metadata = failure.get("metadata") or {}
        validation = metadata.get("validation") or {}
        quality = metadata.get("quality") or {}

        reasons = list(validation.get("warnings") or [])
        if not reasons:
            reasons = list(quality.get("warnings") or [])

        if reasons:
            detail = "; ".join(str(item) for item in reasons[:2])
            if len(reasons) > 2:
                detail += f"; +{len(reasons) - 2} lainnya"
        else:
            detail = message

        return "Status: Ditolak", detail

    def _set_files(self, paths):
        # Failure adalah status dari run tertentu. Saat user memilih batch baru,
        # state failure lama tidak dibawa ke batch berikutnya.
        self._failure_by_file.clear()
        super()._set_files(paths)

    def _show_selected(self, index):
        if not (0 <= index < len(self.files)):
            return

        path = self.files[index]
        key = str(path)

        self._show_original(path)

        failure = self._failure_by_file.get(key)
        if failure:
            status, detail = self._failure_summary(failure)
            self.status_text.set(
                f"{status} — {detail}"
            )

            metadata = failure.get("metadata") or {}
            if not (metadata.get("quality") or {}):
                self.quality_text.set("Kualitas: Perlu ditinjau")
                self.quality_detail_text.set("")

            self._result_photo = None
            self.result_label.configure(
                image="",
                text=(
                    "Tidak ada output dari proses terbaru.\n"
                    f"{detail}"
                ),
                wraplength=330,
                justify="center",
            )
            return

        output = self._output_by_file.get(key)

        if output is None:
            default_output = (
                self.output_dir
                / f"{path.stem}_scanned.jpg"
            )
            if default_output.exists():
                output = default_output

        if output and Path(output).exists():
            self._show_result(output)
        else:
            self._result_photo = None
            self.result_label.configure(
                image="",
                text="Belum diproses",
            )

    def _process_worker(self, output_mode):
        success = 0
        failed = 0
        errors = []

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, input_path in enumerate(
            self.files,
            start=1,
        ):
            self.after(
                0,
                self.status_text.set,
                (
                    f"Memproses {index}/{len(self.files)}: "
                    f"{input_path.name}"
                ),
            )

            output_path = (
                self.output_dir
                / f"{input_path.stem}_scanned.jpg"
            )
            key = str(input_path)

            try:
                _, corners = self.scanner.scan(
                    input_path,
                    output_path,
                    mode="ktp",
                    output_mode=output_mode,
                )

                self._failure_by_file.pop(key, None)
                self._corners_by_file[key] = corners.copy()
                self._metadata_by_file[key] = dict(
                    self.scanner.last_detection
                    or {}
                )
                self._output_by_file[key] = output_path
                success += 1

            except Exception as exc:
                failed += 1

                metadata = dict(
                    self.scanner.last_detection
                    or {}
                )

                quality = dict(
                    self.scanner.last_quality
                    or {}
                )
                validation = dict(
                    self.scanner.last_validation
                    or {}
                )

                if quality and "quality" not in metadata:
                    metadata["quality"] = quality
                if validation and "validation" not in metadata:
                    metadata["validation"] = validation

                corners = self.scanner.last_corners
                if corners is not None:
                    try:
                        self._corners_by_file[key] = corners.copy()
                    except Exception:
                        pass

                self._metadata_by_file[key] = metadata
                self._output_by_file.pop(key, None)

                failure = {
                    "message": str(exc),
                    "metadata": metadata,
                }
                self._failure_by_file[key] = failure

                _, detail = self._failure_summary(failure)
                errors.append(
                    f"{input_path.name}: {detail}"
                )

        def finish():
            self.process_button.configure(state="normal")

            selection = self.listbox.curselection()
            if selection:
                self._show_selected(selection[0])
            elif self.files:
                self._show_selected(len(self.files) - 1)

            if failed == 0:
                self.status_text.set(
                    f"Selesai — berhasil {success}, gagal 0."
                )
            elif success == 0 and len(self.files) == 1:
                # Untuk satu file gagal, biarkan alasan penolakan yang sudah
                # ditampilkan _show_selected tetap terlihat pada status bar.
                pass
            else:
                self.status_text.set(
                    f"Selesai — berhasil {success}, gagal {failed}."
                )

            if failed:
                detail = "\n".join(errors[:5])
                if len(errors) > 5:
                    detail += (
                        f"\n... dan {len(errors) - 5} lainnya."
                    )

                messagebox.showwarning(
                    "Proses selesai",
                    (
                        f"Berhasil: {success}\n"
                        f"Gagal: {failed}\n\n"
                        f"{detail}"
                    ),
                )

        self.after(0, finish)


def main():
    app = SafeScannerUI()
    app.mainloop()


if __name__ == "__main__":
    main()
