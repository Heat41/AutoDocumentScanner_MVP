from io import BytesIO
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from output_manager import build_output_path
from stage2_processing import process_ktp_output
from ui_manual_document import ManualDocumentPage
from ui_responsive import ResponsiveScannerUI


class Stage2ScannerUI(ResponsiveScannerUI):
    """Stage 2 shell that adds output format without changing KTP detection."""

    DEFAULT_OUTPUT_FORMAT = "image"
    AUTO_KTP_PAGE = "auto_ktp"
    MANUAL_DOCUMENT_PAGE = "manual_document"
    NAVIGATION_ITEMS = (
        ("Auto Koreksi KTP", AUTO_KTP_PAGE),
        ("Koreksi Dokumen Manual", MANUAL_DOCUMENT_PAGE),
    )

    def __init__(self):
        super().__init__()

        self._ktp_page = self._locate_ktp_page()
        self._active_stage2_page = self.AUTO_KTP_PAGE

        self.output_format = tk.StringVar(
            master=self,
            value=self.DEFAULT_OUTPUT_FORMAT,
        )
        self._active_output_format = (
            self.DEFAULT_OUTPUT_FORMAT
        )
        self._pdf_preview_by_output = {}
        self._install_output_format_controls()
        self._install_manual_document_page()
        self._install_top_navigation()
        self._update_navigation_state()

    def _locate_ktp_page(self):
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                return child

        raise RuntimeError(
            "Root halaman Auto Koreksi KTP tidak ditemukan."
        )

    def _install_manual_document_page(self):
        self._manual_document_page = ManualDocumentPage(
            self,
            on_back=None,
            output_dir=self.output_dir,
        )

    def _install_top_navigation(self):
        self._navigation_bar = ttk.Frame(
            self,
            padding=(22, 12, 22, 0),
        )
        self._navigation_bar.pack(
            side="top",
            fill="x",
            before=self._ktp_page,
        )

        nav_card = ttk.Frame(
            self._navigation_bar,
            style="Card.TFrame",
            padding=(10, 8),
        )
        nav_card.pack(fill="x")

        ttk.Label(
            nav_card,
            text="Navigasi",
            style="CardMuted.TLabel",
        ).pack(
            side="left",
            padx=(2, 14),
        )

        self._navigation_buttons = {}

        for label, page_name in self.NAVIGATION_ITEMS:
            command = (
                self.show_auto_ktp
                if page_name == self.AUTO_KTP_PAGE
                else self.show_manual_document
            )

            button = ttk.Button(
                nav_card,
                text=label,
                command=command,
            )
            button.pack(
                side="left",
                padx=(0, 8),
            )
            self._navigation_buttons[
                page_name
            ] = button

    def _update_navigation_state(self):
        buttons = getattr(
            self,
            "_navigation_buttons",
            {},
        )

        for page_name, button in buttons.items():
            if (
                page_name
                == self._active_stage2_page
            ):
                button.configure(
                    style="Accent.TButton"
                )
            else:
                button.configure(
                    style="TButton"
                )

    def show_manual_document(self):
        self._ktp_page.pack_forget()
        self._manual_document_page.pack(
            fill="both",
            expand=True,
        )
        self._active_stage2_page = self.MANUAL_DOCUMENT_PAGE
        self._update_navigation_state()
        self.title(
            "Auto Document Scanner — Koreksi Dokumen Manual"
        )

    def show_auto_ktp(self):
        self._manual_document_page.pack_forget()
        self._ktp_page.pack(
            fill="both",
            expand=True,
        )
        self._active_stage2_page = self.AUTO_KTP_PAGE
        self._update_navigation_state()
        self.title("Auto Document Scanner")

    def _refresh_resized_preview(self):
        if (
            getattr(
                self,
                "_active_stage2_page",
                self.AUTO_KTP_PAGE,
            )
            != self.AUTO_KTP_PAGE
        ):
            self._responsive_after_id = None
            return

        return super()._refresh_resized_preview()

    @staticmethod
    def _is_pdf_path(path):
        return Path(path).suffix.lower() == ".pdf"

    def _install_output_format_controls(self):
        footer = self.process_button.master

        panel = ttk.Frame(footer)
        panel.pack(
            side="right",
            before=self.process_button,
            padx=(0, 4),
        )

        ttk.Label(
            panel,
            text="Format",
            style="Subheader.TLabel",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        ttk.Radiobutton(
            panel,
            text="Gambar",
            variable=self.output_format,
            value="image",
        ).pack(side="left")

        ttk.Radiobutton(
            panel,
            text="PDF",
            variable=self.output_format,
            value="pdf",
        ).pack(
            side="left",
            padx=(8, 0),
        )

        self._format_panel = panel

    def process_files(self):
        self._active_output_format = (
            self.output_format.get()
        )
        return super().process_files()

    def _set_files(self, paths):
        self._pdf_preview_by_output.clear()
        super()._set_files(paths)

    def _show_result(self, path):
        path = Path(path)

        if not self._is_pdf_path(path):
            return super()._show_result(path)

        preview_bytes = (
            self._pdf_preview_by_output.get(
                str(path)
            )
        )

        if not preview_bytes:
            self._result_photo = None
            self.result_label.configure(
                image="",
                text=(
                    "PDF tersimpan\n"
                    f"{path.name}"
                ),
                justify="center",
            )
            return

        try:
            self._result_photo = (
                self._load_preview(
                    BytesIO(preview_bytes)
                )
            )
            self.result_label.configure(
                image=self._result_photo,
                text="",
            )
        except Exception:
            self._result_photo = None
            self.result_label.configure(
                image="",
                text=(
                    "PDF berhasil dibuat, "
                    "preview tidak tersedia.\n"
                    f"{path.name}"
                ),
                justify="center",
            )

    def _process_worker(self, output_mode):
        success = 0
        failed = 0
        errors = []
        output_format = (
            self._active_output_format
        )

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
                    f"Memproses {index}/"
                    f"{len(self.files)}: "
                    f"{input_path.name}"
                ),
            )

            key = str(input_path)
            attempted_output_path = (
                build_output_path(
                    self.output_dir,
                    input_path,
                    output_format,
                )
            )

            try:
                processed = process_ktp_output(
                    self.scanner,
                    input_path,
                    self.output_dir,
                    output_mode=output_mode,
                    output_format=output_format,
                )

                output_path = (
                    processed["output_path"]
                )
                corners = processed["corners"]
                preview_bytes = (
                    processed["preview_bytes"]
                )

                self._failure_by_file.pop(
                    key,
                    None,
                )
                self._corners_by_file[key] = (
                    corners.copy()
                )
                self._metadata_by_file[key] = dict(
                    self.scanner.last_detection
                    or {}
                )
                self._output_by_file[key] = (
                    output_path
                )

                if preview_bytes:
                    self._pdf_preview_by_output[
                        str(output_path)
                    ] = preview_bytes
                else:
                    self._pdf_preview_by_output.pop(
                        str(output_path),
                        None,
                    )

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

                if (
                    quality
                    and "quality" not in metadata
                ):
                    metadata["quality"] = quality
                if (
                    validation
                    and "validation"
                    not in metadata
                ):
                    metadata["validation"] = (
                        validation
                    )

                corners = (
                    self.scanner.last_corners
                )
                if corners is not None:
                    try:
                        self._corners_by_file[
                            key
                        ] = corners.copy()
                    except Exception:
                        pass

                self._metadata_by_file[
                    key
                ] = metadata
                self._output_by_file.pop(
                    key,
                    None,
                )
                self._pdf_preview_by_output.pop(
                    str(attempted_output_path),
                    None,
                )

                failure = {
                    "message": str(exc),
                    "metadata": metadata,
                }
                self._failure_by_file[
                    key
                ] = failure

                _, detail = (
                    self._failure_summary(
                        failure
                    )
                )
                errors.append(
                    f"{input_path.name}: "
                    f"{detail}"
                )

        def finish():
            self.process_button.configure(
                state="normal"
            )

            selection = (
                self.listbox.curselection()
            )
            if selection:
                self._show_selected(
                    selection[0]
                )
            elif self.files:
                self._show_selected(
                    len(self.files) - 1
                )

            if failed == 0:
                self.status_text.set(
                    "Selesai — "
                    f"berhasil {success}, "
                    "gagal 0."
                )
            elif (
                success == 0
                and len(self.files) == 1
            ):
                pass
            else:
                self.status_text.set(
                    "Selesai — "
                    f"berhasil {success}, "
                    f"gagal {failed}."
                )

            if failed:
                detail = "\n".join(
                    errors[:5]
                )
                if len(errors) > 5:
                    detail += (
                        "\n... dan "
                        f"{len(errors) - 5} "
                        "lainnya."
                    )

                messagebox.showwarning(
                    "Proses selesai",
                    (
                        f"Berhasil: {success}\n"
                        f"Gagal: {failed}\n\n"
                        f"{detail}"
                    ),
                )

        self.after(
            0,
            finish,
        )


def main():
    app = Stage2ScannerUI()
    app.mainloop()


if __name__ == "__main__":
    main()
