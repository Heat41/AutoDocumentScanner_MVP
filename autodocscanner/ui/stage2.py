from io import BytesIO
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from autodocscanner.output.manager import build_output_path
from autodocscanner.services.ktp_output import process_ktp_output
from autodocscanner.ui.manual_document import ManualDocumentPage
from autodocscanner.ui.responsive import ResponsiveScannerUI
from autodocscanner.ui.sidebar import CollapsibleSidebar
from autodocscanner.ui.theme import apply_theme, get_theme
from autodocscanner.ui.tracking_placeholder import TrackingPlaceholderPage


class Stage2ScannerUI(ResponsiveScannerUI):
    """Stage 2 shell that adds output format without changing KTP detection."""

    DEFAULT_OUTPUT_FORMAT = "image"
    AUTO_KTP_PAGE = "auto_ktp"
    MANUAL_DOCUMENT_PAGE = "manual_document"
    TRACKING_KTP_PAGE = "tracking_ktp"
    TRACKING_RELEASE_ENABLED = False
    NAVIGATION_ITEMS = (
        {
            "icon": "▣",
            "label": "Auto Koreksi KTP",
            "page": AUTO_KTP_PAGE,
        },
        {
            "icon": "▤",
            "label": "Dokumen Manual",
            "page": MANUAL_DOCUMENT_PAGE,
        },
        {
            "icon": "🔒",
            "label": "Tracking KTP",
            "page": TRACKING_KTP_PAGE,
        },
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
        self.theme_mode = tk.StringVar(
            master=self,
            value="light",
        )
        self._tracking_ktp_page = None
        self._tracking_placeholder_page = None

        self._install_output_format_controls()
        self._install_manual_document_page()

        self._ktp_page.pack_forget()
        self._install_global_header()
        self._install_top_navigation()
        self._install_tracking_placeholder_page()
        self._ktp_page.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self._apply_app_theme()
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

    def _install_tracking_ktp_page(self):
        if not self.TRACKING_RELEASE_ENABLED:
            self._tracking_ktp_page = None
            return

        from autodocscanner.ui.tracking_ktp import TrackingKtpPage

        self._tracking_ktp_page = TrackingKtpPage(
            self,
            scanner=self.scanner,
        )

    def _install_tracking_placeholder_page(self):
        self._tracking_placeholder_page = TrackingPlaceholderPage(
            self
        )

    def _install_global_header(self):
        self._global_header = ttk.Frame(
            self,
            style="Surface.TFrame",
            padding=(16, 10),
        )
        self._global_header.pack(
            side="top",
            fill="x",
        )

        ttk.Label(
            self._global_header,
            text="AutoDocumentScanner",
            style="CardLabel.TLabel",
        ).pack(side="left")

        theme_group = ttk.Frame(
            self._global_header,
            style="Surface.TFrame",
        )
        theme_group.pack(side="right")

        ttk.Label(
            theme_group,
            text="Tema",
            style="CardMuted.TLabel",
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            theme_group,
            text="☀ Terang",
            command=lambda: self.set_theme("light"),
        ).pack(side="left")

        ttk.Button(
            theme_group,
            text="🌙 Gelap",
            command=lambda: self.set_theme("dark"),
        ).pack(side="left", padx=(6, 0))

    def _install_top_navigation(self):
        self._navigation_bar = CollapsibleSidebar(
            self,
            items=self.NAVIGATION_ITEMS,
            on_select=self._show_stage2_page,
            active_page=self._active_stage2_page,
            collapsed=False,
        )
        self._navigation_bar.pack(
            side="left",
            fill="y",
        )

    def _show_stage2_page(self, page_name):
        if page_name == self.AUTO_KTP_PAGE:
            self.show_auto_ktp()
        elif page_name == self.MANUAL_DOCUMENT_PAGE:
            self.show_manual_document()
        elif page_name == self.TRACKING_KTP_PAGE:
            self.show_tracking_ktp()

    def _update_navigation_state(self):
        sidebar = getattr(
            self,
            "_navigation_bar",
            None,
        )
        if sidebar is not None:
            sidebar.set_active(
                self._active_stage2_page
            )

    def set_theme(self, mode):
        if mode not in {"light", "dark"}:
            return

        self.theme_mode.set(mode)
        self._apply_app_theme()

    def _apply_app_theme(self):
        palette = apply_theme(
            self,
            self.theme_mode.get(),
        )

        self._apply_custom_widget_theme(palette)

        manual_page = getattr(
            self,
            "_manual_document_page",
            None,
        )
        if manual_page is not None and hasattr(
            manual_page,
            "apply_theme",
        ):
            manual_page.apply_theme(palette)

    def _apply_custom_widget_theme(self, palette):
        def visit(widget):
            try:
                klass = widget.winfo_class()
            except tk.TclError:
                return

            if klass in {"Frame", "Label"}:
                try:
                    current = str(widget.cget("bg")).upper()
                except tk.TclError:
                    current = ""

                drop_colors = {
                    "#EFF6FF",
                    "#F8FAFC",
                    "#FFFFFF",
                    "#F4F7FB",
                    "#F5F7FA",
                    "#0F172A",
                    "#111827",
                    "#1E293B",
                }

                if current in drop_colors:
                    target = (
                        palette["surface_soft"]
                        if current in {"#EFF6FF", "#F8FAFC", "#1E293B"}
                        else palette["surface"]
                    )
                    try:
                        widget.configure(bg=target)
                    except tk.TclError:
                        pass

                if klass == "Label":
                    try:
                        fg = str(widget.cget("fg")).upper()
                    except tk.TclError:
                        fg = ""

                    if fg in {
                        "#1D4ED8",
                        "#2563EB",
                        "#3B82F6",
                    }:
                        try:
                            widget.configure(
                                fg=palette["accent"]
                            )
                        except tk.TclError:
                            pass
                    elif fg in {
                        "#64748B",
                        "#94A3B8",
                        "#6D747C",
                    }:
                        try:
                            widget.configure(
                                fg=palette["muted"]
                            )
                        except tk.TclError:
                            pass
                    elif fg in {
                        "#0F172A",
                        "#111827",
                        "#172033",
                        "#252A30",
                    }:
                        try:
                            widget.configure(
                                fg=palette["text"]
                            )
                        except tk.TclError:
                            pass

            if klass == "Listbox":
                try:
                    widget.configure(
                        bg=palette["surface"],
                        fg=palette["text"],
                        selectbackground=palette["active_bg"],
                        selectforeground=palette["text"],
                    )
                except tk.TclError:
                    pass

            try:
                children = widget.winfo_children()
            except tk.TclError:
                children = []

            for child in children:
                visit(child)

        visit(self)

    def _hide_stage2_pages(self):
        self._ktp_page.pack_forget()
        self._manual_document_page.pack_forget()

        if self._tracking_ktp_page is not None:
            self._tracking_ktp_page.pack_forget()

        if self._tracking_placeholder_page is not None:
            self._tracking_placeholder_page.pack_forget()

    def show_manual_document(self):
        self._hide_stage2_pages()
        self._manual_document_page.pack(
            side="left",
            fill="both",
            expand=True,
        )
        self._active_stage2_page = self.MANUAL_DOCUMENT_PAGE
        self._update_navigation_state()
        self.title(
            "Auto Document Scanner — Koreksi Dokumen Manual"
        )

    def show_tracking_ktp(self):
        self._hide_stage2_pages()

        if not self.TRACKING_RELEASE_ENABLED:
            self._tracking_placeholder_page.pack(
                side="left",
                fill="both",
                expand=True,
            )
        else:
            if self._tracking_ktp_page is None:
                self._install_tracking_ktp_page()

            self._tracking_ktp_page.pack(
                side="left",
                fill="both",
                expand=True,
            )

        self._active_stage2_page = self.TRACKING_KTP_PAGE
        self._update_navigation_state()
        self.title(
            "Auto Document Scanner — Tracking KTP"
        )

    def show_auto_ktp(self):
        self._hide_stage2_pages()
        self._ktp_page.pack(
            side="left",
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
