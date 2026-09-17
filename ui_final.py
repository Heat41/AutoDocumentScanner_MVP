from pathlib import Path
import tkinter as tk
from tkinter import ttk

from ui_safe import SafeScannerUI


class FinalScannerUI(SafeScannerUI):
    """Final UI/UX shell for the scanner MVP.

    This layer only changes presentation. Detection, perspective correction,
    validation, safe output, and failure handling continue to use the locked
    scanner pipeline from the previous stages.
    """

    BG = "#F5F6F7"
    CARD = "#FFFFFF"
    TEXT = "#252A30"
    MUTED = "#6D747C"
    BORDER = "#DDE1E5"
    SOFT = "#ECEFF1"
    ACCENT = "#3E4954"

    def _configure_style(self):
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.configure(bg=self.BG)
        self.title("Auto Document Scanner")
        self.geometry("1180x760")
        self.minsize(1000, 680)

        style.configure(
            "TFrame",
            background=self.BG,
        )
        style.configure(
            "Card.TFrame",
            background=self.CARD,
        )
        style.configure(
            "Sidebar.TFrame",
            background=self.CARD,
        )
        style.configure(
            "TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Header.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 19),
        )
        style.configure(
            "Subheader.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "CardLabel.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "CardMuted.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "SidebarTitle.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "SidebarMuted.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Quality.TLabel",
            background=self.BG,
            foreground="#3F454C",
            font=("Segoe UI Semibold", 9),
        )
        style.configure(
            "TButton",
            font=("Segoe UI", 9),
            padding=(12, 8),
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(18, 10),
        )
        style.configure(
            "TRadiobutton",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Horizontal.TSeparator",
            background=self.BORDER,
        )

    def _build_ui(self):
        self.batch_text = tk.StringVar(value="Belum ada file")
        self.selected_text = tk.StringVar(value="Tidak ada file dipilih")

        root = ttk.Frame(self, padding=22)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 16))

        title_column = ttk.Frame(header)
        title_column.pack(side="left", fill="x", expand=True)

        ttk.Label(
            title_column,
            text="Auto Document Scanner",
            style="Header.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            title_column,
            text=(
                "Koreksi perspektif KTP otomatis, validasi kualitas, "
                "dan output aman."
            ),
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        mode_panel = ttk.Frame(header)
        mode_panel.pack(side="right", anchor="e")

        ttk.Label(
            mode_panel,
            text="Output",
            style="Subheader.TLabel",
        ).pack(side="left", padx=(0, 10))

        ttk.Radiobutton(
            mode_panel,
            text="Warna",
            variable=self.output_mode,
            value="color",
        ).pack(side="left")

        ttk.Radiobutton(
            mode_panel,
            text="Grayscale",
            variable=self.output_mode,
            value="grayscale",
        ).pack(side="left", padx=(10, 0))

        ttk.Separator(root, orient="horizontal").pack(
            fill="x",
            pady=(0, 16),
        )

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(
            body,
            style="Sidebar.TFrame",
            padding=15,
            width=250,
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ttk.Label(
            sidebar,
            text="Input",
            style="SidebarTitle.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            sidebar,
            textvariable=self.batch_text,
            style="SidebarMuted.TLabel",
        ).pack(anchor="w", pady=(3, 12))

        ttk.Button(
            sidebar,
            text="Pilih Foto",
            command=self.choose_files,
        ).pack(fill="x")

        ttk.Button(
            sidebar,
            text="Pilih Folder",
            command=self.choose_folder,
        ).pack(fill="x", pady=(7, 0))

        ttk.Button(
            sidebar,
            text="Folder Output",
            command=self.choose_output,
        ).pack(fill="x", pady=(7, 14))

        ttk.Separator(sidebar, orient="horizontal").pack(
            fill="x",
            pady=(0, 12),
        )

        ttk.Label(
            sidebar,
            text="Daftar File",
            style="SidebarTitle.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        list_frame = tk.Frame(
            sidebar,
            bg=self.CARD,
            highlightthickness=0,
        )
        list_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
        )
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            list_frame,
            borderwidth=0,
            highlightthickness=0,
            bg=self.CARD,
            fg=self.TEXT,
            selectbackground="#E2E6E9",
            selectforeground=self.TEXT,
            activestyle="none",
            font=("Segoe UI", 9),
            yscrollcommand=scrollbar.set,
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self.listbox.yview)
        self.listbox.bind(
            "<<ListboxSelect>>",
            self._on_select,
        )

        workspace = ttk.Frame(body)
        workspace.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(14, 0),
        )

        selected_row = ttk.Frame(workspace)
        selected_row.pack(fill="x", pady=(0, 9))

        ttk.Label(
            selected_row,
            textvariable=self.selected_text,
            style="Subheader.TLabel",
        ).pack(side="left")

        previews = ttk.Frame(workspace)
        previews.pack(fill="both", expand=True)

        self.original_label = self._preview_card(
            previews,
            "Original + Corner",
        )
        self.result_label = self._preview_card(
            previews,
            "Hasil Scanner",
        )

        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(14, 0))

        info = ttk.Frame(footer)
        info.pack(side="left", fill="x", expand=True)

        ttk.Label(
            info,
            textvariable=self.status_text,
            style="Subheader.TLabel",
        ).pack(anchor="w")

        quality_row = ttk.Frame(info)
        quality_row.pack(anchor="w", fill="x", pady=(4, 0))

        ttk.Label(
            quality_row,
            textvariable=self.quality_text,
            style="Quality.TLabel",
        ).pack(side="left")

        ttk.Label(
            quality_row,
            textvariable=self.quality_detail_text,
            style="Subheader.TLabel",
        ).pack(side="left", padx=(10, 0))

        self.process_button = ttk.Button(
            footer,
            text="Proses Otomatis",
            style="Accent.TButton",
            command=self.process_files,
        )
        self.process_button.pack(side="right", padx=(16, 0))

    def _preview_card(self, parent, title):
        frame = ttk.Frame(
            parent,
            style="Card.TFrame",
            padding=12,
        )
        frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 6)
            if title.startswith("Original")
            else (6, 0),
        )

        ttk.Label(
            frame,
            text=title,
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            frame,
            text=(
                "Area deteksi fisik"
                if title.startswith("Original")
                else "Perspective corrected"
            ),
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(2, 9))

        label = tk.Label(
            frame,
            text="Belum ada gambar",
            bg="#ECEFF1",
            fg="#737A82",
            bd=0,
            font=("Segoe UI", 9),
        )
        label.pack(fill="both", expand=True)
        return label

    @staticmethod
    def _batch_summary(total):
        total = int(total or 0)
        if total <= 0:
            return "Belum ada file"
        if total == 1:
            return "1 file siap diproses"
        return f"{total} file siap diproses"

    def _set_files(self, paths):
        super()._set_files(paths)
        self.batch_text.set(
            self._batch_summary(len(self.files))
        )

    def _show_selected(self, index):
        if 0 <= index < len(self.files):
            self.selected_text.set(
                f"File aktif: {Path(self.files[index]).name}"
            )
        else:
            self.selected_text.set("Tidak ada file dipilih")

        super()._show_selected(index)


def main():
    app = FinalScannerUI()
    app.mainloop()


if __name__ == "__main__":
    main()
