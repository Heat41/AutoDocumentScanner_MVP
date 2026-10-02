from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk

try:
    import windnd
except ImportError:
    windnd = None

from autodocscanner.ui.safe import SafeScannerUI


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

    @staticmethod
    def _responsive_window_size(screen_width, screen_height):
        """Keep the initial window inside the usable area on smaller PCs.

        Tk geometry describes the client area while Windows still needs room
        for the title bar/taskbar. Reserving vertical space prevents the
        bottom action footer from ending up below the visible desktop on
        common 1366x768 displays and non-default display scaling.
        """
        screen_width = max(int(screen_width or 0), 1)
        screen_height = max(int(screen_height or 0), 1)

        width = min(1180, max(900, screen_width - 80))
        height = min(760, max(620, screen_height - 120))
        return width, height

    def _configure_style(self):
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.configure(bg=self.BG)
        self.title("Auto Document Scanner")

        width, height = self._responsive_window_size(
            self.winfo_screenwidth(),
            self.winfo_screenheight(),
        )
        self.geometry(f"{width}x{height}")
        self.minsize(
            min(1000, width),
            min(620, height),
        )

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
        self._loading_window = None
        self._loading_progress = None
        self._loading_detail_var = None
        self._loading_watch_id = None
        self._processing_active = False
        self._drop_queue = queue.Queue()
        self._drop_poll_id = None

        root = ttk.Frame(self, padding=(28, 24, 28, 20))
        root.pack(fill="both", expand=True)
        self._page_root = root

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 14))

        title_column = ttk.Frame(header)
        title_column.pack(side="left", fill="x", expand=True)

        ttk.Label(
            title_column,
            text="WORKSPACE KTP",
            style="Eyebrow.TLabel",
        ).pack(anchor="w", pady=(0, 3))

        ttk.Label(
            title_column,
            text="Auto Koreksi KTP",
            style="Header.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            title_column,
            text=(
                "Koreksi perspektif KTP secara otomatis, "
                "kemudian simpan sebagai gambar atau PDF."
            ),
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self._drop_zone = tk.Frame(
            root,
            bg="#EFF6FF",
            highlightbackground="#BFDBFE",
            highlightthickness=1,
            bd=0,
            padx=20,
            pady=18,
        )
        self._drop_zone.pack(fill="x", pady=(6, 16))

        drop_text = tk.Frame(
            self._drop_zone,
            bg="#EFF6FF",
        )
        drop_text.pack(side="left", fill="x", expand=True)

        self._drop_zone_title = tk.Label(
            drop_text,
            text="⇧  Upload Foto KTP",
            bg="#EFF6FF",
            fg="#1D4ED8",
            font=("Segoe UI Semibold", 11),
        )
        self._drop_zone_title.pack(anchor="w")

        tk.Label(
            drop_text,
            text=(
                "Tarik & lepas file di area ini, atau gunakan tombol di kanan.  "
                "JPG • JPEG • PNG • BMP • WEBP"
            ),
            bg="#EFF6FF",
            fg="#64748B",
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))

        drop_actions = tk.Frame(
            self._drop_zone,
            bg="#EFF6FF",
        )
        drop_actions.pack(side="right", padx=(16, 0))

        ttk.Button(
            drop_actions,
            text="Pilih Foto",
            style="Secondary.TButton",
            command=self.choose_files,
        ).pack(side="left")

        ttk.Button(
            drop_actions,
            text="Pilih Folder",
            style="Quiet.TButton",
            command=self.choose_folder,
        ).pack(side="left", padx=(8, 0))

        self._drop_zone.bind(
            "<Button-1>",
            lambda _event: self.choose_files(),
        )
        self._drop_zone_title.bind(
            "<Button-1>",
            lambda _event: self.choose_files(),
        )
        self.after_idle(self._install_file_drop)
        self.after_idle(self._start_drop_queue_poll)

        footer = ttk.Frame(root)
        footer.pack(side="bottom", fill="x", pady=(14, 0))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        files_panel = ttk.Frame(
            body,
            style="Card.TFrame",
            padding=14,
            width=230,
        )
        files_panel.pack(side="left", fill="y")
        files_panel.pack_propagate(False)
        self._files_panel = files_panel

        files_header = ttk.Frame(
            files_panel,
            style="Surface.TFrame",
        )
        files_header.pack(fill="x", pady=(0, 10))

        ttk.Label(
            files_header,
            text="Foto Dipilih",
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            files_header,
            textvariable=self.batch_text,
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        list_frame = tk.Frame(
            files_panel,
            bg=self.CARD,
            highlightthickness=0,
        )
        list_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
        )
        self._files_scrollbar = scrollbar

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
        self.listbox.pack(
            side="left",
            fill="both",
            expand=True,
            padx=4,
            pady=4,
        )
        scrollbar.configure(command=self.listbox.yview)
        self.listbox.bind(
            "<<ListboxSelect>>",
            self._on_select,
        )

        file_actions = ttk.Frame(
            files_panel,
            style="Surface.TFrame",
        )
        file_actions.pack(fill="x", pady=(10, 0))

        self.clear_button = ttk.Button(
            file_actions,
            text="Kosongkan",
            style="Quiet.TButton",
            command=self.clear_files,
            state="disabled",
        )
        self.clear_button.pack(side="left")

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
            text="FILE AKTIF",
            style="Eyebrow.TLabel",
        ).pack(side="left")

        ttk.Label(
            selected_row,
            textvariable=self.selected_text,
            style="Subheader.TLabel",
        ).pack(side="left", padx=(10, 0))

        previews = ttk.Frame(workspace)
        previews.pack(fill="both", expand=True)
        self._preview_parent = previews

        self.original_label = self._preview_card(
            previews,
            "Foto Asli + Sudut",
        )
        self.result_label = self._preview_card(
            previews,
            "Hasil Koreksi",
        )

        info = ttk.Frame(footer)
        info.pack(
            side="left",
            fill="x",
            expand=True,
        )

        status_row = ttk.Frame(info)
        status_row.pack(anchor="w", fill="x")

        ttk.Label(
            status_row,
            text="STATUS",
            style="Badge.TLabel",
        ).pack(side="left")

        ttk.Label(
            status_row,
            textvariable=self.status_text,
            style="Subheader.TLabel",
        ).pack(side="left", padx=(8, 0))

        quality_row = ttk.Frame(info)
        quality_row.pack(
            anchor="w",
            fill="x",
            pady=(4, 0),
        )

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

        self._action_panel = ttk.Frame(footer)
        self._action_panel.pack(side="right", padx=(16, 0))

        self.save_button = ttk.Button(
            self._action_panel,
            text="Simpan",
            style="Secondary.TButton",
            command=self.choose_output,
            state="disabled",
            width=11,
        )
        self.save_button.pack(side="left")

        self.process_button = ttk.Button(
            self._action_panel,
            text="Proses Otomatis",
            style="Accent.TButton",
            command=self.process_files,
            state="disabled",
            width=16,
        )
        self.process_button.pack(side="left", padx=(8, 0))

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
                "Area deteksi sudut KTP"
                if title.startswith("Foto Asli")
                else "Hasil koreksi perspektif"
            ),
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(2, 9))

        label = tk.Label(
            frame,
            text=(
                "Belum ada foto KTP\n"
                "Pilih atau tarik foto untuk memulai"
                if title.startswith("Foto Asli")
                else
                "Belum ada hasil\n"
                "Jalankan Proses Otomatis terlebih dahulu"
            ),
            bg="#ECEFF1",
            fg="#737A82",
            bd=0,
            font=("Segoe UI", 9),
        )
        label.pack(fill="both", expand=True)
        return label

    @staticmethod
    def _ktp_mode_label(mode):
        return {
            "color": "Warna",
            "grayscale": "Grayscale",
            "bw": "B&W",
        }.get(mode, "Warna")

    def _choose_ktp_processing_mode(self):
        try:
            from autodocscanner.ui.theme import get_theme
            mode_var = getattr(self, "theme_mode", None)
            theme_name = (
                mode_var.get()
                if mode_var is not None
                else "light"
            )
            palette = get_theme(theme_name)
        except Exception:
            palette = {
                "surface": "#FFFFFF",
                "surface_soft": "#F8FAFC",
                "text": "#172033",
                "muted": "#64748B",
                "accent": "#2563EB",
            }

        current_mode = self.output_mode.get() or "color"

        dialog = tk.Toplevel(self)
        dialog.title("Pilih Mode Hasil")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        dialog.configure(bg=palette["surface"])

        selected_mode = tk.StringVar(
            master=dialog,
            value=current_mode,
        )
        result = {
            "accepted": False,
            "mode": current_mode,
        }

        container = tk.Frame(
            dialog,
            bg=palette["surface"],
            padx=26,
            pady=24,
        )
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="Pilih Mode Hasil KTP",
            bg=palette["surface"],
            fg=palette["text"],
            font=("Segoe UI Semibold", 14),
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            container,
            text=(
                "Pilih tampilan hasil sebelum proses otomatis dijalankan."
            ),
            bg=palette["surface"],
            fg=palette["muted"],
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x", pady=(4, 16))

        options = (
            (
                "color",
                "Warna",
                "Pertahankan warna asli KTP.",
            ),
            (
                "grayscale",
                "Grayscale",
                "Hasil abu-abu yang lebih ringan.",
            ),
            (
                "bw",
                "B&W",
                "Hitam-putih dengan kontras tinggi.",
            ),
        )

        for value, title, description in options:
            row = tk.Frame(
                container,
                bg=palette["surface_soft"],
                padx=12,
                pady=10,
            )
            row.pack(fill="x", pady=(0, 8))

            ttk.Radiobutton(
                row,
                text=title,
                variable=selected_mode,
                value=value,
            ).pack(anchor="w")

            tk.Label(
                row,
                text=description,
                bg=palette["surface_soft"],
                fg=palette["muted"],
                font=("Segoe UI", 8),
                anchor="w",
            ).pack(
                fill="x",
                padx=(24, 0),
                pady=(2, 0),
            )

            row.bind(
                "<Button-1>",
                lambda _event, v=value: selected_mode.set(v),
            )

        buttons = tk.Frame(
            container,
            bg=palette["surface"],
        )
        buttons.pack(fill="x", pady=(8, 0))

        def cancel():
            dialog.destroy()

        def accept():
            result["accepted"] = True
            result["mode"] = selected_mode.get()
            dialog.destroy()

        ttk.Button(
            buttons,
            text="Batal",
            style="Quiet.TButton",
            command=cancel,
        ).pack(side="right")

        ttk.Button(
            buttons,
            text="Lanjutkan",
            style="Accent.TButton",
            command=accept,
        ).pack(side="right", padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.update_idletasks()

        width = max(dialog.winfo_width(), 430)
        height = dialog.winfo_height()
        x = self.winfo_rootx() + max(
            (self.winfo_width() - width) // 2,
            0,
        )
        y = self.winfo_rooty() + max(
            (self.winfo_height() - height) // 2,
            0,
        )
        dialog.geometry(
            f"{width}x{height}+{x}+{y}"
        )

        dialog.wait_window()
        return result

    @staticmethod
    def _decode_drop_path(raw_path):
        if isinstance(raw_path, bytes):
            for encoding in ("utf-8", "mbcs"):
                try:
                    return Path(raw_path.decode(encoding))
                except (UnicodeDecodeError, LookupError):
                    continue
            return None

        try:
            return Path(str(raw_path))
        except (TypeError, ValueError):
            return None

    def _collect_dropped_images(self, raw_paths):
        supported = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
        images = []

        for raw_path in raw_paths or []:
            path = self._decode_drop_path(raw_path)
            if path is None:
                continue

            if path.is_dir():
                images.extend(
                    sorted(
                        child
                        for child in path.iterdir()
                        if (
                            child.is_file()
                            and child.suffix.lower()
                            in supported
                        )
                    )
                )
            elif (
                path.is_file()
                and path.suffix.lower() in supported
            ):
                images.append(path)

        unique = []
        seen = set()
        for path in images:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)

        return unique

    def _install_file_drop(self):
        if windnd is None:
            return

        try:
            windnd.hook_dropfiles(
                self._drop_zone,
                func=self._on_drop_files,
            )
        except Exception:
            return

    def _start_drop_queue_poll(self):
        if self._drop_poll_id is not None:
            return
        self._poll_drop_queue()

    def _poll_drop_queue(self):
        self._drop_poll_id = None

        try:
            while True:
                paths = self._drop_queue.get_nowait()
                if paths:
                    self._set_files(paths)
                else:
                    self.status_text.set(
                        "Tidak ada file gambar yang valid pada drop."
                    )
        except queue.Empty:
            pass
        except (tk.TclError, RuntimeError, OSError, ValueError) as exc:
            try:
                self.status_text.set(
                    f"Drag & drop gagal: {exc}"
                )
            except tk.TclError:
                return

        try:
            if self.winfo_exists():
                self._drop_poll_id = self.after(
                    80,
                    self._poll_drop_queue,
                )
        except tk.TclError:
            self._drop_poll_id = None

    def _on_drop_files(self, raw_paths):
        """Receive windnd payload without touching Tk from its callback."""
        try:
            paths = self._collect_dropped_images(raw_paths)
            self._drop_queue.put(paths)
        except Exception:
            # Never allow an exception inside the native drop callback to
            # unwind through the Windows message handler.
            try:
                self._drop_queue.put([])
            except Exception:
                pass

    @staticmethod
    def _batch_summary(total):
        total = int(total or 0)
        if total <= 0:
            return "Belum ada file"
        if total == 1:
            return "1 file siap diproses"
        return f"{total} file siap diproses"

    @staticmethod
    def _loading_detail(status_text, total):
        status_text = str(status_text or "").strip()
        if status_text.startswith("Memproses"):
            return status_text

        total = int(total or 0)
        if total == 1:
            return "Menyiapkan 1 file..."
        if total > 1:
            return f"Menyiapkan {total} file..."
        return "Menyiapkan proses..."

    def _show_loading_popup(self):
        self._hide_loading_popup()

        palette = {
            "bg": "#F5F7FA",
            "surface": "#FFFFFF",
            "surface_soft": "#F8FAFC",
            "text": "#172033",
            "muted": "#64748B",
            "accent": "#2563EB",
            "border": "#E2E8F0",
        }

        try:
            from autodocscanner.ui.theme import get_theme
            mode_var = getattr(self, "theme_mode", None)
            mode = (
                mode_var.get()
                if mode_var is not None
                else "light"
            )
            palette = get_theme(mode)
        except Exception:
            pass

        window = tk.Toplevel(self)
        self._loading_window = window
        window.title("Memproses")
        window.configure(bg=palette["surface"])
        window.resizable(False, False)
        window.transient(self)
        window.protocol("WM_DELETE_WINDOW", lambda: None)

        try:
            window.attributes("-topmost", True)
            window.after(
                250,
                lambda: window.attributes("-topmost", False),
            )
        except tk.TclError:
            pass

        shell = tk.Frame(
            window,
            bg=palette["surface"],
            bd=0,
            highlightthickness=0,
        )
        shell.pack(fill="both", expand=True)

        accent = tk.Frame(
            shell,
            bg=palette["accent"],
            height=4,
        )
        accent.pack(fill="x")
        accent.pack_propagate(False)

        container = tk.Frame(
            shell,
            bg=palette["surface"],
            padx=28,
            pady=24,
        )
        container.pack(fill="both", expand=True)

        header = tk.Frame(
            container,
            bg=palette["surface"],
        )
        header.pack(fill="x")

        icon = tk.Label(
            header,
            text="◌",
            bg=palette["surface_soft"],
            fg=palette["accent"],
            font=("Segoe UI Semibold", 13),
            width=2,
            height=1,
        )
        icon.pack(side="left", padx=(0, 12))

        title_block = tk.Frame(
            header,
            bg=palette["surface"],
        )
        title_block.pack(side="left", fill="x", expand=True)

        tk.Label(
            title_block,
            text="Memproses dokumen",
            bg=palette["surface"],
            fg=palette["text"],
            font=("Segoe UI Semibold", 13),
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            title_block,
            text="AutoDocumentScanner sedang menyiapkan hasil.",
            bg=palette["surface"],
            fg=palette["muted"],
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

        self._loading_detail_var = tk.StringVar(
            value=self._loading_detail(
                self.status_text.get(),
                len(self.files),
            )
        )

        detail_card = tk.Frame(
            container,
            bg=palette["surface_soft"],
            bd=0,
            padx=14,
            pady=12,
        )
        detail_card.pack(fill="x", pady=(18, 14))

        tk.Label(
            detail_card,
            textvariable=self._loading_detail_var,
            bg=palette["surface_soft"],
            fg=palette["text"],
            font=("Segoe UI Semibold", 9),
            justify="left",
            anchor="w",
            wraplength=360,
        ).pack(fill="x")

        tk.Label(
            detail_card,
            text=(
                f"{len(self.files)} file dalam antrean"
                if len(self.files) != 1
                else "1 file dalam antrean"
            ),
            bg=palette["surface_soft"],
            fg=palette["muted"],
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x", pady=(4, 0))

        self._loading_progress = ttk.Progressbar(
            container,
            mode="indeterminate",
            length=360,
        )
        self._loading_progress.pack(fill="x")
        self._loading_progress.start(10)

        footer = tk.Frame(
            container,
            bg=palette["surface"],
        )
        footer.pack(fill="x", pady=(12, 0))

        tk.Label(
            footer,
            text="Jangan tutup aplikasi saat proses berlangsung.",
            bg=palette["surface"],
            fg=palette["muted"],
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(side="left")

        tk.Label(
            footer,
            text="● Memproses",
            bg=palette["surface"],
            fg=palette["accent"],
            font=("Segoe UI Semibold", 8),
        ).pack(side="right")

        window.update_idletasks()
        popup_width = max(window.winfo_width(), 430)
        popup_height = window.winfo_height()
        x = self.winfo_rootx() + max(
            (self.winfo_width() - popup_width) // 2,
            0,
        )
        y = self.winfo_rooty() + max(
            (self.winfo_height() - popup_height) // 2,
            0,
        )
        window.geometry(
            f"{popup_width}x{popup_height}+{x}+{y}"
        )
        window.lift()

    def _hide_loading_popup(self):
        if self._loading_watch_id is not None:
            try:
                self.after_cancel(self._loading_watch_id)
            except tk.TclError:
                pass
            self._loading_watch_id = None

        if self._loading_progress is not None:
            try:
                self._loading_progress.stop()
            except tk.TclError:
                pass
            self._loading_progress = None

        if self._loading_window is not None:
            try:
                if self._loading_window.winfo_exists():
                    self._loading_window.destroy()
            except tk.TclError:
                pass
            self._loading_window = None

        self._loading_detail_var = None

    def _watch_loading_popup(self):
        window = self._loading_window
        if window is None:
            return

        try:
            if not window.winfo_exists():
                self._loading_window = None
                return

            if self._loading_detail_var is not None:
                self._loading_detail_var.set(
                    self._loading_detail(
                        self.status_text.get(),
                        len(self.files),
                    )
                )

            if str(self.process_button.cget("state")) == "normal":
                self._hide_loading_popup()
                return

            self._loading_watch_id = self.after(
                120,
                self._watch_loading_popup,
            )
        except tk.TclError:
            self._hide_loading_popup()

    def _refresh_file_scrollbar(self):
        scrollbar = getattr(self, "_files_scrollbar", None)
        if scrollbar is None:
            return

        try:
            if len(self.files) > 8:
                if not scrollbar.winfo_ismapped():
                    scrollbar.pack(side="right", fill="y")
            elif scrollbar.winfo_ismapped():
                scrollbar.pack_forget()
        except tk.TclError:
            pass

    def _refresh_ktp_action_state(self):
        has_files = bool(self.files)
        has_outputs = bool(
            getattr(self, "_processed_image_paths", lambda: [])()
        )

        clear_button = getattr(self, "clear_button", None)
        if clear_button is not None:
            clear_button.configure(
                state="normal" if has_files else "disabled"
            )

        process_button = getattr(self, "process_button", None)
        if process_button is not None:
            process_button.configure(
                state=(
                    "disabled"
                    if self._processing_active or not has_files
                    else "normal"
                )
            )

        save_button = getattr(self, "save_button", None)
        if save_button is not None:
            save_button.configure(
                state="normal" if has_outputs else "disabled"
            )

        pdf_button = getattr(self, "save_pdf_button", None)
        if pdf_button is not None:
            pdf_button.configure(
                state="normal" if has_outputs else "disabled"
            )

    def clear_files(self):
        super().clear_files()
        self.batch_text.set("Belum ada file")
        self.selected_text.set("Tidak ada file dipilih")
        self.original_label.configure(
            image="",
            text=(
                "Belum ada foto KTP\n"
                "Pilih atau tarik foto untuk memulai"
            ),
        )
        self.result_label.configure(
            image="",
            text=(
                "Belum ada hasil\n"
                "Jalankan Proses Otomatis terlebih dahulu"
            ),
        )
        self._refresh_file_scrollbar()
        self._refresh_ktp_action_state()

    def process_files(self):
        if not self.files:
            return super().process_files()

        choice = self._choose_ktp_processing_mode()
        if not choice["accepted"]:
            return

        selected_mode = choice["mode"]
        self.output_mode.set(selected_mode)
        self.status_text.set(
            "Mode hasil: "
            f"{self._ktp_mode_label(selected_mode)}"
        )

        self._processing_active = True
        self._refresh_ktp_action_state()
        self._show_loading_popup()

        try:
            result = super().process_files()
        except Exception:
            self._processing_active = False
            self._refresh_ktp_action_state()
            self._hide_loading_popup()
            raise

        if str(self.process_button.cget("state")) == "disabled":
            self._loading_watch_id = self.after(
                120,
                self._watch_loading_popup,
            )
        else:
            self._hide_loading_popup()

        return result

    def _set_files(self, paths):
        super()._set_files(paths)
        self.batch_text.set(
            self._batch_summary(len(self.files))
        )
        if not self.files:
            self.selected_text.set("Tidak ada file dipilih")
        self._refresh_file_scrollbar()
        self._refresh_ktp_action_state()

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
