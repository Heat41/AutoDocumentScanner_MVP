from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import windnd
except ImportError:
    windnd = None

import cv2
from PIL import Image, ImageTk

from autodocscanner.documents.document_canvas import (
    canvas_to_image,
    fit_image_size,
    image_to_canvas,
)
from autodocscanner.documents.document_session import ManualDocumentSession
from autodocscanner.output.manager import save_document_images, save_pdf_pages


class ManualDocumentPage(ttk.Frame):
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    DEFAULT_OUTPUT_FORMAT = "image"
    HANDLE_RADIUS = 8
    HANDLE_HIT_RADIUS = 18
    NARROW_BREAKPOINT = 900
    COMPACT_BREAKPOINT = 1180

    def __init__(self, parent, on_back=None, output_dir="output"):
        super().__init__(parent, padding=18)
        self.on_back = on_back
        self.output_dir = Path(output_dir)
        self.session = ManualDocumentSession()
        self.current_index = None
        self.output_format = tk.StringVar(
            master=self,
            value=self.DEFAULT_OUTPUT_FORMAT,
        )
        self._page_image_modes = {}
        self._session_image_mode = None
        self.status_text = tk.StringVar(
            master=self,
            value="Pilih satu atau beberapa gambar dokumen untuk memulai.",
        )
        self.page_position_text = tk.StringVar(
            master=self,
            value="Halaman 0 / 0",
        )
        self._canvas_photo = None
        self._result_photo = None
        self._display_size = None
        self._display_offset = None
        self._active_corner = None
        self._responsive_mode = None
        self._resize_after_id = None
        self._drop_queue = queue.Queue()
        self._drop_poll_id = None
        self._build_ui()
        self.bind(
            "<Configure>",
            self._on_page_resize,
            add="+",
        )
        self.after_idle(self._apply_responsive_layout)

    @staticmethod
    def page_label(index, path, corrected):
        marker = "✓" if corrected else "•"
        return f"{int(index) + 1:02d}  {marker}  {Path(path).name}"

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 12))

        if self.on_back is not None:
            ttk.Button(
                header,
                text="← Auto Koreksi KTP",
                command=self.on_back,
            ).pack(side="left", padx=(0, 12))

        title = ttk.Frame(header)
        title.pack(side="left", fill="x", expand=True)

        ttk.Label(
            title,
            text="DOKUMEN",
            style="Eyebrow.TLabel",
        ).pack(anchor="w", pady=(0, 3))

        ttk.Label(
            title,
            text="Koreksi Dokumen Manual",
            style="Header.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            title,
            text=(
                "Geser empat titik sudut, koreksi per halaman, "
                "lalu ekspor sebagai gambar atau PDF."
            ),
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        actions = ttk.Frame(header)
        actions.pack(side="right")
        ttk.Button(
            actions,
            text="Pilih Dokumen",
            style="Secondary.TButton",
            command=self.choose_images,
        ).pack(side="left")
        self.clear_button = ttk.Button(
            actions,
            text="Kosongkan",
            style="Quiet.TButton",
            command=self.clear_pages,
            state="disabled",
        )
        self.clear_button.pack(side="left", padx=(8, 0))

        self._drop_zone = tk.Frame(
            self,
            bg="#EFF6FF",
            highlightbackground="#BFDBFE",
            highlightthickness=1,
            bd=0,
            padx=16,
            pady=13,
        )
        self._drop_zone.pack(fill="x", pady=(0, 12))

        self._drop_zone_title = tk.Label(
            self._drop_zone,
            text="⇧  Upload Dokumen",
            bg="#EFF6FF",
            fg="#1D4ED8",
            font=("Segoe UI Semibold", 10),
        )
        self._drop_zone_title.pack(side="left")

        self._drop_zone_hint = tk.Label(
            self._drop_zone,
            text="Tarik & lepas file  •  JPG • PNG • JPEG • BMP • WEBP",
            bg="#EFF6FF",
            fg="#64748B",
            font=("Segoe UI", 8),
        )
        self._drop_zone_hint.pack(side="right")

        self._drop_zone.bind(
            "<Button-1>",
            lambda _event: self.choose_images(),
        )
        self._drop_zone_title.bind(
            "<Button-1>",
            lambda _event: self.choose_images(),
        )
        self.after_idle(self._install_file_drop)
        self.after_idle(self._start_drop_queue_poll)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(
            body,
            style="Card.TFrame",
            padding=12,
            width=220,
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._document_sidebar = sidebar

        ttk.Label(
            sidebar,
            text="Halaman Dokumen",
            style="CardLabel.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            sidebar,
            text="✓ = sudah dikoreksi",
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(2, 4))
        ttk.Label(
            sidebar,
            textvariable=self.page_position_text,
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        list_wrap = tk.Frame(
            sidebar,
            bg="#FFFFFF",
            highlightthickness=0,
        )
        self._page_list_wrap = list_wrap
        list_wrap.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_wrap, orient="vertical")
        self._page_scrollbar = scrollbar

        self.page_list = tk.Listbox(
            list_wrap,
            borderwidth=0,
            highlightthickness=0,
            bg="#FFFFFF",
            fg="#273038",
            selectbackground="#DDE2E6",
            selectforeground="#273038",
            activestyle="none",
            font=("Segoe UI", 9),
            yscrollcommand=scrollbar.set,
        )
        self.page_list.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self.page_list.yview)
        self.page_list.bind("<<ListboxSelect>>", self._on_page_select)

        page_nav = ttk.Frame(sidebar, style="Surface.TFrame")
        page_nav.pack(fill="x", pady=(10, 0))
        self.previous_button = ttk.Button(
            page_nav,
            text="‹",
            style="Quiet.TButton",
            command=self.select_previous_page,
            state="disabled",
        )
        self.previous_button.pack(side="left", fill="x", expand=True)

        self.next_button = ttk.Button(
            page_nav,
            text="›",
            style="Quiet.TButton",
            command=self.select_next_page,
            state="disabled",
        )
        self.next_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(6, 0),
        )

        workspace = ttk.Frame(body)
        workspace.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(14, 0),
        )
        self._document_workspace = workspace

        previews = ttk.Frame(workspace)
        previews.pack(fill="both", expand=True)
        self._document_previews = previews

        previews.grid_rowconfigure(0, weight=1)
        previews.grid_columnconfigure(0, weight=2, uniform="document-preview")
        previews.grid_columnconfigure(1, weight=1, uniform="document-preview")

        canvas_card = ttk.Frame(
            previews,
            style="Card.TFrame",
            padding=10,
        )
        canvas_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 6),
        )
        self._canvas_card = canvas_card
        ttk.Label(
            canvas_card,
            text="Dokumen + 4 Titik",
            style="CardLabel.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(
            canvas_card,
            bg="#E9EDF0",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        result_card = ttk.Frame(
            previews,
            style="Card.TFrame",
            padding=10,
        )
        result_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(6, 0),
        )
        self._result_card = result_card
        ttk.Label(
            result_card,
            text="Preview Hasil",
            style="CardLabel.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        self.result_label = tk.Label(
            result_card,
            text="Preview belum tersedia\nTerapkan koreksi untuk melihat hasil",
            bg="#E9EDF0",
            fg="#737D85",
            bd=0,
            font=("Segoe UI", 9),
        )
        self.result_label.pack(fill="both", expand=True)

        controls = ttk.Frame(workspace)
        controls.pack(fill="x", pady=(12, 0))
        self._document_controls = controls

        self.reset_button = ttk.Button(
            controls,
            text="Reset Titik",
            style="Quiet.TButton",
            command=self.reset_current,
            state="disabled",
        )
        self.reset_button.pack(side="left")

        self.rotate_left_button = ttk.Button(
            controls,
            text="↺ Putar Kiri",
            style="Secondary.TButton",
            command=lambda: self.rotate_current("left"),
            state="disabled",
        )
        self.rotate_left_button.pack(side="left", padx=(8, 0))

        self.rotate_right_button = ttk.Button(
            controls,
            text="Putar Kanan ↻",
            style="Secondary.TButton",
            command=lambda: self.rotate_current("right"),
            state="disabled",
        )
        self.rotate_right_button.pack(side="left", padx=(8, 0))

        self.apply_button = ttk.Button(
            controls,
            text="Terapkan Koreksi",
            style="Accent.TButton",
            command=self.apply_current,
            state="disabled",
        )
        self.apply_button.pack(side="left", padx=(12, 0))

        export_group = ttk.Frame(controls)
        export_group.pack(side="right")

        self.save_images_button = ttk.Button(
            export_group,
            text="Simpan",
            style="Secondary.TButton",
            command=lambda: self.export_results("image"),
            state="disabled",
            width=11,
        )
        self.save_images_button.pack(side="left")

        self.save_pdf_button = ttk.Button(
            export_group,
            text="Simpan PDF",
            style="Secondary.TButton",
            command=lambda: self.export_results("pdf"),
            state="disabled",
            width=12,
        )
        self.save_pdf_button.pack(side="left", padx=(8, 0))

        status_row = ttk.Frame(self)
        status_row.pack(fill="x", pady=(10, 0))

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

        self._refresh_action_state()

    def _on_page_resize(self, event):
        if event.widget is not self:
            return

        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except tk.TclError:
                pass

        self._resize_after_id = self.after(
            120,
            lambda: self._apply_responsive_layout(event.width),
        )

    def _apply_responsive_layout(self, width=None):
        self._resize_after_id = None

        try:
            width = int(width or self.winfo_width())
        except (TypeError, ValueError, tk.TclError):
            return

        if width < self.NARROW_BREAKPOINT:
            mode = "narrow"
        elif width < self.COMPACT_BREAKPOINT:
            mode = "compact"
        else:
            mode = "wide"

        if mode == self._responsive_mode:
            return

        self._responsive_mode = mode

        try:
            if mode == "wide":
                self._document_sidebar.configure(width=220)
                self._pack_document_previews_horizontal(
                    canvas_weight=2,
                    result_weight=1,
                    gap=6,
                )
            elif mode == "compact":
                self._document_sidebar.configure(width=185)
                self._pack_document_previews_horizontal(
                    canvas_weight=3,
                    result_weight=2,
                    gap=5,
                )
            else:
                self._document_sidebar.configure(width=155)
                self._pack_document_previews_vertical()
        except tk.TclError:
            return

        if self.current_index is not None:
            self.after_idle(self._render_current)

    def _pack_document_previews_horizontal(
        self,
        canvas_weight,
        result_weight,
        gap,
    ):
        previews = self._document_previews
        self._canvas_card.grid_forget()
        self._result_card.grid_forget()

        previews.grid_rowconfigure(0, weight=1)
        previews.grid_rowconfigure(1, weight=0)
        previews.grid_columnconfigure(
            0,
            weight=canvas_weight,
            uniform="",
        )
        previews.grid_columnconfigure(
            1,
            weight=result_weight,
            uniform="",
        )

        self._canvas_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, gap),
        )
        self._result_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(gap, 0),
        )

    def _pack_document_previews_vertical(self):
        previews = self._document_previews
        self._canvas_card.grid_forget()
        self._result_card.grid_forget()

        previews.grid_columnconfigure(0, weight=1)
        previews.grid_columnconfigure(1, weight=0)
        previews.grid_rowconfigure(0, weight=3)
        previews.grid_rowconfigure(1, weight=2)

        self._canvas_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=0,
            pady=(0, 5),
        )
        self._result_card.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=0,
            pady=(5, 0),
        )

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
                            in self.SUPPORTED_EXTENSIONS
                        )
                    )
                )
            elif (
                path.is_file()
                and path.suffix.lower()
                in self.SUPPORTED_EXTENSIONS
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
                    self._add_image_paths(paths)
                else:
                    self.status_text.set(
                        "Tidak ada gambar dokumen yang valid pada drop."
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
        """Receive native drop payload without touching Tk directly."""
        try:
            paths = self._collect_dropped_images(raw_paths)
            self._drop_queue.put(paths)
        except Exception:
            try:
                self._drop_queue.put([])
            except Exception:
                pass

    def _refresh_page_scrollbar(self):
        scrollbar = getattr(self, "_page_scrollbar", None)
        if scrollbar is None:
            return
        try:
            if len(self.session.pages) > 8:
                if not scrollbar.winfo_ismapped():
                    scrollbar.pack(side="right", fill="y")
            elif scrollbar.winfo_ismapped():
                scrollbar.pack_forget()
        except tk.TclError:
            pass

    def _refresh_action_state(self):
        total = len(self.session.pages)
        has_page = (
            self.current_index is not None
            and 0 <= self.current_index < total
        )
        all_corrected = bool(
            total and self.session.all_corrected()
        )

        for name in (
            "reset_button",
            "rotate_left_button",
            "rotate_right_button",
            "apply_button",
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.configure(
                    state="normal" if has_page else "disabled"
                )

        apply_button = getattr(self, "apply_button", None)
        if apply_button is not None:
            is_corrected = False
            if has_page:
                try:
                    is_corrected = (
                        self.session.pages[
                            self.current_index
                        ].corrected_image
                        is not None
                    )
                except (IndexError, TypeError):
                    is_corrected = False
            apply_button.configure(
                text=(
                    "Terapkan Ulang"
                    if is_corrected
                    else "Terapkan Koreksi"
                )
            )

        clear_button = getattr(self, "clear_button", None)
        if clear_button is not None:
            clear_button.configure(
                state="normal" if total else "disabled"
            )

        previous = getattr(self, "previous_button", None)
        if previous is not None:
            previous.configure(
                state=(
                    "normal"
                    if has_page and self.current_index > 0
                    else "disabled"
                )
            )

        next_button = getattr(self, "next_button", None)
        if next_button is not None:
            next_button.configure(
                state=(
                    "normal"
                    if has_page and self.current_index < total - 1
                    else "disabled"
                )
            )

        for name in ("save_images_button", "save_pdf_button"):
            button = getattr(self, name, None)
            if button is not None:
                button.configure(
                    state="normal" if all_corrected else "disabled"
                )

        self._refresh_page_scrollbar()

    def _add_image_paths(self, paths):
        first_added = len(self.session.pages)
        failed = []

        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                failed.append(Path(path).name)
                continue
            index = self.session.add_image(
                Path(path),
                image,
            )
            if self._session_image_mode is not None:
                self._page_image_modes[
                    index
                ] = self._session_image_mode

        self._refresh_page_list()

        if len(self.session.pages) > first_added:
            self._select_page(first_added)

        if failed:
            messagebox.showwarning(
                "Dokumen",
                "File berikut tidak dapat dibaca:\n"
                + "\n".join(failed[:5]),
            )

        self.status_text.set(
            f"{len(self.session.pages)} halaman siap dikoreksi."
        )
        self._refresh_action_state()

    def choose_images(self):
        paths = filedialog.askopenfilenames(
            title="Pilih gambar dokumen",
            filetypes=[
                ("Gambar", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ],
        )
        if not paths:
            return

        self._add_image_paths(
            [Path(raw_path) for raw_path in paths]
        )

    def clear_pages(self):
        self.session = ManualDocumentSession()
        self.current_index = None
        self._active_corner = None
        self._page_image_modes = {}
        self._session_image_mode = None
        self.page_list.delete(0, tk.END)
        self.canvas.delete("all")
        self._canvas_photo = None
        self._result_photo = None
        self.result_label.configure(
            image="",
            text=(
                "Preview belum tersedia\n"
                "Terapkan koreksi untuk melihat hasil"
            ),
        )
        self._update_page_position()
        self.status_text.set("Daftar halaman dikosongkan.")
        self._refresh_action_state()

    def choose_output_dir(self):
        folder = filedialog.askdirectory(title="Pilih folder output dokumen")
        if folder:
            self.output_dir = Path(folder)
            self.status_text.set(f"Folder output: {self.output_dir}")

    def _refresh_page_list(self):
        selected = self.current_index
        self.page_list.delete(0, tk.END)

        for index, page in enumerate(self.session.pages):
            self.page_list.insert(
                tk.END,
                self.page_label(
                    index,
                    page.source_path,
                    page.corrected_image is not None,
                ),
            )

        if (
            selected is not None
            and 0 <= selected < len(self.session.pages)
        ):
            self.page_list.selection_set(selected)

        self._update_page_position()

    def _select_page(self, index):
        index = int(index)
        if not 0 <= index < len(self.session.pages):
            return

        self.current_index = index
        self.page_list.selection_clear(0, tk.END)
        self.page_list.selection_set(index)
        self.page_list.see(index)
        self._update_page_position()
        self._render_current()
        self._refresh_action_state()

    def _on_page_select(self, _event=None):
        selection = self.page_list.curselection()
        if not selection:
            return
        self.current_index = int(selection[0])
        self._update_page_position()
        self._render_current()

    def _update_page_position(self):
        total = len(self.session.pages)
        if (
            self.current_index is None
            or not 0 <= self.current_index < total
        ):
            self.page_position_text.set(f"Halaman 0 / {total}")
            return

        self.page_position_text.set(
            f"Halaman {self.current_index + 1} / {total}"
        )

    def select_previous_page(self):
        if not self.session.pages:
            return

        if self.current_index is None:
            self._select_page(0)
            return

        self._select_page(max(0, self.current_index - 1))

    def select_next_page(self):
        if not self.session.pages:
            return

        if self.current_index is None:
            self._select_page(0)
            return

        self._select_page(
            min(len(self.session.pages) - 1, self.current_index + 1)
        )

    def _on_canvas_resize(self, _event=None):
        if self.current_index is not None:
            self.after_idle(self._render_current)

    def _render_current(self):
        if self.current_index is None:
            return
        if not 0 <= self.current_index < len(self.session.pages):
            return

        page = self.session.pages[self.current_index]
        image = page.original_image
        height, width = image.shape[:2]
        canvas_width = max(self.canvas.winfo_width(), 120)
        canvas_height = max(self.canvas.winfo_height(), 120)

        display_width, display_height = fit_image_size(
            width,
            height,
            canvas_width,
            canvas_height,
            padding=12,
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )
        pil = Image.fromarray(rgb)
        pil = pil.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS,
        )
        self._canvas_photo = ImageTk.PhotoImage(pil)

        offset_x = (canvas_width - display_width) / 2.0
        offset_y = (canvas_height - display_height) / 2.0
        self._display_size = (display_width, display_height)
        self._display_offset = (offset_x, offset_y)

        self.canvas.delete("all")
        self.canvas.create_image(
            offset_x,
            offset_y,
            anchor="nw",
            image=self._canvas_photo,
        )
        self._draw_overlay()
        self._render_result(page.corrected_image)

    def _corner_canvas_points(self):
        if (
            self.current_index is None
            or self._display_size is None
            or self._display_offset is None
        ):
            return []

        page = self.session.pages[self.current_index]
        height, width = page.original_image.shape[:2]

        return [
            image_to_canvas(
                point,
                (width, height),
                self._display_size,
                self._display_offset,
            )
            for point in page.corners
        ]

    def _draw_overlay(self):
        self.canvas.delete("overlay")
        points = self._corner_canvas_points()
        if len(points) != 4:
            return

        flat = []
        for x, y in points:
            flat.extend((x, y))

        self.canvas.create_polygon(
            *flat,
            outline="#2F3A42",
            fill="",
            width=3,
            tags="overlay",
        )

        labels = ("TL", "TR", "BR", "BL")
        for index, (label, (x, y)) in enumerate(zip(labels, points)):
            radius = self.HANDLE_RADIUS
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#FFFFFF",
                outline="#26323A",
                width=2,
                tags="overlay",
            )

            is_right = index in (1, 2)
            is_bottom = index in (2, 3)

            text_x = x - 14 if is_right else x + 14
            text_y = y - 14 if is_bottom else y + 14
            anchor = "e" if is_right else "w"

            self.canvas.create_text(
                text_x,
                text_y,
                text=label,
                anchor=anchor,
                fill="#26323A",
                font=("Segoe UI Semibold", 8),
                tags="overlay",
            )

    def _nearest_corner(self, x, y):
        points = self._corner_canvas_points()
        if not points:
            return None

        distances = [
            (px - x) ** 2 + (py - y) ** 2
            for px, py in points
        ]
        index = min(range(len(distances)), key=distances.__getitem__)

        if distances[index] <= self.HANDLE_HIT_RADIUS ** 2:
            return index
        return None

    def _on_canvas_press(self, event):
        self._active_corner = self._nearest_corner(event.x, event.y)

    def _on_canvas_drag(self, event):
        if (
            self._active_corner is None
            or self.current_index is None
            or self._display_size is None
            or self._display_offset is None
        ):
            return

        page = self.session.pages[self.current_index]
        height, width = page.original_image.shape[:2]
        point = canvas_to_image(
            (event.x, event.y),
            (width, height),
            self._display_size,
            self._display_offset,
        )

        corners = page.corners.copy()
        corners[self._active_corner] = point
        self.session.set_corners(self.current_index, corners)

        self._draw_overlay()
        self._render_result(None)
        self._refresh_page_list()
        self.page_list.selection_set(self.current_index)

    def _on_canvas_release(self, _event=None):
        self._active_corner = None

    def reset_current(self):
        if self.current_index is None:
            return

        self.session.reset_corners(self.current_index)
        self._refresh_page_list()
        self.page_list.selection_set(self.current_index)
        self._render_current()
        self.status_text.set("Titik sudut dikembalikan ke posisi awal.")
        self._refresh_action_state()

    def rotate_current(self, direction):
        if self.current_index is None:
            return

        self.session.rotate(self.current_index, direction)
        self._refresh_page_list()
        self.page_list.selection_set(self.current_index)
        self._render_current()
        self.status_text.set("Halaman diputar. Titik sudut di-reset.")
        self._refresh_action_state()

    def apply_current(self):
        if self.current_index is None:
            messagebox.showinfo(
                "Koreksi Dokumen",
                "Pilih halaman terlebih dahulu.",
            )
            return

        choice = self._choose_processing_mode()
        if not choice["accepted"]:
            return

        selected_mode = choice["mode"]

        if choice["apply_all"]:
            self._session_image_mode = selected_mode
            self._page_image_modes = {
                index: selected_mode
                for index in range(len(self.session.pages))
            }
        else:
            self._page_image_modes[
                self.current_index
            ] = selected_mode

        try:
            result = self.session.apply_correction(
                self.current_index
            )
        except Exception as exc:
            messagebox.showerror(
                "Koreksi Dokumen",
                str(exc),
            )
            return

        self._render_result(result)
        self._refresh_page_list()
        self.page_list.selection_set(
            self.current_index
        )

        scope = (
            "semua halaman"
            if choice["apply_all"]
            else f"halaman {self.current_index + 1}"
        )
        self.status_text.set(
            f"Koreksi berhasil • "
            f"{self._mode_label(selected_mode)} • {scope}."
        )
        self._refresh_action_state()

    @staticmethod
    def _mode_label(mode):
        return {
            "color": "Warna",
            "grayscale": "Grayscale",
            "bw": "B&W",
        }.get(mode, "Warna")

    def _mode_for_page(self, index):
        index = int(index)
        if index in self._page_image_modes:
            return self._page_image_modes[index]
        if self._session_image_mode is not None:
            return self._session_image_mode
        return "color"

    def _apply_image_mode(self, image, mode="color"):
        if image is None:
            return None

        mode = str(mode or "color").lower()

        if mode == "color":
            return image.copy()

        if image.ndim == 2:
            gray = image.copy()
        else:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        if mode == "grayscale":
            return gray

        if mode == "bw":
            blurred = cv2.GaussianBlur(
                gray,
                (3, 3),
                0,
            )
            _, bw = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            return bw

        return image.copy()

    def _choose_processing_mode(self):
        current_mode = self._mode_for_page(
            self.current_index
        )

        dialog = tk.Toplevel(self)
        dialog.title("Pilih Mode Hasil")
        dialog.transient(self.winfo_toplevel())
        dialog.resizable(False, False)
        dialog.grab_set()

        try:
            from autodocscanner.ui.theme import get_theme
            root = self.winfo_toplevel()
            mode_var = getattr(root, "theme_mode", None)
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

        dialog.configure(bg=palette["surface"])

        result = {
            "accepted": False,
            "mode": current_mode,
            "apply_all": False,
        }

        selected_mode = tk.StringVar(
            master=dialog,
            value=current_mode,
        )
        apply_all = tk.BooleanVar(
            master=dialog,
            value=False,
        )

        container = tk.Frame(
            dialog,
            bg=palette["surface"],
            padx=26,
            pady=24,
        )
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="Pilih Mode Hasil",
            bg=palette["surface"],
            fg=palette["text"],
            font=("Segoe UI Semibold", 14),
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            container,
            text=(
                "Tentukan tampilan hasil koreksi "
                "untuk halaman dokumen ini."
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
                "Pertahankan warna asli dokumen.",
            ),
            (
                "grayscale",
                "Grayscale",
                "Hasil abu-abu yang lebih ringan.",
            ),
            (
                "bw",
                "B&W",
                "Hitam-putih dengan kontras tinggi untuk teks.",
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

            radio = ttk.Radiobutton(
                row,
                text=title,
                variable=selected_mode,
                value=value,
            )
            radio.pack(anchor="w")

            tk.Label(
                row,
                text=description,
                bg=palette["surface_soft"],
                fg=palette["muted"],
                font=("Segoe UI", 8),
                anchor="w",
            ).pack(fill="x", padx=(24, 0), pady=(2, 0))

            row.bind(
                "<Button-1>",
                lambda _event, v=value: selected_mode.set(v),
            )

        apply_row = tk.Frame(
            container,
            bg=palette["surface"],
        )
        apply_row.pack(fill="x", pady=(6, 16))

        ttk.Checkbutton(
            apply_row,
            text="Terapkan mode ini ke semua halaman",
            variable=apply_all,
        ).pack(anchor="w")

        buttons = tk.Frame(
            container,
            bg=palette["surface"],
        )
        buttons.pack(fill="x")

        def cancel():
            dialog.destroy()

        def accept():
            result["accepted"] = True
            result["mode"] = selected_mode.get()
            result["apply_all"] = bool(apply_all.get())
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

        dialog.update_idletasks()
        width = max(dialog.winfo_width(), 430)
        height = dialog.winfo_height()
        root = self.winfo_toplevel()
        x = root.winfo_rootx() + max(
            (root.winfo_width() - width) // 2,
            0,
        )
        y = root.winfo_rooty() + max(
            (root.winfo_height() - height) // 2,
            0,
        )
        dialog.geometry(
            f"{width}x{height}+{x}+{y}"
        )

        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.wait_window()

        return result

    def _render_result(self, image):
        if image is None:
            self._result_photo = None
            self.result_label.configure(
                image="",
                text=(
                    "Preview belum tersedia\n"
                    "Terapkan koreksi untuk melihat hasil"
                ),
            )
            return

        mode = (
            self._mode_for_page(self.current_index)
            if self.current_index is not None
            else "color"
        )
        rendered = self._apply_image_mode(
            image,
            mode,
        )

        if rendered.ndim == 2:
            pil = Image.fromarray(rendered)
        else:
            rgb = cv2.cvtColor(
                rendered,
                cv2.COLOR_BGR2RGB,
            )
            pil = Image.fromarray(rgb)

        try:
            self.update_idletasks()
            target_width = max(
                int(self._result_card.winfo_width()) - 28,
                180,
            )
            target_height = max(
                int(self._result_card.winfo_height()) - 58,
                140,
            )
        except (tk.TclError, TypeError, ValueError):
            target_width, target_height = 420, 520

        pil.thumbnail(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )
        self._result_photo = ImageTk.PhotoImage(pil)
        self.result_label.configure(
            image=self._result_photo,
            text="",
        )

    def apply_theme(self, palette):
        try:
            self.configure(
                style="TFrame"
            )
        except Exception:
            pass

        try:
            self._drop_zone.configure(
                bg=palette["surface_soft"],
                highlightbackground=palette["border"],
            )
            self._drop_zone_title.configure(
                bg=palette["surface_soft"],
                fg=palette["accent"],
            )
            self._drop_zone_hint.configure(
                bg=palette["surface_soft"],
                fg=palette["muted"],
            )
        except Exception:
            pass

        try:
            self.canvas.configure(
                bg=palette["surface_soft"],
            )
            self.result_label.configure(
                bg=palette["surface_soft"],
                fg=palette["muted"],
            )
            self.page_list.configure(
                bg=palette["surface"],
                fg=palette["text"],
                selectbackground=palette["active_bg"],
                selectforeground=palette["text"],
            )
            self._page_list_wrap.configure(
                bg=palette["surface"],
            )
        except Exception:
            pass

    def export_results(self, output_format=None):
        if not self.session.pages:
            messagebox.showinfo(
                "Export Dokumen",
                "Belum ada halaman dokumen.",
            )
            return

        if not self.session.all_corrected():
            messagebox.showwarning(
                "Export Dokumen",
                "Semua halaman harus dikoreksi terlebih dahulu.",
            )
            return

        corrected_images = self.session.corrected_images()
        images = [
            self._apply_image_mode(
                image,
                self._mode_for_page(index),
            )
            for index, image in enumerate(
                corrected_images
            )
        ]

        try:
            output_format = (
                output_format or self.output_format.get()
            )
            if output_format == "pdf":
                self.output_dir.mkdir(parents=True, exist_ok=True)
                target = filedialog.asksaveasfilename(
                    title="Simpan PDF dokumen",
                    initialdir=str(self.output_dir),
                    initialfile="dokumen_corrected.pdf",
                    defaultextension=".pdf",
                    filetypes=[("PDF", "*.pdf")],
                )
                if not target:
                    return
                output = save_pdf_pages(target, images)
                message = f"PDF tersimpan:\n{output}"
            else:
                folder = filedialog.askdirectory(
                    title="Folder hasil gambar",
                    initialdir=str(self.output_dir),
                )
                if not folder:
                    return
                outputs = save_document_images(
                    folder,
                    [page.source_path for page in self.session.pages],
                    images,
                )
                message = f"{len(outputs)} gambar berhasil disimpan."
        except Exception as exc:
            messagebox.showerror("Export Dokumen", str(exc))
            return

        self.status_text.set(message.replace("\n", " "))
        messagebox.showinfo("Export Dokumen", message)
