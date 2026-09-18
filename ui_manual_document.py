from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from document_canvas import (
    canvas_to_image,
    fit_image_size,
    image_to_canvas,
)
from document_session import ManualDocumentSession
from output_manager import save_document_images, save_pdf_pages


class ManualDocumentPage(ttk.Frame):
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    DEFAULT_OUTPUT_FORMAT = "image"
    HANDLE_RADIUS = 8
    HANDLE_HIT_RADIUS = 18

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
        self.status_text = tk.StringVar(
            master=self,
            value="Pilih satu atau beberapa gambar dokumen untuk memulai.",
        )
        self._canvas_photo = None
        self._result_photo = None
        self._display_size = None
        self._display_offset = None
        self._active_corner = None
        self._build_ui()

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
            command=self.choose_images,
        ).pack(side="left")
        ttk.Button(
            actions,
            text="Kosongkan",
            command=self.clear_pages,
        ).pack(side="left", padx=(8, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(
            body,
            style="Sidebar.TFrame",
            padding=12,
            width=235,
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ttk.Label(
            sidebar,
            text="Halaman Dokumen",
            style="SidebarTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            sidebar,
            text="✓ = sudah dikoreksi",
            style="SidebarMuted.TLabel",
        ).pack(anchor="w", pady=(2, 8))

        list_wrap = tk.Frame(sidebar, bg="#FFFFFF")
        list_wrap.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_wrap, orient="vertical")
        scrollbar.pack(side="right", fill="y")

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

        ttk.Button(
            sidebar,
            text="Folder Output",
            command=self.choose_output_dir,
        ).pack(fill="x", pady=(10, 0))

        workspace = ttk.Frame(body)
        workspace.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(14, 0),
        )

        previews = ttk.Frame(workspace)
        previews.pack(fill="both", expand=True)

        canvas_card = ttk.Frame(
            previews,
            style="Card.TFrame",
            padding=10,
        )
        canvas_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 6),
        )
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
        result_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(6, 0),
        )
        ttk.Label(
            result_card,
            text="Preview Hasil",
            style="CardLabel.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        self.result_label = tk.Label(
            result_card,
            text="Belum dikoreksi",
            bg="#E9EDF0",
            fg="#737D85",
            bd=0,
            font=("Segoe UI", 9),
        )
        self.result_label.pack(fill="both", expand=True)

        controls = ttk.Frame(workspace)
        controls.pack(fill="x", pady=(12, 0))

        ttk.Button(
            controls,
            text="Reset Titik",
            command=self.reset_current,
        ).pack(side="left")
        ttk.Button(
            controls,
            text="↺ Putar Kiri",
            command=lambda: self.rotate_current("left"),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            controls,
            text="Putar Kanan ↻",
            command=lambda: self.rotate_current("right"),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            controls,
            text="Terapkan Koreksi",
            style="Accent.TButton",
            command=self.apply_current,
        ).pack(side="left", padx=(12, 0))

        export_group = ttk.Frame(controls)
        export_group.pack(side="right")
        ttk.Label(
            export_group,
            text="Output",
            style="Subheader.TLabel",
        ).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(
            export_group,
            text="Gambar",
            variable=self.output_format,
            value="image",
        ).pack(side="left")
        ttk.Radiobutton(
            export_group,
            text="PDF",
            variable=self.output_format,
            value="pdf",
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            export_group,
            text="Export",
            command=self.export_results,
        ).pack(side="left", padx=(12, 0))

        ttk.Label(
            self,
            textvariable=self.status_text,
            style="Subheader.TLabel",
        ).pack(fill="x", pady=(10, 0))

    def choose_images(self):
        paths = filedialog.askopenfilenames(
            title="Pilih gambar dokumen",
            filetypes=[
                ("Gambar", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ],
        )
        if not paths:
            return

        first_added = len(self.session.pages)
        failed = []

        for raw_path in paths:
            path = Path(raw_path)
            image = cv2.imread(str(path))
            if image is None:
                failed.append(path.name)
                continue
            self.session.add_image(path, image)

        self._refresh_page_list()

        if len(self.session.pages) > first_added:
            self._select_page(first_added)

        if failed:
            messagebox.showwarning(
                "Dokumen",
                "File berikut tidak dapat dibaca:\n" + "\n".join(failed[:5]),
            )

        self.status_text.set(
            f"{len(self.session.pages)} halaman siap dikoreksi."
        )

    def clear_pages(self):
        self.session = ManualDocumentSession()
        self.current_index = None
        self._active_corner = None
        self.page_list.delete(0, tk.END)
        self.canvas.delete("all")
        self._canvas_photo = None
        self._result_photo = None
        self.result_label.configure(image="", text="Belum dikoreksi")
        self.status_text.set("Daftar halaman dikosongkan.")

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

    def _select_page(self, index):
        index = int(index)
        if not 0 <= index < len(self.session.pages):
            return

        self.current_index = index
        self.page_list.selection_clear(0, tk.END)
        self.page_list.selection_set(index)
        self.page_list.see(index)
        self._render_current()

    def _on_page_select(self, _event=None):
        selection = self.page_list.curselection()
        if not selection:
            return
        self.current_index = int(selection[0])
        self._render_current()

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
            padding=22,
        )

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        for label, (x, y) in zip(("TL", "TR", "BR", "BL"), points):
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
            self.canvas.create_text(
                x + 14,
                y - 14,
                text=label,
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

    def rotate_current(self, direction):
        if self.current_index is None:
            return

        self.session.rotate(self.current_index, direction)
        self._refresh_page_list()
        self.page_list.selection_set(self.current_index)
        self._render_current()
        self.status_text.set("Halaman diputar. Titik sudut di-reset.")

    def apply_current(self):
        if self.current_index is None:
            messagebox.showinfo(
                "Koreksi Dokumen",
                "Pilih halaman terlebih dahulu.",
            )
            return

        try:
            result = self.session.apply_correction(self.current_index)
        except Exception as exc:
            messagebox.showerror("Koreksi Dokumen", str(exc))
            return

        self._render_result(result)
        self._refresh_page_list()
        self.page_list.selection_set(self.current_index)
        self.status_text.set(
            f"Halaman {self.current_index + 1} berhasil dikoreksi."
        )

    def _render_result(self, image):
        if image is None:
            self._result_photo = None
            self.result_label.configure(image="", text="Belum dikoreksi")
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((420, 520), Image.Resampling.LANCZOS)
        self._result_photo = ImageTk.PhotoImage(pil)
        self.result_label.configure(image=self._result_photo, text="")

    def export_results(self):
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

        images = self.session.corrected_images()

        try:
            if self.output_format.get() == "pdf":
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
