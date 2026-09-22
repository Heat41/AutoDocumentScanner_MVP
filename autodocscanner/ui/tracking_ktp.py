from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from autodocscanner.services.ktp_tracking import (
    process_tracking_input,
)


class TrackingKtpPage(ttk.Frame):
    PAGE_TITLE = "Tracking Data KTP"
    SECTIONS = (
        "Input KTP",
        "Preview KTP",
        "Review Tracking",
    )

    REVIEW_FIELDS = (
        "nik",
        "nama",
        "tempat_lahir",
        "tanggal_lahir",
        "jenis_kelamin",
        "golongan_darah",
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
        "berlaku_hingga",
    )

    FIELD_LABELS = {
        "nik": "NIK",
        "nama": "Nama",
        "tempat_lahir": "Tempat Lahir",
        "tanggal_lahir": "Tanggal Lahir",
        "jenis_kelamin": "Jenis Kelamin",
        "golongan_darah": "Gol. Darah",
        "alamat": "Alamat",
        "rt": "RT",
        "rw": "RW",
        "kelurahan_desa": "Kelurahan / Desa",
        "kecamatan": "Kecamatan",
        "kabupaten_kota": "Kabupaten / Kota",
        "provinsi": "Provinsi",
        "agama": "Agama",
        "status_perkawinan": "Status Perkawinan",
        "pekerjaan": "Pekerjaan",
        "kewarganegaraan": "Kewarganegaraan",
        "berlaku_hingga": "Berlaku Hingga",
    }

    SUPPORTED_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    )

    def __init__(
        self,
        parent,
        scanner=None,
    ):
        super().__init__(
            parent,
            padding=18,
        )
        self.scanner = scanner
        self.input_path = None
        self._worker_running = False
        self._last_tracking = None
        self._preview_photo = None
        self._face_photo = None
        self._review_vars = {}
        self._review_markers = {}

        self.status_text = tk.StringVar(
            master=self,
            value=(
                "Pilih foto KTP untuk "
                "memulai Tracking Data KTP."
            ),
        )
        self.selected_text = tk.StringVar(
            master=self,
            value="Belum ada foto KTP dipilih.",
        )

        self._build_ui()

    @classmethod
    def review_value_mapping(
        cls,
        tracking_result,
    ):
        identity = dict(
            getattr(
                tracking_result,
                "identity",
                {},
            )
            or {}
        )

        return {
            name: str(
                identity.get(
                    name,
                    "",
                )
                or ""
            )
            for name in cls.REVIEW_FIELDS
        }

    @staticmethod
    def review_status_text(
        review_fields,
    ):
        count = len(
            list(
                review_fields
                or []
            )
        )

        if count == 0:
            return (
                "Hasil OCR siap direview."
            )

        return (
            f"{count} field perlu diperiksa."
        )

    @staticmethod
    def _expanded_review_fields(
        review_fields,
    ):
        result = set()

        for name in (
            review_fields
            or []
        ):
            if name == "ttl":
                result.update(
                    {
                        "tempat_lahir",
                        "tanggal_lahir",
                    }
                )
            elif name == "rt_rw":
                result.update(
                    {
                        "rt",
                        "rw",
                    }
                )
            else:
                result.add(name)

        return result

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(
            fill="x",
            pady=(0, 14),
        )

        ttk.Label(
            header,
            text=self.PAGE_TITLE,
            style="Header.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            header,
            text=(
                "Pilih KTP, proses auto perspective + OCR, "
                "lalu periksa hasil sebelum direkam."
            ),
            style="Subheader.TLabel",
        ).pack(
            anchor="w",
            pady=(3, 0),
        )

        content = ttk.Frame(self)
        content.pack(
            fill="both",
            expand=True,
        )

        self._build_input_card(
            content
        )
        self._build_preview_card(
            content
        )
        self._build_review_card(
            content
        )

        ttk.Label(
            self,
            textvariable=self.status_text,
            style="Subheader.TLabel",
        ).pack(
            fill="x",
            pady=(10, 0),
        )

    def _build_input_card(
        self,
        parent,
    ):
        card = ttk.Frame(
            parent,
            style="Card.TFrame",
            padding=14,
        )
        card.pack(
            side="left",
            fill="y",
            padx=(0, 7),
        )

        ttk.Label(
            card,
            text="Input KTP",
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            card,
            textvariable=self.selected_text,
            style="CardMuted.TLabel",
            wraplength=210,
            justify="left",
        ).pack(
            anchor="w",
            pady=(6, 12),
        )

        self.select_button = ttk.Button(
            card,
            text="Pilih Foto KTP",
            command=self.choose_input,
        )
        self.select_button.pack(
            fill="x",
        )

        self.process_button = ttk.Button(
            card,
            text="Proses Tracking",
            style="Accent.TButton",
            command=self.process_selected,
            state="disabled",
        )
        self.process_button.pack(
            fill="x",
            pady=(8, 0),
        )

        ttk.Separator(
            card,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=14,
        )

        ttk.Label(
            card,
            text=(
                "Proses menggunakan pipeline Auto Koreksi KTP "
                "yang sama, kemudian OCR dijalankan per field."
            ),
            style="CardMuted.TLabel",
            wraplength=210,
            justify="left",
        ).pack(anchor="w")

    def _build_preview_card(
        self,
        parent,
    ):
        card = ttk.Frame(
            parent,
            style="Card.TFrame",
            padding=14,
        )
        card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=7,
        )

        ttk.Label(
            card,
            text="Preview KTP",
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        self.preview_label = tk.Label(
            card,
            text="Belum ada KTP diproses",
            bg="#E9EDF0",
            fg="#737D85",
            bd=0,
            font=("Segoe UI", 9),
        )
        self.preview_label.pack(
            fill="both",
            expand=True,
            pady=(8, 8),
        )

        ttk.Label(
            card,
            text="Foto Wajah",
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        self.face_label = tk.Label(
            card,
            text="Belum tersedia",
            bg="#E9EDF0",
            fg="#737D85",
            bd=0,
            font=("Segoe UI", 9),
            height=6,
        )
        self.face_label.pack(
            fill="x",
            pady=(6, 0),
        )

    def _build_review_card(
        self,
        parent,
    ):
        card = ttk.Frame(
            parent,
            style="Card.TFrame",
            padding=14,
        )
        card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(7, 0),
        )

        ttk.Label(
            card,
            text="Review Tracking",
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        self.review_summary = ttk.Label(
            card,
            text="Belum ada hasil OCR.",
            style="CardMuted.TLabel",
        )
        self.review_summary.pack(
            anchor="w",
            pady=(4, 8),
        )

        canvas = tk.Canvas(
            card,
            highlightthickness=0,
            bd=0,
            bg="#FCFDFD",
        )
        scrollbar = ttk.Scrollbar(
            card,
            orient="vertical",
            command=canvas.yview,
        )
        canvas.configure(
            yscrollcommand=(
                scrollbar.set
            )
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )
        canvas.pack(
            fill="both",
            expand=True,
        )

        form = ttk.Frame(
            canvas,
            style="Card.TFrame",
        )
        window_id = canvas.create_window(
            (0, 0),
            window=form,
            anchor="nw",
        )

        def update_scrollregion(
            _event=None,
        ):
            canvas.configure(
                scrollregion=(
                    canvas.bbox("all")
                )
            )

        def stretch_form(event):
            canvas.itemconfigure(
                window_id,
                width=event.width,
            )

        form.bind(
            "<Configure>",
            update_scrollregion,
        )
        canvas.bind(
            "<Configure>",
            stretch_form,
        )

        for row, name in enumerate(
            self.REVIEW_FIELDS
        ):
            ttk.Label(
                form,
                text=self.FIELD_LABELS[
                    name
                ],
                style="CardMuted.TLabel",
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=3,
            )

            variable = tk.StringVar(
                master=self,
                value="",
            )
            entry = ttk.Entry(
                form,
                textvariable=variable,
                width=31,
            )
            entry.grid(
                row=row,
                column=1,
                sticky="ew",
                pady=3,
            )

            marker = ttk.Label(
                form,
                text="",
                style="CardMuted.TLabel",
            )
            marker.grid(
                row=row,
                column=2,
                sticky="w",
                padx=(7, 0),
                pady=3,
            )

            self._review_vars[
                name
            ] = variable
            self._review_markers[
                name
            ] = marker

        form.columnconfigure(
            1,
            weight=1,
        )

        ttk.Separator(
            card,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=10,
        )

        self.record_button = ttk.Button(
            card,
            text="Rekam Data KTP",
            state="disabled",
        )
        self.record_button.pack(
            anchor="e",
        )

        ttk.Label(
            card,
            text=(
                "Perekaman database diaktifkan pada Stage 2F "
                "setelah review dan validasi final."
            ),
            style="CardMuted.TLabel",
            wraplength=300,
            justify="right",
        ).pack(
            anchor="e",
            pady=(5, 0),
        )

    def choose_input(self):
        if self._worker_running:
            return

        selected = (
            filedialog.askopenfilename(
                title="Pilih Foto KTP",
                filetypes=[
                    (
                        "Gambar KTP",
                        "*.jpg *.jpeg *.png *.bmp *.webp",
                    ),
                    (
                        "Semua file",
                        "*.*",
                    ),
                ],
            )
        )

        if not selected:
            return

        path = Path(selected)

        if (
            path.suffix.lower()
            not in self.SUPPORTED_EXTENSIONS
        ):
            messagebox.showwarning(
                "Format tidak didukung",
                "Pilih file gambar JPG, PNG, BMP, atau WEBP.",
            )
            return

        self.input_path = path
        self.selected_text.set(
            path.name
        )
        self.status_text.set(
            "Foto KTP dipilih. Tekan Proses Tracking."
        )
        self.process_button.configure(
            state="normal",
        )

        self._clear_result()
        self._show_source_preview(
            path
        )

    def _clear_result(self):
        self._last_tracking = None
        self._face_photo = None
        self.face_label.configure(
            image="",
            text="Belum tersedia",
        )
        self.review_summary.configure(
            text="Belum ada hasil OCR.",
        )

        for name in self.REVIEW_FIELDS:
            self._review_vars[
                name
            ].set("")
            self._review_markers[
                name
            ].configure(
                text=""
            )

    def _show_source_preview(
        self,
        path,
    ):
        image = cv2.imread(
            str(path)
        )
        if image is None:
            self.preview_label.configure(
                image="",
                text="Preview tidak tersedia",
            )
            self._preview_photo = None
            return

        self._set_preview_image(
            image,
            target="ktp",
        )

    @staticmethod
    def _pil_from_bgr(
        image,
    ):
        if image.ndim == 2:
            return Image.fromarray(
                image
            ).convert("RGB")

        if (
            image.ndim == 3
            and image.shape[2] == 3
        ):
            return Image.fromarray(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )
            )

        if (
            image.ndim == 3
            and image.shape[2] == 4
        ):
            return Image.fromarray(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGRA2RGBA,
                )
            ).convert("RGB")

        raise ValueError(
            "Format preview tidak didukung."
        )

    def _set_preview_image(
        self,
        image,
        target,
    ):
        pil_image = self._pil_from_bgr(
            image
        )

        if target == "ktp":
            pil_image.thumbnail(
                (520, 330),
                Image.Resampling.LANCZOS,
            )
            photo = ImageTk.PhotoImage(
                pil_image
            )
            self._preview_photo = photo
            self.preview_label.configure(
                image=photo,
                text="",
            )
            return

        pil_image.thumbnail(
            (170, 150),
            Image.Resampling.LANCZOS,
        )
        photo = ImageTk.PhotoImage(
            pil_image
        )
        self._face_photo = photo
        self.face_label.configure(
            image=photo,
            text="",
        )

    def process_selected(self):
        if (
            self._worker_running
            or self.input_path is None
        ):
            return

        if self.scanner is None:
            messagebox.showerror(
                "Tracking KTP",
                "Scanner KTP belum tersedia.",
            )
            return

        self._worker_running = True
        self.select_button.configure(
            state="disabled",
        )
        self.process_button.configure(
            state="disabled",
        )
        self.status_text.set(
            "Memproses auto perspective dan OCR KTP..."
        )
        self.review_summary.configure(
            text="OCR sedang berjalan...",
        )

        worker = threading.Thread(
            target=self._process_worker,
            args=(self.input_path,),
            daemon=True,
        )
        worker.start()

    def _process_worker(
        self,
        input_path,
    ):
        try:
            result = (
                process_tracking_input(
                    self.scanner,
                    input_path,
                )
            )
        except Exception as exc:
            try:
                self.after(
                    0,
                    self._finish_failure,
                    str(exc),
                )
            except tk.TclError:
                pass
            return

        try:
            self.after(
                0,
                self._finish_success,
                result,
            )
        except tk.TclError:
            pass

    def _finish_failure(
        self,
        message,
    ):
        self._worker_running = False
        self.select_button.configure(
            state="normal",
        )
        self.process_button.configure(
            state=(
                "normal"
                if self.input_path
                else "disabled"
            ),
        )
        self.status_text.set(
            "Tracking KTP gagal."
        )
        self.review_summary.configure(
            text="Hasil OCR belum tersedia.",
        )

        messagebox.showerror(
            "Tracking KTP gagal",
            message,
        )

    def _finish_success(
        self,
        result,
    ):
        tracking = result[
            "tracking"
        ]
        self._last_tracking = tracking

        values = (
            self.review_value_mapping(
                tracking
            )
        )
        review_identity_fields = (
            self._expanded_review_fields(
                tracking.review_fields
            )
        )

        for name in self.REVIEW_FIELDS:
            self._review_vars[
                name
            ].set(
                values.get(
                    name,
                    "",
                )
            )
            self._review_markers[
                name
            ].configure(
                text=(
                    "⚠ Periksa"
                    if name
                    in review_identity_fields
                    else "✓"
                )
            )

        self._set_preview_image(
            result[
                "corrected_image"
            ],
            target="ktp",
        )
        self._set_preview_image(
            tracking.face_image,
            target="face",
        )

        summary = (
            self.review_status_text(
                tracking.review_fields
            )
        )
        self.review_summary.configure(
            text=summary,
        )
        self.status_text.set(
            "Tracking selesai. Periksa dan koreksi field OCR bila diperlukan."
        )

        self._worker_running = False
        self.select_button.configure(
            state="normal",
        )
        self.process_button.configure(
            state="normal",
        )
