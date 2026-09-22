from pathlib import Path
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from autodocscanner.ktp.tracking import (
    extract_tracking_data,
)
from autodocscanner.ktp.roi_debug import (
    build_detection_overlay,
)
from autodocscanner.services.pdf_preview import (
    build_tracking_report_pdf,
)


class TrackingKtpPage(ttk.Frame):
    PAGE_TITLE = "Tracking Data KTP"
    SECTIONS = (
        "Input KTP",
        "Preview KTP",
        "Review Tracking",
    )

    COLUMN_WEIGHTS = (
        1,
        3,
        4,
    )

    REVIEW_GROUPS = {
        "Identitas Utama": (
            "nik",
            "nama",
            "tempat_lahir",
            "tanggal_lahir",
            "jenis_kelamin",
            "golongan_darah",
        ),
        "Alamat": (
            "alamat",
            "rt",
            "rw",
            "kelurahan_desa",
            "kecamatan",
            "kabupaten_kota",
            "provinsi",
        ),
        "Data Lainnya": (
            "agama",
            "status_perkawinan",
            "pekerjaan",
            "kewarganegaraan",
            "berlaku_hingga",
        ),
    }

    REVIEW_FIELDS = tuple(
        field_name
        for fields in REVIEW_GROUPS.values()
        for field_name in fields
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
        self._corrected_preview_image = None
        self._tracking_debug_image = None
        self._tracking_detections = None
        self._roi_overlay_visible = False
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
            return "Hasil OCR siap direview."

        return (
            f"{count} field perlu diperiksa."
        )

    @staticmethod
    def field_status_text(
        value,
        needs_review,
    ):
        if not str(
            value or ""
        ).strip():
            return (
                "Kosong"
                if needs_review
                else ""
            )

        if needs_review:
            return "Periksa"

        return ""

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
        self.columnconfigure(
            0,
            weight=1,
        )
        self.rowconfigure(
            1,
            weight=1,
        )

        header = ttk.Frame(self)
        header.grid(
            row=0,
            column=0,
            sticky="ew",
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
        content.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        for index, weight in enumerate(
            self.COLUMN_WEIGHTS
        ):
            content.columnconfigure(
                index,
                weight=weight,
                uniform="tracking-columns",
            )

        content.rowconfigure(
            0,
            weight=1,
        )

        self._build_input_card(
            content,
            column=0,
        )
        self._build_preview_card(
            content,
            column=1,
        )
        self._build_review_card(
            content,
            column=2,
        )

        ttk.Label(
            self,
            textvariable=self.status_text,
            style="Subheader.TLabel",
        ).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )

    def _build_input_card(
        self,
        parent,
        column,
    ):
        card = ttk.Frame(
            parent,
            style="Card.TFrame",
            padding=14,
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0, 7),
        )
        card.columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            card,
            text="Input KTP",
            style="CardLabel.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Label(
            card,
            textvariable=self.selected_text,
            style="CardMuted.TLabel",
            wraplength=180,
            justify="left",
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(6, 12),
        )

        self.select_button = ttk.Button(
            card,
            text="Pilih Foto KTP",
            command=self.choose_input,
        )
        self.select_button.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        self.process_button = ttk.Button(
            card,
            text="Proses Tracking",
            style="Accent.TButton",
            command=self.process_selected,
            state="disabled",
        )
        self.process_button.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        self.preview_pdf_button = ttk.Button(
            card,
            text="Preview PDF",
            command=self.preview_pdf,
            state="disabled",
        )
        self.preview_pdf_button.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        self.save_dataset_button = ttk.Button(
            card,
            text="Simpan Sampel Detector",
            command=self.save_detector_sample,
            state="disabled",
        )
        self.save_dataset_button.grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        ttk.Separator(
            card,
            orient="horizontal",
        ).grid(
            row=6,
            column=0,
            sticky="ew",
            pady=14,
        )

        ttk.Label(
            card,
            text=(
                "Pipeline:\n"
                "Auto Koreksi KTP\n"
                "→ OCR per field\n"
                "→ Review manual"
            ),
            style="CardMuted.TLabel",
            wraplength=180,
            justify="left",
        ).grid(
            row=7,
            column=0,
            sticky="nw",
        )

        ttk.Label(
            card,
            text=(
                "Tombol Rekam akan aktif "
                "setelah tahap review final."
            ),
            style="CardMuted.TLabel",
            wraplength=180,
            justify="left",
        ).grid(
            row=8,
            column=0,
            sticky="sw",
            pady=(18, 0),
        )

        card.rowconfigure(
            8,
            weight=1,
        )

    def _build_preview_card(
        self,
        parent,
        column,
    ):
        card = ttk.Frame(
            parent,
            style="Card.TFrame",
            padding=14,
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=7,
        )

        card.columnconfigure(
            0,
            weight=1,
        )
        card.rowconfigure(
            1,
            weight=1,
        )

        preview_header = ttk.Frame(
            card,
        )
        preview_header.grid(
            row=0,
            column=0,
            sticky="ew",
        )
        preview_header.columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            preview_header,
            text="Preview KTP",
            style="CardLabel.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.roi_button = ttk.Button(
            preview_header,
            text="Tampilkan Bounding Box",
            command=self.toggle_roi_overlay,
            state="disabled",
        )
        self.roi_button.grid(
            row=0,
            column=1,
            sticky="e",
        )

        self.preview_label = tk.Label(
            card,
            text="Belum ada KTP diproses",
            bg="#E9EDF0",
            fg="#737D85",
            bd=0,
            font=("Segoe UI", 9),
        )
        self.preview_label.grid(
            row=1,
            column=0,
            sticky="nsew",
            pady=(8, 12),
        )

        face_header = ttk.Frame(
            card,
        )
        face_header.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        ttk.Label(
            face_header,
            text="Foto Wajah Terdeteksi",
            style="CardLabel.TLabel",
        ).pack(
            side="left",
        )

        ttk.Label(
            face_header,
            text="hasil crop bbox foto",
            style="CardMuted.TLabel",
        ).pack(
            side="right",
        )

        self.face_label = tk.Label(
            card,
            text="Belum tersedia",
            bg="#E9EDF0",
            fg="#737D85",
            bd=0,
            font=("Segoe UI", 9),
            height=7,
        )
        self.face_label.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(6, 0),
        )

    def _build_review_card(
        self,
        parent,
        column,
    ):
        card = ttk.Frame(
            parent,
            style="Card.TFrame",
            padding=14,
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(7, 0),
        )

        card.columnconfigure(
            0,
            weight=1,
        )
        card.rowconfigure(
            2,
            weight=1,
        )

        title_row = ttk.Frame(card)
        title_row.grid(
            row=0,
            column=0,
            sticky="ew",
        )
        title_row.columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            title_row,
            text="Review Tracking",
            style="CardLabel.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.review_summary = ttk.Label(
            title_row,
            text="Belum ada hasil OCR.",
            style="CardMuted.TLabel",
        )
        self.review_summary.grid(
            row=0,
            column=1,
            sticky="e",
        )

        ttk.Separator(
            card,
            orient="horizontal",
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(9, 8),
        )

        form_shell = ttk.Frame(
            card,
            style="Card.TFrame",
        )
        form_shell.grid(
            row=2,
            column=0,
            sticky="nsew",
        )
        form_shell.columnconfigure(
            0,
            weight=1,
        )
        form_shell.rowconfigure(
            0,
            weight=1,
        )

        canvas = tk.Canvas(
            form_shell,
            highlightthickness=0,
            bd=0,
            bg="#FCFDFD",
        )
        scrollbar = ttk.Scrollbar(
            form_shell,
            orient="vertical",
            command=canvas.yview,
        )

        canvas.configure(
            yscrollcommand=scrollbar.set,
        )

        canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )
        scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
            padx=(5, 0),
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

        row = 0

        for (
            group_title,
            group_fields,
        ) in self.REVIEW_GROUPS.items():
            group_label = ttk.Label(
                form,
                text=group_title,
                style="CardLabel.TLabel",
            )
            group_label.grid(
                row=row,
                column=0,
                columnspan=3,
                sticky="w",
                pady=(10 if row else 0, 5),
            )
            row += 1

            for name in group_fields:
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
                    pady=4,
                )

                variable = tk.StringVar(
                    master=self,
                    value="",
                )

                entry = ttk.Entry(
                    form,
                    textvariable=variable,
                )
                entry.grid(
                    row=row,
                    column=1,
                    sticky="ew",
                    pady=4,
                )

                marker = ttk.Label(
                    form,
                    text="",
                    style="CardMuted.TLabel",
                    width=8,
                    anchor="w",
                )
                marker.grid(
                    row=row,
                    column=2,
                    sticky="w",
                    padx=(8, 0),
                    pady=4,
                )

                self._review_vars[
                    name
                ] = variable
                self._review_markers[
                    name
                ] = marker

                row += 1

        form.columnconfigure(
            1,
            weight=1,
        )

        footer = ttk.Frame(card)
        footer.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )
        footer.columnconfigure(
            0,
            weight=1,
        )

        ttk.Label(
            footer,
            text=(
                "Periksa hanya field yang ditandai. "
                "Field lain tetap dapat diedit."
            ),
            style="CardMuted.TLabel",
            wraplength=360,
            justify="left",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.record_button = ttk.Button(
            footer,
            text="Rekam Data KTP",
            state="disabled",
        )
        self.record_button.grid(
            row=0,
            column=1,
            sticky="e",
            padx=(10, 0),
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
        self.preview_label.configure(
            image="",
            text=(
                "Foto dipilih.\n"
                "Hasil corrected KTP akan tampil "
                "setelah Auto Perspective selesai."
            ),
        )
        self._preview_photo = None

    def _clear_result(self):
        self._last_tracking = None
        self._face_photo = None
        self._corrected_preview_image = None
        self._tracking_debug_image = None
        self._tracking_detections = None
        self._roi_overlay_visible = False
        self.roi_button.configure(
            text="Tampilkan Bounding Box",
            state="disabled",
        )
        self.preview_pdf_button.configure(
            state="disabled",
        )
        self.save_dataset_button.configure(
            state="disabled",
        )

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
            # Match the Auto Koreksi KTP preview contract: keep the
            # entire card visible inside the preview panel without
            # clipping/zooming when the Tracking column is narrower.
            pil_image.thumbnail(
                (390, 430),
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
            (190, 155),
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

    def toggle_roi_overlay(self):
        if (
            self._corrected_preview_image is None
            or self._tracking_debug_image is None
        ):
            return

        self._roi_overlay_visible = (
            not self._roi_overlay_visible
        )

        if self._roi_overlay_visible:
            preview = build_detection_overlay(
                self._tracking_debug_image,
                self._tracking_detections,
            )
            self.roi_button.configure(
                text="Sembunyikan Bounding Box",
            )
            self.status_text.set(
                "Mode Field Detector aktif — bounding box menunjukkan area yang benar-benar dibaca OCR."
            )
        else:
            preview = (
                self._corrected_preview_image
            )
            self.roi_button.configure(
                text="Tampilkan Bounding Box",
            )
            self.status_text.set(
                "Mode Field Detector nonaktif."
            )

        self._set_preview_image(
            preview,
            target="ktp",
        )

    def current_review_values(self):
        return {
            name: self._review_vars[
                name
            ].get().strip()
            for name in self.REVIEW_FIELDS
        }

    def save_detector_sample(self):
        tracking = self._last_tracking

        if (
            tracking is None
            or tracking.debug_tracking_image is None
        ):
            return

        try:
            dataset_dir = (
                Path("dataset")
                / "ktp_fields"
                / "unlabeled"
            )
            dataset_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            stem = (
                self.input_path.stem
                if self.input_path is not None
                else "ktp_sample"
            )

            target = (
                dataset_dir
                / f"{stem}_canonical.png"
            )

            ok = cv2.imwrite(
                str(target),
                tracking.debug_tracking_image,
            )

            if not ok:
                raise RuntimeError(
                    "OpenCV gagal menyimpan sampel detector."
                )

            self.status_text.set(
                f"Sampel detector tersimpan: {target.name}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Simpan sampel detector gagal",
                str(exc),
            )

    def preview_pdf(self):
        tracking = self._last_tracking

        if tracking is None:
            return

        try:
            stem = (
                self.input_path.stem
                if self.input_path is not None
                else "tracking_ktp"
            )
            pdf_path = build_tracking_report_pdf(
                tracking.corrected_image,
                tracking.face_image,
                self.current_review_values(),
                stem=f"{stem}_tracking",
            )

            if os.name != "nt":
                raise RuntimeError(
                    "Preview PDF otomatis saat ini hanya didukung di Windows."
                )

            os.startfile(
                str(pdf_path)
            )
            self.status_text.set(
                f"Preview PDF dibuka: {pdf_path.name}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Preview PDF gagal",
                str(exc),
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
            "Tahap 1/2 — Auto Perspective KTP..."
        )
        self.review_summary.configure(
            text="Menunggu hasil corrected KTP...",
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
            corrected_image, corners = (
                self.scanner.scan(
                    input_path,
                    output_path=None,
                    mode="ktp",
                    output_mode="color",
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

        corrected = {
            "corrected_image": corrected_image,
            "corners": corners,
        }

        try:
            self.after(
                0,
                self._show_corrected_stage,
                corrected,
            )
        except tk.TclError:
            return

        try:
            tracking = extract_tracking_data(
                corrected_image
            )
            result = {
                "corrected_image": corrected_image,
                "corners": corners,
                "tracking": tracking,
            }
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

    def _show_corrected_stage(
        self,
        corrected,
    ):
        self._corrected_preview_image = (
            corrected[
                "corrected_image"
            ].copy()
        )
        self._set_preview_image(
            self._corrected_preview_image,
            target="ktp",
        )
        self.status_text.set(
            "Tahap 2/2 — KTP terkoreksi. Menjalankan Tracking/OCR..."
        )
        self.review_summary.configure(
            text="OCR sedang berjalan dari corrected KTP...",
        )

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
            value = values.get(
                name,
                "",
            )

            self._review_vars[
                name
            ].set(
                value
            )

            self._review_markers[
                name
            ].configure(
                text=self.field_status_text(
                    value=value,
                    needs_review=(
                        name
                        in review_identity_fields
                    ),
                )
            )

        self._corrected_preview_image = (
            result[
                "corrected_image"
            ].copy()
        )
        self._tracking_debug_image = (
            tracking.debug_tracking_image.copy()
            if tracking.debug_tracking_image is not None
            else None
        )
        self._tracking_detections = (
            list(
                tracking.detections
            )
            if tracking.detections is not None
            else None
        )
        self._roi_overlay_visible = False

        self._set_preview_image(
            self._corrected_preview_image,
            target="ktp",
        )
        self._set_preview_image(
            tracking.face_image,
            target="face",
        )

        self.roi_button.configure(
            text="Tampilkan Bounding Box",
            state=(
                "normal"
                if self._tracking_debug_image is not None
                else "disabled"
            ),
        )

        self.review_summary.configure(
            text=self.review_status_text(
                tracking.review_fields
            ),
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
        self.preview_pdf_button.configure(
            state="normal",
        )
        self.save_dataset_button.configure(
            state="normal",
        )
