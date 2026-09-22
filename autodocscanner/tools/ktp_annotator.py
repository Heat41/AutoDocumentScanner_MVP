from pathlib import Path
import shutil
import tkinter as tk

import cv2
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from autodocscanner.core.scanner import (
    AutoDocumentScanner,
)
from autodocscanner.ktp.annotation import (
    Annotation,
    read_yolo_annotations,
    write_yolo_annotations,
)
from autodocscanner.ktp.field_detection import (
    FIELD_CLASSES,
)
from autodocscanner.ktp.layout import (
    normalize_ktp_for_tracking,
)


DATASET_ROOT = (
    Path("dataset")
    / "ktp_fields"
)
IMAGE_DIR = (
    DATASET_ROOT
    / "images"
    / "train"
)
LABEL_DIR = (
    DATASET_ROOT
    / "labels"
    / "train"
)
CANONICAL_DIR = (
    DATASET_ROOT
    / "canonical"
)


class KtpFieldAnnotator(tk.Tk):
    CANVAS_WIDTH = 900
    CANVAS_HEIGHT = 600

    def __init__(
        self,
        initial_path=None,
        scanner=None,
    ):
        super().__init__()
        self.title(
            "KTP Field Detector — Annotation Tool"
        )
        self.geometry(
            "1250x760"
        )
        self.minsize(
            1050,
            650,
        )

        self.image_path = None
        self.image = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.annotations = []
        self.drag_start = None
        self.preview_rect = None
        self.scanner = (
            scanner
            if scanner is not None
            else AutoDocumentScanner()
        )
        self.initial_path = (
            Path(initial_path)
            if initial_path
            else None
        )

        self.selected_class = tk.StringVar(
            value=FIELD_CLASSES[0]
        )
        self.status_text = tk.StringVar(
            value=(
                "Buka foto mentah atau canonical KTP. Foto mentah akan Auto Perspective otomatis sebelum anotasi."
            )
        )

        self._build_ui()

        if self.initial_path is not None:
            self.after(
                50,
                self._open_path,
                self.initial_path,
            )

    def _build_ui(self):
        root = ttk.Frame(
            self,
            padding=12,
        )
        root.pack(
            fill="both",
            expand=True,
        )

        toolbar = ttk.Frame(
            root
        )
        toolbar.pack(
            fill="x",
            pady=(0, 10),
        )

        ttk.Button(
            toolbar,
            text="Buka Foto / KTP",
            command=self.open_image,
        ).pack(
            side="left"
        )

        ttk.Label(
            toolbar,
            text="Class:",
        ).pack(
            side="left",
            padx=(14, 6),
        )

        self.class_combo = ttk.Combobox(
            toolbar,
            textvariable=self.selected_class,
            values=FIELD_CLASSES,
            state="readonly",
            width=22,
        )
        self.class_combo.pack(
            side="left"
        )

        ttk.Button(
            toolbar,
            text="Undo",
            command=self.undo,
        ).pack(
            side="left",
            padx=(10, 0),
        )

        ttk.Button(
            toolbar,
            text="Hapus Terpilih",
            command=self.delete_selected,
        ).pack(
            side="left",
            padx=(6, 0),
        )

        ttk.Button(
            toolbar,
            text="Simpan Anotasi",
            command=self.save_annotations,
        ).pack(
            side="right"
        )

        body = ttk.Frame(
            root
        )
        body.pack(
            fill="both",
            expand=True,
        )

        self.canvas = tk.Canvas(
            body,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            bg="#E8ECEF",
            highlightthickness=1,
            highlightbackground="#B9C0C5",
        )
        self.canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.canvas.bind(
            "<ButtonPress-1>",
            self._drag_start,
        )
        self.canvas.bind(
            "<B1-Motion>",
            self._drag_move,
        )
        self.canvas.bind(
            "<ButtonRelease-1>",
            self._drag_end,
        )
        self.canvas.bind(
            "<Configure>",
            self._canvas_resized,
        )

        side = ttk.Frame(
            body,
            padding=(12, 0, 0, 0),
            width=260,
        )
        side.pack(
            side="right",
            fill="y",
        )

        ttk.Label(
            side,
            text="Bounding Box",
            font=("Segoe UI Semibold", 11),
        ).pack(
            anchor="w"
        )

        ttk.Label(
            side,
            text=(
                "Kotak harus membungkus VALUE, bukan label.\n"
                "Contoh class nama: box hanya pada 'DJONG FUK HIE'."
            ),
            wraplength=240,
            justify="left",
        ).pack(
            anchor="w",
            pady=(6, 10),
        )

        self.listbox = tk.Listbox(
            side,
            width=34,
            height=28,
        )
        self.listbox.pack(
            fill="both",
            expand=True,
        )
        self.listbox.bind(
            "<<ListboxSelect>>",
            lambda _event: self._draw(),
        )

        ttk.Label(
            root,
            textvariable=self.status_text,
        ).pack(
            fill="x",
            pady=(8, 0),
        )

    def open_image(self):
        selected = filedialog.askopenfilename(
            title="Pilih foto mentah atau corrected KTP",
            filetypes=[
                (
                    "Gambar",
                    "*.jpg *.jpeg *.png *.bmp *.webp",
                ),
                (
                    "Semua file",
                    "*.*",
                ),
            ],
        )

        if not selected:
            return

        self._open_path(
            Path(selected)
        )

    def _canonical_target(
        self,
        source_path,
    ):
        CANONICAL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        return (
            CANONICAL_DIR
            / (
                source_path.stem
                + "_canonical.png"
            )
        )

    def _prepare_canonical(
        self,
        source_path,
    ):
        source_path = Path(
            source_path
        )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"File tidak ditemukan: {source_path}"
            )

        corrected, _corners = (
            self.scanner.scan(
                source_path,
                output_path=None,
                mode="ktp",
                output_mode="color",
            )
        )

        canonical = (
            normalize_ktp_for_tracking(
                corrected
            )
        )

        target = (
            self._canonical_target(
                source_path
            )
        )

        ok = cv2.imwrite(
            str(target),
            canonical,
        )

        if not ok:
            raise RuntimeError(
                "OpenCV gagal menyimpan canonical KTP untuk anotasi."
            )

        return target

    def _load_canonical_image(
        self,
        canonical_path,
    ):
        image = Image.open(
            canonical_path
        ).convert(
            "RGB"
        )

        self.image_path = Path(
            canonical_path
        )
        self.image = image

        label_path = (
            LABEL_DIR
            / (
                self.image_path.stem
                + ".txt"
            )
        )
        self.annotations = (
            read_yolo_annotations(
                label_path,
                image_width=image.width,
                image_height=image.height,
            )
        )

        self.status_text.set(
            (
                f"Canonical aktif: {self.image_path.name} — "
                f"{len(self.annotations)} box."
            )
        )
        self._refresh_list()
        self._draw()

    def _open_path(
        self,
        source_path,
    ):
        source_path = Path(
            source_path
        )

        try:
            self.status_text.set(
                "Menjalankan Auto Perspective KTP..."
            )
            self.update_idletasks()

            canonical_path = (
                self._prepare_canonical(
                    source_path
                )
            )

            self._load_canonical_image(
                canonical_path
            )

        except Exception as exc:
            messagebox.showerror(
                "Persiapan KTP gagal",
                str(exc),
            )
            self.status_text.set(
                "KTP gagal dipersiapkan untuk anotasi."
            )

    def _canvas_resized(
        self,
        _event=None,
    ):
        if self.image is not None:
            self._draw()

    def _display_geometry(self):
        if self.image is None:
            return None

        canvas_width = max(
            self.canvas.winfo_width(),
            1,
        )
        canvas_height = max(
            self.canvas.winfo_height(),
            1,
        )

        image_width, image_height = (
            self.image.size
        )

        scale = min(
            canvas_width
            / image_width,
            canvas_height
            / image_height,
        )
        scale = max(
            scale,
            0.01,
        )

        display_width = int(
            round(
                image_width
                * scale
            )
        )
        display_height = int(
            round(
                image_height
                * scale
            )
        )

        offset_x = (
            canvas_width
            - display_width
        ) // 2
        offset_y = (
            canvas_height
            - display_height
        ) // 2

        return (
            scale,
            offset_x,
            offset_y,
            display_width,
            display_height,
        )

    def _draw(self):
        self.canvas.delete(
            "all"
        )

        if self.image is None:
            self.canvas.create_text(
                30,
                30,
                anchor="nw",
                text="Belum ada KTP.",
                fill="#66717A",
            )
            return

        geometry = (
            self._display_geometry()
        )

        if geometry is None:
            return

        (
            self.scale,
            self.offset_x,
            self.offset_y,
            display_width,
            display_height,
        ) = geometry

        display = self.image.resize(
            (
                display_width,
                display_height,
            ),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(
            display
        )

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor="nw",
            image=self.photo,
        )

        selection = (
            self.listbox.curselection()
        )
        selected_index = (
            int(selection[0])
            if selection
            else -1
        )

        for index, item in enumerate(
            self.annotations
        ):
            x1, y1, x2, y2 = (
                item.bbox
            )

            cx1 = (
                self.offset_x
                + x1 * self.scale
            )
            cy1 = (
                self.offset_y
                + y1 * self.scale
            )
            cx2 = (
                self.offset_x
                + x2 * self.scale
            )
            cy2 = (
                self.offset_y
                + y2 * self.scale
            )

            width = (
                3
                if index == selected_index
                else 2
            )

            self.canvas.create_rectangle(
                cx1,
                cy1,
                cx2,
                cy2,
                outline="#FFD21E",
                width=width,
            )
            self.canvas.create_text(
                cx1 + 3,
                max(
                    self.offset_y + 8,
                    cy1 - 4,
                ),
                anchor="sw",
                text=item.class_name.upper(),
                fill="#111111",
                font=(
                    "Segoe UI Semibold",
                    8,
                ),
            )

    def _canvas_to_image(
        self,
        x,
        y,
    ):
        if (
            self.image is None
            or self.scale <= 0
        ):
            return None

        image_width, image_height = (
            self.image.size
        )

        image_x = int(
            round(
                (
                    x
                    - self.offset_x
                )
                / self.scale
            )
        )
        image_y = int(
            round(
                (
                    y
                    - self.offset_y
                )
                / self.scale
            )
        )

        image_x = max(
            0,
            min(
                image_width,
                image_x,
            ),
        )
        image_y = max(
            0,
            min(
                image_height,
                image_y,
            ),
        )

        return (
            image_x,
            image_y,
        )

    def _drag_start(
        self,
        event,
    ):
        point = self._canvas_to_image(
            event.x,
            event.y,
        )

        if point is None:
            return

        self.drag_start = point

    def _drag_move(
        self,
        event,
    ):
        if (
            self.drag_start is None
            or self.image is None
        ):
            return

        current = self._canvas_to_image(
            event.x,
            event.y,
        )

        if current is None:
            return

        self._draw()

        x1, y1 = self.drag_start
        x2, y2 = current

        self.preview_rect = (
            self.canvas.create_rectangle(
                self.offset_x
                + x1 * self.scale,
                self.offset_y
                + y1 * self.scale,
                self.offset_x
                + x2 * self.scale,
                self.offset_y
                + y2 * self.scale,
                outline="#FF7A00",
                width=2,
                dash=(5, 3),
            )
        )

    def _drag_end(
        self,
        event,
    ):
        if self.drag_start is None:
            return

        current = self._canvas_to_image(
            event.x,
            event.y,
        )
        start = self.drag_start
        self.drag_start = None

        if current is None:
            self._draw()
            return

        x1 = min(
            start[0],
            current[0],
        )
        y1 = min(
            start[1],
            current[1],
        )
        x2 = max(
            start[0],
            current[0],
        )
        y2 = max(
            start[1],
            current[1],
        )

        if (
            x2 - x1 < 3
            or y2 - y1 < 3
        ):
            self._draw()
            return

        class_name = (
            self.selected_class.get()
        )

        # Keep one annotation per KTP field class.
        self.annotations = [
            item
            for item in self.annotations
            if item.class_name
            != class_name
        ]
        self.annotations.append(
            Annotation(
                class_name=class_name,
                bbox=(
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            )
        )

        self._refresh_list()
        self._draw()

    def _refresh_list(self):
        self.listbox.delete(
            0,
            tk.END,
        )

        for item in self.annotations:
            self.listbox.insert(
                tk.END,
                (
                    f"{item.class_name:<20} "
                    f"{item.bbox}"
                ),
            )

    def undo(self):
        if not self.annotations:
            return

        self.annotations.pop()
        self._refresh_list()
        self._draw()

    def delete_selected(self):
        selection = (
            self.listbox.curselection()
        )

        if not selection:
            return

        index = int(
            selection[0]
        )

        if (
            0
            <= index
            < len(
                self.annotations
            )
        ):
            self.annotations.pop(
                index
            )

        self._refresh_list()
        self._draw()

    def save_annotations(self):
        if (
            self.image is None
            or self.image_path is None
        ):
            return

        IMAGE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        LABEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_target = (
            IMAGE_DIR
            / self.image_path.name
        )
        label_target = (
            LABEL_DIR
            / (
                self.image_path.stem
                + ".txt"
            )
        )

        try:
            if (
                self.image_path.resolve()
                != image_target.resolve()
            ):
                shutil.copy2(
                    self.image_path,
                    image_target,
                )

            write_yolo_annotations(
                label_target,
                self.annotations,
                image_width=self.image.width,
                image_height=self.image.height,
            )
        except Exception as exc:
            messagebox.showerror(
                "Simpan anotasi gagal",
                str(exc),
            )
            return

        self.status_text.set(
            (
                f"Tersimpan: {len(self.annotations)} box → "
                f"{label_target}"
            )
        )
        messagebox.showinfo(
            "Anotasi tersimpan",
            (
                f"Image: {image_target}\n"
                f"Label: {label_target}"
            ),
        )


def main(
    initial_path=None,
):
    app = KtpFieldAnnotator(
        initial_path=initial_path
    )
    app.mainloop()


if __name__ == "__main__":
    main()
