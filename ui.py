from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from scanner import AutoDocumentScanner


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class ScannerUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Auto Document Scanner")
        self.geometry("1040x690")
        self.minsize(920, 620)

        self.configure(
            bg="#F3F4F5"
        )

        self.scanner = AutoDocumentScanner()
        self.files = []
        self.output_dir = Path("output")

        self.output_mode = tk.StringVar(
            value="color"
        )
        self.status_text = tk.StringVar(
            value="Pilih foto KTP untuk memulai."
        )

        self._original_photo = None
        self._result_photo = None

        self._configure_style()
        self._build_ui()

    def _configure_style(self):
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TFrame",
            background="#F3F4F5",
        )
        style.configure(
            "Card.TFrame",
            background="#FFFFFF",
        )
        style.configure(
            "TLabel",
            background="#F3F4F5",
            foreground="#30343B",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Header.TLabel",
            background="#F3F4F5",
            foreground="#252A30",
            font=("Segoe UI Semibold", 17),
        )
        style.configure(
            "Muted.TLabel",
            background="#F3F4F5",
            foreground="#6C737C",
            font=("Segoe UI", 9),
        )
        style.configure(
            "TButton",
            font=("Segoe UI", 10),
            padding=(14, 8),
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(16, 9),
        )

    def _build_ui(self):
        root = ttk.Frame(
            self,
            padding=24,
        )
        root.pack(
            fill="both",
            expand=True,
        )

        header = ttk.Frame(root)
        header.pack(
            fill="x",
            pady=(0, 18),
        )

        ttk.Label(
            header,
            text="Auto Document Scanner",
            style="Header.TLabel",
        ).pack(
            anchor="w",
        )

        ttk.Label(
            header,
            text=(
                "Auto perspective correction untuk KTP. "
                "Output default tetap berwarna."
            ),
            style="Muted.TLabel",
        ).pack(
            anchor="w",
            pady=(4, 0),
        )

        toolbar = ttk.Frame(root)
        toolbar.pack(
            fill="x",
            pady=(0, 14),
        )

        ttk.Button(
            toolbar,
            text="Pilih Foto",
            command=self.choose_files,
        ).pack(
            side="left",
        )

        ttk.Button(
            toolbar,
            text="Pilih Folder",
            command=self.choose_folder,
        ).pack(
            side="left",
            padx=(8, 0),
        )

        ttk.Button(
            toolbar,
            text="Folder Output",
            command=self.choose_output,
        ).pack(
            side="left",
            padx=(8, 0),
        )

        mode_frame = ttk.Frame(
            toolbar
        )
        mode_frame.pack(
            side="right",
        )

        ttk.Radiobutton(
            mode_frame,
            text="Warna",
            variable=self.output_mode,
            value="color",
        ).pack(
            side="left",
        )

        ttk.Radiobutton(
            mode_frame,
            text="Grayscale",
            variable=self.output_mode,
            value="grayscale",
        ).pack(
            side="left",
            padx=(10, 0),
        )

        content = ttk.Frame(root)
        content.pack(
            fill="both",
            expand=True,
        )

        left = ttk.Frame(
            content,
            style="Card.TFrame",
            padding=14,
        )
        left.pack(
            side="left",
            fill="y",
        )

        ttk.Label(
            left,
            text="File",
            background="#FFFFFF",
            foreground="#30343B",
            font=("Segoe UI Semibold", 10),
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        self.listbox = tk.Listbox(
            left,
            width=31,
            borderwidth=0,
            highlightthickness=0,
            bg="#FFFFFF",
            fg="#30343B",
            selectbackground="#DDE2E6",
            selectforeground="#22262B",
            activestyle="none",
            font=("Segoe UI", 9),
        )
        self.listbox.pack(
            fill="both",
            expand=True,
        )
        self.listbox.bind(
            "<<ListboxSelect>>",
            self._on_select,
        )

        preview_area = ttk.Frame(
            content
        )
        preview_area.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(14, 0),
        )

        previews = ttk.Frame(
            preview_area
        )
        previews.pack(
            fill="both",
            expand=True,
        )

        self.original_label = self._preview_card(
            previews,
            "Original",
        )
        self.result_label = self._preview_card(
            previews,
            "Hasil",
        )

        bottom = ttk.Frame(
            root
        )
        bottom.pack(
            fill="x",
            pady=(14, 0),
        )

        ttk.Label(
            bottom,
            textvariable=self.status_text,
            style="Muted.TLabel",
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        self.process_button = ttk.Button(
            bottom,
            text="Proses Otomatis",
            style="Accent.TButton",
            command=self.process_files,
        )
        self.process_button.pack(
            side="right",
        )

    def _preview_card(
        self,
        parent,
        title,
    ):
        frame = ttk.Frame(
            parent,
            style="Card.TFrame",
            padding=12,
        )
        frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 7)
            if title == "Original"
            else (7, 0),
        )

        ttk.Label(
            frame,
            text=title,
            background="#FFFFFF",
            foreground="#30343B",
            font=("Segoe UI Semibold", 10),
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        label = tk.Label(
            frame,
            text="Belum ada gambar",
            bg="#E9ECEF",
            fg="#747B83",
            bd=0,
            font=("Segoe UI", 9),
        )
        label.pack(
            fill="both",
            expand=True,
        )

        return label

    def choose_files(self):
        paths = filedialog.askopenfilenames(
            title="Pilih foto KTP",
            filetypes=[
                (
                    "Gambar",
                    "*.jpg *.jpeg *.png *.bmp *.webp",
                )
            ],
        )

        if paths:
            self._set_files(
                [Path(path) for path in paths]
            )

    def choose_folder(self):
        folder = filedialog.askdirectory(
            title="Pilih folder foto KTP"
        )

        if not folder:
            return

        paths = sorted(
            path
            for path in Path(folder).iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )

        self._set_files(paths)

    def choose_output(self):
        folder = filedialog.askdirectory(
            title="Pilih folder output"
        )

        if folder:
            self.output_dir = Path(folder)
            self.status_text.set(
                f"Output: {self.output_dir}"
            )

    def _set_files(self, paths):
        self.files = list(paths)
        self.listbox.delete(
            0,
            tk.END,
        )

        for path in self.files:
            self.listbox.insert(
                tk.END,
                path.name,
            )

        if self.files:
            self.listbox.selection_set(0)
            self._show_original(
                self.files[0]
            )

        self.status_text.set(
            f"{len(self.files)} file dipilih."
        )

    def _on_select(self, _event=None):
        selection = self.listbox.curselection()

        if not selection:
            return

        index = selection[0]
        self._show_original(
            self.files[index]
        )

        output = (
            self.output_dir
            / f"{self.files[index].stem}_scanned.jpg"
        )

        if output.exists():
            self._show_result(
                output
            )

    @staticmethod
    def _load_preview(path):
        image = Image.open(path)
        image.thumbnail(
            (390, 430)
        )

        return ImageTk.PhotoImage(
            image
        )

    def _show_original(self, path):
        try:
            self._original_photo = (
                self._load_preview(path)
            )
            self.original_label.configure(
                image=self._original_photo,
                text="",
            )
        except Exception:
            self.original_label.configure(
                image="",
                text="Preview tidak tersedia",
            )

    def _show_result(self, path):
        try:
            self._result_photo = (
                self._load_preview(path)
            )
            self.result_label.configure(
                image=self._result_photo,
                text="",
            )
        except Exception:
            self.result_label.configure(
                image="",
                text="Preview tidak tersedia",
            )

    def process_files(self):
        if not self.files:
            messagebox.showinfo(
                "Auto Document Scanner",
                "Pilih minimal satu foto KTP.",
            )
            return

        self.process_button.configure(
            state="disabled"
        )
        self.status_text.set(
            "Memproses..."
        )

        threading.Thread(
            target=self._process_worker,
            daemon=True,
        ).start()

    def _process_worker(self):
        success = 0
        failed = 0
        last_output = None

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

            try:
                self.scanner.scan(
                    input_path,
                    output_path,
                    mode="ktp",
                    output_mode=self.output_mode.get(),
                )

                success += 1
                last_output = output_path

            except Exception:
                failed += 1

        def finish():
            self.process_button.configure(
                state="normal"
            )
            self.status_text.set(
                f"Selesai — berhasil {success}, gagal {failed}."
            )

            if last_output:
                self._show_result(
                    last_output
                )

            if failed:
                messagebox.showwarning(
                    "Proses selesai",
                    (
                        f"Berhasil: {success}\n"
                        f"Gagal: {failed}\n\n"
                        "File gagal dapat diuji kembali "
                        "setelah engine disempurnakan."
                    ),
                )

        self.after(
            0,
            finish,
        )


def main():
    app = ScannerUI()
    app.mainloop()


if __name__ == "__main__":
    main()
