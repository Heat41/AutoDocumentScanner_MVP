import tkinter as tk
from tkinter import ttk

from autodocscanner.support.branding import brand_asset_paths
from autodocscanner.ui.final import FinalScannerUI


class TexturedScannerUI(FinalScannerUI):
    """Soft-textured visual skin for the final scanner UI.

    This class only changes presentation. The scanner, validation, loading,
    batch handling, safe output, and failure handling continue to come from
    FinalScannerUI and the locked pipeline below it.
    """

    BG = "#F4F7FB"
    CARD = "#FFFFFF"
    TEXT = "#0F172A"
    MUTED = "#64748B"
    BORDER = "#DDE5F0"
    SOFT = "#EEF3F8"
    ACCENT = "#2563EB"

    PANEL = "#0F172A"
    SHADOW = "#D9E2EC"
    TEXTURE_LIGHT = "#F8FAFC"
    TEXTURE_MID = "#E2E8F0"
    TEXTURE_DARK = "#CBD5E1"

    def __init__(self):
        super().__init__()
        self._brand_icon_photo = None
        self._apply_branding()

    def _apply_branding(self):
        """Apply the official logo without changing scanner behavior."""
        try:
            assets = brand_asset_paths()
        except Exception:
            return

        ico_path = assets.get("ico")
        png_path = assets.get("png")

        if ico_path is not None:
            try:
                self.iconbitmap(default=str(ico_path))
            except tk.TclError:
                pass

        if png_path is not None:
            try:
                self._brand_icon_photo = tk.PhotoImage(file=str(png_path))
                self.iconphoto(True, self._brand_icon_photo)
            except tk.TclError:
                self._brand_icon_photo = None

    def _configure_style(self):
        # Keep every functional style/geometry from the final UI, then apply
        # a more tactile skin on top of it.
        super()._configure_style()

        style = ttk.Style(self)

        style.configure(
            "TFrame",
            background=self.BG,
        )
        style.configure(
            "Card.TFrame",
            background=self.CARD,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Surface.TFrame",
            background=self.CARD,
        )
        style.configure(
            "Sidebar.TFrame",
            background=self.PANEL,
            borderwidth=0,
            relief="flat",
        )
        style.configure(
            "PremiumCard.TFrame",
            background=self.CARD,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabel",
            background=self.BG,
            foreground=self.TEXT,
        )
        style.configure(
            "Header.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 20),
        )
        style.configure(
            "SectionTitle.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Badge.TLabel",
            background="#EFF6FF",
            foreground="#1D4ED8",
            font=("Segoe UI Semibold", 8),
            padding=(8, 3),
        )
        style.configure(
            "StatusGood.TLabel",
            background="#12372A",
            foreground="#6EE7B7",
            font=("Segoe UI Semibold", 8),
            padding=(8, 5),
        )
        style.configure(
            "StatusChip.TLabel",
            background="#EFF6FF",
            foreground="#1D4ED8",
            font=("Segoe UI Semibold", 8),
            padding=(9, 5),
        )
        style.configure(
            "NeutralChip.TLabel",
            background="#F1F5F9",
            foreground="#475569",
            font=("Segoe UI Semibold", 8),
            padding=(9, 5),
        )
        style.configure(
            "Eyebrow.TLabel",
            background=self.BG,
            foreground="#2563EB",
            font=("Segoe UI Semibold", 8),
        )
        style.configure(
            "Subheader.TLabel",
            background=self.BG,
            foreground=self.MUTED,
        )
        style.configure(
            "CardLabel.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
        )
        style.configure(
            "CardMuted.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
        )
        style.configure(
            "SidebarTitle.TLabel",
            background=self.PANEL,
            foreground="#F8FAFC",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "SidebarMuted.TLabel",
            background=self.PANEL,
            foreground="#94A3B8",
            font=("Segoe UI", 9),
        )
        style.configure(
            "SidebarSection.TLabel",
            background=self.PANEL,
            foreground="#64748B",
            font=("Segoe UI Semibold", 8),
        )
        style.configure(
            "Quality.TLabel",
            background=self.BG,
            foreground="#43515B",
        )

        style.configure(
            "TButton",
            background="#FFFFFF",
            foreground=self.TEXT,
            borderwidth=1,
            relief="solid",
            padding=(12, 8),
        )
        style.configure(
            "Nav.TButton",
            background=self.PANEL,
            foreground="#CBD5E1",
            borderwidth=0,
            relief="flat",
            padding=(14, 11),
            anchor="w",
            font=("Segoe UI Semibold", 9),
        )
        style.configure(
            "NavActive.TButton",
            background="#1D4ED8",
            foreground="#FFFFFF",
            borderwidth=0,
            relief="flat",
            padding=(14, 11),
            anchor="w",
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "TButton",
            background=[
                ("active", "#F8FAFC"),
                ("pressed", "#F1F5F9"),
                ("disabled", "#F8FAFC"),
            ],
            foreground=[
                ("disabled", "#94A3B8"),
            ],
        )
        style.map(
            "Nav.TButton",
            background=[
                ("active", "#1E293B"),
                ("pressed", "#1E293B"),
            ],
            foreground=[
                ("active", "#FFFFFF"),
            ],
        )
        style.map(
            "NavActive.TButton",
            background=[
                ("active", "#2563EB"),
                ("pressed", "#1D4ED8"),
            ],
            foreground=[
                ("active", "#FFFFFF"),
            ],
        )

        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="#FFFFFF",
            borderwidth=1,
            relief="solid",
            padding=(20, 10),
        )
        style.map(
            "Accent.TButton",
            background=[
                ("active", "#1D4ED8"),
                ("pressed", "#1E40AF"),
                ("disabled", "#94A3B8"),
            ],
            foreground=[
                ("disabled", "#F8FAFC"),
            ],
        )

        style.configure(
            "TRadiobutton",
            background=self.BG,
            foreground="#334155",
            font=("Segoe UI", 9),
        )
        style.map(
            "TRadiobutton",
            background=[("active", self.BG)],
        )

        style.configure(
            "Horizontal.TProgressbar",
            background=self.ACCENT,
            troughcolor="#E1E6E9",
            bordercolor="#D7DDE1",
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
            thickness=9,
        )

    @staticmethod
    def _texture_positions(width, step=16):
        width = max(int(width or 0), 0)
        step = max(int(step or 1), 1)
        return list(range(0, width, step))

    def _paint_texture_strip(self, canvas):
        try:
            width = max(canvas.winfo_width(), 1)
        except tk.TclError:
            return

        canvas.delete("texture")
        canvas.create_line(
            0,
            1,
            width,
            1,
            fill=self.TEXTURE_MID,
            tags="texture",
        )

        positions = self._texture_positions(width)
        for index, x in enumerate(positions):
            tone = (
                self.TEXTURE_DARK
                if index % 2 == 0
                else self.TEXTURE_MID
            )
            canvas.create_line(
                x,
                4,
                min(x + 7, width),
                4,
                fill=tone,
                tags="texture",
            )
            dot_x = min(x + 10, width)
            canvas.create_rectangle(
                dot_x,
                6,
                min(dot_x + 1, width),
                7,
                fill=self.TEXTURE_DARK,
                outline="",
                tags="texture",
            )

    def _preview_card(self, parent, title):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(
            0,
            weight=1,
            uniform="preview",
        )
        parent.grid_columnconfigure(
            1,
            weight=1,
            uniform="preview",
        )

        column = 0 if title.startswith("Foto Asli") else 1

        shadow = tk.Frame(
            parent,
            bg=self.BG,
            bd=0,
        )
        shadow.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0, 7) if column == 0 else (7, 0),
            pady=(0, 2),
        )

        frame = tk.Frame(
            shadow,
            bg=self.CARD,
            bd=0,
            highlightthickness=0,
        )
        frame.pack(
            fill="both",
            expand=True,
            padx=(0, 2),
            pady=(0, 2),
        )

        header = tk.Frame(
            frame,
            bg=self.CARD,
            padx=14,
            pady=12,
        )
        header.pack(fill="x")

        tk.Label(
            header,
            text=title,
            bg=self.CARD,
            fg=self.TEXT,
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w")

        tk.Label(
            header,
            text=(
                "Foto sumber KTP"
                if title.startswith("Foto Asli")
                else "Hasil koreksi perspektif"
            ),
            bg=self.CARD,
            fg=self.MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(3, 0))

        content = tk.Frame(
            frame,
            bg=self.CARD,
            padx=8,
            pady=8,
        )
        content.pack(fill="both", expand=True)

        label = tk.Label(
            content,
            text=(
                "Belum ada preview\n"
                "Pilih file untuk memulai"
            ),
            bg="#F8FAFC",
            fg="#94A3B8",
            bd=0,
            relief="flat",
            font=("Segoe UI", 9),
            justify="center",
        )
        label.pack(fill="both", expand=True)

        label._responsive_card = shadow
        return label

    def _show_loading_popup(self):
        # The final UI now owns the complete themed loading experience.
        super()._show_loading_popup()


def main():
    app = TexturedScannerUI()
    app.mainloop()


if __name__ == "__main__":
    main()
