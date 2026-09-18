import tkinter as tk
from tkinter import ttk


class TrackingKtpPage(ttk.Frame):
    PAGE_TITLE = "Tracking Data KTP"
    SECTIONS = (
        "Input KTP",
        "Preview KTP",
        "Review Tracking",
    )

    def __init__(self, parent):
        super().__init__(
            parent,
            padding=18,
        )
        self.status_text = tk.StringVar(
            master=self,
            value=(
                "Stage 2D siap untuk "
                "bounding box dan OCR KTP."
            ),
        )
        self._build_ui()

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
                "Pilih KTP, proses ekstraksi, "
                "review hasil, lalu Rekam Data KTP."
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

        input_card = ttk.Frame(
            content,
            style="Card.TFrame",
            padding=14,
        )
        input_card.pack(
            side="left",
            fill="y",
            padx=(0, 7),
        )

        ttk.Label(
            input_card,
            text="Input KTP",
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            input_card,
            text=(
                "Area pemilihan foto KTP "
                "akan diaktifkan pada Stage 2D."
            ),
            style="CardMuted.TLabel",
            wraplength=210,
            justify="left",
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

        preview_card = ttk.Frame(
            content,
            style="Card.TFrame",
            padding=14,
        )
        preview_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=7,
        )

        ttk.Label(
            preview_card,
            text="Preview KTP",
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        preview = tk.Label(
            preview_card,
            text="Belum ada KTP diproses",
            bg="#E9EDF0",
            fg="#737D85",
            bd=0,
            font=("Segoe UI", 9),
        )
        preview.pack(
            fill="both",
            expand=True,
            pady=(8, 0),
        )

        review_card = ttk.Frame(
            content,
            style="Card.TFrame",
            padding=14,
        )
        review_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(7, 0),
        )

        ttk.Label(
            review_card,
            text="Review Tracking",
            style="CardLabel.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            review_card,
            text=(
                "Field identitas, alamat, "
                "status validasi, dan foto "
                "akan tampil di sini."
            ),
            style="CardMuted.TLabel",
            wraplength=260,
            justify="left",
        ).pack(
            anchor="w",
            pady=(6, 0),
        )

        ttk.Separator(
            review_card,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=14,
        )

        ttk.Button(
            review_card,
            text="Rekam Data KTP",
            state="disabled",
        ).pack(
            anchor="e",
        )

        ttk.Label(
            self,
            textvariable=self.status_text,
            style="Subheader.TLabel",
        ).pack(
            fill="x",
            pady=(10, 0),
        )
