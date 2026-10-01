from tkinter import ttk


class TrackingPlaceholderPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=28)
        self._build()

    def _build(self):
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 18))

        ttk.Label(
            header,
            text="TRACKING KTP",
            style="Eyebrow.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Tracking KTP",
            style="Header.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        ttk.Label(
            header,
            text="Fitur khusus perangkat Supervisor.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        card = ttk.Frame(
            self,
            style="Card.TFrame",
            padding=34,
        )
        card.pack(fill="both", expand=True)

        ttk.Label(
            card,
            text="🔒",
            style="SectionTitle.TLabel",
        ).pack(pady=(36, 10))

        ttk.Label(
            card,
            text="Coming Soon",
            style="SectionTitle.TLabel",
        ).pack()

        ttk.Label(
            card,
            text=(
                "OCR identitas, review data, dan rekam langsung "
                "ke website induk akan tersedia pada versi Supervisor."
            ),
            style="CardMuted.TLabel",
            wraplength=520,
            justify="center",
        ).pack(pady=(10, 24))

        features = ttk.Frame(card, style="Surface.TFrame")
        features.pack()

        for title, desc in (
            ("OCR Identitas", "Ekstraksi field KTP"),
            ("Review Data", "Verifikasi sebelum rekam"),
            ("Website Induk", "Integrasi API kantor"),
        ):
            box = ttk.Frame(
                features,
                style="Card.TFrame",
                padding=16,
            )
            box.pack(
                side="left",
                padx=6,
            )
            ttk.Label(
                box,
                text=title,
                style="CardLabel.TLabel",
            ).pack()
            ttk.Label(
                box,
                text=desc,
                style="CardMuted.TLabel",
            ).pack(pady=(4, 0))
