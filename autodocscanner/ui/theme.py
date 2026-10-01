import tkinter as tk
from tkinter import ttk


LIGHT_THEME = {
    "name": "light",
    "bg": "#F5F7FA",
    "surface": "#FFFFFF",
    "surface_soft": "#F8FAFC",
    "text": "#172033",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "active_bg": "#EFF6FF",
    "success": "#16A34A",
    "success_bg": "#ECFDF5",
    "warning": "#D97706",
    "sidebar": "#FFFFFF",
    "sidebar_text": "#334155",
    "sidebar_muted": "#94A3B8",
}

DARK_THEME = {
    "name": "dark",
    "bg": "#0F172A",
    "surface": "#111827",
    "surface_soft": "#1E293B",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "border": "#334155",
    "accent": "#3B82F6",
    "accent_hover": "#60A5FA",
    "active_bg": "#1E3A8A",
    "success": "#34D399",
    "success_bg": "#12372A",
    "warning": "#FBBF24",
    "sidebar": "#111827",
    "sidebar_text": "#E5E7EB",
    "sidebar_muted": "#94A3B8",
}


def get_theme(mode):
    return DARK_THEME if str(mode).lower() == "dark" else LIGHT_THEME


def apply_windows_titlebar(root, palette):
    """Keep the native Windows title bar fully system-managed.

    Tkinter's native window chrome can render inconsistently when DWM
    attributes are forced across different Windows builds and scaling setups.
    The application theme therefore affects only the client area.
    """
    return False


def apply_theme(root, mode):
    palette = get_theme(mode)
    style = ttk.Style(root)

    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.configure(bg=palette["bg"])

    style.configure(
        "TFrame",
        background=palette["bg"],
    )
    style.configure(
        "Surface.TFrame",
        background=palette["surface"],
    )
    style.configure(
        "Card.TFrame",
        background=palette["surface"],
        borderwidth=0,
        relief="flat",
    )
    style.configure(
        "Sidebar.TFrame",
        background=palette["sidebar"],
    )

    style.configure(
        "TLabel",
        background=palette["bg"],
        foreground=palette["text"],
        font=("Segoe UI", 10),
    )
    style.configure(
        "Header.TLabel",
        background=palette["bg"],
        foreground=palette["text"],
        font=("Segoe UI Semibold", 19),
    )
    style.configure(
        "Subheader.TLabel",
        background=palette["bg"],
        foreground=palette["muted"],
        font=("Segoe UI", 9),
    )
    style.configure(
        "Eyebrow.TLabel",
        background=palette["bg"],
        foreground=palette["accent"],
        font=("Segoe UI Semibold", 8),
    )
    style.configure(
        "CardLabel.TLabel",
        background=palette["surface"],
        foreground=palette["text"],
        font=("Segoe UI Semibold", 10),
    )
    style.configure(
        "CardMuted.TLabel",
        background=palette["surface"],
        foreground=palette["muted"],
        font=("Segoe UI", 9),
    )
    style.configure(
        "SectionTitle.TLabel",
        background=palette["surface"],
        foreground=palette["text"],
        font=("Segoe UI Semibold", 11),
    )
    style.configure(
        "SidebarTitle.TLabel",
        background=palette["sidebar"],
        foreground=palette["sidebar_text"],
        font=("Segoe UI Semibold", 10),
    )
    style.configure(
        "SidebarMuted.TLabel",
        background=palette["sidebar"],
        foreground=palette["sidebar_muted"],
        font=("Segoe UI", 9),
    )
    style.configure(
        "SidebarSection.TLabel",
        background=palette["sidebar"],
        foreground=palette["sidebar_muted"],
        font=("Segoe UI Semibold", 8),
    )
    style.configure(
        "Quality.TLabel",
        background=palette["bg"],
        foreground=palette["text"],
        font=("Segoe UI Semibold", 9),
    )
    style.configure(
        "Badge.TLabel",
        background=palette["active_bg"],
        foreground=palette["accent"],
        font=("Segoe UI Semibold", 8),
        padding=(8, 3),
    )
    style.configure(
        "StatusGood.TLabel",
        background=palette["success_bg"],
        foreground=palette["success"],
        font=("Segoe UI Semibold", 8),
        padding=(8, 5),
    )

    # Global button system: flatter, softer, and less boxy.
    style.configure(
        "TButton",
        background=palette["surface"],
        foreground=palette["text"],
        borderwidth=0,
        relief="flat",
        padding=(14, 9),
        font=("Segoe UI", 9),
        focusthickness=0,
        focuscolor=palette["surface"],
    )
    style.map(
        "TButton",
        background=[
            ("active", palette["surface_soft"]),
            ("pressed", palette["active_bg"]),
            ("disabled", palette["surface_soft"]),
        ],
        foreground=[
            ("disabled", palette["sidebar_muted"]),
        ],
        relief=[
            ("pressed", "flat"),
            ("active", "flat"),
        ],
    )

    style.configure(
        "Secondary.TButton",
        background=palette["surface_soft"],
        foreground=palette["text"],
        borderwidth=0,
        relief="flat",
        padding=(14, 9),
        font=("Segoe UI Semibold", 9),
        focusthickness=0,
        focuscolor=palette["surface_soft"],
    )
    style.map(
        "Secondary.TButton",
        background=[
            ("active", palette["active_bg"]),
            ("pressed", palette["active_bg"]),
        ],
        foreground=[
            ("active", palette["accent"]),
            ("pressed", palette["accent"]),
        ],
    )

    style.configure(
        "Quiet.TButton",
        background=palette["bg"],
        foreground=palette["muted"],
        borderwidth=0,
        relief="flat",
        padding=(10, 7),
        font=("Segoe UI", 9),
        focusthickness=0,
        focuscolor=palette["bg"],
    )
    style.map(
        "Quiet.TButton",
        background=[
            ("active", palette["surface_soft"]),
            ("pressed", palette["surface_soft"]),
        ],
        foreground=[
            ("active", palette["text"]),
        ],
    )

    style.configure(
        "Accent.TButton",
        background=palette["accent"],
        foreground="#FFFFFF",
        borderwidth=0,
        relief="flat",
        padding=(20, 11),
        font=("Segoe UI Semibold", 10),
        focusthickness=0,
        focuscolor=palette["accent"],
    )
    style.map(
        "Accent.TButton",
        background=[
            ("active", palette["accent_hover"]),
            ("pressed", palette["accent_hover"]),
            ("disabled", palette["sidebar_muted"]),
        ],
        foreground=[
            ("disabled", "#E2E8F0"),
        ],
        relief=[
            ("pressed", "flat"),
            ("active", "flat"),
        ],
    )

    style.configure(
        "Nav.TButton",
        background=palette["sidebar"],
        foreground=palette["sidebar_text"],
        borderwidth=0,
        relief="flat",
        padding=(14, 11),
        anchor="w",
        font=("Segoe UI Semibold", 9),
    )
    style.configure(
        "NavActive.TButton",
        background=palette["active_bg"],
        foreground=palette["accent"],
        borderwidth=0,
        relief="flat",
        padding=(14, 11),
        anchor="w",
        font=("Segoe UI Semibold", 9),
    )
    style.map(
        "Nav.TButton",
        background=[("active", palette["surface_soft"])],
    )
    style.map(
        "NavActive.TButton",
        background=[
            ("active", palette["active_bg"]),
            ("pressed", palette["active_bg"]),
        ],
    )

    style.configure(
        "TRadiobutton",
        background=palette["bg"],
        foreground=palette["text"],
        font=("Segoe UI", 9),
    )
    style.map(
        "TRadiobutton",
        background=[("active", palette["bg"])],
    )

    style.configure(
        "Horizontal.TSeparator",
        background=palette["border"],
    )

    # Global scrollbar styling. Every ttk.Scrollbar in the application
    # inherits these styles automatically.
    style.configure(
        "Vertical.TScrollbar",
        background=palette["surface_soft"],
        troughcolor=palette["surface"],
        bordercolor=palette["border"],
        arrowcolor=palette["muted"],
        lightcolor=palette["surface_soft"],
        darkcolor=palette["surface_soft"],
        relief="flat",
        borderwidth=0,
        width=12,
    )
    style.map(
        "Vertical.TScrollbar",
        background=[
            ("active", palette["accent"]),
            ("pressed", palette["accent_hover"]),
        ],
        arrowcolor=[
            ("active", "#FFFFFF"),
            ("pressed", "#FFFFFF"),
        ],
    )

    style.configure(
        "Horizontal.TScrollbar",
        background=palette["surface_soft"],
        troughcolor=palette["surface"],
        bordercolor=palette["border"],
        arrowcolor=palette["muted"],
        lightcolor=palette["surface_soft"],
        darkcolor=palette["surface_soft"],
        relief="flat",
        borderwidth=0,
        width=12,
    )
    style.map(
        "Horizontal.TScrollbar",
        background=[
            ("active", palette["accent"]),
            ("pressed", palette["accent_hover"]),
        ],
        arrowcolor=[
            ("active", "#FFFFFF"),
            ("pressed", "#FFFFFF"),
        ],
    )

    apply_windows_titlebar(
        root,
        palette,
    )

    return palette
