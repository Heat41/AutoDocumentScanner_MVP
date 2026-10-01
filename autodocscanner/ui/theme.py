import ctypes
import sys
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


def _window_handles(root):
    try:
        root.update_idletasks()
        child = int(root.winfo_id())
    except (tk.TclError, TypeError, ValueError):
        return ()

    handles = []

    try:
        parent = int(
            ctypes.windll.user32.GetParent(
                ctypes.c_void_p(child)
            )
            or 0
        )
        if parent:
            handles.append(parent)
    except Exception:
        pass

    if child not in handles:
        handles.append(child)

    return tuple(handles)


def apply_windows_titlebar(root, palette):
    """Ask Windows to use native light/dark caption rendering.

    We intentionally do not force caption/text/border colors. Windows owns
    those system buttons and forcing their colors can make minimize/maximize/
    close unreadable on some Windows 10/11 builds.
    """
    if sys.platform != "win32":
        return False

    try:
        dwm = ctypes.windll.dwmapi
        user32 = ctypes.windll.user32
    except Exception:
        return False

    dark_value = ctypes.c_int(
        1 if palette.get("name") == "dark" else 0
    )
    applied = False

    for hwnd in _window_handles(root):
        handle = ctypes.c_void_p(hwnd)

        for attribute in (20, 19):
            try:
                result = dwm.DwmSetWindowAttribute(
                    handle,
                    attribute,
                    ctypes.byref(dark_value),
                    ctypes.sizeof(dark_value),
                )
                if result == 0:
                    applied = True
                    break
            except Exception:
                continue

        # Refresh the non-client area so Windows redraws the native caption
        # and system buttons immediately.
        try:
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            SWP_FRAMECHANGED = 0x0020
            user32.SetWindowPos(
                handle,
                None,
                0,
                0,
                0,
                0,
                SWP_NOMOVE
                | SWP_NOSIZE
                | SWP_NOZORDER
                | SWP_NOACTIVATE
                | SWP_FRAMECHANGED,
            )
        except Exception:
            pass

    return applied


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
        borderwidth=1,
        relief="solid",
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

    style.configure(
        "TButton",
        background=palette["surface"],
        foreground=palette["text"],
        borderwidth=1,
        relief="solid",
        padding=(12, 8),
        font=("Segoe UI", 9),
    )
    style.map(
        "TButton",
        background=[
            ("active", palette["surface_soft"]),
            ("pressed", palette["surface_soft"]),
        ],
    )

    style.configure(
        "Accent.TButton",
        background=palette["accent"],
        foreground="#FFFFFF",
        borderwidth=1,
        relief="solid",
        padding=(18, 10),
        font=("Segoe UI Semibold", 10),
    )
    style.map(
        "Accent.TButton",
        background=[
            ("active", palette["accent_hover"]),
            ("pressed", palette["accent_hover"]),
        ],
        foreground=[("disabled", "#CBD5E1")],
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

    apply_windows_titlebar(
        root,
        palette,
    )

    return palette
