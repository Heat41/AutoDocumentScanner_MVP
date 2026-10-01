import tkinter as tk
from tkinter import ttk


class CollapsibleSidebar(ttk.Frame):
    EXPANDED_WIDTH = 252
    COLLAPSED_WIDTH = 72

    def __init__(
        self,
        parent,
        items,
        on_select,
        active_page,
        collapsed=False,
    ):
        super().__init__(
            parent,
            style="Sidebar.TFrame",
            padding=(12, 16),
            width=(
                self.COLLAPSED_WIDTH
                if collapsed
                else self.EXPANDED_WIDTH
            ),
        )
        self.pack_propagate(False)

        self.items = tuple(items)
        self.on_select = on_select
        self.active_page = active_page
        self.collapsed = bool(collapsed)
        self._buttons = {}
        self._labels = []

        self._build()

    def _build(self):
        for child in self.winfo_children():
            child.destroy()

        top = ttk.Frame(self, style="Sidebar.TFrame")
        top.pack(fill="x", pady=(0, 12))

        toggle = ttk.Button(
            top,
            text="☰",
            style="Quiet.TButton",
            command=self.toggle,
        )
        toggle.pack(
            side="left",
            fill="x" if self.collapsed else "none",
            expand=self.collapsed,
        )

        if not self.collapsed:
            brand = ttk.Frame(
                self,
                style="Sidebar.TFrame",
            )
            brand.pack(
                fill="x",
                pady=(0, 18),
            )

            ttk.Label(
                brand,
                text="AUTODOCUMENT",
                style="SidebarTitle.TLabel",
            ).pack(anchor="w")

            ttk.Label(
                brand,
                text="Office Scanner",
                style="SidebarMuted.TLabel",
            ).pack(anchor="w", pady=(2, 0))

            ttk.Label(
                self,
                text="WORKSPACE",
                style="SidebarSection.TLabel",
            ).pack(anchor="w", pady=(0, 8))

        self._buttons = {}

        for item in self.items:
            page = item["page"]
            text = (
                item["icon"]
                if self.collapsed
                else f'{item["icon"]}  {item["label"]}'
            )
            style = (
                "NavActive.TButton"
                if page == self.active_page
                else "Nav.TButton"
            )
            button = ttk.Button(
                self,
                text=text,
                style=style,
                command=lambda p=page: self.on_select(p),
            )
            button.pack(fill="x", pady=(0, 7))
            self._buttons[page] = button

        ttk.Separator(
            self,
            orient="horizontal",
        ).pack(fill="x", pady=(10, 14))

        if self.collapsed:
            ttk.Label(
                self,
                text="●",
                style="SidebarMuted.TLabel",
            ).pack()
        else:
            ttk.Label(
                self,
                text="●  Sistem siap digunakan",
                style="StatusGood.TLabel",
            ).pack(anchor="w")

            ttk.Label(
                self,
                text="SUPERVISOR",
                style="SidebarSection.TLabel",
            ).pack(anchor="w", pady=(18, 5))
            ttk.Label(
                self,
                text="Tracking KTP",
                style="SidebarTitle.TLabel",
            ).pack(anchor="w")
            ttk.Label(
                self,
                text="Coming Soon • integrasi website induk",
                style="SidebarMuted.TLabel",
                wraplength=198,
                justify="left",
            ).pack(anchor="w", pady=(3, 0))

    def set_active(self, page):
        self.active_page = page
        for key, button in self._buttons.items():
            button.configure(
                style=(
                    "NavActive.TButton"
                    if key == page
                    else "Nav.TButton"
                )
            )

    def toggle(self):
        self.collapsed = not self.collapsed
        self.configure(
            width=(
                self.COLLAPSED_WIDTH
                if self.collapsed
                else self.EXPANDED_WIDTH
            )
        )
        self._build()
