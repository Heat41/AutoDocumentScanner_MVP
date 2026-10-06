import tkinter as tk

from autodocscanner.ui.textured import TexturedScannerUI


class ResponsiveScannerUI(TexturedScannerUI):
    """Responsive presentation layer for desktop and windowed modes."""

    NARROW_BREAKPOINT = 980
    COMPACT_BREAKPOINT = 1200
    RESIZE_DEBOUNCE_MS = 160

    def _configure_style(self):
        super()._configure_style()
        self.minsize(820, 600)

    def _build_ui(self):
        super()._build_ui()

        self._responsive_mode = None
        self._responsive_after_id = None

        self._responsive_sidebar = getattr(
            self,
            "_files_panel",
            self.listbox.master.master,
        )
        self._responsive_root = getattr(
            self,
            "_page_root",
            None,
        )
        self._responsive_preview_parent = getattr(
            self,
            "_preview_parent",
            None,
        )

        self._responsive_preview_cards = (
            getattr(
                self.original_label,
                "_responsive_card",
                None,
            ),
            getattr(
                self.result_label,
                "_responsive_card",
                None,
            ),
        )

        self.bind(
            "<Configure>",
            self._on_responsive_configure,
            add="+",
        )
        self.after_idle(self._apply_responsive_layout)

    @staticmethod
    def _layout_mode(width):
        width = max(int(width or 0), 1)
        if width < ResponsiveScannerUI.NARROW_BREAKPOINT:
            return "narrow"
        if width < ResponsiveScannerUI.COMPACT_BREAKPOINT:
            return "compact"
        return "wide"

    def _on_responsive_configure(self, event):
        if event.widget is not self:
            return

        self._apply_responsive_layout(event.width)

        if self._responsive_after_id is not None:
            try:
                self.after_cancel(
                    self._responsive_after_id
                )
            except tk.TclError:
                pass

        self._responsive_after_id = self.after(
            self.RESIZE_DEBOUNCE_MS,
            self._refresh_resized_preview,
        )

    def _apply_responsive_layout(self, width=None):
        try:
            width = int(width or self.winfo_width())
        except (TypeError, ValueError, tk.TclError):
            return

        mode = self._layout_mode(width)
        if mode == self._responsive_mode:
            return

        self._responsive_mode = mode

        try:
            if mode == "wide":
                self._set_root_padding(
                    (28, 24, 28, 20)
                )
                self._responsive_sidebar.configure(
                    width=230
                )
                self._pack_preview_cards_horizontal(
                    gap=7
                )

            elif mode == "compact":
                self._set_root_padding(
                    (18, 18, 18, 16)
                )
                self._responsive_sidebar.configure(
                    width=190
                )
                self._pack_preview_cards_horizontal(
                    gap=5
                )

            else:
                self._set_root_padding(
                    (12, 14, 12, 12)
                )
                self._responsive_sidebar.configure(
                    width=160
                )
                self._pack_preview_cards_vertical()

        except tk.TclError:
            return

    def _set_root_padding(self, padding):
        root = self._responsive_root
        if root is not None:
            root.configure(padding=padding)

    def _pack_preview_cards_horizontal(self, gap=7):
        parent = self._responsive_preview_parent
        original, result = self._responsive_preview_cards

        if (
            parent is None
            or original is None
            or result is None
        ):
            return

        original.grid_forget()
        result.grid_forget()

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
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

        original.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, gap),
            pady=(0, 2),
        )
        result.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(gap, 0),
            pady=(0, 2),
        )

    def _pack_preview_cards_vertical(self):
        parent = self._responsive_preview_parent
        original, result = self._responsive_preview_cards

        if (
            parent is None
            or original is None
            or result is None
        ):
            return

        original.grid_forget()
        result.grid_forget()

        parent.grid_columnconfigure(
            0,
            weight=1,
            uniform="",
        )
        parent.grid_columnconfigure(1, weight=0)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        original.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=0,
            pady=(0, 5),
        )
        result.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=0,
            pady=(5, 0),
        )

    def _refresh_resized_preview(self):
        self._responsive_after_id = None

        try:
            selection = self.listbox.curselection()
            if selection:
                self._show_selected(
                    int(selection[0])
                )
        except (
            tk.TclError,
            ValueError,
            IndexError,
        ):
            pass


def main():
    app = ResponsiveScannerUI()
    app.mainloop()


if __name__ == "__main__":
    main()
