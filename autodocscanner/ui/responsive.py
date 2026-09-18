import tkinter as tk

from autodocscanner.ui.textured import TexturedScannerUI


class ResponsiveScannerUI(TexturedScannerUI):
    """Responsive shell on top of the locked final/textured UI.

    The scanner pipeline and processing behavior stay untouched. This layer
    only adapts layout when the user resizes, maximizes, or restores the
    application window.
    """

    NARROW_BREAKPOINT = 900
    COMPACT_BREAKPOINT = 1050
    RESIZE_DEBOUNCE_MS = 160

    def _configure_style(self):
        super()._configure_style()

        # Allow the window to become meaningfully smaller while keeping the
        # footer usable. The layout itself will reflow at narrow widths.
        self.minsize(760, 560)

    def _build_ui(self):
        super()._build_ui()

        self._responsive_mode = None
        self._responsive_after_id = None

        # These references are derived from the widgets created by the locked
        # UI hierarchy, so no scanner/UI behavior has to be duplicated here.
        self._responsive_sidebar = self.listbox.master.master

        original_shadow = self.original_label.master.master.master
        result_shadow = self.result_label.master.master.master
        self._responsive_preview_parent = original_shadow.master
        self._responsive_preview_cards = (
            original_shadow,
            result_shadow,
        )

        self.bind("<Configure>", self._on_responsive_configure, add="+")
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
                self.after_cancel(self._responsive_after_id)
            except tk.TclError:
                pass

        # Re-render the currently selected preview after resizing has stopped,
        # so images also follow the new preview area instead of keeping their
        # old dimensions.
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
                self._responsive_sidebar.configure(width=250)
                self._pack_preview_cards_horizontal()
            elif mode == "compact":
                self._responsive_sidebar.configure(width=220)
                self._pack_preview_cards_horizontal()
            else:
                self._responsive_sidebar.configure(width=190)
                self._pack_preview_cards_vertical()
        except tk.TclError:
            return

    def _pack_preview_cards_horizontal(self):
        original, result = self._responsive_preview_cards

        original.pack_forget()
        result.pack_forget()

        original.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 7),
            pady=(0, 2),
        )
        result.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(7, 0),
            pady=(0, 2),
        )

    def _pack_preview_cards_vertical(self):
        original, result = self._responsive_preview_cards

        original.pack_forget()
        result.pack_forget()

        original.pack(
            side="top",
            fill="both",
            expand=True,
            padx=0,
            pady=(0, 6),
        )
        result.pack(
            side="top",
            fill="both",
            expand=True,
            padx=0,
            pady=(6, 0),
        )

    def _refresh_resized_preview(self):
        self._responsive_after_id = None

        try:
            selection = self.listbox.curselection()
            if selection:
                self._show_selected(int(selection[0]))
        except (tk.TclError, ValueError, IndexError):
            pass


def main():
    app = ResponsiveScannerUI()
    app.mainloop()


if __name__ == "__main__":
    main()
