"""PR6 UI layout regression: use only synthetic status text, no KTP photos."""
from types import SimpleNamespace

from autodocscanner.ui.final import FinalScannerUI


def test_footer_status_wrap_tracks_available_width():
    status_widths = []
    quality_widths = []
    ui = SimpleNamespace(
        _status_detail_label=SimpleNamespace(
            configure=lambda **kw: status_widths.append(kw["wraplength"])
        ),
        _quality_detail_label=SimpleNamespace(
            configure=lambda **kw: quality_widths.append(kw["wraplength"])
        ),
    )
    for width in (900, 1100, 1360, 200):
        FinalScannerUI._resize_ktp_footer_text(
            ui, SimpleNamespace(width=width)
        )
    assert status_widths == [770, 970, 1230, 160]
    assert quality_widths == [650, 850, 1110, 160]


def test_footer_wrap_ignores_invalid_event_and_missing_widgets():
    for width in (None, "garbage", -5):
        FinalScannerUI._resize_ktp_footer_text(
            SimpleNamespace(), SimpleNamespace(width=width)
        )


def test_real_tk_ui_buttons_not_cut_off_at_narrow_and_compact(monkeypatch):
    """Run with Xvfb on Linux CI or a normal desktop DISPLAY on Windows."""
    import os
    import tkinter as tk

    if os.name != "nt" and not os.environ.get("DISPLAY"):
        import pytest
        pytest.skip("Requires virtual or desktop display")

    from autodocscanner.ui.stage2 import Stage2ScannerUI

    monkeypatch.setenv("AUTODOCSCANNER_PR6_RUNTIME_REVIEW", "1")
    root = Stage2ScannerUI()
    try:
        root.status_text.set(
            "PR6: Foto menunggu review empat sudut. "
            "Pilih foto kemudian klik Koreksi Manual. " * 4
        )
        root.quality_detail_text.set(
            "Periksa bukti batas fisik sebelum memakai hasil. " * 2
        )
        assert root.pr6_review_button is not None

        for scaling in (1.0, 1.4):
            root.tk.call("tk", "scaling", scaling)
            for width in (900, 1120, 1360):
                root.geometry(f"{width}x760+20+20")
                root.update_idletasks()
                root._apply_responsive_layout(width)
                root.update_idletasks()
                # Body may resize in narrow mode but buttons must never
                # disappear beyond the application's visible client area.
                left = root.winfo_rootx()
                right = left + root.winfo_width()
                top = root.winfo_rooty()
                bottom = top + root.winfo_height()
                for button in (root.process_button, root.manual_button,
                               root.save_button, root.save_pdf_button,
                               root.pr6_review_button):
                    assert button.winfo_ismapped(), button.cget("text")
                    assert button.winfo_width() > 35, button.cget("text")
                    assert button.winfo_rootx() >= left, button.cget("text")
                    assert (button.winfo_rootx() + button.winfo_width()
                            <= right - 4), button.cget("text")
                    assert (button.winfo_rooty() + button.winfo_height()
                            <= bottom - 4), button.cget("text")
                # The experimental button takes the entire sidebar row,
                # rather than competing horizontally with Kosongkan.
                assert (root.pr6_review_button.winfo_rooty() >=
                        root.clear_button.winfo_rooty()
                        + root.clear_button.winfo_height())
                assert (root.process_button.winfo_rooty() >
                        root._status_detail_label.winfo_rooty())
    finally:
        root.destroy()


def test_real_tk_legacy_ui_not_rearranged_into_pr6_button(monkeypatch):
    """Opt-out hides Review PR6 without changing legacy button captions."""
    import os
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        import pytest
        pytest.skip("Requires virtual or desktop display")

    from autodocscanner.ui.stage2 import Stage2ScannerUI
    monkeypatch.delenv("AUTODOCSCANNER_PR6_RUNTIME_REVIEW", raising=False)
    root = Stage2ScannerUI()
    try:
        root.update_idletasks()
        assert root.pr6_review_button is None
        assert root.process_button.cget("text") == "Proses Otomatis"
        assert root.manual_button.cget("text") == "Koreksi Manual"
    finally:
        root.destroy()
