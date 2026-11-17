"""PR6 opt-in Auto KTP manual editor bridge tests — synthetic image only."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from autodocscanner.ui.manual_fallback_bridge_pr6 import (
    FLAG, experimental_manual_bridge_enabled, manual_editor_available,
    fallback_starting_corners, validate_user_confirmed_corners,
)


def test_bridge_is_disabled_by_default_and_only_exact_one_enables(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not experimental_manual_bridge_enabled()
    for value in ("", "true", "yes", "0", "2"):
        assert not experimental_manual_bridge_enabled({FLAG: value})
    assert experimental_manual_bridge_enabled({FLAG: "1"})


@pytest.mark.parametrize("has_source,detected,processing,experimental,expected", [
    (False, False, False, True, False),
    (True, False, True, True, False),
    (True, False, False, False, False),
    (True, True, False, False, True),
    (True, True, True, True, False),
    (True, False, False, True, True),
])
def test_editor_enabled_only_for_selected_source_and_safe_state(
        has_source, detected, processing, experimental, expected):
    assert manual_editor_available(
        has_source=has_source, has_detected_corners=detected,
        processing_active=processing, experimental_enabled=experimental
    ) is expected


@pytest.mark.parametrize("size", [(400, 300), (300, 400), (2000, 300), (300, 2000)])
def test_seed_is_unmodified_source_image_coordinate_quad(size):
    corners = fallback_starting_corners(size)
    assert len(corners) == 4
    assert validate_user_confirmed_corners(corners, size) is not None
    assert corners[0][0] < corners[1][0] and corners[0][1] < corners[3][1]
    assert all(0 <= x < size[0] and 0 <= y < size[1] for x, y in corners)


@pytest.mark.parametrize("size", [None, [], (300,), (0, 300), ("300", 400),
                                  (True, 300), (300.0, 400), (300, -1)])
def test_invalid_image_size_is_never_valid_for_manual_apply(size):
    assert validate_user_confirmed_corners([[10, 10], [90, 10], [90, 90], [10, 90]], size) is None


@pytest.mark.parametrize("corners", [
    None, [], [[1, 2]], [[10, 10]] * 4,
    [[10, 10], [200, 140], [200, 10], [10, 140]],
    [[10, 10], [10, 140], [200, 140], [200, 10]],
    [[10, 10], [200, 10], [200, 140], [-1, 140]],
    [[10, 10], [200, 10], [200, 140], [10, 1000]],
    [[10, 10], [200, 10], [200, 140], [float("nan"), 140]],
    [[10, 10], [200, 10], [200, 140], [float("inf"), 140]],
    [[10, 10], [10, 10], [200, 140], [10, 140]],
])
def test_invalid_crossed_or_outside_corners_never_pass(corners):
    assert validate_user_confirmed_corners(corners, (240, 180)) is None


def test_manual_quad_near_all_frame_edges_can_pass_human_confirmation():
    size = (240, 180)
    corners = [[0, 0], [239, 0], [239, 179], [0, 179]]
    assert validate_user_confirmed_corners(corners, size) == corners


def test_confirmation_copies_input_not_mutating_user_points():
    original = [[20, 20], [200, 20], [200, 150], [20, 150]]
    confirmed = validate_user_confirmed_corners(original, (240, 180))
    confirmed[0][0] = 30
    assert original[0][0] == 20


def test_gui_availability_calls_experimental_flag_without_tk_root(monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI

    fake = SimpleNamespace(files=[Path("synthetic.jpg")],
                           _active_file_index=lambda: 0,
                           _active_file_has_corners=lambda: False,
                           _processing_active=False)
    monkeypatch.delenv(FLAG, raising=False)
    assert not FinalScannerUI._manual_correction_available(fake)
    monkeypatch.setenv(FLAG, "1")
    assert FinalScannerUI._manual_correction_available(fake)
    fake._processing_active = True
    assert not FinalScannerUI._manual_correction_available(fake)


def _fake_editor(monkeypatch):
    """Exercise real dialog callbacks without Windows or a display server."""
    from autodocscanner.ui import final

    widgets, buttons = [], {}

    class Widget:
        def __init__(self, *args, **kwargs):
            self.command = kwargs.get("command")
            self.closed = False
            self.bindings = {}
            self.created = []
            self.deleted = []
            self.idle = None
            widgets.append(self)
            if "text" in kwargs and self.command:
                buttons[kwargs["text"]] = self
        def pack(self, *args, **kwargs):
            return self
        def bind(self, event, callback):
            self.bindings[event] = callback
        def winfo_width(self):
            return 700
        def winfo_height(self):
            return 490
        def update_idletasks(self):
            pass
        def delete(self, tag):
            self.deleted.append(tag)
        def create_image(self, *args, **kwargs):
            self.created.append(("image", kwargs.get("tags")))
            return len(self.created)
        def create_rectangle(self, *args, **kwargs):
            self.created.append(("rectangle", kwargs.get("tags")))
            return len(self.created)
        def create_line(self, *args, **kwargs):
            self.created.append(("line", kwargs.get("tags")))
            return len(self.created)
        def create_oval(self, *args, **kwargs):
            self.created.append(("oval", kwargs.get("tags")))
            return len(self.created)
        def create_text(self, *args, **kwargs):
            self.created.append(("text", kwargs.get("tags")))
            return len(self.created)
        def configure(self, *args, **kwargs):
            pass
        def title(self, *args, **kwargs):
            pass
        def transient(self, *args, **kwargs):
            pass
        def grab_set(self, *args, **kwargs):
            pass
        def minsize(self, *args, **kwargs):
            pass
        def after_idle(self, fn):
            # Existing apply tests do not run render; magnifier tests opt in
            # explicitly to invoke the genuine deferred canvas callbacks.
            self.idle = fn
        def destroy(self):
            self.closed = True

    for klass in ("Toplevel", "Frame", "Label", "Canvas"):
        monkeypatch.setattr(final.tk, klass, Widget)
    monkeypatch.setattr(final.ttk, "Button", Widget)
    monkeypatch.setattr(final.ImageTk, "PhotoImage", lambda *a, **kw: "synthetic-photo")
    return final, widgets, buttons


def _fake_ui(tmp_path):
    image_path = tmp_path / "synthetic.jpg"
    Image.new("RGB", (400, 300), (120, 120, 120)).save(image_path)
    status = SimpleNamespace(value="", set=lambda text: setattr(status, "value", text))
    calls = []

    class Scanner:
        last_detection = {"selected_source": "manual_correction"}
        def scan_with_corners(self, path, corners, output_path, **kwargs):
            calls.append((path, [point[:] for point in corners], output_path, kwargs))
            return np.zeros((200, 320, 3), dtype=np.uint8), np.asarray(corners, np.float32)

    fake = SimpleNamespace(
        files=[image_path], _active_file_index=lambda: 0,
        _corners_by_file={}, _output_by_file={},
        _metadata_by_file={}, _failure_by_file={str(image_path): {"message": "not detected"}},
        _processing_active=False, scanner=Scanner(), output_dir=tmp_path / "out",
        status_text=status, output_mode=SimpleNamespace(get=lambda: "color"),
        _show_selected=lambda index: None, _refresh_ktp_action_state=lambda: None,
    )
    return fake, image_path, calls, status


def test_gui_no_candidate_cancel_keeps_image_and_state_untouched(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.setenv(FLAG, "1")
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    before = path.read_bytes()
    FinalScannerUI.open_manual_correction(fake)
    assert "Reset Titik" in buttons
    assert "Terapkan Koreksi" in buttons
    assert str(path) not in fake._corners_by_file
    assert str(path) not in fake._output_by_file
    buttons["Batal"].command()
    assert widgets[0].closed
    assert not calls and path.read_bytes() == before
    assert str(path) in fake._failure_by_file


def test_gui_no_candidate_apply_only_after_click_and_clears_failure(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.setenv(FLAG, "1")
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    before = path.read_bytes()
    FinalScannerUI.open_manual_correction(fake)
    assert not calls
    buttons["Terapkan Koreksi"].command()
    assert len(calls) == 1
    assert calls[0][0] == path
    assert calls[0][1] == fallback_starting_corners((400, 300))
    assert calls[0][3]["mode"] == "ktp"
    assert str(path) in fake._corners_by_file
    assert str(path) in fake._output_by_file
    assert str(path) not in fake._failure_by_file
    assert widgets[0].closed
    assert path.read_bytes() == before


def test_gui_invalid_confirm_never_calls_scanner(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.setenv(FLAG, "1")
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    monkeypatch.setattr(
        "autodocscanner.ui.final.validate_user_confirmed_corners",
        lambda corners, size: None
    )
    FinalScannerUI.open_manual_correction(fake)
    buttons["Terapkan Koreksi"].command()
    assert not calls
    assert not widgets[0].closed
    assert "tidak valid" in status.value
    assert str(path) in fake._failure_by_file


def test_gui_feature_flag_off_keeps_old_no_corners_guard(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.delenv(FLAG, raising=False)
    fake, path, calls, status = _fake_ui(tmp_path)
    FinalScannerUI.open_manual_correction(fake)
    assert "Jalankan Proses Otomatis" in status.value
    assert not calls


def test_exif_rotated_photo_uses_same_corners_as_opencv(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    import cv2

    monkeypatch.setenv(FLAG, "1")
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    photo = Image.new("RGB", (400, 300), (90, 90, 90))
    exif = photo.getexif()
    exif[274] = 6  # clockwise 90 degrees
    photo.save(path, exif=exif)
    decoded = cv2.imread(str(path))
    assert decoded.shape[:2] == (400, 300)

    FinalScannerUI.open_manual_correction(fake)
    buttons["Terapkan Koreksi"].command()
    assert len(calls) == 1
    assert calls[0][1] == fallback_starting_corners((300, 400))
    assert widgets[0].closed


def test_gui_decoder_dimension_disagreement_fails_closed(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.setenv(FLAG, "1")
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    monkeypatch.setattr("autodocscanner.ui.final.cv2.imread",
                        lambda _path: np.zeros((55, 99, 3), dtype=np.uint8))
    FinalScannerUI.open_manual_correction(fake)
    buttons["Terapkan Koreksi"].command()
    assert not calls
    assert not widgets[0].closed
    assert "Orientasi" in status.value


def test_gui_apply_during_processing_cannot_write(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.setenv(FLAG, "1")
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    FinalScannerUI.open_manual_correction(fake)
    fake._processing_active = True
    buttons["Terapkan Koreksi"].command()
    assert not calls
    assert not widgets[0].closed
    assert "Proses otomatis" in status.value


def test_gui_failed_auto_ignores_stale_cached_detector_points(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.setenv(FLAG, "1")
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    # Even if a failed automatic attempt left previous corners in the cache,
    # the user must start from a neutral editable seed, not stale geometry.
    fake._corners_by_file[str(path)] = np.asarray(
        [[0, 0], [399, 0], [399, 299], [0, 299]], dtype=np.float32)
    FinalScannerUI.open_manual_correction(fake)
    buttons["Terapkan Koreksi"].command()
    assert len(calls) == 1
    assert calls[0][1] == fallback_starting_corners((400, 300))
    assert widgets[0].closed



def test_gui_magnifier_appears_on_press_tracks_drag_and_hides_on_release(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.setenv(FLAG, "1")
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    FinalScannerUI.open_manual_correction(fake)

    dialog = widgets[0]
    canvas = next(w for w in widgets if "<Button-1>" in w.bindings)
    assert dialog.idle is not None
    dialog.idle()  # render source preview in fake canvas
    assert not any(tags == ("manual-magnifier",) for _, tags in canvas.created)

    # Source 400 x 300, fake canvas 700 x 490. Initial TL is (24, 18).
    # Scale=min(652/400,442/300), offset=(55.3,24).
    source_scale = min((700 - 48) / 400, (490 - 48) / 300)
    left = (700 - 400 * source_scale) / 2
    top = (490 - 300 * source_scale) / 2
    click = SimpleNamespace(
        x=int(round(left + 24 * source_scale)),
        y=int(round(top + 18 * source_scale)),
    )
    canvas.bindings["<Button-1>"](click)
    assert any(tags == ("manual-magnifier",) for _, tags in canvas.created)

    before = len(canvas.created)
    canvas.bindings["<B1-Motion>"](
        SimpleNamespace(x=click.x + 15, y=click.y + 10)
    )
    assert len(canvas.created) > before
    assert len(calls) == 0  # zoom / drag never invokes scanner
    assert "manual-magnifier" in canvas.deleted

    canvas.bindings["<ButtonRelease-1>"](SimpleNamespace())
    assert canvas.deleted[-1] == "manual-magnifier"
    buttons["Batal"].command()
    assert not calls


def test_gui_magnifier_does_not_change_legacy_editor_without_flag(tmp_path, monkeypatch):
    from autodocscanner.ui.final import FinalScannerUI
    monkeypatch.delenv(FLAG, raising=False)
    _, widgets, buttons = _fake_editor(monkeypatch)
    fake, path, calls, status = _fake_ui(tmp_path)
    fake._failure_by_file.clear()
    fake._corners_by_file[str(path)] = np.asarray(
        fallback_starting_corners((400, 300)), dtype=np.float32
    )
    FinalScannerUI.open_manual_correction(fake)
    dialog = widgets[0]
    canvas = next(w for w in widgets if "<Button-1>" in w.bindings)
    dialog.idle()
    source_scale = min((700 - 48) / 400, (490 - 48) / 300)
    left = (700 - 400 * source_scale) / 2
    top = (490 - 300 * source_scale) / 2
    canvas.bindings["<Button-1>"](
        SimpleNamespace(
            x=int(round(left + 24 * source_scale)),
            y=int(round(top + 18 * source_scale)),
        )
    )
    assert not any(tags == ("manual-magnifier",) for _, tags in canvas.created)
    buttons["Batal"].command()
    assert not calls
