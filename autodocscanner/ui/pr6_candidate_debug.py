"""Read-only Tk/Pillow visualization for PR6 quadrilateral hypotheses.

Displayed colors summarize evidence against the proposed sides; they do not
assert that any candidate is the physical KTP contour. This module never
changes image data or calls scanners, exports, network, or persistence.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageTk


SIDE_NAMES = ("Atas", "Kanan", "Bawah", "Kiri")
VERTEX_NAMES = ("TL", "TR", "BR", "BL")
TONE_GOOD = "#16A34A"
TONE_WEAK = "#EA580C"
TONE_UNKNOWN = "#64748B"


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _side_tone(side):
    """Display hint only, reproducing existing PR6 combined evidence bands."""
    if not isinstance(side, dict):
        return TONE_UNKNOWN
    external = _finite(side.get("external_support"))
    coverage = _finite(side.get("local_coverage"))
    offset = _finite(side.get("median_offset_fraction"))
    spread = _finite(side.get("offset_spread"))
    contrast = _finite(side.get("inner_outer_contrast"))
    contrast_coverage = _finite(side.get("contrast_coverage"))
    if None in (external, coverage, offset, spread, contrast, contrast_coverage):
        return TONE_UNKNOWN
    if (external >= .20 and coverage >= .5 and abs(offset) <= .25
            and spread <= .25 and contrast >= 5 and contrast_coverage >= .5):
        return TONE_GOOD
    return TONE_WEAK


def draw_pr6_candidate_overlay(source, candidate, *, canvas_size=(620, 510)):
    """Return (RGB image, offset) with a correctly mapped 4-corner overlay.

    source must be the EXIF-transposed ORIGINAL local photo. No output file.
    Draw only inside the image; no coordinate clamping or repositioning of
    the proposed geometry.
    """
    if not isinstance(source, Image.Image):
        raise ValueError("Foto asli tidak valid.")
    if (not isinstance(canvas_size, (tuple, list))
            or len(canvas_size) != 2
            or any(type(v) is not int or v < 64 for v in canvas_size)):
        raise ValueError("Ukuran preview tidak valid.")
    if source.width < 2 or source.height < 2:
        raise ValueError("Foto terlalu kecil.")
    candidate = candidate if isinstance(candidate, dict) else {}
    points = candidate.get("corners")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("Kandidat tidak memiliki empat sudut.")
    try:
        xy = [(_finite(p[0]), _finite(p[1])) for p in points]
    except (ValueError, TypeError, IndexError):
        raise ValueError("Koordinat kandidat tidak valid.") from None
    if any(x is None or y is None for x, y in xy):
        raise ValueError("Koordinat kandidat tidak valid.")
    if any(x < -.05 * source.width or x > source.width * 1.05 or
           y < -.05 * source.height or y > source.height * 1.05
           for x, y in xy):
        raise ValueError("Koordinat kandidat di luar foto.")

    w, h = canvas_size
    scale = min((w - 16) / source.width, (h - 16) / source.height)
    sw = max(1, int(round(source.width * scale)))
    sh = max(1, int(round(source.height * scale)))
    x0 = (w - sw) // 2
    y0 = (h - sh) // 2
    base = Image.new("RGB", (w, h), (242, 245, 249))
    resized = source.convert("RGB").resize((sw, sh), Image.Resampling.BILINEAR)
    base.paste(resized, (x0, y0))
    draw = ImageDraw.Draw(base)
    coords = [(x0 + x * sw / source.width,
               y0 + y * sh / source.height) for x, y in xy]
    sides = candidate.get("side_evidence")
    sides = sides if isinstance(sides, list) else []
    width = max(2, min(5, int(round(min(sw, sh) / 130))))
    for index, start in enumerate(coords):
        end = coords[(index + 1) % 4]
        color = _side_tone(sides[index] if index < len(sides) else None)
        draw.line((start, end), fill=(255, 255, 255), width=width + 3)
        draw.line((start, end), fill=color, width=width)
    for index, (x, y) in enumerate(coords):
        radius = 4
        draw.ellipse((x-radius-1, y-radius-1, x+radius+1, y+radius+1),
                     fill="#FFFFFF", outline="#1D4ED8", width=2)
        label_y = y - 18 if index < 2 else y + 8
        draw.text((x + 6, label_y), VERTEX_NAMES[index],
                  fill="#1D4ED8", stroke_width=2, stroke_fill="#FFFFFF")
    return base, (x0, y0)


def _num(value):
    v = _finite(value)
    return "–" if v is None else f"{v:.2f}"


def format_pr6_candidate_details(candidate, diagnostics):
    """Human-readable per-side evidence and refusal reasons, no PII."""
    candidate = candidate if isinstance(candidate, dict) else {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    info = []
    source = str(candidate.get("source", "?"))
    family = str(candidate.get("family", "Tidak diketahui"))
    info.append(f"KANDIDAT: {family} ({source})")
    info.append("Empat garis adalah HIPOTESIS batas kartu, bukan batas terverifikasi.")
    info.append(f"Skor ranking (bukan probabilitas): {_num(candidate.get('ranking_score'))}")
    info.append(f"Geometri valid: {'ya' if candidate.get('geometry_valid') else 'tidak'}")
    rep = candidate.get("selection_representative")
    if rep and rep != source:
        info.append(f"Perwakilan grup untuk keputusan PR6: {rep}")
        if candidate.get("measurements_visual_only"):
            info.append(
                "Metrik empat sisi di bawah dihitung khusus kandidat ini "
                "untuk tampilan saja; tidak mengubah keputusan PR6."
            )
        else:
            info.append(
                "Metrik fisik perwakilan tidak diterapkan ke kandidat ini."
            )
    elif candidate.get("measurements_visual_only"):
        info.append(
            "Metrik sisi dihitung ulang khusus untuk tampilan; bukan "
            "dasar pemilihan otomatis."
        )
    relocated = candidate.get("side_recovery_details")
    if isinstance(relocated, dict):
        info.append("")
        info.append("PEMULIHAN SATU SISI — HANYA PREVIEW")
        info.append(
            f"Kandidat asal: {relocated.get('parent_source', '?')}"
        )
        side_index = relocated.get("changed_side_index")
        info.append(
            "Sisi yang dicari ulang: "
            + (
                str(side_index + 1)
                if isinstance(side_index, int) and 0 <= side_index < 4
                else "tidak diketahui"
            )
            + " dari 4 (urutan poligon)."
        )
        info.append(
            "Perpindahan relatif: "
            + _num(relocated.get("shift_fraction_of_parent_gap"))
        )
        info.append(
            "Bukti perubahan warna sepanjang sisi: "
            + _num(relocated.get("side_score_before"))
            + " → "
            + _num(relocated.get("side_score_after"))
        )
        info.append(
            "Skor ini bukan probabilitas benar. Tiga sisi lainnya "
            "dipertahankan, dan hasil belum diperiksa sebagai KTP."
        )
        info.append(
            "Kandidat ini tidak ikut ranking, Review Gate, atau auto-crop."
        )
    outer = candidate.get("outer_recovery_check")
    if isinstance(outer, dict):
        info.append("")
        info.append("PEMULIHAN SUDUT LUAR — DIAGNOSTIK SAJA")
        info.append(
            "Homography valid secara numerik: "
            + ("ya" if outer.get("perspective_feasible") is True else "belum")
        )
        info.append(
            "Keempat sisi terukur: "
            + ("ya" if outer.get("all_sides_measurable") is True else "tidak")
        )
        info.append(
            "Risiko persegi panjang internal: "
            + ("ada" if outer.get("nested_in_larger_quad") is True else "tidak teramati")
        )
        steps = outer.get("chromatic_step_by_side")
        if isinstance(steps, list):
            info.append(
                "Perubahan warna Lab per sisi: "
                + ", ".join(
                    _num(row.get("median_lab_step"))
                    for row in steps[:4] if isinstance(row, dict)
                )
            )
        codes = outer.get("reason_codes")
        if isinstance(codes, list) and codes:
            info.extend(
                f"  • {code}" for code in codes[:12] if isinstance(code, str)
            )
        context = outer.get("region_context")
        if isinstance(context, dict):
            info.append("")
            info.append("KONTEKS WILAYAH (BUKAN IDENTIFIKASI KTP)")
            info.append(
                "Struktur internal: "
                f"{_num(context.get('interior_spatial_coverage'))} cakupan sel"
                " • " + _num(context.get("interior_edge_density")) + " kerapatan tepi"
            )
            info.append(
                "Kelompok garis internal: "
                f"{context.get('interior_axis_line_bands', 0)}"
            )
            info.append(
                "Sumber lain dengan geometri mirip: "
                + (", ".join(context.get("corroborating_families") or [])
                   or "tidak ada")
            )
            info.append(
                "Kandidat kecil yang terkandung: "
                f"{context.get('nested_inner_candidates', 0)}"
            )
            risk = context.get("reason_codes")
            if isinstance(risk, list) and risk:
                info.append("Peringatan konteks:")
                info.extend(f"  • {code}" for code in risk[:12])
            info.append(
                "Objek lain bisa memiliki struktur dan bukti sisi serupa."
            )
        info.append(
            "Kejelasan garis atau homography tidak membuktikan objek ini KTP."
        )
    info.append("")
    info.append("BUKTI EMPAT SISI")
    for index, side in enumerate(candidate.get("side_evidence") or []):
        tone = _side_tone(side)
        status = ("indikasi kuat" if tone == TONE_GOOD else
                  "indikasi lemah" if tone == TONE_WEAK else "belum cukup data")
        info.append(
            f"{SIDE_NAMES[index] if index < 4 else 'Sisi'}: {status}\n"
            f"  Dukungan tepi: {_num(side.get('external_support'))} | "
            f"Cakupan: {_num(side.get('local_coverage'))}\n"
            f"  Offset: {_num(side.get('median_offset_fraction'))} | "
            f"Sebaran: {_num(side.get('offset_spread'))}\n"
            f"  Kontras dalam/luar: {_num(side.get('inner_outer_contrast'))}"
        )
    for title, codes in (
        ("Peringatan kandidat", candidate.get("candidate_warnings")),
        ("Peringatan ranking", candidate.get("ranking_warnings")),
        ("Alasan asesmen", candidate.get("assessment_reasons")),
    ):
        info.append("")
        info.append(f"{title}:")
        if isinstance(codes, list) and codes:
            info.extend(f"  • {code}" for code in codes[:24] if isinstance(code, str))
        else:
            info.append("  Tidak ada kode khusus pada laporan.")
    info.append("")
    info.append("KEPUTUSAN AKHIR PR6")
    info.append(str(diagnostics.get("review_decision", "manual_review")))
    reasons = diagnostics.get("decision_reason_codes")
    if isinstance(reasons, list) and reasons:
        info.append("Penghambat preview:")
        info.extend(f"  • {code}" for code in reasons[:30] if isinstance(code, str))
    else:
        info.append("Tidak ada kode penghambat dari gate konsolidasi.")
    info.append("")
    info.append("DEBUG SAJA — tidak ada crop, simpan, atau pemilihan otomatis.")
    return "\n".join(info)


def open_pr6_candidate_debug_dialog(parent, path, diagnostics):
    """Show local original image with selectable PR6 candidate overlays.

    Must be called from the Tk main thread. The dialog is read-only and
    never modifies parent scanner, selection, cached outputs, or photo.
    """
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    if (diagnostics.get("status") != "offline_visual_diagnostic_only" or
            diagnostics.get("auto_crop_allowed") is not False or
            diagnostics.get("auto_select", "unexpected") is not None):
        raise ValueError("Snapshot diagnostik PR6 tidak valid.")
    candidates = diagnostics.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Daftar kandidat PR6 tidak valid.")

    with Image.open(Path(path)) as raw:
        source = ImageOps.exif_transpose(raw).convert("RGB")
    shape = diagnostics.get("image_shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or [source.height, source.width] != shape):
        raise ValueError("Orientasi foto sumber berbeda dari analisis PR6.")

    popup = tk.Toplevel(parent)
    popup.title("Visual Candidate Debugging PR6 — hanya pemeriksaan")
    popup.configure(background="#F4F7FB")
    available_w = max(parent.winfo_screenwidth() - 100, 680)
    available_h = max(parent.winfo_screenheight() - 120, 500)
    popup.geometry(f"{min(1070, available_w)}x{min(710, available_h)}")
    popup.minsize(650, 470)
    popup.transient(parent)
    popup.grab_set()

    root = ttk.Frame(popup, padding=12)
    root.pack(fill="both", expand=True)
    ttk.Label(
        root, text="Perbandingan kandidat batas fisik KTP",
        font=("Segoe UI Semibold", 14),
    ).pack(anchor="w")
    ttk.Label(
        root, text="Overlay menunjukkan hipotesis — tidak ada sudut yang otomatis diterapkan.",
        wraplength=890,
    ).pack(anchor="w", pady=(2, 8))

    picker_bar = ttk.Frame(root)
    picker_bar.pack(fill="x", pady=(0, 8))
    ttk.Label(picker_bar, text="Kandidat:").pack(side="left", padx=(0, 8))
    labels = [
        f"{row.get('family', '?')} — {row.get('source', '?')}"
        for row in candidates
    ]
    picker = ttk.Combobox(
        picker_bar, state="readonly", width=32,
        values=labels if labels else ["Tidak ditemukan kandidat"],
    )
    picker.pack(side="left", fill="x", expand=False)
    # Do not use a temporary StringVar here: when its Python owner is
    # collected, Tk loses the selection and current() becomes -1.
    # The Combobox itself owns and maintains the selected index.
    picker.current(0)
    ttk.Label(
        picker_bar, text=f"{len(candidates)} kandidat (bukan jumlah kartu)",
    ).pack(side="left", padx=(10, 0))

    counts = diagnostics.get("family_counts")
    counts = counts if isinstance(counts, dict) else {}
    summary = (
        f"Per sumber: Stable {counts.get('Stable', 0)} • "
        f"Adaptive {counts.get('Adaptive', 0)} • "
        f"Kontur V2 {counts.get('Kontur V2', 0)} • "
        f"Four-Line {counts.get('Four-Line', 0)} • "
        f"Chroma {counts.get('Chroma', 0)} • "
        f"Outer Region {counts.get('Outer Region', 0)} • "
        f"Side Recovery {counts.get('Side Recovery', 0)}"
    )
    ttk.Label(root, text=summary, wraplength=880).pack(
        anchor="w", pady=(0, 6)
    )

    main = ttk.Frame(root)
    main.pack(fill="both", expand=True)
    preview_container = ttk.Frame(main)
    preview_container.pack(side="left", fill="both", expand=True)
    canvas = tk.Canvas(
        preview_container, highlightthickness=0, background="#F2F5F9",
        width=560, height=470,
    )
    canvas.pack(fill="both", expand=True)

    detail_container = ttk.Frame(main, width=320)
    detail_container.pack(side="right", fill="y", padx=(10, 0))
    detail_container.pack_propagate(False)
    details = tk.Text(
        detail_container, wrap="word", width=38, height=27,
        font=("Consolas", 9), background="#FFFFFF", foreground="#172033",
        relief="flat", padx=8, pady=8,
    )
    scrollbar = ttk.Scrollbar(detail_container, orient="vertical", command=details.yview)
    details.configure(yscrollcommand=scrollbar.set)
    details.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    render_state = {"photo": None}

    def refresh(_event=None):
        if not popup.winfo_exists():
            return
        canvas.update_idletasks()
        width = max(160, canvas.winfo_width())
        height = max(180, canvas.winfo_height())
        canvas.delete("all")
        index = picker.current() if candidates else -1
        row = candidates[index] if 0 <= index < len(candidates) else None
        if row is not None:
            preview, _ = draw_pr6_candidate_overlay(
                source, row, canvas_size=(width, height),
            )
            render_state["photo"] = ImageTk.PhotoImage(preview, master=popup)
            canvas.create_image(0, 0, anchor="nw", image=render_state["photo"])
        else:
            canvas.create_text(
                width // 2, height // 2,
                text="Tidak ada kandidat geometri yang valid.\n"
                     "Koreksi Manual tetap tersedia.",
                fill="#475569", justify="center",
            )
        details.configure(state="normal")
        details.delete("1.0", "end")
        details.insert(
            "end", format_pr6_candidate_details(row, diagnostics)
            if row is not None else
            "Tidak ada kandidat empat sudut untuk ditampilkan.\n"
            "Gunakan Koreksi Manual pada foto asli.\n\n"
            + "\n".join(diagnostics.get("decision_reason_codes") or []),
        )
        details.configure(state="disabled")

    picker.bind("<<ComboboxSelected>>", refresh)
    pending = {"id": None}

    def on_resize(event):
        if event.widget is not canvas:
            return
        if pending["id"] is not None:
            canvas.after_cancel(pending["id"])
        pending["id"] = canvas.after(100, refresh)

    canvas.bind("<Configure>", on_resize)
    actions = ttk.Frame(root)
    actions.pack(fill="x", pady=(8, 0))
    ttk.Label(
        actions, text="Hijau: indikasi sisi kuat • Oranye: lemah • Abu: belum lengkap",
    ).pack(side="left")
    ttk.Button(actions, text="Tutup", command=popup.destroy).pack(side="right")
    popup.after_idle(refresh)
    return popup
