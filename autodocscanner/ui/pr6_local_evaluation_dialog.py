"""Read-only local PR6 candidate error display; no image or JSON export."""
import tkinter as tk
from tkinter import ttk


def format_local_pr6_evaluation(report):
    if (not isinstance(report, dict)
            or report.get("status") != "post_manual_reference_evaluation_only"
            or report.get("auto_select", "unsafe") is not None
            or report.get("auto_crop_allowed") is not False):
        raise ValueError("Laporan evaluasi PR6 tidak valid.")
    lines = [
        "EVALUASI EMPAT SUDUT — HANYA DI KOMPUTER INI",
        "Acuan: empat titik yang kamu tandai pada foto asli.",
        "Kesalahan relatif terhadap diagonal foto asli.",
        "Nilai terbaik setiap metode adalah ORACLE SETELAH anotasi,",
        "BUKAN pemilihan otomatis atau probabilitas benar.",
        "",
        f"Kandidat terukur: {report['candidate_count']}; "
        f"dilewati: {report['skipped_count']}",
        "", "PER METODE (terbaik / median / p90 dalam % diagonal foto):",
    ]
    for row in report.get("families", []):
        lines.append(
            f"{row['family']:<19} {row['best_source']:<22} "
            f"{100*row['best_error']:.2f}% / "
            f"{100*row['median_error']:.2f}% / "
            f"{100*row['p90_error']:.2f}%"
        )
    if not report.get("families"):
        lines.append("Tidak ada kandidat yang dapat diukur.")
    generation = report.get("generation") or {}
    if generation:
        lines.extend((
            "",
            "KETERSEDIAAN KANDIDAT PR6 (bukan akurasi):",
            "  Side Recovery: "
            + str(generation.get("side_recovery", 0))
            + " | Paired Alignment: "
            + str(generation.get("paired_alignment", 0))
            + " | Independent Sides: "
            + str(generation.get("independent_sides", 0)),
            "  Induk diuji: Paired "
            + str(generation.get("paired_parents_examined", 0))
            + " | Independent "
            + str(generation.get("independent_parents_examined", 0)),
        ))
    hidden = report.get("undisplayed_hypothesis_count", 0)
    if hidden:
        lines.append(
            f"  PERHATIAN: {hidden} hipotesis tidak masuk tampilan; "
            "angka kesalahan hanya mencakup kandidat yang ditampilkan."
        )
    skipped_details = report.get("skipped_details") or []
    if skipped_details:
        lines.extend((
            "",
            "ALASAN KANDIDAT DILEWATI (tidak masuk ranking in-frame):",
        ))
        labels = {
            "corner_outside_original_photo": (
                "Ada sudut di luar foto asli; hanya boleh evaluasi geometri"
            ),
            "outside_visual_diagnostic_extent": (
                "Sudut terlalu jauh di luar foto untuk evaluasi"
            ),
            "invalid_corner_geometry": (
                "Koordinat kandidat rusak atau tidak terbaca"
            ),
        }
        for item in skipped_details:
            lines.append(
                f"  {item['source']} ({item['family']}): "
                + labels.get(item["reason"], "Tidak valid")
            )
    outside = report.get("outside_frame_evaluations") or []
    if outside:
        lines.extend((
            "",
            "KANDIDAT DI LUAR FOTO — UKURAN GEOMETRI SAJA, BUKAN CROP:",
            "Hasil di bawah TIDAK masuk statistik metode atau ranking "
            "kandidat dalam foto.",
        ))
        for row in outside:
            lines.append(
                f"  {row['source']} ({row['family']}): "
                f"{row['mean_error_px']:.2f} px | "
                f"{100*row['normalized_error']:.2f}% diagonal | "
                f"maks {row['max_error_px']:.2f} px"
            )
            for label, error, shift in zip(
                ("Kiri-atas", "Kanan-atas", "Kanan-bawah", "Kiri-bawah"),
                row["corner_errors_px"], row["corner_offsets_xy_px"]
            ):
                lines.append(
                    f"    {label:<12} {error:.2f} px "
                    f"(dx {shift[0]:+.2f}, dy {shift[1]:+.2f})"
                )
    lines.extend(("", "PER KANDIDAT (urutan kesalahan setelah acuan manual):"))
    for row in report.get("candidates", []):
        lines.append(
            f"{row['source']:<24} {row['family']:<19} "
            f"{row['mean_error_px']:>6.2f} px | "
            f"{100*row['normalized_error']:>6.2f}% | "
            f"maks {row['max_error_px']:.2f} px"
        )
    lines.extend((
        "",
        "PER SUDUT — 5 KANDIDAT PALING DEKAT (selisih acuan manual):",
        "dx positif: kandidat di kanan acuan; dy positif: di bawah.",
    ))
    names = ("Kiri-atas", "Kanan-atas", "Kanan-bawah", "Kiri-bawah")
    for row in report.get("candidates", [])[:5]:
        if (not isinstance(row.get("corner_errors_px"), list)
                or not isinstance(row.get("corner_offsets_xy_px"), list)
                or len(row["corner_errors_px"]) != 4
                or len(row["corner_offsets_xy_px"]) != 4):
            continue
        lines.append(f"  {row['source']} ({row['family']})")
        for corner, error, offsets in zip(
            names, row["corner_errors_px"], row["corner_offsets_xy_px"]
        ):
            dx, dy = offsets
            lines.append(
                f"    {corner:<12} {error:>7.2f} px "
                f"(dx {dx:+.2f}, dy {dy:+.2f})"
            )
    lines.extend((
        "", "Bukan hasil auto-crop, tidak dipakai memilih kandidat.",
        "Tidak ada gambar, koordinat acuan atau laporan yang disimpan.",
    ))
    return "\n".join(lines)


def open_pr6_local_evaluation_dialog(parent, report):
    text = format_local_pr6_evaluation(report)
    popup = tk.Toplevel(parent)
    popup.title("Evaluasi lokal kandidat PR6 — hanya pemeriksaan")
    popup.transient(parent)
    popup.geometry(
        f"{min(950, max(640, popup.winfo_screenwidth()-80))}x"
        f"{min(650, max(400, popup.winfo_screenheight()-120))}"
    )
    shell = ttk.Frame(popup, padding=14)
    shell.pack(fill="both", expand=True)
    ttk.Label(
        shell, text="Perbandingan dengan sudut yang kamu tandai",
        font=("Segoe UI Semibold", 12),
    ).pack(anchor="w", pady=(0, 8))
    ttk.Label(
        shell, text="Evaluasi saja — tidak ada crop, simpan atau upload."
    ).pack(anchor="w", pady=(0, 8))
    frame = ttk.Frame(shell)
    frame.pack(fill="both", expand=True)
    view = tk.Text(frame, font=("Consolas", 10), wrap="none")
    scrollbar = ttk.Scrollbar(
        frame, orient="vertical", command=view.yview,
    )
    view.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    view.pack(side="left", fill="both", expand=True)
    view.insert("1.0", text)
    view.configure(state="disabled")

    def close():
        popup.destroy()
        try:
            if parent.winfo_exists():
                parent.grab_set()
        except tk.TclError:
            pass

    ttk.Button(shell, text="Tutup", command=close).pack(
        anchor="e", pady=(10, 0)
    )
    popup.protocol("WM_DELETE_WINDOW", close)
    popup.grab_set()
    return popup
