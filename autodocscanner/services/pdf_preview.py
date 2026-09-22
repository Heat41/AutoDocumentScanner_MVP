from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autodocscanner.output.manager import save_pdf


REPORT_GROUPS = (
    (
        "Identitas Utama",
        (
            ("nik", "NIK"),
            ("nama", "Nama"),
            ("tempat_lahir", "Tempat Lahir"),
            ("tanggal_lahir", "Tanggal Lahir"),
            ("jenis_kelamin", "Jenis Kelamin"),
            ("golongan_darah", "Gol. Darah"),
        ),
    ),
    (
        "Alamat",
        (
            ("alamat", "Alamat"),
            ("rt", "RT"),
            ("rw", "RW"),
            ("kelurahan_desa", "Kelurahan / Desa"),
            ("kecamatan", "Kecamatan"),
            ("kabupaten_kota", "Kabupaten / Kota"),
            ("provinsi", "Provinsi"),
        ),
    ),
    (
        "Data Lainnya",
        (
            ("agama", "Agama"),
            ("status_perkawinan", "Status Perkawinan"),
            ("pekerjaan", "Pekerjaan"),
            ("kewarganegaraan", "Kewarganegaraan"),
            ("berlaku_hingga", "Berlaku Hingga"),
        ),
    ),
)


def _validate_image(image, label):
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            f"{label} harus berupa numpy array yang tidak kosong."
        )


def _safe_stem(stem, fallback):
    value = (
        str(stem or fallback)
        .strip()
        .replace(" ", "_")
    )
    return value or fallback


def _temp_output_path(
    stem,
    temp_dir,
):
    root = Path(
        temp_dir
        if temp_dir is not None
        else gettempdir()
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return root / (
        f"{stem}_{uuid4().hex[:8]}.pdf"
    )


def build_ktp_preview_pdf(
    image,
    stem="ktp_preview",
    temp_dir=None,
):
    _validate_image(
        image,
        "image preview PDF",
    )

    output_path = _temp_output_path(
        _safe_stem(
            stem,
            "ktp_preview",
        ),
        temp_dir,
    )

    return save_pdf(
        output_path,
        image,
    )


def _to_pil_rgb(image):
    _validate_image(
        image,
        "image report",
    )

    if image.ndim == 2:
        return Image.fromarray(
            image
        ).convert("RGB")

    if (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        return Image.fromarray(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )
        )

    if (
        image.ndim == 3
        and image.shape[2] == 4
    ):
        return Image.fromarray(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGBA,
            )
        ).convert("RGB")

    raise ValueError(
        "Format image report tidak didukung."
    )


def _font(size):
    candidates = (
        "arial.ttf",
        "DejaVuSans.ttf",
    )

    for name in candidates:
        try:
            return ImageFont.truetype(
                name,
                size=size,
            )
        except OSError:
            continue

    return ImageFont.load_default()


def _wrap_text(
    draw,
    text,
    font,
    max_width,
):
    value = str(
        text or "-"
    ).strip() or "-"

    words = value.split()
    if not words:
        return ["-"]

    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = (
            current
            + " "
            + word
        )

        left, top, right, bottom = (
            draw.textbbox(
                (0, 0),
                candidate,
                font=font,
            )
        )

        if (
            right - left
            <= max_width
        ):
            current = candidate
        else:
            lines.append(
                current
            )
            current = word

    lines.append(
        current
    )
    return lines


def _paste_thumbnail(
    canvas,
    image,
    box,
):
    x1, y1, x2, y2 = box
    width = max(
        1,
        x2 - x1,
    )
    height = max(
        1,
        y2 - y1,
    )

    preview = image.copy()
    preview.thumbnail(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    offset_x = (
        x1
        + (width - preview.width)
        // 2
    )
    offset_y = (
        y1
        + (height - preview.height)
        // 2
    )

    canvas.paste(
        preview,
        (offset_x, offset_y),
    )


def build_tracking_report_image(
    corrected_image,
    face_image,
    fields,
):
    _validate_image(
        corrected_image,
        "corrected_image",
    )

    values = dict(
        fields or {}
    )

    page_width = 1240
    page_height = 1754
    margin = 70

    canvas = Image.new(
        "RGB",
        (
            page_width,
            page_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(
        canvas
    )

    title_font = _font(34)
    group_font = _font(23)
    label_font = _font(18)
    value_font = _font(18)
    small_font = _font(15)

    draw.text(
        (margin, 55),
        "Hasil Tracking Data KTP",
        fill="black",
        font=title_font,
    )
    draw.text(
        (margin, 102),
        (
            "Preview hasil koreksi KTP dan "
            "data review tracking."
        ),
        fill=(90, 90, 90),
        font=small_font,
    )

    corrected = _to_pil_rgb(
        corrected_image
    )

    ktp_box = (
        margin,
        145,
        850,
        605,
    )
    draw.rectangle(
        ktp_box,
        outline=(190, 190, 190),
        width=2,
    )
    _paste_thumbnail(
        canvas,
        corrected,
        ktp_box,
    )

    face_box = (
        900,
        180,
        1165,
        505,
    )
    draw.rectangle(
        face_box,
        outline=(190, 190, 190),
        width=2,
    )
    draw.text(
        (900, 145),
        "Foto Wajah",
        fill="black",
        font=group_font,
    )

    if (
        isinstance(
            face_image,
            np.ndarray,
        )
        and face_image.size > 0
    ):
        face = _to_pil_rgb(
            face_image
        )
        _paste_thumbnail(
            canvas,
            face,
            face_box,
        )
    else:
        draw.text(
            (940, 325),
            "Tidak tersedia",
            fill=(120, 120, 120),
            font=small_font,
        )

    y = 650
    label_x = margin
    value_x = 340
    status_x = 1110
    value_width = (
        status_x
        - value_x
        - 25
    )

    for (
        group_title,
        group_fields,
    ) in REPORT_GROUPS:
        draw.text(
            (margin, y),
            group_title,
            fill="black",
            font=group_font,
        )
        y += 38

        draw.line(
            (
                margin,
                y,
                page_width - margin,
                y,
            ),
            fill=(205, 205, 205),
            width=2,
        )
        y += 16

        for key, label in group_fields:
            value = str(
                values.get(
                    key,
                    "",
                )
                or "-"
            ).strip() or "-"

            lines = _wrap_text(
                draw,
                value,
                value_font,
                value_width,
            )

            row_height = max(
                34,
                26 * len(lines),
            )

            draw.text(
                (label_x, y),
                label,
                fill=(70, 70, 70),
                font=label_font,
            )

            line_y = y
            for line in lines:
                draw.text(
                    (value_x, line_y),
                    line,
                    fill="black",
                    font=value_font,
                )
                line_y += 25

            draw.line(
                (
                    value_x,
                    y + row_height - 5,
                    page_width - margin,
                    y + row_height - 5,
                ),
                fill=(235, 235, 235),
                width=1,
            )

            y += row_height

        y += 14

    draw.text(
        (
            margin,
            page_height - 65,
        ),
        (
            "Nilai pada laporan mengikuti "
            "field review saat Preview PDF dibuat."
        ),
        fill=(105, 105, 105),
        font=small_font,
    )

    return canvas


def build_tracking_report_pdf(
    corrected_image,
    face_image,
    fields,
    stem="tracking_ktp",
    temp_dir=None,
):
    report = build_tracking_report_image(
        corrected_image,
        face_image,
        fields,
    )

    output_path = _temp_output_path(
        _safe_stem(
            stem,
            "tracking_ktp",
        ),
        temp_dir,
    )

    report.save(
        output_path,
        format="PDF",
        resolution=150.0,
    )

    return output_path
