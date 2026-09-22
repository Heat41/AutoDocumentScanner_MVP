from dataclasses import dataclass
from pathlib import Path

from autodocscanner.ktp.field_detection import (
    FIELD_CLASSES,
)


CLASS_INDEX = {
    name: index
    for index, name in enumerate(
        FIELD_CLASSES
    )
}


@dataclass(frozen=True)
class Annotation:
    class_name: str
    bbox: tuple[int, int, int, int]

    def __post_init__(self):
        if self.class_name not in CLASS_INDEX:
            raise ValueError(
                f"Class anotasi tidak dikenal: {self.class_name}"
            )

        if len(self.bbox) != 4:
            raise ValueError(
                "bbox anotasi harus berisi x1, y1, x2, y2."
            )


def normalize_bbox_to_yolo(
    bbox,
    image_width,
    image_height,
):
    width = max(
        int(image_width),
        1,
    )
    height = max(
        int(image_height),
        1,
    )

    x1, y1, x2, y2 = [
        float(value)
        for value in bbox
    ]

    x1 = max(
        0.0,
        min(
            float(width),
            x1,
        ),
    )
    x2 = max(
        0.0,
        min(
            float(width),
            x2,
        ),
    )
    y1 = max(
        0.0,
        min(
            float(height),
            y1,
        ),
    )
    y2 = max(
        0.0,
        min(
            float(height),
            y2,
        ),
    )

    left = min(
        x1,
        x2,
    )
    right = max(
        x1,
        x2,
    )
    top = min(
        y1,
        y2,
    )
    bottom = max(
        y1,
        y2,
    )

    box_width = max(
        1.0,
        right - left,
    )
    box_height = max(
        1.0,
        bottom - top,
    )

    center_x = (
        left
        + box_width / 2.0
    ) / width
    center_y = (
        top
        + box_height / 2.0
    ) / height

    return (
        center_x,
        center_y,
        box_width / width,
        box_height / height,
    )


def denormalize_yolo_bbox(
    yolo_bbox,
    image_width,
    image_height,
):
    width = max(
        int(image_width),
        1,
    )
    height = max(
        int(image_height),
        1,
    )

    center_x, center_y, box_width, box_height = [
        float(value)
        for value in yolo_bbox
    ]

    pixel_width = (
        box_width
        * width
    )
    pixel_height = (
        box_height
        * height
    )
    pixel_center_x = (
        center_x
        * width
    )
    pixel_center_y = (
        center_y
        * height
    )

    x1 = int(
        round(
            pixel_center_x
            - pixel_width / 2.0
        )
    )
    y1 = int(
        round(
            pixel_center_y
            - pixel_height / 2.0
        )
    )
    x2 = int(
        round(
            pixel_center_x
            + pixel_width / 2.0
        )
    )
    y2 = int(
        round(
            pixel_center_y
            + pixel_height / 2.0
        )
    )

    return (
        max(
            0,
            min(
                width,
                x1,
            ),
        ),
        max(
            0,
            min(
                height,
                y1,
            ),
        ),
        max(
            0,
            min(
                width,
                x2,
            ),
        ),
        max(
            0,
            min(
                height,
                y2,
            ),
        ),
    )


def write_yolo_annotations(
    path,
    annotations,
    image_width,
    image_height,
):
    path = Path(
        path
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = []

    for item in annotations:
        if not isinstance(
            item,
            Annotation,
        ):
            continue

        class_id = CLASS_INDEX[
            item.class_name
        ]
        x, y, w, h = (
            normalize_bbox_to_yolo(
                item.bbox,
                image_width=image_width,
                image_height=image_height,
            )
        )
        lines.append(
            (
                f"{class_id} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )
        )

    path.write_text(
        "\n".join(
            lines
        )
        + (
            "\n"
            if lines
            else ""
        ),
        encoding="utf-8",
    )


def read_yolo_annotations(
    path,
    image_width,
    image_height,
):
    path = Path(
        path
    )

    if not path.is_file():
        return []

    result = []

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(
                parts[0]
            )
            values = tuple(
                float(value)
                for value in parts[
                    1:
                ]
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            class_id < 0
            or class_id
            >= len(
                FIELD_CLASSES
            )
        ):
            continue

        bbox = (
            denormalize_yolo_bbox(
                values,
                image_width=image_width,
                image_height=image_height,
            )
        )

        result.append(
            Annotation(
                class_name=FIELD_CLASSES[
                    class_id
                ],
                bbox=bbox,
            )
        )

    return result
