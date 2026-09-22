from dataclasses import dataclass
from pathlib import Path

from autodocscanner.ktp.annotation import (
    Annotation,
    read_yolo_annotations,
    write_yolo_annotations,
)


@dataclass(frozen=True)
class AnnotationSeedResult:
    annotations: list[Annotation]
    source: str


def save_annotation_template_if_missing(
    template_path,
    annotations,
    image_width,
    image_height,
):
    template_path = Path(
        template_path
    )

    if template_path.is_file():
        return False

    annotations = [
        item
        for item in annotations
        if isinstance(
            item,
            Annotation,
        )
    ]

    if not annotations:
        return False

    write_yolo_annotations(
        template_path,
        annotations,
        image_width=image_width,
        image_height=image_height,
    )
    return True


def load_annotation_seed(
    own_label_path,
    template_label_path,
    image_width,
    image_height,
):
    own_label_path = Path(
        own_label_path
    )
    template_label_path = Path(
        template_label_path
    )

    if own_label_path.is_file():
        return AnnotationSeedResult(
            annotations=(
                read_yolo_annotations(
                    own_label_path,
                    image_width=image_width,
                    image_height=image_height,
                )
            ),
            source="image",
        )

    if template_label_path.is_file():
        return AnnotationSeedResult(
            annotations=(
                read_yolo_annotations(
                    template_label_path,
                    image_width=image_width,
                    image_height=image_height,
                )
            ),
            source="template",
        )

    return AnnotationSeedResult(
        annotations=[],
        source="empty",
    )



def bootstrap_annotation_template(
    template_path,
    label_dir,
):
    template_path = Path(
        template_path
    )
    label_dir = Path(
        label_dir
    )

    if template_path.is_file():
        return False

    if not label_dir.is_dir():
        return False

    candidates = sorted(
        (
            path
            for path in label_dir.glob(
                "*.txt"
            )
            if path.is_file()
            and path.stat().st_size > 0
        ),
        key=lambda path: (
            path.stat().st_mtime,
            path.name.lower(),
        ),
    )

    if not candidates:
        return False

    template_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    template_path.write_text(
        candidates[0].read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    return True
