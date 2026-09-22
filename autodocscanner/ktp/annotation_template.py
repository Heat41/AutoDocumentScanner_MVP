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
