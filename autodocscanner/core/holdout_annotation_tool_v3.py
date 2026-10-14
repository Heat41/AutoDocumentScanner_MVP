"""Local-only, human-reviewed holdout corner annotation tool.

OpenCV GUI: left click the four *physical* KTP corners in any order.
Keys: r=reset current photo, s=save four points, n=skip, q=quit.
No automatic detector output is used as ground truth. No photo is uploaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from autodocscanner.core.quad_validation import validate_quad

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_corners(corners, shape):
    """Validate a human annotation and return four cyclic pixel coordinates."""
    points = np.asarray(corners, dtype=np.float64)
    valid, reasons, _ = validate_quad(points, shape)
    if not valid:
        raise ValueError("Invalid four corners: " + ", ".join(reasons))
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.round(np.roll(ordered, -start, axis=0), 2).tolist()


def save_annotation(path, filename, corners, shape):
    """Merge a single verified annotation; write atomically and stay local."""
    output = Path(path)
    current = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    if not isinstance(current, dict):
        raise ValueError("Annotation file must be a JSON object")
    current[filename] = normalize_corners(corners, shape)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    return current[filename]


def annotate_folder(folder, output):
    photos = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in EXTENSIONS]
    if not photos:
        print("No photos found.")
        return
    window = "KTP Holdout: click corners | S save | R reset | N skip | Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for file in photos:
            image = cv2.imread(str(file))
            if image is None:
                print(f"Skipping unreadable file: {file.name}")
                continue
            points = []
            def on_mouse(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                    points.append([float(x), float(y)])
            cv2.setMouseCallback(window, on_mouse)
            while True:
                preview = image.copy()
                for i, point in enumerate(points):
                    p = tuple(map(int, point))
                    cv2.circle(preview, p, 6, (0, 255, 255), -1)
                    cv2.putText(preview, str(i + 1), p, cv2.FONT_HERSHEY_SIMPLEX,
                                .75, (0, 0, 255), 2)
                if len(points) == 4:
                    polygon = np.asarray(points, dtype=np.int32)
                    cv2.polylines(preview, [polygon], True, (0, 255, 0), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    return
                if key == ord("r"):
                    points.clear()
                if key == ord("n"):
                    break
                if key == ord("s"):
                    if len(points) != 4:
                        print(f"Select exactly four corners first ({len(points)}/4).")
                        continue
                    try:
                        save_annotation(output, file.name, points, image.shape)
                    except ValueError as exc:
                        print(f"{file.name}: {exc}; press R to retry.")
                        continue
                    print(f"Saved annotation: {file.name}")
                    break
    finally:
        cv2.destroyWindow(window)


def main():
    parser = argparse.ArgumentParser(description="Annotate physical KTP corners locally for holdout evaluation")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, default=Path("holdout_annotations.json"))
    args = parser.parse_args()
    annotate_folder(args.folder, args.output)


if __name__ == "__main__":
    main()
