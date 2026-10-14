"""Local-only, human-reviewed holdout corner annotation tool.

OpenCV GUI: left click the four *physical* KTP corners in any order.
Keys: a/d=rotate preview CCW/CW, r=reset points, s=save, n=skip, q=quit.
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


def rotated_to_original(point, image_shape, turns):
    """Map a clicked pixel on a clockwise-rotated preview to original pixels."""
    x, y = map(float, point)
    h, w = image_shape[:2]
    turns %= 4
    if turns == 1:
        return [y, h - 1 - x]
    if turns == 2:
        return [w - 1 - x, h - 1 - y]
    if turns == 3:
        return [w - 1 - y, x]
    return [x, y]


def original_to_rotated(point, image_shape, turns):
    """Forward mapping for drawing previously selected corners on a preview."""
    x, y = map(float, point)
    h, w = image_shape[:2]
    turns %= 4
    if turns == 1:
        return [h - 1 - y, x]
    if turns == 2:
        return [w - 1 - x, h - 1 - y]
    if turns == 3:
        return [y, w - 1 - x]
    return [x, y]


def preview_geometry(image_shape, turns, canvas_width, canvas_height, header=54):
    """Fit the oriented photo into a fixed canvas WITHOUT stretching."""
    h, w = image_shape[:2]
    rotated_w, rotated_h = (h, w) if turns % 2 else (w, h)
    usable_h = canvas_height - header
    scale = min(canvas_width / rotated_w, usable_h / rotated_h)
    draw_w = max(1, int(round(rotated_w * scale)))
    draw_h = max(1, int(round(rotated_h * scale)))
    left = (canvas_width - draw_w) // 2
    top = header + (usable_h - draw_h) // 2
    return left, top, draw_w, draw_h


def annotate_folder(folder, output, *, width=960, height=680):
    photos = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in EXTENSIONS]
    if not photos:
        print("No photos found.")
        return
    window = "KTP Holdout V3 - A/D rotate | S save | R reset | N skip | Q quit"
    # Autosize ensures mouse callback coordinates match the rendered canvas.
    # The image itself is letterboxed into this canvas without distortion.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        for file in photos:
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping unreadable file: {file.name}")
                continue
            points = []  # Always coordinates in the ORIGINAL image.
            state = {"turns": 0, "rect": (0, 0, 1, 1)}

            def on_mouse(event, x, y, flags, userdata):
                if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
                    return
                left, top, draw_w, draw_h = state["rect"]
                if not (left <= x < left + draw_w and top <= y < top + draw_h):
                    return  # Ignore header and black letterbox padding.
                turns = state["turns"]
                oriented_h, oriented_w = (image.shape[1], image.shape[0]) if turns % 2 else image.shape[:2]
                rx = min(oriented_w - 1, max(0, (x - left) * oriented_w / draw_w))
                ry = min(oriented_h - 1, max(0, (y - top) * oriented_h / draw_h))
                points.append(rotated_to_original((rx, ry), image.shape, turns))

            cv2.setMouseCallback(window, on_mouse)
            while True:
                turns = state["turns"]
                oriented = np.rot90(image, -turns).copy()
                rh, rw = oriented.shape[:2]
                left, top, dw, dh = preview_geometry(image.shape, turns, width, height)
                state["rect"] = (left, top, dw, dh)
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                display = cv2.resize(oriented, (dw, dh), interpolation=cv2.INTER_AREA)
                canvas[top:top + dh, left:left + dw] = display
                cv2.putText(canvas, "A/D: putar 90   R: ulang   S: simpan   N: lewati   Q: keluar",
                            (14, 23), cv2.FONT_HERSHEY_SIMPLEX, .57, (240, 240, 240), 1)
                cv2.putText(canvas, f"{file.name[:65]}   Titik: {len(points)}/4   Rotasi: {turns * 90} deg",
                            (14, 45), cv2.FONT_HERSHEY_SIMPLEX, .46, (205, 205, 205), 1)
                shown = []
                for i, original in enumerate(points):
                    px, py = original_to_rotated(original, image.shape, turns)
                    sx = left + int(round(px * dw / rw))
                    sy = top + int(round(py * dh / rh))
                    shown.append([sx, sy])
                    cv2.circle(canvas, (sx, sy), 5, (0, 255, 255), -1)
                    cv2.putText(canvas, str(i + 1), (sx + 7, sy - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                if len(shown) == 4:
                    center = np.mean(shown, axis=0)
                    ordered = sorted(shown, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
                    cv2.polylines(canvas, [np.asarray(ordered, dtype=np.int32)],
                                  True, (0, 255, 0), 2)
                cv2.imshow(window, canvas)
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    return
                if key == ord("a"):
                    state["turns"] = (state["turns"] - 1) % 4
                if key == ord("d"):
                    state["turns"] = (state["turns"] + 1) % 4
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
                    print(f"Saved annotation in original coordinates: {file.name}")
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
