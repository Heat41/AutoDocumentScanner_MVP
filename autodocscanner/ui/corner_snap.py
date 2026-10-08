"""Conservative optional OpenCV corner snap for manual editors.

Only snaps to the intersection of TWO strong, nonparallel local edges.
Does not modify automatic KTP detection, and never changes source pixels.
"""
import cv2
import numpy as np


def snap_corner(image, point, *, radius=18, min_strength=35.0):
    """Return nearest reliable edge intersection or the unchanged point.

    Input may be BGR ndarray or grayscale ndarray. Radius is in ORIGINAL
    image pixels. A conservative line-intersection test avoids snapping
    arbitrarily to text and internal image corners.
    """
    if image is None or point is None:
        return point
    x, y = float(point[0]), float(point[1])
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return point

    r = max(8, int(radius))
    x0, y0 = max(0, int(x)-r*2), max(0, int(y)-r*2)
    x1, y1 = min(w, int(x)+r*2+1), min(h, int(y)+r*2+1)
    if x1-x0 < 20 or y1-y0 < 20:
        return point
    roi = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 65, 160)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=12,
        minLineLength=max(10, int(r*.9)), maxLineGap=7
    )
    if segments is None:
        return point

    lines = []
    for segment in segments.reshape(-1, 4):
        ax, ay, bx, by = map(float, segment)
        vec = np.array([bx-ax, by-ay], dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 8:
            continue
        vec /= norm
        lines.append((np.array([ax+x0, ay+y0]), vec, norm))
    best = None
    best_dist = float(r)
    for i, (a, u, len_a) in enumerate(lines):
        for b, v, len_b in lines[i+1:]:
            cross = float(np.linalg.det(np.column_stack((u, -v))))
            if abs(cross) < .40:
                continue
            t, s = np.linalg.solve(np.column_stack((u, -v)), b-a)
            # Allow only a small extension past either detected segment.
            if not (-8 <= t <= len_a+8 and -8 <= s <= len_b+8):
                continue
            p = a + t*u
            d = float(np.hypot(p[0]-x, p[1]-y))
            if d < best_dist and d <= r:
                # Require visible edge energy near both intersecting lines.
                px, py = int(round(p[0]-x0)), int(round(p[1]-y0))
                if not (2 <= px < edges.shape[1]-2 and 2 <= py < edges.shape[0]-2):
                    continue
                strength = float(np.max(edges[py-2:py+3, px-2:px+3]))
                if strength < min_strength:
                    continue
                best, best_dist = p, d
    if best is None:
        return point
    return [float(best[0]), float(best[1])]
