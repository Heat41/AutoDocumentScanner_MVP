"""Experimental PR6 long-edge quadrilateral proposals (OpenCV only).

Unlike contour-only proposals, these use four independently visible straight
segments. Partial, non-closed physical borders may therefore be recovered.
A line intersection is never proof of a KTP: results remain *candidate
hypotheses* for the existing physical-boundary and review gates.

No sample-specific colors/coordinates, learning, annotations or file I/O.
"""
from __future__ import annotations

import cv2
import numpy as np


def _angle_difference(a, b):
    value = abs(a - b) % np.pi
    return min(value, np.pi - value)


def _intersection(a, b):
    pa, qa = a
    pb, qb = b
    u, v = qa - pa, qb - pb
    determinant = float(u[0] * v[1] - u[1] * v[0])
    if abs(determinant) < 1e-5:
        return None
    t = float(((pb - pa)[0] * v[1] - (pb - pa)[1] * v[0]) / determinant)
    return pa + t * u


def _overlap(line, p, q):
    a, b = line
    vector = q - p
    length_squared = float(vector @ vector)
    if length_squared < 1:
        return 0.
    start = float((a - p) @ vector / length_squared)
    end = float((b - p) @ vector / length_squared)
    return max(0., min(max(start, end), 1.) - max(min(start, end), 0.))


def _order(points):
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    return np.roll(points, -start, axis=0)


def line_quad_candidates(image, *, max_candidates=6, max_longest=1150):
    """Generate bounded-size proposals from four long physical-looking edges.

    Ratios and support checks are broad prefilters, NOT reliability scores.
    No candidate is accepted or saved by this function.
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.dtype != np.uint8
            or type(max_candidates) is not int or max_candidates < 1
            or type(max_longest) is not int or max_longest < 100):
        return []
    h, w = image.shape[:2]
    if min(h, w) < 75:
        return []

    factor = min(1., float(max_longest) / max(w, h))
    small = (cv2.resize(
        image, (max(2, int(round(w * factor))), max(2, int(round(h * factor)))),
        interpolation=cv2.INTER_AREA,
    ) if factor < 1 else image)
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.bitwise_or(cv2.Canny(gray, 50, 135), cv2.Canny(gray, 85, 190))

    min_length = max(20, int(.14 * min(sh, sw)))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(22, int(.045 * min(sh, sw))),
        minLineLength=min_length,
        maxLineGap=max(5, int(.022 * min(sh, sw))),
    )
    if lines is None:
        return []

    segments = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        a = np.array((float(x1), float(y1)))
        b = np.array((float(x2), float(y2)))
        vector = b - a
        length = float(np.linalg.norm(vector))
        if length >= min_length:
            angle = float(np.arctan2(vector[1], vector[0]) % np.pi)
            segments.append((a, b, angle, length))
    if len(segments) < 4:
        return []

    segments.sort(key=lambda row: -row[3])
    segments = segments[:100]
    modes = []
    for segment in segments:
        angle = segment[2]
        if all(_angle_difference(angle, mode) > .24 for mode in modes):
            modes.append(angle)
        if len(modes) == 8:
            break

    def side_pairs(group):
        pairs = []
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                if _angle_difference(first[2], second[2]) > .22:
                    continue
                direction = first[1] - first[0]
                direction /= max(float(np.linalg.norm(direction)), 1.)
                normal = np.array([-direction[1], direction[0]])
                mid_a = (first[0] + first[1]) / 2.
                mid_b = (second[0] + second[1]) / 2.
                separation = abs(float((mid_a - mid_b) @ normal))
                if (.12 * min(sw, sh) <= separation
                        <= .90 * max(sw, sh)):
                    pairs.append((min(first[3], second[3]), first, second))
        pairs.sort(key=lambda row: -row[0])
        return pairs[:18]

    hypotheses = []
    for angle_a in modes:
        group_a = [s for s in segments
                   if _angle_difference(s[2], angle_a) < .22][:16]
        if len(group_a) < 2:
            continue
        pairs_a = side_pairs(group_a)
        for angle_b in modes:
            difference = _angle_difference(angle_a, angle_b)
            if not .86 <= difference <= 2.28:
                continue
            group_b = [s for s in segments
                       if _angle_difference(s[2], angle_b) < .22][:16]
            if len(group_b) < 2:
                continue
            for _, a0, a1 in pairs_a:
                for _, b0, b1 in side_pairs(group_b):
                    line_a0 = (a0[0], a0[1])
                    line_a1 = (a1[0], a1[1])
                    line_b0 = (b0[0], b0[1])
                    line_b1 = (b1[0], b1[1])
                    p00 = _intersection(line_a0, line_b0)
                    p01 = _intersection(line_a0, line_b1)
                    p11 = _intersection(line_a1, line_b1)
                    p10 = _intersection(line_a1, line_b0)
                    if any(p is None for p in (p00, p01, p11, p10)):
                        continue
                    corners = np.asarray((p00, p01, p11, p10), dtype=np.float32)
                    if (not np.isfinite(corners).all()
                            or np.any(corners[:, 0] < -1)
                            or np.any(corners[:, 1] < -1)
                            or np.any(corners[:, 0] > sw)
                            or np.any(corners[:, 1] > sh)
                            or not cv2.isContourConvex(corners)):
                        continue
                    area = abs(float(cv2.contourArea(corners)))
                    if not .018 * sh * sw <= area <= .85 * sh * sw:
                        continue
                    sides = np.linalg.norm(
                        np.roll(corners, -1, axis=0) - corners, axis=1
                    )
                    if float(min(sides)) < .10 * min(sw, sh):
                        continue
                    pair_lengths = (
                        float(np.mean(sides[::2])),
                        float(np.mean(sides[1::2])),
                    )
                    ratio = max(pair_lengths) / max(min(pair_lengths), 1.)
                    if not 1.12 <= ratio <= 3.4:
                        continue
                    coverage = (
                        _overlap(line_a0, p00, p01),
                        _overlap(line_a1, p10, p11),
                        _overlap(line_b0, p00, p10),
                        _overlap(line_b1, p01, p11),
                    )
                    if min(coverage) < .48:
                        continue
                    proposal = _order(corners) / factor
                    # Length support, not an assumed 'largest contour wins'.
                    score = float(min(coverage) + .2 * np.mean(coverage))
                    hypotheses.append((score, proposal))

    hypotheses.sort(key=lambda item: -item[0])
    unique = []
    diagonal = float(np.hypot(w, h))
    for _, quad in hypotheses:
        if any(
            float(np.linalg.norm(quad - previous, axis=1).mean()) / diagonal < .02
            for previous in unique
        ):
            continue
        unique.append(quad)
        if len(unique) >= max_candidates:
            break
    return unique
