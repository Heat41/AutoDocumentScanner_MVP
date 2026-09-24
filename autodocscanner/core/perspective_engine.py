from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PerspectiveCandidate:
    points: np.ndarray
    score: float
    source: str


class AutoPerspectiveEngine:
    """
    Auto perspective correction seperti document scanner di HP:
    cari beberapa kandidat 4-sudut, beri skor, warp kandidat terbaik,
    lalu validasi hasilnya. Warna/rasio hanya membantu penilaian; bukan
    penentu tunggal sudut dokumen.
    """

    def __init__(
        self,
        target_ratio=85.60 / 53.98,
        detection_height=1000,
        min_area_ratio=0.035,
    ):
        self.target_ratio = target_ratio
        self.detection_height = detection_height
        self.min_area_ratio = min_area_ratio

    @staticmethod
    def order_points(points):
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)

        rect = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)

        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]

        return rect

    def _resize_for_detection(self, image):
        h, w = image.shape[:2]

        if h <= self.detection_height:
            return image.copy(), 1.0

        scale = self.detection_height / float(h)
        resized = cv2.resize(
            image,
            (int(round(w * scale)), self.detection_height),
            interpolation=cv2.INTER_AREA,
        )

        return resized, scale

    @staticmethod
    def _auto_canny(gray, sigma=0.33):
        median = float(np.median(gray))
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))

        if upper <= lower:
            lower, upper = 50, 150

        return cv2.Canny(gray, lower, upper)

    def _build_detection_maps(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        normalized = clahe.apply(gray)

        smooth = cv2.bilateralFilter(
            normalized,
            7,
            45,
            45,
        )

        edges_auto = self._auto_canny(smooth)
        edges_fixed = cv2.Canny(smooth, 45, 135)

        adaptive = cv2.adaptiveThreshold(
            smooth,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        adaptive_edges = cv2.Canny(adaptive, 50, 150)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        )

        maps = []

        for edge_map in (edges_auto, edges_fixed, adaptive_edges):
            closed = cv2.morphologyEx(
                edge_map,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2,
            )
            closed = cv2.dilate(
                closed,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            maps.append(closed)

        # Union dipakai untuk edge-support scoring.
        union = cv2.bitwise_or(maps[0], maps[1])
        union = cv2.bitwise_or(union, maps[2])

        return maps, union

    def _contour_candidates(self, detection_maps):
        candidates = []

        for map_index, edge_map in enumerate(detection_maps):
            contours, _ = cv2.findContours(
                edge_map,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                continue

            image_area = edge_map.shape[0] * edge_map.shape[1]

            for contour in sorted(
                contours,
                key=cv2.contourArea,
                reverse=True,
            )[:80]:
                area = cv2.contourArea(contour)

                if area < image_area * self.min_area_ratio:
                    continue

                perimeter = cv2.arcLength(contour, True)

                if perimeter <= 0:
                    continue

                hull = cv2.convexHull(contour)

                for base in (contour, hull):
                    base_perimeter = cv2.arcLength(base, True)

                    if base_perimeter <= 0:
                        continue

                    for eps in (0.012, 0.018, 0.024, 0.032, 0.045, 0.060):
                        approx = cv2.approxPolyDP(
                            base,
                            eps * base_perimeter,
                            True,
                        )

                        if len(approx) != 4:
                            continue

                        quad = approx.reshape(4, 2).astype(np.float32)

                        if not cv2.isContourConvex(
                            quad.astype(np.int32)
                        ):
                            continue

                        candidates.append(
                            PerspectiveCandidate(
                                points=self.order_points(quad),
                                score=0.0,
                                source=f"contour_{map_index}",
                            )
                        )

                        break

        return candidates

    @staticmethod
    def _ktp_color_mask(image):
        """
        Mask badan KTP yang konservatif.

        Jangan gabungkan HSV lebar dengan background karena permukaan laptop,
        meja abu-abu, atau pantulan dapat ikut memiliki hue biru. Yang dipakai
        adalah dominasi kanal cyan/biru terhadap merah sehingga background
        netral jauh lebih sulit menyatu dengan badan kartu.
        """
        b, g, r = cv2.split(image)

        b16 = b.astype(np.int16)
        g16 = g.astype(np.int16)
        r16 = r.astype(np.int16)

        mask = (
            (b16 - r16 > 18)
            & (g16 - r16 > 3)
            & (b > 60)
        ).astype(np.uint8) * 255

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        )
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=3,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=1,
        )

        return mask

    def _fit_quad_to_contour(self, contour, initial_quad):
        """
        Refine 4 sudut dari rounded-corner KTP.

        ApproxPolyDP biasanya berhenti di lengkungan sudut, bukan di
        perpotongan teoritis sisi lurus kartu. Untuk meniru corner manual
        scanner HP, fit garis pada bagian tengah keempat sisi contour lalu
        hitung intersection-nya.
        """
        points = contour.reshape(-1, 2).astype(np.float32)

        if len(points) < 20:
            return self.order_points(initial_quad)

        quad = self.order_points(initial_quad)
        fitted_lines = []

        for index in range(4):
            p1 = quad[index]
            p2 = quad[(index + 1) % 4]

            vector = p2 - p1
            length = float(np.linalg.norm(vector))

            if length < 10:
                return quad

            unit = vector / length
            relative = points - p1

            projection = (
                relative @ unit
            ) / max(length, 1.0)

            perpendicular = np.abs(
                relative[:, 0] * unit[1]
                - relative[:, 1] * unit[0]
            )

            # Hindari rounded corner; pakai bagian tengah edge saja.
            max_distance = max(
                10.0,
                length * 0.045,
            )

            selected = (
                (projection >= 0.12)
                & (projection <= 0.88)
                & (perpendicular <= max_distance)
            )

            edge_points = points[selected]

            if len(edge_points) < 10:
                return quad

            vx, vy, x0, y0 = cv2.fitLine(
                edge_points.reshape(-1, 1, 2),
                cv2.DIST_HUBER,
                0,
                0.01,
                0.01,
            ).flatten()

            fitted_lines.append(
                (
                    np.array(
                        [x0, y0],
                        dtype=np.float32,
                    ),
                    np.array(
                        [vx, vy],
                        dtype=np.float32,
                    ),
                )
            )

        tl = self._intersect_lines(
            fitted_lines[3],
            fitted_lines[0],
        )
        tr = self._intersect_lines(
            fitted_lines[0],
            fitted_lines[1],
        )
        br = self._intersect_lines(
            fitted_lines[1],
            fitted_lines[2],
        )
        bl = self._intersect_lines(
            fitted_lines[2],
            fitted_lines[3],
        )

        if any(
            point is None
            for point in (tl, tr, br, bl)
        ):
            return quad

        refined = np.array(
            [tl, tr, br, bl],
            dtype=np.float32,
        )

        if not cv2.isContourConvex(
            refined.astype(np.int32)
        ):
            return quad

        original_area = abs(
            cv2.contourArea(
                quad.astype(np.float32)
            )
        )
        refined_area = abs(
            cv2.contourArea(
                refined.astype(np.float32)
            )
        )

        if original_area <= 1:
            return quad

        change = refined_area / original_area

        if not (0.72 <= change <= 1.32):
            return quad

        return self.order_points(refined)

    def _color_candidate(self, image):
        """
        Candidate boundary KTP dari badan cyan/biru.

        Candidate warna tetap ikut scoring umum, tetapi corner-nya terlebih
        dahulu di-refine dengan line fitting agar rounded corner tidak membuat
        crop terlalu masuk ke dalam.
        """
        mask = self._ktp_color_mask(image)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        image_area = image.shape[0] * image.shape[1]
        result = []

        for contour in sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )[:8]:
            contour_area = cv2.contourArea(contour)

            if contour_area < image_area * 0.035:
                continue

            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            if perimeter <= 0:
                continue

            initial = None

            for eps in (
                0.008,
                0.012,
                0.016,
                0.020,
                0.026,
                0.034,
                0.045,
                0.060,
            ):
                approx = cv2.approxPolyDP(
                    hull,
                    eps * perimeter,
                    True,
                )

                if len(approx) != 4:
                    continue

                quad = approx.reshape(4, 2).astype(np.float32)

                if cv2.isContourConvex(
                    quad.astype(np.int32)
                ):
                    initial = self.order_points(quad)
                    break

            if initial is None:
                hull_points = hull.reshape(-1, 2).astype(np.float32)

                if len(hull_points) < 4:
                    continue

                sums = hull_points.sum(axis=1)
                diffs = np.diff(
                    hull_points,
                    axis=1,
                ).reshape(-1)

                initial = self.order_points(
                    np.array(
                        [
                            hull_points[np.argmin(sums)],
                            hull_points[np.argmin(diffs)],
                            hull_points[np.argmax(sums)],
                            hull_points[np.argmax(diffs)],
                        ],
                        dtype=np.float32,
                    )
                )

            refined = self._fit_quad_to_contour(
                contour,
                initial,
            )

            if (
                abs(
                    cv2.contourArea(
                        refined.astype(np.float32)
                    )
                )
                < image_area * 0.035
            ):
                continue

            result.append(
                PerspectiveCandidate(
                    points=refined,
                    score=0.0,
                    source="color_refined",
                )
            )

        return result

    @staticmethod
    def _point_line_samples(p1, p2, count=40):
        t = np.linspace(0.0, 1.0, count)
        points = (
            p1[None, :] * (1.0 - t[:, None])
            + p2[None, :] * t[:, None]
        )
        return np.rint(points).astype(np.int32)

    @staticmethod
    def _edge_support(edge_map, quad):
        h, w = edge_map.shape[:2]
        supports = []

        for index in range(4):
            p1 = quad[index]
            p2 = quad[(index + 1) % 4]
            samples = AutoPerspectiveEngine._point_line_samples(
                p1,
                p2,
                count=48,
            )

            hit_count = 0

            for x, y in samples:
                if not (0 <= x < w and 0 <= y < h):
                    continue

                x1 = max(0, x - 3)
                x2 = min(w, x + 4)
                y1 = max(0, y - 3)
                y2 = min(h, y + 4)

                if np.any(edge_map[y1:y2, x1:x2] > 0):
                    hit_count += 1

            supports.append(hit_count / max(len(samples), 1))

        return float(np.mean(supports))

    @staticmethod
    def _boundary_contrast(image, quad):
        """
        Nilai apakah quad benar-benar berada di batas fisik kartu:
        band tipis di dalam dan di luar quad seharusnya memiliki perbedaan.
        """
        h, w = image.shape[:2]

        mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        polygon = np.rint(
            quad
        ).astype(np.int32)

        cv2.fillConvexPoly(
            mask,
            polygon,
            255,
        )

        radius = max(
            3,
            int(round(min(h, w) * 0.012)),
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )

        eroded = cv2.erode(
            mask,
            kernel,
            iterations=1,
        )
        dilated = cv2.dilate(
            mask,
            kernel,
            iterations=1,
        )

        inside_ring = (
            (mask > 0)
            & (eroded == 0)
        )
        outside_ring = (
            (dilated > 0)
            & (mask == 0)
        )

        if (
            np.count_nonzero(inside_ring) < 20
            or np.count_nonzero(outside_ring) < 20
        ):
            return 0.0

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)

        inside_mean = np.mean(
            lab[inside_ring],
            axis=0,
        )
        outside_mean = np.mean(
            lab[outside_ring],
            axis=0,
        )

        distance = float(
            np.linalg.norm(
                inside_mean - outside_mean
            )
        )

        return min(
            distance / 45.0,
            1.0,
        )

    def _card_color_coverage(self, image):
        mask = self._ktp_color_mask(
            image
        )

        h, w = mask.shape[:2]

        if h < 10 or w < 10:
            return 0.0, 0.0

        overall = float(
            np.mean(mask > 0)
        )

        border = max(
            2,
            int(round(min(h, w) * 0.055)),
        )

        border_mask = np.zeros_like(
            mask,
            dtype=np.uint8,
        )
        border_mask[:border, :] = 1
        border_mask[-border:, :] = 1
        border_mask[:, :border] = 1
        border_mask[:, -border:] = 1

        border_pixels = mask[
            border_mask > 0
        ]

        if border_pixels.size == 0:
            return overall, 0.0

        border_coverage = float(
            np.mean(border_pixels > 0)
        )

        return overall, border_coverage

    @staticmethod
    def _sample_map(image, points):
        h, w = image.shape[:2]

        xs = np.clip(
            np.rint(points[:, 0]).astype(np.int32),
            0,
            w - 1,
        )
        ys = np.clip(
            np.rint(points[:, 1]).astype(np.int32),
            0,
            h - 1,
        )

        return image[ys, xs]

    @staticmethod
    def _intersect_lines(line_a, line_b):
        p1, v1 = line_a
        p2, v2 = line_b

        matrix = np.column_stack(
            (v1, -v2)
        )

        det = float(
            np.linalg.det(matrix)
        )

        if abs(det) < 1e-6:
            return None

        t = np.linalg.solve(
            matrix,
            p2 - p1,
        )[0]

        return (
            p1 + t * v1
        ).astype(np.float32)

    def _snap_quad_to_boundary(self, image, quad):
        """
        Snap empat sisi kandidat ke boundary fisik kartu.

        Ini meniru perilaku document scanner HP: kandidat awal hanya memberi
        lokasi kasar, lalu setiap sisi digeser tegak-lurus sampai menemukan
        transisi kuat dari background -> badan kartu. Karena sisi digeser
        sebagai garis, perspektif/trapezoid tetap dipertahankan.
        """
        quad = self.order_points(
            quad
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        gradient = cv2.magnitude(
            gx,
            gy,
        )

        gradient = cv2.GaussianBlur(
            gradient,
            (3, 3),
            0,
        )

        color_mask = self._ktp_color_mask(
            image
        )

        h, w = image.shape[:2]

        side_lengths = [
            np.linalg.norm(
                quad[(index + 1) % 4]
                - quad[index]
            )
            for index in range(4)
        ]

        short_side = max(
            min(side_lengths),
            1.0,
        )

        search_radius = int(
            np.clip(
                short_side * 0.10,
                10,
                70,
            )
        )

        sample_gap = float(
            np.clip(
                short_side * 0.018,
                3.0,
                10.0,
            )
        )

        snapped_lines = []

        for index in range(4):
            p1 = quad[index].astype(
                np.float32
            )
            p2 = quad[
                (index + 1) % 4
            ].astype(
                np.float32
            )

            tangent = p2 - p1
            length = float(
                np.linalg.norm(tangent)
            )

            if length < 5:
                return quad

            tangent /= length

            # Dengan urutan TL,TR,BR,BL pada koordinat gambar,
            # normal ini mengarah ke bagian dalam kartu.
            inward = np.array(
                [
                    -tangent[1],
                    tangent[0],
                ],
                dtype=np.float32,
            )

            # Hindari rounded corner/glare: nilai hanya 72% bagian tengah edge.
            t_values = np.linspace(
                0.14,
                0.86,
                72,
                dtype=np.float32,
            )

            base_points = (
                p1[None, :]
                + (
                    p2 - p1
                )[None, :]
                * t_values[:, None]
            )

            best_offset = 0.0
            best_score = -1e9

            for offset in range(
                -search_radius,
                search_radius + 1,
                2,
            ):
                line_points = (
                    base_points
                    + inward[None, :]
                    * float(offset)
                )

                inside_points = (
                    line_points
                    + inward[None, :]
                    * sample_gap
                )
                outside_points = (
                    line_points
                    - inward[None, :]
                    * sample_gap
                )

                # Jangan memilih garis yang sebagian besar keluar frame.
                valid = (
                    (line_points[:, 0] >= 1)
                    & (line_points[:, 0] < w - 1)
                    & (line_points[:, 1] >= 1)
                    & (line_points[:, 1] < h - 1)
                )

                if float(
                    np.mean(valid)
                ) < 0.82:
                    continue

                line_points = line_points[
                    valid
                ]
                inside_points = inside_points[
                    valid
                ]
                outside_points = outside_points[
                    valid
                ]

                grad_values = self._sample_map(
                    gradient,
                    line_points,
                )

                # Median/percentile lebih tahan terhadap teks yang kebetulan
                # memotong garis dibanding mean biasa.
                gradient_score = float(
                    np.percentile(
                        grad_values,
                        68,
                    )
                ) / 255.0

                inside_color = self._sample_map(
                    color_mask,
                    inside_points,
                )
                outside_color = self._sample_map(
                    color_mask,
                    outside_points,
                )

                inside_fraction = float(
                    np.mean(
                        inside_color > 0
                    )
                )
                outside_fraction = float(
                    np.mean(
                        outside_color > 0
                    )
                )

                transition = max(
                    0.0,
                    inside_fraction
                    - outside_fraction,
                )

                # Preferensi kecil terhadap perpindahan minimum agar edge tidak
                # "meloncat" ke garis internal lain bila dua skor mirip.
                movement_penalty = (
                    abs(offset)
                    / max(search_radius, 1)
                ) * 0.08

                score = (
                    gradient_score * 0.52
                    + transition * 0.48
                    - movement_penalty
                )

                if score > best_score:
                    best_score = score
                    best_offset = float(
                        offset
                    )

            shifted_p1 = (
                p1
                + inward
                * best_offset
            )
            shifted_p2 = (
                p2
                + inward
                * best_offset
            )

            snapped_lines.append(
                (
                    shifted_p1,
                    shifted_p2 - shifted_p1,
                )
            )

        tl = self._intersect_lines(
            snapped_lines[3],
            snapped_lines[0],
        )
        tr = self._intersect_lines(
            snapped_lines[0],
            snapped_lines[1],
        )
        br = self._intersect_lines(
            snapped_lines[1],
            snapped_lines[2],
        )
        bl = self._intersect_lines(
            snapped_lines[2],
            snapped_lines[3],
        )

        if any(
            point is None
            for point in (
                tl,
                tr,
                br,
                bl,
            )
        ):
            return quad

        snapped = np.array(
            [tl, tr, br, bl],
            dtype=np.float32,
        )

        if not cv2.isContourConvex(
            snapped.astype(np.int32)
        ):
            return quad

        original_area = abs(
            cv2.contourArea(
                quad.astype(np.float32)
            )
        )
        snapped_area = abs(
            cv2.contourArea(
                snapped.astype(np.float32)
            )
        )

        if original_area <= 1:
            return quad

        area_ratio = (
            snapped_area
            / original_area
        )

        if not (
            0.68
            <= area_ratio
            <= 1.28
        ):
            return quad

        diagonal = float(
            np.hypot(w, h)
        )
        movement = float(
            np.mean(
                np.linalg.norm(
                    snapped - quad,
                    axis=1,
                )
            )
            / max(diagonal, 1.0)
        )

        if movement > 0.11:
            return quad

        return self.order_points(
            snapped
        )

    def _candidate_score(self, candidate, image, edge_map):
        h, w = image.shape[:2]
        image_area = float(h * w)
        quad = self.order_points(candidate.points)

        area = abs(cv2.contourArea(quad.astype(np.float32)))
        area_ratio = area / max(image_area, 1.0)

        if area_ratio < self.min_area_ratio:
            return -1.0

        tl, tr, br, bl = quad

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        if min(
            width_top,
            width_bottom,
            height_left,
            height_right,
        ) < 12:
            return -1.0

        avg_width = (width_top + width_bottom) / 2.0
        avg_height = (height_left + height_right) / 2.0

        long_side = max(avg_width, avg_height)
        short_side = min(avg_width, avg_height)

        ratio = long_side / max(short_side, 1.0)
        ratio_error = abs(ratio - self.target_ratio) / self.target_ratio
        ratio_score = max(0.0, 1.0 - (ratio_error / 0.55))

        opposite_width = min(width_top, width_bottom) / max(
            width_top,
            width_bottom,
        )
        opposite_height = min(height_left, height_right) / max(
            height_left,
            height_right,
        )
        perspective_score = (
            opposite_width + opposite_height
        ) / 2.0

        bounding = cv2.boundingRect(
            quad.astype(np.int32)
        )
        bx, by, bw, bh = bounding
        bbox_area = max(float(bw * bh), 1.0)
        fill_score = min(area / bbox_area, 1.0)

        center = quad.mean(axis=0)
        image_center = np.array(
            [w / 2.0, h / 2.0],
            dtype=np.float32,
        )
        center_distance = np.linalg.norm(center - image_center)
        max_distance = np.linalg.norm(image_center)
        center_score = max(
            0.0,
            1.0 - (center_distance / max(max_distance, 1.0)),
        )

        edge_score = self._edge_support(
            edge_map,
            quad,
        )

        # Candidate yang terlalu memenuhi 99% frame sering sebenarnya frame
        # foto, bukan dokumen. Beri penalti ringan.
        frame_penalty = 0.0
        if area_ratio > 0.96:
            frame_penalty = 0.20

        boundary_score = self._boundary_contrast(
            image,
            quad,
        )

        # Apparent ratio dapat berubah cukup besar karena perspektif.
        # Karena itu ratio hanya validator ringan, bukan faktor dominan.
        if candidate.source.startswith("color_refined"):
            source_bonus = 0.16
        elif candidate.source.startswith("color"):
            source_bonus = 0.055
        else:
            source_bonus = 0.0

        score = (
            min(area_ratio / 0.65, 1.0) * 0.22
            + ratio_score * 0.08
            + edge_score * 0.26
            + fill_score * 0.08
            + perspective_score * 0.08
            + center_score * 0.05
            + boundary_score * 0.18
            + source_bonus
            - frame_penalty
        )

        return float(score)

    @staticmethod
    def _candidate_distance(a, b, diagonal):
        a = AutoPerspectiveEngine.order_points(a)
        b = AutoPerspectiveEngine.order_points(b)

        return float(
            np.mean(
                np.linalg.norm(a - b, axis=1)
            ) / max(diagonal, 1.0)
        )

    def _deduplicate(self, candidates, image_shape):
        h, w = image_shape[:2]
        diagonal = float(np.hypot(w, h))
        unique = []

        for candidate in sorted(
            candidates,
            key=lambda item: item.score,
            reverse=True,
        ):
            duplicate = False

            for existing in unique:
                if self._candidate_distance(
                    candidate.points,
                    existing.points,
                    diagonal,
                ) < 0.035:
                    duplicate = True
                    break

            if not duplicate:
                unique.append(candidate)

        return unique

    def warp(self, image, points):
        rect = self.order_points(points)
        tl, tr, br, bl = rect

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        width = max(
            int(round(max(width_top, width_bottom))),
            2,
        )
        height = max(
            int(round(max(height_left, height_right))),
            2,
        )

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            rect,
            destination,
        )

        return cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _warp_quality(self, warped):
        if warped is None or warped.size == 0:
            return -1.0

        h, w = warped.shape[:2]

        if min(h, w) < 60:
            return -1.0

        ratio = max(w, h) / max(min(w, h), 1)
        ratio_error = abs(ratio - self.target_ratio) / self.target_ratio

        # Rasio hanya indikator ringan pada hasil raw warp.
        ratio_score = max(
            0.0,
            1.0 - ratio_error / 0.65,
        )

        gray = cv2.cvtColor(
            warped,
            cv2.COLOR_BGR2GRAY,
        )

        contrast_score = min(
            float(np.std(gray)) / 55.0,
            1.0,
        )

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        edge_density = float(
            np.mean(edges > 0)
        )
        detail_score = min(
            edge_density / 0.12,
            1.0,
        )

        color_coverage, border_coverage = (
            self._card_color_coverage(
                warped
            )
        )

        # Kandidat boundary KTP yang benar biasanya menghasilkan frame yang
        # mayoritas berisi badan kartu, termasuk area dekat tepi. Background
        # laptop/meja yang ikut ter-crop menurunkan dua nilai ini.
        card_score = min(
            color_coverage / 0.78,
            1.0,
        )
        border_score = min(
            border_coverage / 0.62,
            1.0,
        )

        return (
            ratio_score * 0.12
            + contrast_score * 0.13
            + detail_score * 0.18
            + card_score * 0.32
            + border_score * 0.25
        )

    def detect(self, image):
        resized, scale = self._resize_for_detection(
            image
        )

        detection_maps, union_edges = self._build_detection_maps(
            resized
        )

        candidates = self._contour_candidates(
            detection_maps
        )
        candidates.extend(
            self._color_candidate(resized)
        )

        if not candidates:
            return None, {
                "candidate_count": 0,
                "selected_source": None,
                "score": 0.0,
            }

        expanded_candidates = []

        for candidate in candidates:
            raw_points = self.order_points(
                candidate.points
            )

            if candidate.source == "color_refined":
                # Pertahankan fitted physical edges sebagai baseline.
                candidate.points = raw_points
                expanded_candidates.append(
                    candidate
                )

                # Beberapa foto KTP memiliki glare/warna pudar sehingga hasil
                # line-fitting bisa sedikit masuk/keluar dari boundary asli.
                # Coba versi snapped sebagai kandidat tambahan, bukan sebagai
                # pengganti paksa, agar perilaku lama tetap aman.
                snapped_points = self._snap_quad_to_boundary(
                    resized,
                    raw_points,
                )

                moved = float(
                    np.mean(
                        np.linalg.norm(
                            snapped_points - raw_points,
                            axis=1,
                        )
                    )
                )

                if moved >= 1.5:
                    expanded_candidates.append(
                        PerspectiveCandidate(
                            points=snapped_points,
                            score=0.0,
                            source="color_refined_snapped",
                        )
                    )
                continue

            candidate.points = self._snap_quad_to_boundary(
                resized,
                raw_points,
            )

            moved = float(
                np.mean(
                    np.linalg.norm(
                        candidate.points - raw_points,
                        axis=1,
                    )
                )
            )

            if moved >= 1.5:
                candidate.source = (
                    f"{candidate.source}_snapped"
                )

            expanded_candidates.append(
                candidate
            )

        candidates = expanded_candidates

        for candidate in candidates:
            candidate.score = self._candidate_score(
                candidate,
                resized,
                union_edges,
            )

        candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= 0.0
        ]

        candidates = self._deduplicate(
            candidates,
            resized.shape,
        )

        best = None
        best_total = -1.0

        # Jangan langsung percaya kandidat #1. Coba beberapa kandidat terbaik,
        # warp, lalu validasi hasilnya.
        for candidate in candidates[:14]:
            points_original = (
                candidate.points / scale
            ).astype(np.float32)

            warped = self.warp(
                image,
                points_original,
            )

            quality = self._warp_quality(
                warped
            )

            total = (
                candidate.score * 0.58
                + quality * 0.42
            )

            if total > best_total:
                best_total = total
                best = PerspectiveCandidate(
                    points=points_original,
                    score=total,
                    source=candidate.source,
                )

        if best is None or best.score < 0.32:
            return None, {
                "candidate_count": len(candidates),
                "selected_source": None,
                "score": max(best_total, 0.0),
            }

        return best.points, {
            "candidate_count": len(candidates),
            "selected_source": best.source,
            "score": float(best.score),
        }

    def correct(self, image):
        corners, metadata = self.detect(
            image
        )

        if corners is None:
            return None, None, metadata

        warped = self.warp(
            image,
            corners,
        )

        return warped, corners, metadata
