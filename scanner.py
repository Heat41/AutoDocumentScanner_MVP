from pathlib import Path

import cv2
import numpy as np


class AutoDocumentScanner:
    KTP_ASPECT_RATIO = 85.60 / 53.98

    def __init__(
        self,
        detection_height=900,
        ktp_corner_padding=0.0,
        ktp_aspect_ratio=KTP_ASPECT_RATIO,
        ktp_safe_margin=0.0,
        ktp_edge_trim=0.006,
    ):
        self.detection_height = detection_height
        self.ktp_corner_padding = ktp_corner_padding
        self.ktp_aspect_ratio = ktp_aspect_ratio
        self.ktp_safe_margin = ktp_safe_margin
        self.ktp_edge_trim = ktp_edge_trim

    # ============================================================
    # POINT ORDERING
    # ============================================================

    @staticmethod
    def order_points(points):
        points = np.asarray(points, dtype=np.float32)

        if points.shape != (4, 2):
            raise ValueError("Perspective transform membutuhkan tepat 4 titik.")

        rect = np.zeros((4, 2), dtype=np.float32)

        s = points.sum(axis=1)
        diff = np.diff(points, axis=1).reshape(-1)

        rect[0] = points[np.argmin(s)]      # top-left
        rect[2] = points[np.argmax(s)]      # bottom-right
        rect[1] = points[np.argmin(diff)]   # top-right
        rect[3] = points[np.argmax(diff)]   # bottom-left

        return rect

    # ============================================================
    # CORNER REFINEMENT
    # ============================================================

    def expand_corners(self, points, image_shape, mode="document"):
        """
        Sedikit memperluas quadrilateral agar tepi kartu/dokumen tidak
        terpotong terlalu rapat, terutama pada KTP yang bersudut membulat.
        """
        rect = self.order_points(points)

        if mode != "ktp" or self.ktp_corner_padding <= 0:
            return rect

        center = rect.mean(axis=0)
        expanded = center + (rect - center) * (1.0 + self.ktp_corner_padding)

        height, width = image_shape[:2]
        expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)

        return expanded.astype(np.float32)

    # ============================================================
    # PERSPECTIVE TRANSFORM
    # ============================================================

    def perspective_transform(self, image, points, mode="document"):
        rect = self.expand_corners(
            points,
            image.shape,
            mode=mode,
        )

        tl, tr, br, bl = rect

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = int(round(max(width_top, width_bottom)))

        height_right = np.linalg.norm(br - tr)
        height_left = np.linalg.norm(bl - tl)
        max_height = int(round(max(height_right, height_left)))

        max_width = max(max_width, 2)
        max_height = max(max_height, 2)

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            rect,
            destination,
        )

        warped = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return warped

    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================

    @staticmethod
    def preprocess(image):
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.bilateralFilter(
            gray,
            7,
            45,
            45,
        )

        median = float(np.median(gray))

        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))

        if upper <= lower:
            lower, upper = 50, 150

        edges = cv2.Canny(
            gray,
            lower,
            upper,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        )

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        edges = cv2.dilate(
            edges,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        return edges

    # ============================================================
    # KTP COLOR CANDIDATE
    # ============================================================

    def _detect_ktp_color_candidate(self, image):
        """
        Fallback khusus KTP. Area cyan-biru digabungkan lalu convex hull-nya
        dipakai untuk memperkirakan empat sudut kartu. Kandidat warna hanya
        dipakai bila contour tepi biasa gagal, agar tidak merusak kasus yang
        sudah terdeteksi dengan baik.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            np.array([70, 14, 55], dtype=np.uint8),
            np.array([122, 255, 255], dtype=np.uint8),
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=4,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]
        useful = [
            contour
            for contour in contours
            if cv2.contourArea(contour) >= image_area * 0.004
        ]

        if not useful:
            return None

        points = np.vstack(useful)
        hull = cv2.convexHull(points)

        perimeter = cv2.arcLength(hull, True)

        for epsilon_ratio in (0.012, 0.018, 0.025, 0.035, 0.05):
            approx = cv2.approxPolyDP(
                hull,
                epsilon_ratio * perimeter,
                True,
            )

            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)

                if cv2.isContourConvex(quad.astype(np.int32)):
                    return quad

        hull_points = hull.reshape(-1, 2).astype(np.float32)

        sums = hull_points.sum(axis=1)
        diffs = np.diff(hull_points, axis=1).reshape(-1)

        quad = np.array(
            [
                hull_points[np.argmin(sums)],
                hull_points[np.argmin(diffs)],
                hull_points[np.argmax(sums)],
                hull_points[np.argmax(diffs)],
            ],
            dtype=np.float32,
        )

        if abs(cv2.contourArea(quad)) < image_area * 0.08:
            return None

        return quad

    def _detect_ktp_border_line_candidate(self, image):
        """
        Deteksi empat sisi KTP dari boundary warna cyan, lalu fit garis
        top/right/bottom/left dan ambil titik perpotongannya.

        Ini lebih cocok untuk kasus KTP miring/perspektif di atas laptop/meja
        dibanding minAreaRect karena mempertahankan trapezoid sebenarnya.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            np.array([72, 40, 50], dtype=np.uint8),
            np.array([122, 255, 255], dtype=np.uint8),
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=3,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            return None

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        image_area = image.shape[0] * image.shape[1]

        if cv2.contourArea(contour) < image_area * 0.20:
            return None

        points = contour.reshape(-1, 2).astype(np.float32)

        if len(points) < 20:
            return None

        x = points[:, 0]
        y = points[:, 1]

        x10 = np.percentile(x, 10)
        x90 = np.percentile(x, 90)
        y10 = np.percentile(y, 10)
        y90 = np.percentile(y, 90)

        left_points = points[x <= x10]
        right_points = points[x >= x90]
        top_points = points[y <= y10]
        bottom_points = points[y >= y90]

        if min(
            len(left_points),
            len(right_points),
            len(top_points),
            len(bottom_points),
        ) < 8:
            return None

        def fit_line(pts):
            vx, vy, x0, y0 = cv2.fitLine(
                pts.reshape(-1, 1, 2),
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            ).flatten()

            return (
                np.array([x0, y0], dtype=np.float32),
                np.array([vx, vy], dtype=np.float32),
            )

        def intersect(line_a, line_b):
            p1, v1 = line_a
            p2, v2 = line_b

            matrix = np.column_stack(
                (v1, -v2)
            )

            det = np.linalg.det(matrix)

            if abs(det) < 1e-6:
                return None

            t = np.linalg.solve(
                matrix,
                p2 - p1,
            )[0]

            return p1 + (t * v1)

        left_line = fit_line(left_points)
        right_line = fit_line(right_points)
        top_line = fit_line(top_points)
        bottom_line = fit_line(bottom_points)

        tl = intersect(top_line, left_line)
        tr = intersect(top_line, right_line)
        br = intersect(bottom_line, right_line)
        bl = intersect(bottom_line, left_line)

        if any(point is None for point in (tl, tr, br, bl)):
            return None

        quad = np.array(
            [tl, tr, br, bl],
            dtype=np.float32,
        )

        height, width = image.shape[:2]

        # Beri toleransi kecil untuk hasil fit yang sedikit keluar frame.
        quad[:, 0] = np.clip(
            quad[:, 0],
            -0.03 * width,
            1.03 * width,
        )
        quad[:, 1] = np.clip(
            quad[:, 1],
            -0.03 * height,
            1.03 * height,
        )

        area = abs(
            cv2.contourArea(
                quad.astype(np.float32)
            )
        )

        if area < image_area * 0.20:
            return None

        return quad

    # ============================================================
    # DOCUMENT DETECTION
    # ============================================================

    def _quad_score(self, points, image_area, mode="document"):
        rect = AutoDocumentScanner.order_points(points)
        tl, tr, br, bl = rect

        area = abs(cv2.contourArea(rect.astype(np.float32)))

        if area <= 0:
            return -1.0

        width = (
            np.linalg.norm(tr - tl)
            + np.linalg.norm(br - bl)
        ) / 2.0

        height = (
            np.linalg.norm(bl - tl)
            + np.linalg.norm(br - tr)
        ) / 2.0

        if width <= 1 or height <= 1:
            return -1.0

        bbox_area = width * height
        rectangularity = min(area / bbox_area, 1.0) if bbox_area > 0 else 0.0
        area_ratio = min(area / image_area, 1.0)

        if mode == "ktp":
            long_side = max(width, height)
            short_side = min(width, height)

            if short_side <= 1:
                return -1.0

            ratio = long_side / short_side
            ratio_error = abs(ratio - self.ktp_aspect_ratio) / self.ktp_aspect_ratio
            ratio_score = max(0.0, 1.0 - (ratio_error / 0.45))

            # Untuk KTP, ukuran tetap penting tetapi bentuk kartu lebih dominan
            # agar contour meja/kertas besar tidak otomatis terpilih.
            return (
                (area_ratio * 0.45)
                + (rectangularity * 0.20)
                + (ratio_score * 0.35)
            )

        return (area_ratio * 0.8) + (rectangularity * 0.2)

    def detect_document(self, image, mode="document"):
        original_height, original_width = image.shape[:2]

        scale = 1.0

        if original_height > self.detection_height:
            scale = self.detection_height / float(original_height)
            resized_width = int(round(original_width * scale))

            resized = cv2.resize(
                image,
                (resized_width, self.detection_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image.copy()

        edges = self.preprocess(resized)

        color_quad = None
        border_line_quad = None

        if mode == "ktp":
            color_quad = self._detect_ktp_color_candidate(
                resized
            )

            border_line_quad = self._detect_ktp_border_line_candidate(
                resized
            )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            if color_quad is not None:
                color_quad /= scale
                return color_quad.astype(np.float32)

            return None

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        image_area = resized.shape[0] * resized.shape[1]
        # KTP yang difoto agak jauh tetap perlu dianggap kandidat.
        min_area = image_area * 0.04

        best_quad = None
        best_score = -1.0
        fallback_contour = None

        if border_line_quad is not None:
            line_score = self._quad_score(
                border_line_quad,
                image_area,
                mode=mode,
            )

            # Border-line candidate mendapat sedikit prioritas karena
            # memakai empat sisi aktual kartu dan lebih presisi untuk
            # koreksi perspektif dibanding contour internal.
            best_quad = border_line_quad.copy()
            best_score = line_score + 0.12

        for contour in contours[:80]:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            if fallback_contour is None:
                fallback_contour = contour

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0:
                continue

            for epsilon_ratio in (0.015, 0.02, 0.025, 0.03):
                approx = cv2.approxPolyDP(
                    contour,
                    epsilon_ratio * perimeter,
                    True,
                )

                if len(approx) != 4:
                    continue

                quad = approx.reshape(4, 2).astype(np.float32)

                if not cv2.isContourConvex(
                    quad.astype(np.int32)
                ):
                    continue

                score = self._quad_score(
                    quad,
                    image_area,
                    mode=mode,
                )

                if score > best_score:
                    best_score = score
                    best_quad = quad

        if best_quad is None:
            if color_quad is not None:
                color_quad /= scale
                return color_quad.astype(np.float32)

            if fallback_contour is None:
                return None

            # Coba beberapa contour besar sebagai fallback dan tetap
            # pilih yang paling menyerupai KTP saat mode KTP aktif.
            fallback_best = None
            fallback_best_score = -1.0

            for contour in contours[:12]:
                area = cv2.contourArea(contour)

                if area < min_area:
                    continue

                rect = cv2.minAreaRect(contour)
                quad = cv2.boxPoints(rect).astype(np.float32)

                score = self._quad_score(
                    quad,
                    image_area,
                    mode=mode,
                )

                if score > fallback_best_score:
                    fallback_best_score = score
                    fallback_best = quad

            if fallback_best is None:
                rect = cv2.minAreaRect(
                    fallback_contour
                )
                fallback_best = cv2.boxPoints(
                    rect
                ).astype(np.float32)

            best_quad = fallback_best

        best_quad /= scale

        return best_quad.astype(np.float32)

    def refine_ktp_perspective(self, image):
        """
        Refinement perspektif konservatif.

        Hanya memakai kandidat empat sisi bila:
        - kandidat menutupi sebagian besar hasil warp,
        - bentuknya masih masuk akal sebagai KTP,
        - dan mismatch sisi atas/bawah atau kiri/kanan memang cukup besar.
        Jika syarat tidak terpenuhi, hasil warp pertama dipertahankan.
        """
        quad = self._detect_ktp_border_line_candidate(image)

        if quad is None:
            return image

        ordered = self.order_points(quad)

        image_area = image.shape[0] * image.shape[1]
        quad_area = abs(cv2.contourArea(ordered.astype(np.float32)))
        area_ratio = quad_area / float(max(image_area, 1))

        if area_ratio < 0.72:
            return image

        tl, tr, br, bl = ordered

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        if min(width_top, width_bottom, height_left, height_right) < 20:
            return image

        ratio = (
            (width_top + width_bottom) / 2.0
        ) / max(
            (height_left + height_right) / 2.0,
            1.0,
        )

        if not (1.15 <= ratio <= 2.20):
            return image

        width_mismatch = abs(width_top - width_bottom) / max(
            width_top,
            width_bottom,
            1.0,
        )
        height_mismatch = abs(height_left - height_right) / max(
            height_left,
            height_right,
            1.0,
        )

        # Kalau hasil awal sudah cukup rectangular, jangan dipoles lagi.
        if max(width_mismatch, height_mismatch) < 0.035:
            return image

        refined = self.perspective_transform(
            image,
            ordered,
            mode="document",
        )

        # Jangan menerima refinement yang mengubah ukuran secara ekstrem.
        rh, rw = refined.shape[:2]
        ih, iw = image.shape[:2]

        if (
            rw < iw * 0.70
            or rh < ih * 0.70
            or rw > iw * 1.15
            or rh > ih * 1.15
        ):
            return image

        return refined

    # ============================================================
    # AUTO ROTATE
    # ============================================================

    @staticmethod
    def _detect_face_score(image):
        """
        Nilai kandidat orientasi berdasarkan deteksi wajah.
        KTP Indonesia menempatkan foto wajah di sisi kanan saat orientasi benar.
        """
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)

            if detector.empty():
                return 0.0

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Haar lebih stabil pada ukuran sedang.
            h, w = gray.shape[:2]
            scale = min(1.0, 900.0 / max(h, w))
            if scale < 1.0:
                gray = cv2.resize(
                    gray,
                    (int(round(w * scale)), int(round(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.06,
                minNeighbors=3,
                minSize=(28, 28),
            )

            if len(faces) == 0:
                return 0.0

            gh, gw = gray.shape[:2]
            best = 0.0

            for x, y, fw, fh in faces:
                area_ratio = (fw * fh) / float(gw * gh)
                cx = x + (fw / 2.0)

                # Orientasi KTP benar: foto berada di sisi kanan.
                right_bonus = 1.0 if cx >= (gw * 0.55) else 0.25
                score = area_ratio * right_bonus

                if score > best:
                    best = score

            return best

        except Exception:
            return 0.0

    @staticmethod
    def _photo_side_score(image):
        """
        Heuristik fallback orientasi KTP berdasarkan lokasi area foto.
        Dalam orientasi normal, foto identitas berada di sisi kanan.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, w = gray.shape[:2]

        y1 = int(round(h * 0.18))
        y2 = int(round(h * 0.82))

        left = slice(int(round(w * 0.03)), int(round(w * 0.31)))
        right = slice(int(round(w * 0.69)), int(round(w * 0.97)))

        def score(region_slice):
            g = gray[y1:y2, region_slice]
            s = hsv[y1:y2, region_slice, 1]

            if g.size == 0:
                return 0.0

            dark_fraction = float(np.mean(g < 120))
            saturation = float(np.mean(s)) / 255.0
            contrast = min(float(np.std(g)) / 64.0, 1.0)

            return (
                (dark_fraction * 0.45)
                + (saturation * 0.35)
                + (contrast * 0.20)
            )

        right_score = score(right)
        left_score = score(left)

        return right_score - left_score

    @staticmethod
    def _portrait_block_score(image):
        """
        Skor orientasi berdasarkan blok foto paspor. Berbeda dari teks,
        area foto membentuk komponen gelap/berkontras yang relatif besar.
        Nilai positif berarti blok foto lebih kuat di sisi kanan.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        y1 = int(round(h * 0.14))
        y2 = int(round(h * 0.82))

        def side_score(x1, x2):
            roi = gray[y1:y2, x1:x2]

            if roi.size == 0:
                return 0.0

            threshold = min(175, int(np.mean(roi) * 0.92))
            dark = (roi < threshold).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (9, 9),
            )
            dark = cv2.morphologyEx(
                dark,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2,
            )
            dark = cv2.morphologyEx(
                dark,
                cv2.MORPH_OPEN,
                kernel,
                iterations=1,
            )

            contours, _ = cv2.findContours(
                dark,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                return 0.0

            roi_area = roi.shape[0] * roi.shape[1]
            largest = max(cv2.contourArea(contour) for contour in contours)

            return largest / float(max(roi_area, 1))

        left_score = side_score(
            int(round(w * 0.02)),
            int(round(w * 0.38)),
        )
        right_score = side_score(
            int(round(w * 0.62)),
            int(round(w * 0.98)),
        )

        return right_score - left_score

    def auto_rotate(self, image, mode="document"):
        """
        Koreksi orientasi KTP secara konservatif.
        - Portrait: uji CW dan CCW; jika deteksi wajah tidak tegas, default CCW.
        - Landscape: hanya balik 180° bila wajah upright terdeteksi lebih kuat.
        - Tanpa bukti wajah, jangan membalik 180°.
        """
        if mode != "ktp":
            return image

        height, width = image.shape[:2]

        if height > width:
            candidate_cw = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )
            candidate_ccw = cv2.rotate(
                image,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

            cw_face = self._detect_face_score(candidate_cw)
            ccw_face = self._detect_face_score(candidate_ccw)

            if cw_face > 0 and ccw_face <= 0:
                return candidate_cw

            if ccw_face > 0 and cw_face <= 0:
                return candidate_ccw

            if cw_face > 0 and ccw_face > 0:
                if cw_face > ccw_face * 1.15:
                    return candidate_cw
                if ccw_face > cw_face * 1.15:
                    return candidate_ccw

            # Default aman untuk foto HP yang diputar ke kanan:
            # kembalikan dengan rotasi counter-clockwise.
            return candidate_ccw

        normal = image
        rotated_180 = cv2.rotate(
            image,
            cv2.ROTATE_180,
        )

        normal_face = self._detect_face_score(normal)
        rotated_face = self._detect_face_score(rotated_180)

        if rotated_face > 0 and normal_face <= 0:
            return rotated_180

        if normal_face > 0 and rotated_face <= 0:
            return normal

        if normal_face > 0 and rotated_face > 0:
            if rotated_face > normal_face * 1.20:
                return rotated_180
            return normal

        # Penting: tanpa bukti wajah yang jelas, jangan membalik 180°.
        return normal

    def deskew_ktp(self, image):
        """
        Koreksi kemiringan kecil setelah perspective transform.
        Hanya sudut horizontal kecil yang digunakan agar tidak merusak
        perspective correction utama.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(50, image.shape[1] // 8),
            minLineLength=max(60, image.shape[1] // 5),
            maxLineGap=20,
        )

        if lines is None:
            return image

        angles = []

        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = [int(value) for value in line]
            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1:
                continue

            angle = np.degrees(np.arctan2(dy, dx))

            if -8.0 <= angle <= 8.0:
                angles.append(angle)

        if len(angles) < 3:
            return image

        angle = float(np.median(angles))

        if abs(angle) < 0.35:
            return image

        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (w / 2.0, h / 2.0),
            angle,
            1.0,
        )

        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ============================================================
    # KTP GEOMETRY NORMALIZATION
    # ============================================================

    def normalize_ktp_aspect_ratio(self, image):
        """
        Normalisasi ke rasio kartu ID-1/KTP sekitar 1.586:1.
        """
        height, width = image.shape[:2]

        if height <= 1 or width <= 1:
            return image

        target_height = max(
            2,
            int(round(width / self.ktp_aspect_ratio)),
        )

        return cv2.resize(
            image,
            (width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )

    def trim_ktp_edges(self, image):
        """
        Trim sangat tipis setelah warp untuk membuang sisa background
        yang mungkin ikut terbawa di tepi kartu tanpa memotong isi penting.
        """
        if self.ktp_edge_trim <= 0:
            return image

        height, width = image.shape[:2]

        trim_x = int(round(width * self.ktp_edge_trim))
        trim_y = int(round(height * self.ktp_edge_trim))

        if (
            trim_x <= 0
            and trim_y <= 0
        ):
            return image

        x1 = min(max(trim_x, 0), max(width - 2, 0))
        y1 = min(max(trim_y, 0), max(height - 2, 0))
        x2 = max(width - trim_x, x1 + 2)
        y2 = max(height - trim_y, y1 + 2)

        return image[y1:y2, x1:x2]

    def add_ktp_safe_margin(self, image):
        """
        Membuat margin luar yang benar-benar terlihat tanpa mengubah
        rasio output KTP. Kartu diperkecil sedikit lalu ditempatkan di
        canvas netral dengan rasio yang sama.
        """
        if self.ktp_safe_margin <= 0:
            return image

        height, width = image.shape[:2]

        margin = min(max(self.ktp_safe_margin, 0.0), 0.10)

        inner_width = max(
            2,
            int(round(width * (1.0 - (2.0 * margin)))),
        )
        inner_height = max(
            2,
            int(round(height * (1.0 - (2.0 * margin)))),
        )

        inner = cv2.resize(
            image,
            (inner_width, inner_height),
            interpolation=cv2.INTER_AREA,
        )

        # Latar netral seperti hasil scanner.
        canvas = np.full(
            (height, width, 3),
            245,
            dtype=np.uint8,
        )

        x = (width - inner_width) // 2
        y = (height - inner_height) // 2

        canvas[
            y:y + inner_height,
            x:x + inner_width,
        ] = inner

        return canvas

    # ============================================================
    # OUTPUT MODE
    # ============================================================

    @staticmethod
    def apply_output_mode(image, output_mode="color"):
        """
        Output final default tetap berwarna.

        output_mode:
        - "color"     : pertahankan warna asli/enhanced
        - "grayscale" : keluarkan grayscale 3-channel agar pipeline/output
                        tetap konsisten untuk penyimpanan dan UI
        """
        if output_mode == "color":
            return image

        if output_mode == "grayscale":
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            return cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR,
            )

        raise ValueError(
            "output_mode harus 'color' atau 'grayscale'."
        )

    # ============================================================
    # IMAGE ENHANCEMENT
    # ============================================================

    @staticmethod
    def enhance(image):
        """
        Enhancement ringan untuk menjaga warna dokumen tetap natural.
        """
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=1.35,
            tileGridSize=(8, 8),
        )

        enhanced_l = clahe.apply(l)

        merged = cv2.merge(
            (enhanced_l, a, b)
        )

        enhanced = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2BGR,
        )

        return cv2.addWeighted(
            image,
            0.72,
            enhanced,
            0.28,
            0,
        )

    # ============================================================
    # MAIN SCANNER
    # ============================================================

    def scan(
        self,
        image_path,
        output_path=None,
        mode="document",
        output_mode="color",
    ):
        image_path = Path(image_path)

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Gambar tidak dapat dibaca: {image_path}"
            )

        corners = self.detect_document(
            image,
            mode=mode,
        )

        if corners is None:
            raise RuntimeError(
                "Dokumen tidak berhasil dideteksi."
            )

        result = self.perspective_transform(
            image,
            corners,
            mode=mode,
        )

        if mode == "ktp":
            result = self.refine_ktp_perspective(
                result
            )

        result = self.auto_rotate(
            result,
            mode=mode,
        )

        if mode == "ktp":
            result = self.deskew_ktp(
                result
            )

            result = self.trim_ktp_edges(
                result
            )

            result = self.normalize_ktp_aspect_ratio(
                result
            )

        result = self.enhance(
            result
        )

        if mode == "ktp" and self.ktp_safe_margin > 0:
            result = self.add_ktp_safe_margin(
                result
            )

        result = self.apply_output_mode(
            result,
            output_mode=output_mode,
        )

        if output_path:
            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            ok = cv2.imwrite(
                str(output_path),
                result,
            )

            if not ok:
                raise RuntimeError(
                    f"Gagal menyimpan hasil: {output_path}"
                )

        return result, corners
