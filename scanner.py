from pathlib import Path

import cv2
import numpy as np


class AutoDocumentScanner:
    KTP_ASPECT_RATIO = 85.60 / 53.98

    def __init__(
        self,
        detection_height=900,
        ktp_corner_padding=0.012,
        ktp_aspect_ratio=KTP_ASPECT_RATIO,
        ktp_safe_margin=0.02,
    ):
        self.detection_height = detection_height
        self.ktp_corner_padding = ktp_corner_padding
        self.ktp_aspect_ratio = ktp_aspect_ratio
        self.ktp_safe_margin = ktp_safe_margin

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
    # DOCUMENT DETECTION
    # ============================================================

    @staticmethod
    def _quad_score(points, image_area):
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

        return (area_ratio * 0.8) + (rectangularity * 0.2)

    def detect_document(self, image):
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

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        image_area = resized.shape[0] * resized.shape[1]
        min_area = image_area * 0.08

        best_quad = None
        best_score = -1.0
        fallback_contour = None

        for contour in contours[:40]:
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
                )

                if score > best_score:
                    best_score = score
                    best_quad = quad

        if best_quad is None:
            if fallback_contour is None:
                return None

            rect = cv2.minAreaRect(
                fallback_contour
            )

            best_quad = cv2.boxPoints(
                rect
            ).astype(np.float32)

        best_quad /= scale

        return best_quad.astype(np.float32)

    # ============================================================
    # AUTO ROTATE
    # ============================================================

    @staticmethod
    def auto_rotate(image, mode="document"):
        height, width = image.shape[:2]

        if mode == "ktp" and height > width:
            image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )

        return image

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
            image
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

        result = self.auto_rotate(
            result,
            mode=mode,
        )

        if mode == "ktp":
            result = self.normalize_ktp_aspect_ratio(
                result
            )

        result = self.enhance(
            result
        )

        if mode == "ktp":
            result = self.add_ktp_safe_margin(
                result
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
