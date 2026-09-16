from pathlib import Path
import cv2
import numpy as np


class AutoDocumentScanner:
    def __init__(self, detection_height=900):
        self.detection_height = detection_height

    # ============================================================
    # POINT ORDERING
    # ============================================================

    @staticmethod
    def order_points(points):
        points = np.asarray(points, dtype=np.float32)

        rect = np.zeros((4, 2), dtype=np.float32)

        s = points.sum(axis=1)
        diff = np.diff(points, axis=1).reshape(-1)

        rect[0] = points[np.argmin(s)]      # top-left
        rect[2] = points[np.argmax(s)]      # bottom-right
        rect[1] = points[np.argmin(diff)]   # top-right
        rect[3] = points[np.argmax(diff)]   # bottom-left

        return rect

    # ============================================================
    # PERSPECTIVE TRANSFORM
    # ============================================================

    def perspective_transform(self, image, points):
        rect = self.order_points(points)

        tl, tr, br, bl = rect

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)

        max_width = int(max(width_top, width_bottom))

        height_right = np.linalg.norm(br - tr)
        height_left = np.linalg.norm(bl - tl)

        max_height = int(max(height_right, height_left))

        max_width = max(max_width, 1)
        max_height = max(max_height, 1)

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(rect, destination)

        warped = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height),
        )

        return warped

    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        edges = cv2.Canny(
            gray,
            50,
            150,
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

        return edges

    # ============================================================
    # DOCUMENT DETECTION
    # ============================================================

    def detect_document(self, image):
        original_height, original_width = image.shape[:2]

        scale = 1.0

        if original_height > self.detection_height:
            scale = self.detection_height / float(original_height)

            resized_width = int(original_width * scale)

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

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        image_area = resized.shape[0] * resized.shape[1]

        document_contour = None

        for contour in contours[:20]:
            area = cv2.contourArea(contour)

            # abaikan object terlalu kecil
            if area < image_area * 0.10:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            approx = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True,
            )

            if len(approx) == 4:
                document_contour = approx.reshape(4, 2)
                break

        # ========================================================
        # FALLBACK
        # ========================================================

        if document_contour is None:

            if not contours:
                return None

            largest = contours[0]

            rect = cv2.minAreaRect(largest)

            box = cv2.boxPoints(rect)

            document_contour = np.asarray(
                box,
                dtype=np.float32,
            )

        document_contour = document_contour.astype(
            np.float32
        )

        # kembali ke koordinat image asli
        document_contour /= scale

        return document_contour

    # ============================================================
    # AUTO ROTATE
    # ============================================================

    @staticmethod
    def auto_rotate(image, mode="document"):
        height, width = image.shape[:2]

        # KTP normalnya horizontal
        if mode == "ktp":
            if height > width:
                image = cv2.rotate(
                    image,
                    cv2.ROTATE_90_CLOCKWISE,
                )

        return image

    # ============================================================
    # IMAGE ENHANCEMENT
    # ============================================================

    @staticmethod
    def enhance(image):
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        l = clahe.apply(l)

        merged = cv2.merge(
            (l, a, b)
        )

        enhanced = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2BGR,
        )

        return enhanced

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

        corners = self.detect_document(image)

        if corners is None:
            raise RuntimeError(
                "Dokumen tidak berhasil dideteksi."
            )

        result = self.perspective_transform(
            image,
            corners,
        )

        result = self.auto_rotate(
            result,
            mode=mode,
        )

        result = self.enhance(result)

        if output_path:
            output_path = Path(output_path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(output_path),
                result,
            )

        return result, corners