from pathlib import Path

import cv2
import numpy as np

from quality_check import DocumentQualityChecker
from robustness_engine import RobustPerspectiveEngine


class AutoDocumentScanner:
    KTP_ASPECT_RATIO = 85.60 / 53.98

    def __init__(
        self,
        detection_height=1000,
        ktp_aspect_ratio=KTP_ASPECT_RATIO,
        ktp_edge_trim=0.004,
    ):
        self.ktp_aspect_ratio = ktp_aspect_ratio
        self.ktp_edge_trim = ktp_edge_trim

        self.perspective_engine = RobustPerspectiveEngine(
            target_ratio=ktp_aspect_ratio,
            detection_height=detection_height,
        )
        self.quality_checker = DocumentQualityChecker(
            target_ratio=ktp_aspect_ratio,
        )

        self.last_detection = {}
        self.last_corners = None
        self.last_quality = {}

    @staticmethod
    def _detect_face_score(image):
        try:
            cascade_path = (
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )
            detector = cv2.CascadeClassifier(
                cascade_path
            )

            if detector.empty():
                return 0.0

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            h, w = gray.shape[:2]
            scale = min(
                1.0,
                900.0 / max(h, w),
            )

            if scale < 1.0:
                gray = cv2.resize(
                    gray,
                    (
                        int(round(w * scale)),
                        int(round(h * scale)),
                    ),
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
                area_ratio = (
                    fw * fh
                ) / float(
                    max(gw * gh, 1)
                )

                cx = x + fw / 2.0
                right_bonus = (
                    1.0
                    if cx >= gw * 0.55
                    else 0.20
                )

                best = max(
                    best,
                    area_ratio * right_bonus,
                )

            return best

        except Exception:
            return 0.0

    def auto_rotate(self, image, mode="ktp"):
        """
        Orientation dipisahkan dari perspective engine.
        Koreksi 180 derajat tetap konservatif.
        """
        if mode != "ktp":
            return image

        h, w = image.shape[:2]

        if h > w:
            cw = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )
            ccw = cv2.rotate(
                image,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

            cw_face = self._detect_face_score(
                cw
            )
            ccw_face = self._detect_face_score(
                ccw
            )

            if cw_face > ccw_face * 1.15:
                return cw

            return ccw

        normal = image
        rotated = cv2.rotate(
            image,
            cv2.ROTATE_180,
        )

        normal_face = self._detect_face_score(
            normal
        )
        rotated_face = self._detect_face_score(
            rotated
        )

        if (
            rotated_face > 0
            and rotated_face > normal_face * 1.25
        ):
            return rotated

        return normal

    @staticmethod
    def deskew_small_angle(image):
        """
        Hanya mengoreksi residual skew kecil setelah perspective correction.
        """
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        edges = cv2.Canny(
            gray,
            60,
            160,
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(
                45,
                image.shape[1] // 9,
            ),
            minLineLength=max(
                50,
                image.shape[1] // 5,
            ),
            maxLineGap=18,
        )

        if lines is None:
            return image

        angles = []

        for line in np.asarray(
            lines
        ).reshape(-1, 4):
            x1, y1, x2, y2 = [
                int(value)
                for value in line
            ]

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1:
                continue

            angle = np.degrees(
                np.arctan2(
                    dy,
                    dx,
                )
            )

            if -6.0 <= angle <= 6.0:
                angles.append(angle)

        if len(angles) < 3:
            return image

        angle = float(
            np.median(angles)
        )

        if abs(angle) < 0.40:
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

    def trim_edges(self, image):
        if self.ktp_edge_trim <= 0:
            return image

        h, w = image.shape[:2]

        trim_x = max(
            1,
            int(round(w * self.ktp_edge_trim)),
        )
        trim_y = max(
            1,
            int(round(h * self.ktp_edge_trim)),
        )

        if (
            w - 2 * trim_x < 20
            or h - 2 * trim_y < 20
        ):
            return image

        return image[
            trim_y:h - trim_y,
            trim_x:w - trim_x,
        ]

    def normalize_ktp_ratio(self, image):
        h, w = image.shape[:2]

        if h > w:
            image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )
            h, w = image.shape[:2]

        target_height = max(
            2,
            int(round(
                w / self.ktp_aspect_ratio
            )),
        )

        # Hanya koreksi geometri kecil setelah perspective selesai.
        current_ratio = w / max(h, 1)
        error = abs(
            current_ratio - self.ktp_aspect_ratio
        ) / self.ktp_aspect_ratio

        if error <= 0.025:
            return image

        return cv2.resize(
            image,
            (w, target_height),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def enhance(image):
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=1.30,
            tileGridSize=(8, 8),
        )

        enhanced_l = clahe.apply(l)

        enhanced = cv2.cvtColor(
            cv2.merge(
                (enhanced_l, a, b)
            ),
            cv2.COLOR_LAB2BGR,
        )

        return cv2.addWeighted(
            image,
            0.78,
            enhanced,
            0.22,
            0,
        )

    @staticmethod
    def apply_output_mode(
        image,
        output_mode="color",
    ):
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

    def scan(
        self,
        image_path,
        output_path=None,
        mode="ktp",
        output_mode="color",
    ):
        image_path = Path(
            image_path
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Gambar tidak dapat dibaca: {image_path}"
            )

        corrected, corners, metadata = (
            self.perspective_engine.correct(
                image
            )
        )

        self.last_detection = dict(
            metadata or {}
        )
        self.last_corners = corners

        if corrected is None:
            self.last_quality = {
                "status": "review",
                "score": 0.0,
                "warnings": [
                    "batas KTP tidak terdeteksi"
                ],
            }
            self.last_detection[
                "quality"
            ] = self.last_quality

            raise RuntimeError(
                "Batas KTP tidak berhasil dideteksi otomatis."
            )

        result = corrected

        result = self.auto_rotate(
            result,
            mode=mode,
        )

        # Untuk KTP, perspective engine sudah memetakan empat sisi fisik
        # langsung ke rectangle. Deskew berbasis Hough setelah homography
        # justru bisa memiringkan kembali dimensi kartu karena garis teks
        # internal ikut terbaca sebagai acuan.
        #
        # Deskew kecil hanya dipakai untuk mode dokumen umum.
        if mode != "ktp":
            result = self.deskew_small_angle(
                result
            )

        result = self.trim_edges(
            result
        )

        if mode == "ktp":
            result = self.normalize_ktp_ratio(
                result
            )

        result = self.enhance(
            result
        )

        # Quality check sengaja dilakukan sebelum opsi grayscale sehingga
        # kualitas geometri/cahaya dinilai dari output warna hasil pipeline.
        # Hasil quality check tidak mengubah gambar dan tidak membatalkan save.
        self.last_quality = self.quality_checker.assess(
            result,
            detection_metadata=self.last_detection,
        )
        self.last_detection[
            "quality"
        ] = self.last_quality

        result = self.apply_output_mode(
            result,
            output_mode=output_mode,
        )

        if output_path:
            output_path = Path(
                output_path
            )
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not cv2.imwrite(
                str(output_path),
                result,
            ):
                raise RuntimeError(
                    f"Gagal menyimpan hasil: {output_path}"
                )

        return result, corners
