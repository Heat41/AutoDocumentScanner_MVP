from pathlib import Path

import cv2
import numpy as np

from autodocscanner.core.final_validation import FinalScanValidator
from autodocscanner.core.quality_check import DocumentQualityChecker
from autodocscanner.core.robustness_engine import RobustPerspectiveEngine
from autodocscanner.output.safe import atomic_imwrite


class AutoDocumentScanner:
    KTP_ASPECT_RATIO = 85.60 / 53.98

    def __init__(
        self,
        detection_height=1000,
        ktp_aspect_ratio=KTP_ASPECT_RATIO,
        ktp_edge_trim=0.0,
        ktp_safe_margin=0.0,
    ):
        self.ktp_aspect_ratio = ktp_aspect_ratio
        self.ktp_edge_trim = ktp_edge_trim
        self.ktp_safe_margin = ktp_safe_margin

        self.perspective_engine = RobustPerspectiveEngine(
            target_ratio=ktp_aspect_ratio,
            detection_height=detection_height,
        )
        self.quality_checker = DocumentQualityChecker(
            target_ratio=ktp_aspect_ratio,
        )
        self.final_validator = FinalScanValidator(
            target_ratio=ktp_aspect_ratio,
        )

        self.last_detection = {}
        self.last_corners = None
        self.last_quality = {}
        self.last_validation = {}

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

    def expand_ktp_corners(
        self,
        image_shape,
        corners,
    ):
        """
        Tambahkan safety margin geometris kecil di luar quad KTP.

        Detector tetap bertugas menemukan batas fisik kartu. Margin ini hanya
        dipakai saat warp final agar rounded corner, glare, atau edge yang
        terbaca sedikit masuk ke badan kartu tidak memotong teks/foto di tepi.

        Nilainya bersifat relatif terhadap ukuran kartu, bukan pixel tetap,
        sehingga berlaku konsisten untuk foto KTP random beresolusi berbeda.
        """
        if corners is None:
            return corners

        quad = np.asarray(
            corners,
            dtype=np.float32,
        ).reshape(4, 2)

        margin = max(
            float(self.ktp_safe_margin),
            0.0,
        )

        if margin <= 0:
            return quad

        center = np.mean(
            quad,
            axis=0,
        )

        # margin adalah tambahan per sisi. Faktor 2 karena vektor center->corner
        # hanya merepresentasikan setengah dimensi kartu.
        scale = 1.0 + 2.0 * margin
        expanded = (
            center
            + (
                quad - center
            ) * scale
        )

        h, w = image_shape[:2]

        expanded[:, 0] = np.clip(
            expanded[:, 0],
            0.0,
            max(float(w - 1), 0.0),
        )
        expanded[:, 1] = np.clip(
            expanded[:, 1],
            0.0,
            max(float(h - 1), 0.0),
        )

        if not cv2.isContourConvex(
            expanded.astype(
                np.int32
            )
        ):
            return quad

        return self.perspective_engine.order_points(
            expanded
        )

    def warp_ktp_to_ratio(
        self,
        image,
        corners,
    ):
        """
        Warp empat sudut langsung ke rasio fisik KTP.

        Berbeda dari resize setelah warp, rasio target dimasukkan ke destination
        homography. Dengan begitu seluruh area yang dipilih tetap dipertahankan
        dan hasil tidak perlu dipotong/ditarik lagi sesudah perspective.
        """
        rect = self.perspective_engine.order_points(
            corners
        )
        tl, tr, br, bl = rect

        width_top = float(
            np.linalg.norm(
                tr - tl
            )
        )
        width_bottom = float(
            np.linalg.norm(
                br - bl
            )
        )
        height_left = float(
            np.linalg.norm(
                bl - tl
            )
        )
        height_right = float(
            np.linalg.norm(
                br - tr
            )
        )

        avg_width = max(
            (
                width_top
                + width_bottom
            ) / 2.0,
            2.0,
        )
        avg_height = max(
            (
                height_left
                + height_right
            ) / 2.0,
            2.0,
        )

        if avg_width >= avg_height:
            target_width = max(
                int(round(avg_width)),
                2,
            )
            target_height = max(
                int(round(
                    target_width
                    / self.ktp_aspect_ratio
                )),
                2,
            )
        else:
            target_height = max(
                int(round(avg_height)),
                2,
            )
            target_width = max(
                int(round(
                    target_height
                    / self.ktp_aspect_ratio
                )),
                2,
            )

        destination = np.array(
            [
                [0, 0],
                [
                    target_width - 1,
                    0,
                ],
                [
                    target_width - 1,
                    target_height - 1,
                ],
                [
                    0,
                    target_height - 1,
                ],
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
            (
                target_width,
                target_height,
            ),
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
        """
        Paksa hasil akhir KTP ke rasio fisik ID-1 (85.60 x 53.98 mm).

        Perspective detector menentukan empat sisi fisik kartu. Setelah warp,
        output dinormalisasi tepat ke rasio KTP agar hasil dari foto random
        selalu berbentuk kartu, bukan mengikuti rasio frame/background kamera.
        """
        h, w = image.shape[:2]

        if h > w:
            image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )
            h, w = image.shape[:2]

        if h < 2 or w < 2:
            return image

        # Pertahankan sisi panjang/resolusi hasil deteksi, lalu hitung tinggi
        # tepat dari rasio fisik KTP. Tidak ada toleransi 2.5% lagi karena
        # output KTP memang harus selalu konsisten.
        target_width = max(
            int(w),
            2,
        )
        target_height = max(
            2,
            int(round(
                target_width
                / self.ktp_aspect_ratio
            )),
        )

        if (
            w == target_width
            and h == target_height
        ):
            return image

        interpolation = (
            cv2.INTER_AREA
            if target_height < h
            else cv2.INTER_CUBIC
        )

        return cv2.resize(
            image,
            (
                target_width,
                target_height,
            ),
            interpolation=interpolation,
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
        output_mode = str(
            output_mode or "color"
        ).strip().lower()

        if output_mode == "color":
            return image

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        if output_mode == "grayscale":
            return cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR,
            )

        if output_mode == "bw":
            blurred = cv2.GaussianBlur(
                gray,
                (3, 3),
                0,
            )
            _, bw = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            return cv2.cvtColor(
                bw,
                cv2.COLOR_GRAY2BGR,
            )

        raise ValueError(
            "output_mode harus 'color', "
            "'grayscale', atau 'bw'."
        )

    def scan_with_corners(
        self,
        image_path,
        corners,
        output_path=None,
        mode="ktp",
        output_mode="color",
    ):
        """
        Koreksi perspektif menggunakan empat sudut pilihan pengguna.

        Dipakai sebagai fallback setelah proses otomatis. Jalur output tetap
        memakai validasi, normalisasi rasio KTP, enhancement, dan mode hasil
        yang sama dengan proses otomatis.
        """
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

        points = np.asarray(
            corners,
            dtype=np.float32,
        ).reshape(4, 2)

        h, w = image.shape[:2]
        points[:, 0] = np.clip(
            points[:, 0],
            0,
            max(w - 1, 0),
        )
        points[:, 1] = np.clip(
            points[:, 1],
            0,
            max(h - 1, 0),
        )

        points = self.perspective_engine.order_points(
            points
        )

        area = abs(
            cv2.contourArea(
                points.astype(
                    np.float32
                )
            )
        )

        if area < 4.0:
            raise ValueError(
                "Empat titik manual menghasilkan area terlalu kecil."
            )

        self.last_corners = points.copy()
        self.last_detection = {
            "candidate_count": 1,
            "selected_source": "manual_correction",
            "score": 1.0,
            "manual": True,
        }
        self.last_validation = {}

        corner_validation = self.final_validator.validate_corners(
            image.shape,
            points,
        )

        if not corner_validation.get(
            "hard_valid",
            False,
        ):
            self.last_validation = {
                "status": "review",
                "hard_valid": False,
                "warnings": list(
                    corner_validation.get(
                        "warnings"
                    ) or []
                ),
                "corner": corner_validation,
            }
            self.last_detection[
                "validation"
            ] = self.last_validation
            raise RuntimeError(
                "Geometri empat sudut manual gagal validasi."
            )

        # Untuk koreksi manual, titik pengguna adalah boundary final.
        # Warp langsung ke rasio fisik KTP agar area pilihan tidak dipotong
        # atau ditarik ulang setelah perspective.
        if mode == "ktp":
            result = self.warp_ktp_to_ratio(
                image,
                points,
            )
        else:
            result = self.perspective_engine.warp(
                image,
                points,
            )

        result = self.auto_rotate(
            result,
            mode=mode,
        )

        if mode != "ktp":
            result = self.deskew_small_angle(
                result
            )
            result = self.trim_edges(
                result
            )

        result = self.enhance(
            result
        )

        self.last_quality = self.quality_checker.assess(
            result,
            detection_metadata=self.last_detection,
        )
        self.last_detection[
            "quality"
        ] = self.last_quality

        output_validation = self.final_validator.validate_output(
            result,
            detection_metadata=self.last_detection,
            quality=self.last_quality,
        )
        self.last_validation = self.final_validator.combine(
            corner_validation,
            output_validation,
        )
        self.last_detection[
            "validation"
        ] = self.last_validation

        if not self.last_validation.get(
            "hard_valid",
            False,
        ):
            raise RuntimeError(
                "Hasil koreksi manual gagal validasi geometri akhir."
            )

        result = self.apply_output_mode(
            result,
            output_mode=output_mode,
        )

        if output_path:
            atomic_imwrite(
                output_path,
                result,
            )

        return result, points

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
        self.last_validation = {}

        if corrected is None:
            self.last_quality = {
                "status": "review",
                "score": 0.0,
                "warnings": [
                    "batas KTP tidak terdeteksi"
                ],
            }
            self.last_validation = {
                "status": "review",
                "hard_valid": False,
                "warnings": [
                    "perspective engine tidak menghasilkan output"
                ],
            }
            self.last_detection["quality"] = self.last_quality
            self.last_detection["validation"] = self.last_validation

            raise RuntimeError(
                "Batas KTP tidak berhasil dideteksi otomatis."
            )

        corner_validation = self.final_validator.validate_corners(
            image.shape,
            corners,
        )

        if not corner_validation.get("hard_valid", False):
            self.last_validation = {
                "status": "review",
                "hard_valid": False,
                "warnings": list(
                    corner_validation.get("warnings") or []
                ),
                "corner": corner_validation,
            }
            self.last_detection["validation"] = self.last_validation

            raise RuntimeError(
                "Geometri empat sudut gagal validasi akhir."
            )

        if mode == "ktp":
            # Gunakan boundary hasil detector apa adanya. Safety expansion
            # sebelumnya dapat membawa background masuk secara tidak merata
            # pada foto yang sangat miring. Rasio KTP sekarang diterapkan
            # langsung pada destination homography.
            result = self.warp_ktp_to_ratio(
                image,
                corners,
            )
            self.last_detection[
                "safe_margin"
            ] = 0.0
        else:
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

        # Untuk KTP jangan trim ke dalam lagi setelah warp. Hasil detector
        # sudah di-warp dengan safety margin kecil agar konten tepi tidak
        # terpotong. Trim tetap dipertahankan untuk mode dokumen umum.
        if mode != "ktp":
            result = self.trim_edges(
                result
            )

        result = self.enhance(
            result
        )

        # Quality check dilakukan sebelum opsi grayscale agar geometri/cahaya
        # dinilai dari output warna hasil pipeline. Quality gate tidak mengubah
        # gambar dan tidak otomatis menggagalkan output.
        self.last_quality = self.quality_checker.assess(
            result,
            detection_metadata=self.last_detection,
        )
        self.last_detection["quality"] = self.last_quality

        output_validation = self.final_validator.validate_output(
            result,
            detection_metadata=self.last_detection,
            quality=self.last_quality,
        )
        self.last_validation = self.final_validator.combine(
            corner_validation,
            output_validation,
        )
        self.last_detection["validation"] = self.last_validation

        # Hanya kondisi yang secara geometri mustahil/korup yang dihentikan.
        # Warning kualitas tetap boleh disimpan agar baseline tidak menjadi
        # terlalu agresif menolak foto yang masih layak.
        if not self.last_validation.get("hard_valid", False):
            raise RuntimeError(
                "Hasil scan gagal validasi geometri akhir."
            )

        result = self.apply_output_mode(
            result,
            output_mode=output_mode,
        )

        if output_path:
            atomic_imwrite(
                output_path,
                result,
            )

        return result, corners
