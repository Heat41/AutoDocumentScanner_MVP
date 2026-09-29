from pathlib import Path

import cv2
import numpy as np

from autodocscanner.core.final_validation import FinalScanValidator
from autodocscanner.core.quality_check import DocumentQualityChecker
from autodocscanner.core.robustness_engine import RobustPerspectiveEngine
from autodocscanner.output.safe import atomic_imwrite


class AutoDocumentScanner:
    KTP_ASPECT_RATIO = 85.60 / 53.98
    KTP_CANONICAL_WIDTH = 856
    KTP_CANONICAL_HEIGHT = 540
    KTP_RESIDUAL_SKEW_MIN_DEG = 0.45
    KTP_RESIDUAL_SKEW_MAX_DEG = 2.50
    KTP_RESIDUAL_SKEW_MAX_SPREAD_DEG = 0.80
    KTP_RESIDUAL_SKEW_MIN_LINES = 4

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

    @classmethod
    def estimate_ktp_residual_skew(
        cls,
        image,
    ):
        """
        Estimasi residual skew kecil sesudah perspective warp.

        Hanya garis horizontal panjang dan konsisten yang dipakai agar
        tekstur/security pattern KTP tidak memicu rotasi palsu.
        """
        if (
            not isinstance(
                image,
                np.ndarray,
            )
            or image.size == 0
        ):
            raise ValueError(
                "image residual skew tidak boleh kosong."
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        height, width = image.shape[:2]

        # Fokus ke area teks kiri/utama. Ini menghindari tepi foto,
        # tanda tangan, dan sebagian besar security pattern kanan.
        roi_x1 = max(
            0,
            int(
                round(
                    width * 0.035
                )
            ),
        )
        roi_x2 = min(
            width,
            int(
                round(
                    width * 0.73
                )
            ),
        )
        roi_y1 = max(
            0,
            int(
                round(
                    height * 0.10
                )
            ),
        )
        roi_y2 = min(
            height,
            int(
                round(
                    height * 0.82
                )
            ),
        )

        gray_roi = gray[
            roi_y1:roi_y2,
            roi_x1:roi_x2,
        ]

        # Downsample ringan bila perlu agar Stage A tetap murah di CPU.
        roi_height, roi_width = gray_roi.shape[:2]
        scale = min(
            1.0,
            640.0
            / max(
                roi_width,
                1,
            ),
        )

        if scale < 1.0:
            gray_roi = cv2.resize(
                gray_roi,
                (
                    max(
                        2,
                        int(
                            round(
                                roi_width * scale
                            )
                        ),
                    ),
                    max(
                        2,
                        int(
                            round(
                                roi_height * scale
                            )
                        ),
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )

        gray_roi = cv2.GaussianBlur(
            gray_roi,
            (3, 3),
            0,
        )

        edges = cv2.Canny(
            gray_roi,
            65,
            155,
        )

        analysis_width = edges.shape[1]

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(
                24,
                analysis_width // 18,
            ),
            minLineLength=max(
                42,
                int(
                    round(
                        analysis_width * 0.08
                    )
                ),
            ),
            maxLineGap=max(
                8,
                int(
                    round(
                        analysis_width * 0.018
                    )
                ),
            ),
        )

        if lines is None:
            return {
                "angle": 0.0,
                "applied": False,
                "line_count": 0,
                "spread": None,
                "reason": "no_lines",
            }

        angles = []
        weights = []

        for line in np.asarray(
            lines
        ).reshape(-1, 4):
            x1, y1, x2, y2 = [
                float(value)
                for value in line
            ]

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1.0:
                continue

            length = float(
                np.hypot(
                    dx,
                    dy,
                )
            )

            if length < analysis_width * 0.08:
                continue

            angle = float(
                np.degrees(
                    np.arctan2(
                        dy,
                        dx,
                    )
                )
            )

            # Residual KTP setelah homography seharusnya kecil.
            if not (
                -cls.KTP_RESIDUAL_SKEW_MAX_DEG
                <= angle
                <= cls.KTP_RESIDUAL_SKEW_MAX_DEG
            ):
                continue

            # ROI analisis sudah membuang border/foto; tidak perlu filter
            # tambahan yang berisiko membuang baris teks valid.

            angles.append(
                angle
            )
            weights.append(
                max(
                    length,
                    1.0,
                )
            )

        if len(angles) < cls.KTP_RESIDUAL_SKEW_MIN_LINES:
            return {
                "angle": 0.0,
                "applied": False,
                "line_count": len(
                    angles
                ),
                "spread": None,
                "reason": "insufficient_lines",
            }

        angle_array = np.asarray(
            angles,
            dtype=np.float32,
        )

        median = float(
            np.median(
                angle_array
            )
        )
        deviation = np.abs(
            angle_array
            - median
        )
        mad = float(
            np.median(
                deviation
            )
        )

        tolerance = max(
            0.35,
            3.0 * mad,
        )
        keep = (
            deviation
            <= tolerance
        )

        filtered_angles = angle_array[
            keep
        ]
        filtered_weights = np.asarray(
            weights,
            dtype=np.float32,
        )[
            keep
        ]

        if (
            filtered_angles.size
            < cls.KTP_RESIDUAL_SKEW_MIN_LINES
        ):
            return {
                "angle": 0.0,
                "applied": False,
                "line_count": int(
                    filtered_angles.size
                ),
                "spread": mad,
                "reason": "inconsistent_lines",
            }

        # Weighted mean setelah robust median rejection.
        angle = float(
            np.average(
                filtered_angles,
                weights=filtered_weights,
            )
        )
        spread = float(
            np.median(
                np.abs(
                    filtered_angles
                    - np.median(
                        filtered_angles
                    )
                )
            )
        )

        if (
            spread
            > cls.KTP_RESIDUAL_SKEW_MAX_SPREAD_DEG
        ):
            return {
                "angle": angle,
                "applied": False,
                "line_count": int(
                    filtered_angles.size
                ),
                "spread": spread,
                "reason": "spread_too_high",
            }

        if (
            abs(angle)
            < cls.KTP_RESIDUAL_SKEW_MIN_DEG
        ):
            return {
                "angle": angle,
                "applied": False,
                "line_count": int(
                    filtered_angles.size
                ),
                "spread": spread,
                "reason": "already_straight",
            }

        if (
            abs(angle)
            > cls.KTP_RESIDUAL_SKEW_MAX_DEG
        ):
            return {
                "angle": angle,
                "applied": False,
                "line_count": int(
                    filtered_angles.size
                ),
                "spread": spread,
                "reason": "angle_too_large",
            }

        return {
            "angle": angle,
            "applied": True,
            "line_count": int(
                filtered_angles.size
            ),
            "spread": spread,
            "reason": "residual_skew",
        }

    @classmethod
    def stabilize_ktp_residual_skew(
        cls,
        image,
    ):
        metadata = (
            cls.estimate_ktp_residual_skew(
                image
            )
        )

        if not metadata.get(
            "applied",
            False,
        ):
            return (
                image.copy(),
                metadata,
            )

        angle = float(
            metadata.get(
                "angle",
                0.0,
            )
        )

        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (
                width / 2.0,
                height / 2.0,
            ),
            angle,
            1.0,
        )

        corrected = cv2.warpAffine(
            image,
            matrix,
            (
                width,
                height,
            ),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        corrected = cls.normalize_ktp_canvas(
            corrected
        )

        return (
            corrected,
            metadata,
        )


    @classmethod
    def estimate_ktp_internal_axes(
        cls,
        image,
    ):
        """
        Estimasi sumbu internal KTP setelah homography.

        Horizontal diambil terutama dari baris teks kiri, sedangkan vertical
        diambil dari garis panjang seperti sisi foto/batas internal.
        """
        if (
            not isinstance(
                image,
                np.ndarray,
            )
            or image.size == 0
        ):
            raise ValueError(
                "image internal axis tidak boleh kosong."
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        height, width = gray.shape[:2]
        scale = min(
            1.0,
            640.0
            / max(
                width,
                1,
            ),
        )

        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (
                    max(
                        2,
                        int(
                            round(
                                width * scale
                            )
                        ),
                    ),
                    max(
                        2,
                        int(
                            round(
                                height * scale
                            )
                        ),
                    ),
                ),
                interpolation=cv2.INTER_AREA,
            )

        h, w = gray.shape[:2]
        blurred = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )
        edges = cv2.Canny(
            blurred,
            60,
            150,
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(
                24,
                w // 24,
            ),
            minLineLength=max(
                34,
                int(
                    round(
                        w * 0.065
                    )
                ),
            ),
            maxLineGap=max(
                8,
                int(
                    round(
                        w * 0.018
                    )
                ),
            ),
        )

        horizontal = []
        vertical = []

        if lines is not None:
            for line in np.asarray(
                lines
            ).reshape(-1, 4):
                x1, y1, x2, y2 = [
                    float(
                        value
                    )
                    for value in line
                ]

                dx = x2 - x1
                dy = y2 - y1
                length = float(
                    np.hypot(
                        dx,
                        dy,
                    )
                )

                if length < 10.0:
                    continue

                angle = float(
                    np.degrees(
                        np.arctan2(
                            dy,
                            dx,
                        )
                    )
                )

                # Horizontal evidence utama di area teks kiri.
                center_x = (
                    x1 + x2
                ) / 2.0
                center_y = (
                    y1 + y2
                ) / 2.0

                if (
                    center_x
                    <= w * 0.74
                    and h * 0.10
                    <= center_y
                    <= h * 0.84
                    and -6.0
                    <= angle
                    <= 6.0
                ):
                    horizontal.append(
                        (
                            angle,
                            length,
                        )
                    )

                # Vertical evidence boleh dari seluruh kartu, terutama foto.
                vertical_angle = angle

                if vertical_angle < -90.0:
                    vertical_angle += 180.0

                if vertical_angle > 90.0:
                    vertical_angle -= 180.0

                if (
                    abs(
                        abs(
                            vertical_angle
                        )
                        - 90.0
                    )
                    <= 7.0
                    and length
                    >= h * 0.13
                ):
                    # Samakan orientasi ke +90 agar median stabil.
                    if vertical_angle < 0.0:
                        vertical_angle += 180.0

                    vertical.append(
                        (
                            vertical_angle,
                            length,
                        )
                    )

        def robust_weighted_angle(
            samples,
            minimum,
        ):
            if len(
                samples
            ) < minimum:
                return None

            values = np.asarray(
                [
                    item[0]
                    for item
                    in samples
                ],
                dtype=np.float32,
            )
            weights = np.asarray(
                [
                    max(
                        item[1],
                        1.0,
                    )
                    for item
                    in samples
                ],
                dtype=np.float32,
            )

            median = float(
                np.median(
                    values
                )
            )
            deviation = np.abs(
                values
                - median
            )
            mad = float(
                np.median(
                    deviation
                )
            )
            keep = (
                deviation
                <= max(
                    0.45,
                    3.0 * mad,
                )
            )

            if int(
                np.count_nonzero(
                    keep
                )
            ) < minimum:
                return None

            return float(
                np.average(
                    values[
                        keep
                    ],
                    weights=weights[
                        keep
                    ],
                )
            )

        horizontal_angle = (
            robust_weighted_angle(
                horizontal,
                minimum=4,
            )
        )
        vertical_angle = (
            robust_weighted_angle(
                vertical,
                minimum=2,
            )
        )

        if (
            horizontal_angle is None
            or vertical_angle is None
        ):
            return {
                "available": False,
                "horizontal_angle": (
                    horizontal_angle
                ),
                "vertical_angle": (
                    vertical_angle
                ),
                "horizontal_count": len(
                    horizontal
                ),
                "vertical_count": len(
                    vertical
                ),
                "orthogonality_error": None,
            }

        orthogonality_error = abs(
            (
                vertical_angle
                - horizontal_angle
            )
            - 90.0
        )

        return {
            "available": True,
            "horizontal_angle": float(
                horizontal_angle
            ),
            "vertical_angle": float(
                vertical_angle
            ),
            "horizontal_count": len(
                horizontal
            ),
            "vertical_count": len(
                vertical
            ),
            "orthogonality_error": float(
                orthogonality_error
            ),
        }

    def rectify_ktp_internal_affine(
        self,
        image,
    ):
        """
        Hilangkan residual affine shear pada isi KTP tanpa mengubah corner.

        Hasil hanya diterima jika sumbu internal membaik dan post-warp
        geometry tidak turun.
        """
        before_axes = (
            self.estimate_ktp_internal_axes(
                image
            )
        )

        if not before_axes.get(
            "available",
            False,
        ):
            return (
                image.copy(),
                {
                    "applied": False,
                    "reason": "insufficient_axes",
                    "before": before_axes,
                },
            )

        horizontal_angle = float(
            before_axes[
                "horizontal_angle"
            ]
        )
        vertical_angle = float(
            before_axes[
                "vertical_angle"
            ]
        )
        orthogonality_error = float(
            before_axes[
                "orthogonality_error"
            ]
        )

        # KTP yang sudah cukup ortogonal tidak disentuh.
        if (
            abs(
                horizontal_angle
            ) < 0.40
            and orthogonality_error
            < 0.55
        ):
            return (
                image.copy(),
                {
                    "applied": False,
                    "reason": "already_orthogonal",
                    "before": before_axes,
                },
            )

        # Jangan gunakan affine correction untuk geometri ekstrem.
        if (
            abs(
                horizontal_angle
            ) > 4.0
            or orthogonality_error
            > 4.0
        ):
            return (
                image.copy(),
                {
                    "applied": False,
                    "reason": "axis_error_too_large",
                    "before": before_axes,
                },
            )

        horizontal_rad = np.radians(
            horizontal_angle
        )
        vertical_rad = np.radians(
            vertical_angle
        )

        basis = np.array(
            [
                [
                    np.cos(
                        horizontal_rad
                    ),
                    np.cos(
                        vertical_rad
                    ),
                ],
                [
                    np.sin(
                        horizontal_rad
                    ),
                    np.sin(
                        vertical_rad
                    ),
                ],
            ],
            dtype=np.float32,
        )

        determinant = float(
            np.linalg.det(
                basis
            )
        )

        if abs(
            determinant
        ) < 0.90:
            return (
                image.copy(),
                {
                    "applied": False,
                    "reason": "unstable_basis",
                    "before": before_axes,
                },
            )

        affine_2x2 = np.linalg.inv(
            basis
        ).astype(
            np.float32
        )

        # Batasi perubahan affine agar tidak over-correct.
        if (
            np.max(
                np.abs(
                    affine_2x2
                    - np.eye(
                        2,
                        dtype=np.float32,
                    )
                )
            )
            > 0.09
        ):
            return (
                image.copy(),
                {
                    "applied": False,
                    "reason": "affine_change_too_large",
                    "before": before_axes,
                },
            )

        height, width = image.shape[:2]
        center = np.array(
            [
                width / 2.0,
                height / 2.0,
            ],
            dtype=np.float32,
        )

        translation = (
            center
            - affine_2x2
            @ center
        )

        matrix = np.column_stack(
            (
                affine_2x2,
                translation,
            )
        ).astype(
            np.float32
        )

        corrected = cv2.warpAffine(
            image,
            matrix,
            (
                width,
                height,
            ),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        corrected = self.normalize_ktp_canvas(
            corrected
        )

        after_axes = (
            self.estimate_ktp_internal_axes(
                corrected
            )
        )

        if not after_axes.get(
            "available",
            False,
        ):
            return (
                image.copy(),
                {
                    "applied": False,
                    "reason": "after_axes_unavailable",
                    "before": before_axes,
                    "after": after_axes,
                },
            )

        before_orth = float(
            before_axes[
                "orthogonality_error"
            ]
        )
        after_orth = float(
            after_axes[
                "orthogonality_error"
            ]
        )
        before_horizontal = abs(
            float(
                before_axes[
                    "horizontal_angle"
                ]
            )
        )
        after_horizontal = abs(
            float(
                after_axes[
                    "horizontal_angle"
                ]
            )
        )

        before_geometry, _ = (
            self.perspective_engine
            ._post_warp_geometry_score(
                image
            )
        )
        after_geometry, _ = (
            self.perspective_engine
            ._post_warp_geometry_score(
                corrected
            )
        )

        improved_axes = (
            (
                after_orth
                <= before_orth
                - 0.20
            )
            or (
                after_horizontal
                <= before_horizontal
                - 0.20
            )
        )

        if (
            not improved_axes
            or after_geometry
            < before_geometry
            - 0.01
        ):
            return (
                image.copy(),
                {
                    "applied": False,
                    "reason": "no_clear_internal_gain",
                    "before": before_axes,
                    "after": after_axes,
                    "before_geometry": float(
                        before_geometry
                    ),
                    "after_geometry": float(
                        after_geometry
                    ),
                },
            )

        return (
            corrected,
            {
                "applied": True,
                "reason": "internal_axes_rectified",
                "before": before_axes,
                "after": after_axes,
                "before_geometry": float(
                    before_geometry
                ),
                "after_geometry": float(
                    after_geometry
                ),
                "matrix": (
                    matrix.tolist()
                ),
            },
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

    @classmethod
    def normalize_ktp_canvas(
        cls,
        image,
    ):
        if (
            not isinstance(
                image,
                np.ndarray,
            )
            or image.size == 0
        ):
            raise ValueError(
                "image KTP canonical tidak boleh kosong."
            )

        if (
            image.shape[1]
            == cls.KTP_CANONICAL_WIDTH
            and image.shape[0]
            == cls.KTP_CANONICAL_HEIGHT
        ):
            return image.copy()

        return cv2.resize(
            image,
            (
                cls.KTP_CANONICAL_WIDTH,
                cls.KTP_CANONICAL_HEIGHT,
            ),
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
            result = self.normalize_ktp_canvas(
                result
            )

            (
                result,
                internal_affine,
            ) = self.rectify_ktp_internal_affine(
                result
            )
            self.last_detection[
                "internal_affine_rectification"
            ] = internal_affine

            (
                result,
                residual_skew,
            ) = self.stabilize_ktp_residual_skew(
                result
            )
            self.last_detection[
                "residual_skew"
            ] = residual_skew

            # Contract akhir KTP selalu exact canonical canvas.
            result = self.normalize_ktp_canvas(
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
