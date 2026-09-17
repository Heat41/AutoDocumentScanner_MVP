import cv2
import numpy as np


class FinalScanValidator:
    """Final safety gate after automatic perspective detection.

    Validator ini sengaja konservatif. Ia tidak memperbaiki atau mengubah
    gambar; tugasnya hanya menolak geometri yang jelas tidak masuk akal dan
    menandai hasil meragukan untuk review. Dengan begitu baseline perspective
    yang sudah PASS tetap tidak tersentuh.
    """

    def __init__(self, target_ratio=85.60 / 53.98):
        self.target_ratio = float(target_ratio)

    @staticmethod
    def _result(status, hard_valid, warnings, metrics=None):
        return {
            "status": status,
            "hard_valid": bool(hard_valid),
            "warnings": list(warnings),
            "metrics": dict(metrics or {}),
        }

    @staticmethod
    def _angles(points):
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        result = []

        for index in range(4):
            previous = points[(index - 1) % 4] - points[index]
            following = points[(index + 1) % 4] - points[index]

            previous_length = float(np.linalg.norm(previous))
            following_length = float(np.linalg.norm(following))

            if previous_length < 1e-6 or following_length < 1e-6:
                return []

            cosine = float(
                np.dot(previous, following)
                / (previous_length * following_length)
            )
            cosine = float(np.clip(cosine, -1.0, 1.0))
            result.append(float(np.degrees(np.arccos(cosine))))

        return result

    def validate_corners(self, image_shape, corners):
        if corners is None:
            return self._result(
                "review",
                False,
                ["empat sudut tidak tersedia"],
            )

        points = np.asarray(corners, dtype=np.float32)

        if points.size != 8:
            return self._result(
                "review",
                False,
                ["jumlah sudut tidak valid"],
            )

        points = points.reshape(4, 2)

        if not np.all(np.isfinite(points)):
            return self._result(
                "review",
                False,
                ["koordinat sudut tidak valid"],
            )

        h, w = image_shape[:2]
        image_area = float(max(h * w, 1))
        contour = points.astype(np.float32)
        area = abs(float(cv2.contourArea(contour)))
        area_ratio = area / image_area

        side_lengths = [
            float(np.linalg.norm(points[(i + 1) % 4] - points[i]))
            for i in range(4)
        ]
        min_side = min(side_lengths)

        warnings = []
        hard_valid = True

        if not cv2.isContourConvex(points.astype(np.int32)):
            warnings.append("quadrilateral tidak convex")
            hard_valid = False

        # Fallback small-card menerima area mulai sekitar 1.5%. Batas hard
        # dibuat lebih longgar agar kartu jauh masih sah.
        if area_ratio < 0.006:
            warnings.append("area kartu terlalu kecil")
            hard_valid = False
        elif area_ratio < 0.018:
            warnings.append("kartu sangat kecil di frame")

        minimum_side = max(10.0, min(h, w) * 0.012)
        if min_side < minimum_side:
            warnings.append("sisi kartu terlalu pendek")
            hard_valid = False

        margin_x = w * 0.10
        margin_y = h * 0.10
        outside = (
            (points[:, 0] < -margin_x)
            | (points[:, 0] > w - 1 + margin_x)
            | (points[:, 1] < -margin_y)
            | (points[:, 1] > h - 1 + margin_y)
        )
        outside_count = int(np.sum(outside))

        if outside_count >= 2:
            warnings.append("sudut terlalu jauh di luar frame")
            hard_valid = False
        elif outside_count == 1:
            warnings.append("satu sudut sedikit di luar frame")

        angles = self._angles(points)
        if len(angles) != 4:
            warnings.append("sudut quadrilateral degeneratif")
            hard_valid = False
        else:
            min_angle = min(angles)
            max_angle = max(angles)

            if min_angle < 18.0 or max_angle > 162.0:
                warnings.append("bentuk perspektif terlalu ekstrem")
                hard_valid = False
            elif min_angle < 28.0 or max_angle > 152.0:
                warnings.append("perspektif sangat ekstrem")

        if not hard_valid:
            status = "review"
        elif warnings:
            status = "warning"
        else:
            status = "pass"

        return self._result(
            status,
            hard_valid,
            warnings,
            {
                "area_ratio": round(area_ratio, 4),
                "min_side": round(min_side, 2),
                "outside_count": outside_count,
                "angles": [round(value, 2) for value in angles],
            },
        )

    def validate_output(self, image, detection_metadata=None, quality=None):
        if image is None or image.size == 0:
            return self._result(
                "review",
                False,
                ["hasil scan kosong"],
            )

        h, w = image.shape[:2]
        short_side = min(h, w)
        long_side = max(h, w)
        ratio = long_side / max(short_side, 1)
        ratio_error = abs(ratio - self.target_ratio) / self.target_ratio

        warnings = []
        hard_valid = True

        if short_side < 70:
            warnings.append("dimensi output terlalu kecil")
            hard_valid = False
        elif short_side < 180:
            warnings.append("resolusi output rendah")

        # normalize_ktp_ratio menjaga error maksimal sekitar 2.5%. Beri ruang
        # toleransi sedikit lebih lebar untuk rounding/resampling.
        if ratio_error > 0.045:
            warnings.append("rasio output tidak sesuai KTP")
            hard_valid = False
        elif ratio_error > 0.028:
            warnings.append("rasio output perlu ditinjau")

        metadata = detection_metadata or {}
        detection_score = float(metadata.get("score", 0.0) or 0.0)
        if detection_score < 0.32:
            warnings.append("confidence final terlalu rendah")
            hard_valid = False
        elif detection_score < 0.50:
            warnings.append("confidence final rendah")

        quality = quality or {}
        quality_status = str(quality.get("status", "")).lower()
        if quality_status == "review":
            warnings.append("quality gate meminta review")
        elif quality_status == "warning":
            warnings.append("quality gate memberi peringatan")

        if not hard_valid:
            status = "review"
        elif warnings:
            status = "warning"
        else:
            status = "pass"

        return self._result(
            status,
            hard_valid,
            warnings,
            {
                "ratio": round(ratio, 4),
                "ratio_error": round(ratio_error, 4),
                "short_side": int(short_side),
                "detection_score": round(detection_score, 4),
            },
        )

    def combine(self, corner_validation, output_validation):
        corner_validation = corner_validation or {}
        output_validation = output_validation or {}

        warnings = []
        warnings.extend(corner_validation.get("warnings") or [])
        warnings.extend(output_validation.get("warnings") or [])
        warnings = list(dict.fromkeys(warnings))

        hard_valid = bool(
            corner_validation.get("hard_valid", False)
            and output_validation.get("hard_valid", False)
        )

        statuses = {
            str(corner_validation.get("status", "review")),
            str(output_validation.get("status", "review")),
        }

        if not hard_valid or "review" in statuses:
            status = "review"
        elif "warning" in statuses:
            status = "warning"
        else:
            status = "pass"

        return {
            "status": status,
            "hard_valid": hard_valid,
            "warnings": warnings,
            "corner": corner_validation,
            "output": output_validation,
        }
