import cv2
import numpy as np


class DocumentQualityChecker:
    """
    Quality gate non-destruktif untuk hasil scanner.

    Modul ini tidak mengubah gambar dan tidak menggagalkan output. Tugasnya
    hanya memberi skor + warning agar kasus yang meragukan dapat ditandai
    tanpa merusak baseline perspective correction yang sudah PASS.
    """

    def __init__(
        self,
        target_ratio=85.60 / 53.98,
    ):
        self.target_ratio = float(target_ratio)

    @staticmethod
    def _clip01(value):
        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    def assess(
        self,
        image,
        detection_metadata=None,
    ):
        if image is None or image.size == 0:
            return {
                "status": "review",
                "score": 0.0,
                "warnings": ["hasil kosong"],
            }

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        h, w = gray.shape[:2]
        short_side = min(h, w)
        long_side = max(h, w)

        ratio = (
            long_side
            / max(short_side, 1)
        )
        ratio_error = abs(
            ratio - self.target_ratio
        ) / self.target_ratio
        ratio_score = self._clip01(
            1.0 - ratio_error / 0.08
        )

        laplacian_variance = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )
        sharpness_score = self._clip01(
            laplacian_variance / 180.0
        )

        brightness = float(
            np.mean(gray)
        )
        brightness_score = self._clip01(
            1.0
            - abs(brightness - 145.0)
            / 120.0
        )

        contrast = float(
            np.std(gray)
        )
        contrast_score = self._clip01(
            contrast / 55.0
        )

        dark_clip = float(
            np.mean(gray <= 8)
        )
        bright_clip = float(
            np.mean(gray >= 247)
        )
        clipped_fraction = (
            dark_clip + bright_clip
        )
        exposure_score = self._clip01(
            1.0
            - clipped_fraction / 0.30
        )

        resolution_score = self._clip01(
            short_side / 420.0
        )

        metadata = detection_metadata or {}
        detection_raw = float(
            metadata.get(
                "score",
                0.0,
            )
            or 0.0
        )
        detection_score = self._clip01(
            detection_raw / 0.75
        )

        score = float(
            detection_score * 0.28
            + ratio_score * 0.18
            + sharpness_score * 0.18
            + exposure_score * 0.13
            + contrast_score * 0.10
            + resolution_score * 0.08
            + brightness_score * 0.05
        )

        warnings = []

        if detection_raw < 0.50:
            warnings.append(
                "confidence deteksi rendah"
            )

        if ratio_error > 0.035:
            warnings.append(
                "rasio/geometri perlu ditinjau"
            )

        if laplacian_variance < 35.0:
            warnings.append(
                "gambar cukup blur"
            )

        if brightness < 55.0:
            warnings.append(
                "gambar terlalu gelap"
            )
        elif brightness > 215.0:
            warnings.append(
                "gambar terlalu terang"
            )

        if clipped_fraction > 0.22:
            warnings.append(
                "highlight/shadow terpotong"
            )

        if contrast < 22.0:
            warnings.append(
                "kontras rendah"
            )

        if short_side < 240:
            warnings.append(
                "resolusi hasil rendah"
            )

        if score >= 0.72 and not warnings:
            status = "pass"
        elif score >= 0.52:
            status = "warning"
        else:
            status = "review"

        return {
            "status": status,
            "score": round(score, 4),
            "warnings": warnings,
            "metrics": {
                "detection_score": round(
                    detection_raw,
                    4,
                ),
                "ratio": round(ratio, 4),
                "ratio_error": round(
                    ratio_error,
                    4,
                ),
                "sharpness": round(
                    laplacian_variance,
                    2,
                ),
                "brightness": round(
                    brightness,
                    2,
                ),
                "contrast": round(
                    contrast,
                    2,
                ),
                "clipped_fraction": round(
                    clipped_fraction,
                    4,
                ),
                "short_side": int(
                    short_side
                ),
            },
        }
