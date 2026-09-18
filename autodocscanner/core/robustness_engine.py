import cv2
import numpy as np

from autodocscanner.core.perspective_engine import AutoPerspectiveEngine, PerspectiveCandidate


class RobustPerspectiveEngine(AutoPerspectiveEngine):
    """
    Layer robustness di atas baseline AutoPerspectiveEngine.

    Prinsip penting:
    - baseline yang sudah confidence tinggi TIDAK diubah;
    - retry hanya dilakukan saat baseline gagal / confidence rendah;
    - retry memakai beberapa photometric variant tanpa mengubah geometri;
    - kandidat hasil retry selalu dinilai ulang pada foto original;
    - small-card fallback hanya aktif pada kasus sulit.

    Dengan pendekatan ini kasus yang sudah PASS tetap aman, sementara foto
    gelap, kontras rendah, bayangan, glare ringan, atau KTP lebih kecil punya
    jalur fallback tambahan.
    """

    def __init__(
        self,
        target_ratio=85.60 / 53.98,
        detection_height=1000,
        min_area_ratio=0.035,
        stable_score=0.72,
        retry_score=0.60,
        improvement_margin=0.035,
    ):
        super().__init__(
            target_ratio=target_ratio,
            detection_height=detection_height,
            min_area_ratio=min_area_ratio,
        )

        self.stable_score = stable_score
        self.retry_score = retry_score
        self.improvement_margin = improvement_margin

        # Fallback khusus kartu yang relatif kecil di frame. Engine ini tidak
        # pernah dipakai bila baseline sudah confidence tinggi.
        self.small_card_engine = AutoPerspectiveEngine(
            target_ratio=target_ratio,
            detection_height=detection_height,
            min_area_ratio=0.015,
        )

    @staticmethod
    def _gamma(image, gamma):
        gamma = max(float(gamma), 0.05)
        inv = 1.0 / gamma

        table = np.array(
            [
                ((value / 255.0) ** inv) * 255.0
                for value in range(256)
            ],
            dtype=np.uint8,
        )

        return cv2.LUT(
            image,
            table,
        )

    @staticmethod
    def _clahe_color(image):
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.2,
            tileGridSize=(8, 8),
        )

        l = clahe.apply(l)

        return cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2BGR,
        )

    @staticmethod
    def _shadow_normalized(image):
        """
        Normalisasi pencahayaan hanya untuk tahap detection.
        Output final tetap selalu berasal dari foto original.
        """
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        l, a, b = cv2.split(lab)

        blur = cv2.GaussianBlur(
            l,
            (0, 0),
            sigmaX=21,
            sigmaY=21,
        )

        normalized = cv2.divide(
            l,
            np.maximum(blur, 1),
            scale=155,
        )

        normalized = cv2.normalize(
            normalized,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        return cv2.cvtColor(
            cv2.merge(
                (normalized, a, b)
            ),
            cv2.COLOR_LAB2BGR,
        )

    def _variants(self, image):
        return (
            (
                "clahe",
                self._clahe_color(image),
            ),
            (
                "bright",
                self._gamma(image, 0.72),
            ),
            (
                "dark",
                self._gamma(image, 1.35),
            ),
            (
                "shadow_norm",
                self._shadow_normalized(image),
            ),
        )

    def _evaluate_on_original(
        self,
        image,
        corners,
        source="robust",
    ):
        """
        Kandidat retry wajib lolos scoring ulang pada foto original.
        Ini mencegah variant pencahayaan menghasilkan corner palsu.
        """
        if corners is None:
            return -1.0

        resized, scale = self._resize_for_detection(
            image
        )

        _, union_edges = self._build_detection_maps(
            resized
        )

        candidate = PerspectiveCandidate(
            points=(
                np.asarray(
                    corners,
                    dtype=np.float32,
                )
                * scale
            ),
            score=0.0,
            source=source,
        )

        geometry_score = self._candidate_score(
            candidate,
            resized,
            union_edges,
        )

        if geometry_score < 0:
            return -1.0

        warped = self.warp(
            image,
            corners,
        )

        quality = self._warp_quality(
            warped
        )

        if quality < 0:
            return -1.0

        return float(
            geometry_score * 0.58
            + quality * 0.42
        )

    @staticmethod
    def _metadata(
        metadata,
        **extra,
    ):
        result = dict(
            metadata or {}
        )
        result.update(extra)
        return result

    def detect(self, image):
        base_corners, base_metadata = super().detect(
            image
        )

        base_score = float(
            (base_metadata or {}).get(
                "score",
                0.0,
            )
            or 0.0
        )

        # Baseline yang sudah kuat kita kunci. Ini jalur untuk dua kasus PASS
        # yang sudah diuji sebelumnya dan mencegah regression akibat fallback.
        if (
            base_corners is not None
            and base_score >= self.stable_score
        ):
            return base_corners, self._metadata(
                base_metadata,
                robustness_mode="baseline_locked",
                base_score=base_score,
                retry_count=0,
            )

        best_corners = base_corners
        best_metadata = dict(
            base_metadata or {}
        )

        if base_corners is not None:
            best_score = self._evaluate_on_original(
                image,
                base_corners,
                source=(
                    best_metadata.get(
                        "selected_source",
                        "baseline",
                    )
                    or "baseline"
                ),
            )
        else:
            best_score = -1.0

        retry_count = 0

        # Photometric retry: koordinat tidak berubah karena variant hanya
        # mengubah luminance/contrast.
        for variant_name, variant in self._variants(
            image
        ):
            retry_count += 1

            corners, metadata = super().detect(
                variant
            )

            if corners is None:
                continue

            source = (
                (metadata or {}).get(
                    "selected_source",
                    "unknown",
                )
                or "unknown"
            )

            score = self._evaluate_on_original(
                image,
                corners,
                source=source,
            )

            if score > best_score:
                best_score = score
                best_corners = corners
                best_metadata = self._metadata(
                    metadata,
                    selected_source=(
                        f"robust_{variant_name}:{source}"
                    ),
                    score=score,
                )

        # Jika baseline benar-benar gagal / sangat lemah, izinkan engine yang
        # menerima kartu lebih kecil. Tetap dinilai ulang pada foto original.
        if (
            base_corners is None
            or base_score < self.retry_score
        ):
            small_inputs = [
                ("small_original", image),
            ]

            small_inputs.extend(
                (
                    f"small_{name}",
                    variant,
                )
                for name, variant in self._variants(
                    image
                )
            )

            for variant_name, variant in small_inputs:
                retry_count += 1

                corners, metadata = (
                    self.small_card_engine.detect(
                        variant
                    )
                )

                if corners is None:
                    continue

                source = (
                    (metadata or {}).get(
                        "selected_source",
                        "unknown",
                    )
                    or "unknown"
                )

                score = self._evaluate_on_original(
                    image,
                    corners,
                    source=source,
                )

                if score > best_score:
                    best_score = score
                    best_corners = corners
                    best_metadata = self._metadata(
                        metadata,
                        selected_source=(
                            f"robust_{variant_name}:{source}"
                        ),
                        score=score,
                    )

        if best_corners is None:
            return None, self._metadata(
                base_metadata,
                robustness_mode="failed",
                base_score=base_score,
                retry_count=retry_count,
            )

        # Bila baseline ada, fallback harus benar-benar lebih baik. Kalau hanya
        # beda tipis, pertahankan baseline agar output stabil.
        if base_corners is not None:
            base_original_score = self._evaluate_on_original(
                image,
                base_corners,
                source=(
                    (base_metadata or {}).get(
                        "selected_source",
                        "baseline",
                    )
                    or "baseline"
                ),
            )

            if (
                best_score
                < base_original_score
                + self.improvement_margin
            ):
                return base_corners, self._metadata(
                    base_metadata,
                    robustness_mode="baseline_preserved",
                    base_score=base_score,
                    retry_count=retry_count,
                )

        return best_corners, self._metadata(
            best_metadata,
            robustness_mode="fallback_selected",
            base_score=base_score,
            retry_count=retry_count,
            score=max(
                best_score,
                0.0,
            ),
        )
