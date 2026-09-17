import unittest

import cv2
import numpy as np

from robustness_engine import RobustPerspectiveEngine


class TestRobustPerspectiveEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RobustPerspectiveEngine()

    @staticmethod
    def _synthetic_card():
        card = np.full(
            (340, 540, 3),
            (220, 205, 105),
            dtype=np.uint8,
        )

        cv2.rectangle(
            card,
            (3, 3),
            (536, 336),
            (70, 70, 70),
            3,
        )

        cv2.rectangle(
            card,
            (390, 75),
            (500, 220),
            (80, 85, 90),
            -1,
        )

        for y in range(70, 275, 31):
            cv2.line(
                card,
                (35, y),
                (335, y),
                (45, 45, 45),
                3,
            )

        cv2.line(
            card,
            (35, 42),
            (305, 42),
            (35, 35, 35),
            5,
        )

        return card

    def _place_card(
        self,
        background=(95, 95, 95),
        destination=None,
    ):
        canvas = np.full(
            (720, 960, 3),
            background,
            dtype=np.uint8,
        )

        card = self._synthetic_card()

        if destination is None:
            destination = np.array(
                [
                    [175, 155],
                    [770, 105],
                    [815, 540],
                    [120, 585],
                ],
                dtype=np.float32,
            )

        source = np.array(
            [
                [0, 0],
                [539, 0],
                [539, 339],
                [0, 339],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            source,
            destination,
        )

        warped = cv2.warpPerspective(
            card,
            matrix,
            (960, 720),
        )

        mask = cv2.warpPerspective(
            np.full(
                (340, 540),
                255,
                dtype=np.uint8,
            ),
            matrix,
            (960, 720),
        )

        canvas[mask > 0] = warped[mask > 0]

        return canvas

    def _assert_detected_near(
        self,
        image,
        expected,
        max_normalized_error=0.10,
    ):
        corrected, corners, metadata = self.engine.correct(
            image
        )

        self.assertIsNotNone(corners)
        self.assertIsNotNone(corrected)

        detected = self.engine.order_points(
            corners
        )
        expected = self.engine.order_points(
            expected
        )

        mean_error = float(
            np.mean(
                np.linalg.norm(
                    detected - expected,
                    axis=1,
                )
            )
        )

        image_diagonal = float(
            np.hypot(
                image.shape[1],
                image.shape[0],
            )
        )

        normalized_error = (
            mean_error
            / max(image_diagonal, 1.0)
        )

        self.assertLess(
            normalized_error,
            max_normalized_error,
        )

        self.assertIn(
            "robustness_mode",
            metadata,
        )

    def test_high_confidence_baseline_is_locked(self):
        image = self._place_card(
            background=(85, 85, 85),
        )

        corners, metadata = self.engine.detect(
            image
        )

        self.assertIsNotNone(corners)
        self.assertIn(
            metadata.get("robustness_mode"),
            {
                "baseline_locked",
                "baseline_preserved",
                "fallback_selected",
            },
        )

    def test_low_light_still_produces_candidate(self):
        image = self._place_card(
            background=(40, 40, 40),
        )

        image = cv2.convertScaleAbs(
            image,
            alpha=0.52,
            beta=7,
        )

        corrected, corners, metadata = self.engine.correct(
            image
        )

        self.assertIsNotNone(corners)
        self.assertIsNotNone(corrected)
        self.assertGreater(
            corrected.shape[0],
            60,
        )
        self.assertGreater(
            corrected.shape[1],
            60,
        )
        self.assertIn(
            "robustness_mode",
            metadata,
        )

    def test_red_background_perspective(self):
        destination = np.array(
            [
                [280, 95],
                [690, 135],
                [735, 610],
                [245, 575],
            ],
            dtype=np.float32,
        )

        image = self._place_card(
            background=(55, 65, 145),
            destination=destination,
        )

        self._assert_detected_near(
            image,
            destination,
            max_normalized_error=0.08,
        )

    def test_small_card_in_large_frame(self):
        destination = np.array(
            [
                [345, 245],
                [620, 232],
                [635, 412],
                [330, 425],
            ],
            dtype=np.float32,
        )

        image = self._place_card(
            background=(78, 82, 86),
            destination=destination,
        )

        self._assert_detected_near(
            image,
            destination,
            max_normalized_error=0.12,
        )

    def test_card_near_frame_edge(self):
        destination = np.array(
            [
                [18, 72],
                [650, 42],
                [700, 478],
                [8, 520],
            ],
            dtype=np.float32,
        )

        image = self._place_card(
            background=(105, 102, 98),
            destination=destination,
        )

        self._assert_detected_near(
            image,
            destination,
            max_normalized_error=0.11,
        )

    def test_mild_blur_still_detects_card(self):
        destination = np.array(
            [
                [155, 150],
                [775, 118],
                [810, 535],
                [118, 570],
            ],
            dtype=np.float32,
        )

        image = self._place_card(
            background=(90, 90, 90),
            destination=destination,
        )

        image = cv2.GaussianBlur(
            image,
            (7, 7),
            1.6,
        )

        self._assert_detected_near(
            image,
            destination,
            max_normalized_error=0.12,
        )

    def test_shadow_gradient_still_detects_card(self):
        destination = np.array(
            [
                [170, 125],
                [765, 100],
                [805, 535],
                [125, 565],
            ],
            dtype=np.float32,
        )

        image = self._place_card(
            background=(95, 95, 95),
            destination=destination,
        ).astype(np.float32)

        h, w = image.shape[:2]
        gradient = np.linspace(
            0.48,
            1.0,
            w,
            dtype=np.float32,
        )[None, :, None]

        image = np.clip(
            image * gradient,
            0,
            255,
        ).astype(np.uint8)

        self._assert_detected_near(
            image,
            destination,
            max_normalized_error=0.12,
        )

    def test_glare_band_still_produces_candidate(self):
        destination = np.array(
            [
                [175, 145],
                [770, 112],
                [808, 535],
                [125, 575],
            ],
            dtype=np.float32,
        )

        image = self._place_card(
            background=(82, 82, 82),
            destination=destination,
        )

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (410, 90),
            (520, 640),
            (245, 245, 245),
            -1,
        )
        image = cv2.addWeighted(
            image,
            0.72,
            overlay,
            0.28,
            0,
        )

        corrected, corners, metadata = self.engine.correct(
            image
        )

        self.assertIsNotNone(corners)
        self.assertIsNotNone(corrected)
        self.assertIn(
            "robustness_mode",
            metadata,
        )

    def test_detection_variants_keep_geometry(self):
        image = self._place_card()

        variants = dict(
            self.engine._variants(image)
        )

        self.assertEqual(
            set(variants),
            {
                "clahe",
                "bright",
                "dark",
                "shadow_norm",
            },
        )

        for variant in variants.values():
            self.assertEqual(
                variant.shape,
                image.shape,
            )


if __name__ == "__main__":
    unittest.main()
