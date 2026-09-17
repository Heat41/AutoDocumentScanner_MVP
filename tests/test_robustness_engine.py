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
        image = self._place_card(
            background=(55, 65, 145),
            destination=np.array(
                [
                    [280, 95],
                    [690, 135],
                    [735, 610],
                    [245, 575],
                ],
                dtype=np.float32,
            ),
        )

        corrected, corners, _ = self.engine.correct(
            image
        )

        self.assertIsNotNone(corners)
        self.assertIsNotNone(corrected)

        h, w = corrected.shape[:2]
        ratio = max(w, h) / max(min(w, h), 1)

        self.assertLess(
            abs(ratio - self.engine.target_ratio),
            0.55,
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
