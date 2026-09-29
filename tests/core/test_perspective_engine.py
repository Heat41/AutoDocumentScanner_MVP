import unittest

import cv2
import numpy as np

from autodocscanner.core.perspective_engine import (
    AutoPerspectiveEngine,
    PerspectiveCandidate,
)
from autodocscanner.core.scanner import AutoDocumentScanner


class TestAutoPerspectiveEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AutoPerspectiveEngine()

    def test_order_points(self):
        points = np.array(
            [
                [500, 300],
                [100, 100],
                [120, 320],
                [520, 90],
            ],
            dtype=np.float32,
        )

        ordered = self.engine.order_points(
            points
        )

        self.assertEqual(
            ordered.shape,
            (4, 2),
        )
        self.assertTrue(
            np.allclose(
                ordered[0],
                [100, 100],
            )
        )

    def test_synthetic_perspective_correction(self):
        canvas = np.full(
            (700, 900, 3),
            225,
            dtype=np.uint8,
        )

        source = np.full(
            (340, 540, 3),
            (205, 210, 115),
            dtype=np.uint8,
        )

        cv2.rectangle(
            source,
            (4, 4),
            (535, 335),
            (60, 60, 60),
            4,
        )

        for y in range(
            70,
            270,
            35,
        ):
            cv2.line(
                source,
                (35, y),
                (330, y),
                (40, 40, 40),
                3,
            )

        destination = np.array(
            [
                [180, 145],
                [735, 95],
                [770, 520],
                [125, 555],
            ],
            dtype=np.float32,
        )

        source_points = np.array(
            [
                [0, 0],
                [539, 0],
                [539, 339],
                [0, 339],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            source_points,
            destination,
        )

        warped = cv2.warpPerspective(
            source,
            matrix,
            (900, 700),
            borderValue=(225, 225, 225),
        )

        mask = cv2.warpPerspective(
            np.full(
                (340, 540),
                255,
                dtype=np.uint8,
            ),
            matrix,
            (900, 700),
        )

        canvas[
            mask > 0
        ] = warped[
            mask > 0
        ]

        corrected, corners, metadata = (
            self.engine.correct(
                canvas
            )
        )

        self.assertIsNotNone(
            corrected
        )
        self.assertIsNotNone(
            corners
        )
        self.assertGreater(
            metadata["candidate_count"],
            0,
        )

        h, w = corrected.shape[:2]
        ratio = max(w, h) / min(w, h)

        self.assertLess(
            abs(
                ratio
                - self.engine.target_ratio
            ),
            0.35,
        )


    def test_line_refinement_keeps_quad_when_boundary_support_does_not_improve(self):
        image = np.zeros(
            (400, 640, 3),
            dtype=np.uint8,
        )
        edge_map = np.zeros(
            (400, 640),
            dtype=np.uint8,
        )
        quad = np.array(
            [
                [80, 70],
                [560, 70],
                [560, 330],
                [80, 330],
            ],
            dtype=np.float32,
        )

        refined = (
            self.engine
            ._refine_quad_from_edge_lines(
                image,
                quad,
                edge_map,
            )
        )

        self.assertTrue(
            np.allclose(
                refined,
                quad,
            )
        )

    def test_line_refinement_can_improve_small_shifted_boundary(self):
        image = np.zeros(
            (400, 640, 3),
            dtype=np.uint8,
        )
        edge_map = np.zeros(
            (400, 640),
            dtype=np.uint8,
        )

        # Boundary aktual sedikit di luar quad awal.
        cv2.line(
            edge_map,
            (76, 66),
            (564, 66),
            255,
            2,
        )
        cv2.line(
            edge_map,
            (564, 66),
            (564, 334),
            255,
            2,
        )
        cv2.line(
            edge_map,
            (564, 334),
            (76, 334),
            255,
            2,
        )
        cv2.line(
            edge_map,
            (76, 334),
            (76, 66),
            255,
            2,
        )

        quad = np.array(
            [
                [80, 70],
                [560, 70],
                [560, 330],
                [80, 330],
            ],
            dtype=np.float32,
        )

        refined = (
            self.engine
            ._refine_quad_from_edge_lines(
                image,
                quad,
                edge_map,
            )
        )

        self.assertEqual(
            refined.shape,
            (4, 2),
        )
        self.assertLessEqual(
            float(
                np.mean(
                    np.linalg.norm(
                        refined - quad,
                        axis=1,
                    )
                )
            ),
            40.0,
        )

    def test_post_warp_geometry_prefers_straight_card(self):
        straight = np.full(
            (340, 540, 3),
            235,
            dtype=np.uint8,
        )

        for y in range(
            80,
            270,
            28,
        ):
            cv2.line(
                straight,
                (35, y),
                (330, y),
                (30, 30, 30),
                3,
            )

        cv2.rectangle(
            straight,
            (390, 85),
            (500, 245),
            (40, 40, 40),
            3,
        )

        matrix = cv2.getRotationMatrix2D(
            (270.0, 170.0),
            2.0,
            1.0,
        )
        tilted = cv2.warpAffine(
            straight,
            matrix,
            (540, 340),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        straight_score, straight_meta = (
            self.engine
            ._post_warp_geometry_score(
                straight
            )
        )
        tilted_score, tilted_meta = (
            self.engine
            ._post_warp_geometry_score(
                tilted
            )
        )

        self.assertGreater(
            straight_score,
            tilted_score,
        )
        self.assertGreaterEqual(
            straight_score,
            0.70,
        )
        self.assertIsNotNone(
            straight_meta[
                "horizontal"
            ]
        )
        self.assertIsNotNone(
            tilted_meta[
                "horizontal"
            ]
        )

    def test_post_warp_geometry_is_neutral_without_enough_lines(self):
        blank = np.full(
            (340, 540, 3),
            220,
            dtype=np.uint8,
        )

        score, metadata = (
            self.engine
            ._post_warp_geometry_score(
                blank
            )
        )

        self.assertAlmostEqual(
            score,
            0.50,
            places=3,
        )
        self.assertEqual(
            metadata[
                "line_count"
            ],
            0,
        )

    def test_angle_trend_detects_projective_shear(self):
        stable = [
            (20.0, 0.1),
            (60.0, 0.0),
            (100.0, 0.1),
            (140.0, -0.1),
            (180.0, 0.0),
        ]
        sheared = [
            (20.0, -1.0),
            (60.0, -0.5),
            (100.0, 0.0),
            (140.0, 0.6),
            (180.0, 1.2),
        ]

        stable_result = (
            self.engine
            ._angle_trend_score(
                stable,
                axis_span=200.0,
            )
        )
        shear_result = (
            self.engine
            ._angle_trend_score(
                sheared,
                axis_span=200.0,
            )
        )

        self.assertTrue(
            stable_result[
                "available"
            ]
        )
        self.assertTrue(
            shear_result[
                "available"
            ]
        )
        self.assertGreater(
            stable_result[
                "score"
            ],
            shear_result[
                "score"
            ],
        )

    def test_angle_trend_is_neutral_with_too_few_samples(self):
        result = (
            self.engine
            ._angle_trend_score(
                [
                    (20.0, 0.0),
                    (80.0, 0.4),
                ],
                axis_span=200.0,
            )
        )

        self.assertFalse(
            result[
                "available"
            ]
        )
        self.assertAlmostEqual(
            result[
                "score"
            ],
            0.50,
            places=3,
        )

    def test_projective_rectification_can_use_coupled_shear_pattern(self):
        source = np.full(
            (340, 540, 3),
            235,
            dtype=np.uint8,
        )

        for y in range(
            80,
            270,
            28,
        ):
            cv2.line(
                source,
                (40, y),
                (335, y),
                (30, 30, 30),
                3,
            )

        cv2.rectangle(
            source,
            (390, 85),
            (500, 245),
            (40, 40, 40),
            3,
        )

        destination = np.array(
            [
                [45, 25],
                [520, 45],
                [500, 320],
                [25, 300],
            ],
            dtype=np.float32,
        )
        source_points = np.array(
            [
                [0, 0],
                [539, 0],
                [539, 339],
                [0, 339],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            source_points,
            destination,
        )

        canvas = cv2.warpPerspective(
            source,
            matrix,
            (560, 350),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Quad sengaja sedikit salah sehingga menyisakan shear.
        base_quad = destination.copy()
        base_quad[0, 0] += 7.0
        base_quad[1, 0] += 7.0
        base_quad[2, 0] -= 7.0
        base_quad[3, 0] -= 7.0

        before_warp = self.engine.warp(
            canvas,
            base_quad,
        )
        before_score, _before_meta = (
            self.engine
            ._post_warp_geometry_score(
                before_warp
            )
        )

        optimized, metadata = (
            self.engine
            ._projective_rectify_best_quad(
                canvas,
                base_quad,
                base_geometry_score=(
                    before_score
                ),
            )
        )

        self.assertEqual(
            optimized.shape,
            (4, 2),
        )
        self.assertIn(
            "applied",
            metadata,
        )

        if metadata[
            "applied"
        ]:
            after_warp = self.engine.warp(
                canvas,
                optimized,
            )
            after_score, _after_meta = (
                self.engine
                ._post_warp_geometry_score(
                    after_warp
                )
            )

            self.assertGreater(
                after_score,
                before_score,
            )

    def test_projective_rectification_keeps_good_quad_when_no_clear_gain(self):
        image = np.full(
            (400, 640, 3),
            235,
            dtype=np.uint8,
        )

        for y in range(
            100,
            300,
            30,
        ):
            cv2.line(
                image,
                (100, y),
                (430, y),
                (30, 30, 30),
                3,
            )

        quad = np.array(
            [
                [70, 55],
                [570, 55],
                [570, 345],
                [70, 345],
            ],
            dtype=np.float32,
        )

        optimized, metadata = (
            self.engine
            ._projective_rectify_best_quad(
                image,
                quad,
                base_geometry_score=1.0,
            )
        )

        self.assertEqual(
            optimized.shape,
            (4, 2),
        )
        self.assertLessEqual(
            float(
                np.mean(
                    np.linalg.norm(
                        optimized - quad,
                        axis=1,
                    )
                )
            ),
            30.0,
        )
        self.assertIn(
            "applied",
            metadata,
        )

    def test_color_refined_candidate_can_use_snapped_boundary_variant(self):
        image = np.zeros(
            (400, 640, 3),
            dtype=np.uint8,
        )
        raw = np.array(
            [
                [80, 70],
                [560, 70],
                [560, 330],
                [80, 330],
            ],
            dtype=np.float32,
        )
        snapped = raw + np.array(
            [
                [-3, -3],
                [3, -3],
                [3, 3],
                [-3, 3],
            ],
            dtype=np.float32,
        )

        self.engine._resize_for_detection = (
            lambda source: (
                source.copy(),
                1.0,
            )
        )
        self.engine._build_detection_maps = (
            lambda source: (
                [
                    np.zeros(
                        source.shape[:2],
                        dtype=np.uint8,
                    )
                ],
                np.zeros(
                    source.shape[:2],
                    dtype=np.uint8,
                ),
            )
        )
        self.engine._contour_candidates = (
            lambda maps: []
        )
        self.engine._color_candidate = (
            lambda source: [
                PerspectiveCandidate(
                    points=raw.copy(),
                    score=0.0,
                    source="color_refined",
                )
            ]
        )
        self.engine._snap_quad_to_boundary = (
            lambda source, points: (
                snapped.copy()
            )
        )
        self.engine._candidate_score = (
            lambda candidate, source, edges: (
                0.92
                if candidate.source
                == "color_refined_snapped"
                else 0.70
            )
        )
        self.engine._warp_quality = (
            lambda warped: 0.80
        )

        corners, metadata = (
            self.engine.detect(
                image
            )
        )

        self.assertIsNotNone(
            corners
        )
        self.assertTrue(
            np.allclose(
                corners,
                snapped,
            )
        )
        self.assertEqual(
            metadata[
                "selected_source"
            ],
            "color_refined_snapped",
        )


class TestScannerOutput(unittest.TestCase):
    def test_grayscale_mode_keeps_three_channels(self):
        image = np.zeros(
            (20, 30, 3),
            dtype=np.uint8,
        )
        image[:, :, 1] = 150

        result = AutoDocumentScanner.apply_output_mode(
            image,
            "grayscale",
        )

        self.assertEqual(
            result.shape,
            (20, 30, 3),
        )
        self.assertTrue(
            np.array_equal(
                result[:, :, 0],
                result[:, :, 1],
            )
        )


if __name__ == "__main__":
    unittest.main()
