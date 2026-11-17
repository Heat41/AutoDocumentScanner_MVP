"""Synthetic PR6 visual snapshot tests; no private document images or names."""
from copy import deepcopy

import numpy as np

from autodocscanner.core.pr6_visual_diagnostics import (
    build_pr6_visual_diagnostics,
)


CORNERS = [[22, 27], [310, 22], [315, 191], [20, 188]]
OTHERS = [[32, 32], [305, 30], [312, 183], [30, 185]]


def _report():
    return {
        "stable": {
            "corners": deepcopy(CORNERS), "geometry_valid": True,
            "area_ratio": .35,
            "boundary_evidence": {"side_support": [.88, .62, .31, .25]},
        },
        "adaptive": {"found": False},
        "v2": [
            {
                "corners": deepcopy(OTHERS), "geometry_valid": True,
                "area_ratio": .31,
                "boundary_evidence": {"side_support": [.18, .9, .6, .7]},
            },
            {
                "corners": deepcopy(CORNERS), "geometry_valid": True,
                "area_ratio": .35,
                "boundary_evidence": {"side_support": [.8, .2, .5, .1]},
            },
        ],
        "pr6_v2_proposal_origins": ["contour", "four_line"],
        "pr6_selection_shadow": {"groups": [
            {"representative": "stable", "sources": ["stable", "v2_1"]},
            {"representative": "v2_0", "sources": ["v2_0"]},
        ], "warnings": ["competing_geometries"]},
        "pr6_combined_candidate_shadow": {"candidates": [
            {
                "source": "stable", "warnings": ["weak_inner_outer_separation"],
                "side_checks": {
                    "supported_external": 4, "localized_aligned": 3,
                    "measurable_contrast": 2,
                },
            },
            {"source": "v2_0", "warnings": ["poor_edge_alignment"]},
        ]},
        "pr6_boundary_localization_shadow": {
            "stable": {"side_localization": [
                {"coverage": .9, "median_offset_fraction": .1, "offset_spread": .1}
                for _ in range(4)
            ]}
        },
        "pr6_physical_boundary_shadow": {
            "stable": {"side_measurements": [
                {"contrast": 9, "coverage": .9} for _ in range(4)
            ]}
        },
        "pr6_candidate_decision_shadow": {"candidate_assessments": [
            {"source": "stable", "evidence_state": "requires_manual_review",
             "reasons": ["alignment_missing"]},
        ]},
        "v3_ranking": {"ranking": [
            {"source": "stable", "score": .79, "review_reasons": []},
            {"source": "v2_0", "score": .62, "review_reasons": ["weak_side"]},
        ]},
        "pr6_review_consolidation_shadow": {
            "status": "manual_review",
            "action": "manual_corners",
            "mode": "offline_shadow_review_only",
            "auto_select": None, "auto_crop_allowed": False,
            "requires_explicit_confirmation": True,
            "physical_boundary_verified": False,
            "reason_codes": ["pr6_candidate_decision_shadow:requires_manual_review"],
        },
        "pr6_manual_fallback_shadow": {
            "status": "manual_review",
            "action": "manual_corners",
            "status_scope": "offline_diagnostic_only",
            "ui_integration_status": "not_connected_shadow_only",
            "auto_select": None, "auto_crop_allowed": False,
            "requires_explicit_confirmation": True,
            "file_write_performed": False,
            "source_image_policy": "unmodified_original_image_required",
            "source_image_shape": [240, 360],
        },
    }


def test_all_source_families_and_proposal_origins_visible():
    input_report = _report()
    before = deepcopy(input_report)
    diagnostics = build_pr6_visual_diagnostics(input_report, (240, 360, 3))
    assert diagnostics["status"] == "offline_visual_diagnostic_only"
    assert diagnostics["candidate_count"] == 3
    assert [(c["source"], c["family"]) for c in diagnostics["candidates"]] == [
        ("stable", "Stable"), ("v2_0", "Kontur V2"),
        ("v2_1", "Four-Line"),
    ]
    assert diagnostics["auto_select"] is None
    assert diagnostics["auto_crop_allowed"] is False
    assert diagnostics["physical_boundary_verified"] is False
    assert input_report == before


def test_measurements_represent_the_exact_selected_quad_only():
    diagnostics = build_pr6_visual_diagnostics(_report(), (240, 360))
    stable, contour, equivalent = diagnostics["candidates"]
    assert stable["side_evidence"][0]["local_coverage"] == .9
    assert stable["side_evidence"][0]["inner_outer_contrast"] == 9
    assert stable["physical_measurements_for_this_quad"] is True
    assert equivalent["selection_representative"] == "stable"
    assert equivalent["physical_measurements_for_this_quad"] is False
    assert equivalent["side_evidence"][0]["inner_outer_contrast"] is None
    assert contour["candidate_warnings"] == ["poor_edge_alignment"]


def test_diagnostic_code_and_manual_decision_visible_not_upgraded_to_preview():
    result = build_pr6_visual_diagnostics(_report(), (240, 360))
    assert result["review_decision"] == "manual_review"
    assert "pr6_candidate_decision_shadow:requires_manual_review" in result[
        "decision_reason_codes"
    ]
    assert result["candidates"][0]["assessment_reasons"] == [
        "alignment_missing"
    ]


def test_ground_truth_and_private_metadata_never_exported_to_debug_snapshot():
    report = _report()
    first = build_pr6_visual_diagnostics(report, (240, 360))
    report["expected"] = [[1, 2]] * 4
    report["normalized_corner_error"] = 0.
    report["subject_name"] = "REDACTED"
    report["private_file_path"] = "REDACTED"
    report["pr6_multi_refinement_ground_truth_evaluation"] = {"error": 0}
    assert build_pr6_visual_diagnostics(report, (240, 360)) == first
    assert "REDACTED" not in str(first)


def test_missing_broken_candidates_fail_closed():
    report = _report()
    report["stable"]["corners"] = [[1, 1]] * 3
    report["v2"][0]["corners"] = [
        [float("nan"), 3], [12, 4], [13, 8], [2, 8],
    ]
    report["v2"][1]["corners"] = [
        [-9000, 0], [100, 0], [100, 100], [0, 100],
    ]
    out = build_pr6_visual_diagnostics(report, (240, 360))
    assert out["candidate_count"] == 0
    assert out["review_decision"] == "manual_review"
    assert out["auto_crop_allowed"] is False


def test_no_candidates_and_bad_report():
    for report in ({}, [], None, {"v2": "wrong"}):
        result = build_pr6_visual_diagnostics(report, (200, 310))
        assert result["candidate_count"] == 0
        assert result["auto_select"] is None


def test_wrong_shape_prevents_trusting_overlay_coordinates():
    for shape in ((0, 10), (10,), None, (-100, 200)):
        result = build_pr6_visual_diagnostics(_report(), shape)
        assert result["status"] == "invalid_image"
        assert result["image_shape"] is None


def test_runtime_review_populates_visual_diagnostics_but_never_crop():
    from autodocscanner.core.pr6_runtime_review import evaluate_pr6_runtime_image

    image = np.full((240, 360, 3), 110, np.uint8)
    report = _report()
    received = []
    def analyzer(image, *, expected=None):
        received.append(expected)
        return report
    result = evaluate_pr6_runtime_image(image, analyzer=analyzer)
    assert received == [None]
    assert result["status"] == "manual_review"
    assert result["file_write_performed"] is False
    assert result["visual_debug"]["candidate_count"] == 3
    assert not result["visual_debug"]["auto_crop_allowed"]


def test_candidate_snapshot_is_limited_even_with_many_v2_candidates():
    report = _report()
    report["v2"] = [deepcopy(report["v2"][0]) for _ in range(100)]
    snapshot = build_pr6_visual_diagnostics(report, (240, 360))
    assert snapshot["candidate_count"] <= 40



def test_chroma_family_and_source_counts_visible_without_auto_crop():
    report = _report()
    report["pr6_v2_proposal_origins"] = ["chroma", "four_line"]
    report["pr6_chroma_proposal_count"] = 1
    report["pr6_line_proposal_count"] = 1
    report["v2_rejected_post_order"] = 3
    result = build_pr6_visual_diagnostics(report, (240, 360, 3))
    assert result["family_counts"] == {
        "Stable": 1, "Adaptive": 0, "Kontur V2": 0,
        "Four-Line": 1, "Chroma": 1, "Outer Region": 0,
        "Side Recovery": 0, "Paired Alignment": 0,
    }
    assert result["generated_chroma_count"] == 1
    assert result["generated_four_line_count"] == 1
    assert result["rejected_after_geometry"] == 3
    assert result["auto_crop_allowed"] is False


def test_per_candidate_visual_measurements_do_not_change_selection_gate():
    image = np.full((240, 360, 3), 130, dtype=np.uint8)
    report = _report()
    before = deepcopy(report)
    visual = build_pr6_visual_diagnostics(
        report, image.shape, image=image,
    )
    stable, contour, equivalent = visual["candidates"]
    assert all(c["measurements_visual_only"] for c in visual["candidates"])
    assert stable["physical_measurements_for_this_quad"] is True
    assert equivalent["physical_measurements_for_this_quad"] is True
    assert contour["physical_measurements_for_this_quad"] is True
    assert equivalent["side_evidence"][0]["local_coverage"] is not None
    assert contour["side_evidence"][0]["local_coverage"] is not None
    assert visual["review_decision"] == "manual_review"
    assert visual["auto_crop_allowed"] is False
    assert report == before



def test_outer_region_family_and_perspective_risk_visible_without_auto_crop():
    report = _report()
    report["pr6_v2_proposal_origins"] = ["outer_region", "four_line"]
    report["pr6_region_proposal_count"] = 1
    report["pr6_outer_region_recovery_shadow"] = {
        "status": "offline_diagnostic_only", "auto_select": None,
        "auto_crop_allowed": False,
        "candidate_checks": [
            {
                "source": "v2_0", "status": "manual_review_only",
                "perspective_feasible": True,
                "all_sides_measurable": True,
                "nested_in_larger_quad": True,
                "chromatic_step_by_side": [
                    {"coverage": 1.0, "median_lab_step": 10} for _ in range(4)
                ],
                "reason_codes": ["possible_internal_rectangle"],
            },
        ],
    }
    visual = build_pr6_visual_diagnostics(report, (240, 360))
    assert visual["family_counts"]["Outer Region"] == 1
    assert visual["generated_outer_region_count"] == 1
    row = next(r for r in visual["candidates"] if r["family"] == "Outer Region")
    assert row["outer_recovery_check"]["nested_in_larger_quad"] is True
    assert "possible_internal_rectangle" in row["outer_recovery_check"]["reason_codes"]
    assert visual["auto_select"] is None
    assert visual["auto_crop_allowed"] is False



def test_visual_display_balances_outer_region_even_if_contour_list_is_long():
    report = _report()
    report["v2"] = [deepcopy(report["v2"][0]) for _ in range(60)]
    report["v2"][-1] = deepcopy(report["v2"][1])
    report["pr6_v2_proposal_origins"] = ["contour"] * 59 + ["outer_region"]
    display = build_pr6_visual_diagnostics(report, (240, 360))
    assert display["candidate_count"] <= 40
    assert display["source_hypotheses_hidden"] > 0
    assert any(row["family"] == "Outer Region" and row["source"] == "v2_59"
               for row in display["candidates"])
    assert display["auto_select"] is None
    assert display["auto_crop_allowed"] is False


def test_region_context_warnings_survive_read_only_visual_projection():
    report = _report()
    report["pr6_v2_proposal_origins"] = ["outer_region", "contour"]
    report["pr6_outer_region_recovery_shadow"] = {
        "status": "offline_diagnostic_only",
        "auto_select": None, "auto_crop_allowed": False,
        "candidate_checks": [{
            "source": "v2_0",
            "status": "manual_review_only",
            "perspective_feasible": True,
            "reason_codes": [],
            "region_context": {
                "interior_edge_density": 0.0,
                "interior_spatial_coverage": 0.0,
                "interior_axis_line_bands": 0,
                "corroborating_families": [],
                "nested_inner_candidates": 0,
                "reason_codes": ["sparse_internal_structure",
                                 "no_independent_geometric_corroboration"],
                "auto_select": None, "auto_crop_allowed": False,
            }
        }],
    }
    snapshot = build_pr6_visual_diagnostics(report, (240, 360))
    row = next(row for row in snapshot["candidates"]
               if row["family"] == "Outer Region")
    reasons = row["outer_recovery_check"]["region_context"]["reason_codes"]
    assert "sparse_internal_structure" in reasons
    assert "no_independent_geometric_corroboration" in reasons
    assert snapshot["auto_crop_allowed"] is False
