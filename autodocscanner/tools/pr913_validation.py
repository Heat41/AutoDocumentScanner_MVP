"""PR9.13 positive-only spatial evidence audit. No coordinates or identities."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from autodocscanner.core.pr913_spatial_verification import verify_spatial, inspect_spatial


def evaluate_photo(image, annotation, *, detectors=None, parent=None, audit=verify_spatial):
    if parent is None:
        from autodocscanner.tools.pr912_validation import evaluate_photo as prior, _parent_capture
        def parent(im, ref, detectors):
            captured = []
            def capture(i, a, d):
                row, raw, seeds = _parent_capture(i, a, d)
                captured.append((raw, seeds))
                return row, raw, seeds
            row = prior(im, ref, detectors=detectors, parent=capture)
            return row, *(captured[0] if captured else ([], []))
    row, raw, seeds = parent(image, annotation, detectors)
    if row['status'] != 'evaluated':
        return row
    prior = row['reconciliation']
    # Strip retrospective errors and reference controls before the runtime audit.
    fields = ('candidate_id', 'family', 'group_id', 'surface_boundary_supported', 'joint_supported', 'partial_joint_supported')
    runtime = dict(candidates=[{k: c[k] for k in fields} for c in prior['candidates']],
                   groups=[{'group_id': g['group_id'], 'members': list(g['members'])} for g in prior['groups']],
                   **{k: prior[k] for k in ('auto_select', 'auto_crop_allowed', 'requires_manual_confirmation')})
    result = audit(image.copy(), raw, seeds, runtime)
    if result.get('auto_select', 'missing') is not None or result.get('auto_crop_allowed') is not False or result.get('requires_manual_confirmation') is not True:
        raise ValueError('unsafe spatial evidence contract')
    if len(result['candidates']) != len(prior['candidates']) or len(raw) != len(prior['candidates']):
        raise ValueError('candidate count changed')
    for c, old in zip(result['candidates'], prior['candidates'], strict=True):
        if any(c[k] != old[k] for k in ('candidate_id', 'family', 'group_id')):
            raise ValueError('candidate alignment changed')
        c['error'] = dict(old['error'])
    # Retrospective reference control, only after all runtime evidence is fixed.
    from autodocscanner.tools.pr9_holdout_validation import _quad
    ref = _quad(annotation, image.shape)
    if ref is None:
        raise ValueError('invalid reference control')
    result['reference_control'] = inspect_spatial(image, ref, seeds, prior['reference_control']['structure_supported'])
    row['spatial_verification'] = result
    return row


def metrics(rows):
    pos = [r['spatial_verification'] for r in rows if r['status'] == 'evaluated']
    flat = [c for r in pos for c in r['candidates']]
    def good(c): return c['error']['mean'] <= .02
    def stats(v):
        return dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(max(v))) if v else dict.fromkeys(('mean', 'median', 'p90', 'max'))
    measures = {}
    for key in ('spatial_spread_supported', 'spatial_structure_supported', 'spatial_joint_supported', 'spatial_partial_supported', 'spatial_without_full_boundary', 'boundary_without_spatial'):
        supported = [c for c in flat if c[key]]
        groups = [[r['candidates'][i] for i in g['members'] if r['candidates'][i][key]] for r in pos for g in r['groups']]
        groups = [g for g in groups if g]
        measures[key] = dict(supported_candidates=len(supported), accurate_supported=sum(map(good, supported)),
            wrong_supported=sum(not good(c) for c in supported), severe_supported=sum(c['error']['mean'] > .1 for c in supported),
            accurate_unsupported=sum(good(c) and not c[key] for c in flat),
            photos_with_accurate_supported=sum(any(c[key] and good(c) for c in r['candidates']) for r in pos),
            supported_groups=len(groups), groups_with_accurate_member=sum(any(map(good, g)) for g in groups),
            groups_with_wrong_member=sum(any(not good(c) for c in g) for g in groups),
            supported_error=stats([c['error']['mean'] for c in supported]),
            supported_worst_corner_error=stats([c['error']['max'] for c in supported]))
    fields = ('occupied_tiles', 'occupied_quadrants', 'hull_union_fraction', 'unique_quantized_sites', 'surface_neighbor_agreement')
    best = [min(c['error']['mean'] for c in r['candidates']) for r in pos if r['candidates']]
    return dict(candidate_count=len(flat), accurate_candidates=sum(map(good, flat)),
        geometry_group_count=sum(len(r['groups']) for r in pos), oracle_photos_within_2pct=sum(e <= .02 for e in best), oracle_error=stats(best),
        evidence=measures, feature_distributions={name: {k: stats([c[k] for c in flat if pred(c) and c[k] is not None]) for k in fields} for name, pred in [('accurate', good), ('wrong', lambda c: not good(c)), ('severe', lambda c: c['error']['mean'] > .1)]},
        reference_controls=len(pos), reference_spatial_supported=sum(r['reference_control']['spatial_structure_supported'] for r in pos),
        reference_features={k: stats([r['reference_control'][k] for r in pos if r['reference_control'][k] is not None]) for k in fields},
        photos_with_conflicts=sum(bool(r['conflicts']) for r in pos), conflict_types=dict(Counter(c['kind'] for r in pos for c in r['conflicts'])))


def evaluate_folder(folder, annotations, role, *, evaluator=evaluate_photo):
    from autodocscanner.tools.pr912_validation import evaluate_folder as prior
    out = prior(folder, annotations, role, evaluator=evaluator)
    out['spatial_verification_metrics'] = metrics(out['photos'])
    out['mode'] = 'positive_only_spatial_verification_audit'
    out['note'] = 'All prior hypotheses retained. Spatial spread and surface adjacency are advisory, not KTP identity or confidence. Oracle metrics are retrospective. No automatic selection.'
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description='PR9.13 local positive-only spatial evidence audit')
    p.add_argument('folder', type=Path)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--dataset-role', choices=('development', 'holdout'), required=True)
    a = p.parse_args(argv)
    try:
        r = evaluate_folder(a.folder, json.loads(a.annotations.read_text(encoding='utf-8-sig')), a.dataset_role)
    except (OSError, ValueError, TypeError, cv2.error):
        p.error('Dataset atau kontrak evaluasi tidak valid')
    print(json.dumps(r, indent=2, allow_nan=False))
    return 0 if r['evaluated_photo_count'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
