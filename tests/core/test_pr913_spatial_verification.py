"""Synthetic positive document layouts and safety/contract checks."""
import json
import sys
import types
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr913_spatial_verification import verify_spatial
from autodocscanner.tools.pr913_validation import evaluate_photo, metrics


def scene(layout='distributed'):
    image = np.full((600, 900, 3), 40, np.uint8)
    q = np.float32([[140, 120], [740, 120], [740, 480], [140, 480]])
    cv2.fillConvexPoly(image, q.astype(int), (180, 220, 210))
    if layout == 'distributed':
        # Separate areas, no one fixed position or portrait template.
        boxes = [(180, 160, 480, 255), (220, 285, 670, 420)]
    elif layout == 'asymmetric':
        boxes = [(180, 165, 685, 235), (180, 270, 430, 430)]
    elif layout == 'localized':
        boxes = [(260, 230, 410, 270)]
    elif layout == 'line':
        boxes = [(175, 285, 700, 295)]
    else:
        boxes = []
    seeds = []
    for j, (x0, y0, x1, y1) in enumerate(boxes):
        points = np.float32([[x, y] for y in range(y0, y1, 15) for x in range(x0, x1, 13)])
        for x, y in points.astype(int):
            cv2.rectangle(image, (x-2, y-3), (x+2, y+3), (25, 25, 25), -1)
        seeds.append(dict(seed_id=j+1, points=points.tolist()))
    return image, q, seeds


def candidate(q, family='synthetic', pattern=True):
    return dict(corners=q.tolist(), family=family, structure={'distributed_stroke_pattern': pattern})


def prior(cs, *, boundary=None, group_ids=None, partial=None):
    n = len(cs)
    boundary = [True]*n if boundary is None else boundary
    partial = boundary if partial is None else partial
    group_ids = list(range(1, n+1)) if group_ids is None else group_ids
    rows = [dict(candidate_id=i+1, family=c['family'], group_id=g, surface_boundary_supported=b,
                 joint_supported=b and c['structure']['distributed_stroke_pattern'],
                 partial_joint_supported=p and c['structure']['distributed_stroke_pattern'])
            for i, (c, g, b, p) in enumerate(zip(cs, group_ids, boundary, partial))]
    groups = [dict(group_id=g, members=[i for i, k in enumerate(group_ids) if g == k]) for g in dict.fromkeys(group_ids)]
    return dict(candidates=rows, groups=groups, auto_select=None, auto_crop_allowed=False, requires_manual_confirmation=True)


def measure(image, q, seeds, **kwargs):
    cs = [candidate(q)]
    return verify_spatial(image, cs, seeds, prior(cs, **kwargs))['candidates'][0]


@pytest.mark.parametrize('layout', ['distributed', 'asymmetric'])
@pytest.mark.parametrize('angle', [0, 37, 90, 163])
def test_spread_rotation(layout, angle):
    im, q, seeds = scene(layout)
    m = cv2.getRotationMatrix2D((450, 300), angle, .8)
    m[:, 2] += [100, 250]
    im = cv2.warpAffine(im, m, (1100, 1100), borderValue=(40, 40, 40))
    q = cv2.transform(q[None], m)[0]
    for s in seeds:
        s['points'] = cv2.transform(np.float32(s['points'])[None], m)[0].tolist()
    c = measure(im, q, seeds)
    assert c['spatial_structure_supported'] and c['spatial_joint_supported']
    assert c['occupied_quadrants'] >= 3


@pytest.mark.parametrize('layout', ['localized', 'line', 'empty'])
def test_concentrated_or_empty_structure(layout):
    im, q, seeds = scene(layout)
    c = measure(im, q, seeds)
    assert not c['spatial_structure_supported'] and c['boundary_without_spatial']


def test_perspective_and_resolution():
    im, q, seeds = scene()
    dest = np.float32([[100, 90], [810, 180], [710, 530], [210, 450]])
    m = cv2.getPerspectiveTransform(q, dest)
    im = cv2.warpPerspective(im, m, (900, 600))
    for s in seeds:
        s['points'] = cv2.perspectiveTransform(np.float32(s['points'])[None], m)[0].tolist()
    for scale in (1., .5, .25):
        small = cv2.resize(im, None, fx=scale, fy=scale)
        ss = [{**s, 'points': (np.asarray(s['points'])*scale).tolist()} for s in seeds]
        c = measure(small, dest[[2, 0, 3, 1]]*scale, ss)
        assert c['spatial_structure_supported']


def test_duplicate_seeds_do_not_inflate_spatial_metrics():
    im, q, seeds = scene()
    a = measure(im, q, seeds)
    b = measure(im, q, seeds + [{**s, 'seed_id': s['seed_id']+20} for s in seeds])
    for key in ('unique_quantized_sites', 'occupied_tiles', 'hull_union_fraction', 'spatial_spread_supported'):
        assert a[key] == b[key]
    assert b['enclosed_groups'] == 2*a['enclosed_groups']


def test_no_boundary_does_not_reject_spatial():
    im, q, seeds = scene()
    c = measure(im, q, seeds, boundary=[False], partial=[True])
    assert c['spatial_without_full_boundary'] and c['spatial_partial_supported'] and not c['spatial_joint_supported']


def test_group_disagreement_and_preservation():
    im, q, seeds = scene()
    other = np.float32([[20, 20], [120, 20], [120, 90], [20, 90]])
    cs = [candidate(q), candidate(q+1, 'other_family'), candidate(other)]
    old = prior(cs, group_ids=[1, 1, 2], boundary=[False, False, True])
    copy = im.copy()
    r = verify_spatial(im, cs, seeds, old)
    assert len(r['candidates']) == 3 and len(r['groups']) == 2
    assert r['conflicts'] == [dict(a=1, b=2, kind='spatial_boundary_disagreement')]
    assert r['auto_select'] is None and not r['auto_crop_allowed'] and r['requires_manual_confirmation']
    assert np.array_equal(im, copy)
    json.dumps(r, allow_nan=False)


def test_two_documents_preserved():
    im, q, seeds = scene()
    im = np.concatenate([im, im], axis=1)
    other = q+[900, 0]
    seeds += [dict(seed_id=s['seed_id']+10, points=(np.asarray(s['points'])+[900, 0]).tolist()) for s in seeds]
    cs = [candidate(q), candidate(other)]
    r = verify_spatial(im, cs, seeds, prior(cs))
    assert all(c['spatial_joint_supported'] for c in r['candidates'])
    assert r['conflicts'][0]['kind'] == 'competing_spatial_groups'


def test_surface_continuity_advisory_and_frame():
    im, q, seeds = scene()
    a = measure(im, q, seeds)
    im[120:480, 440:740] = (20, 20, 220)
    b = measure(im, q, seeds)
    assert a['surface_neighbor_agreement'] > b['surface_neighbor_agreement']
    assert b['spatial_structure_supported']  # advisory, not rejection
    # No fabricated observation for the off-image half.
    c = measure(im, q-[440, 0], seeds)
    assert c['observed_tiles'] < 16


def test_bad_contracts_and_empty():
    im, q, seeds = scene()
    cs = [candidate(q)]
    with pytest.raises(ValueError): verify_spatial(im, cs, seeds, {**prior(cs), 'auto_crop_allowed': True})
    with pytest.raises(ValueError): verify_spatial(im, cs, seeds, prior([]))
    with pytest.raises(ValueError): verify_spatial(im, cs, seeds, {**prior(cs), 'groups': []})
    with pytest.raises(ValueError): verify_spatial(im, [candidate(q*0)], seeds, prior(cs))
    with pytest.raises(ValueError): verify_spatial(im, cs, [{'points': [[float('nan'), 1]]}], prior(cs))
    r = verify_spatial(im, [], seeds, prior([]))
    assert not r['candidates'] and not r['conflicts']
    assert metrics([])['oracle_error']['mean'] is None


def test_runner_annotation_independence_and_privacy(monkeypatch):
    im, q, seeds = scene()
    cs = [candidate(q)]
    calls = []
    mod = types.ModuleType('autodocscanner.tools.pr9_holdout_validation')
    mod._quad = lambda a, shape: q if a < .1 else q + 150
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    def parent(image, annotation, detectors):
        old = prior(cs)
        old['candidates'][0]['error'] = {'mean': annotation, 'max': annotation}
        old['reference_control'] = {'private_test': annotation, 'structure_supported': True}
        return {'status': 'evaluated', 'reconciliation': old}, cs, seeds
    def audit(image, raw, ss, old):
        assert 'reference_control' not in old and 'error' not in old['candidates'][0]
        r = verify_spatial(image, raw, ss, old)
        calls.append(json.dumps(r))
        return r
    a = evaluate_photo(im, .01, parent=parent, audit=audit)
    b = evaluate_photo(im, .15, parent=parent, audit=audit)
    assert calls[0] == calls[1]
    assert a['spatial_verification']['reference_control']['spatial_structure_supported']
    assert not b['spatial_verification']['reference_control']['spatial_structure_supported']
    report = json.dumps(a['spatial_verification'], allow_nan=False)
    assert 'corners' not in report and '"points"' not in report
    m = metrics([a, b]); assert m['candidate_count'] == 2 and m['accurate_candidates'] == 1
    assert m['evidence']['spatial_joint_supported']['severe_supported'] == 1
    with pytest.raises(ValueError): evaluate_photo(im, .01, parent=parent, audit=lambda *args: {**audit(*args), 'auto_crop_allowed': True})


def test_parent_capture_and_folder_contract(monkeypatch):
    from autodocscanner.tools.pr913_validation import evaluate_folder
    im, q, seeds = scene(); cs = [candidate(q)]
    old = prior(cs); old['candidates'][0]['error'] = dict(mean=.01, max=.015)
    old['reference_control'] = {'structure_supported': True}
    gm = types.ModuleType('autodocscanner.tools.pr9_holdout_validation')
    gm._quad = lambda a, shape: q
    monkeypatch.setitem(sys.modules, gm.__name__, gm)
    def capture(i, a, d):
        assert np.array_equal(i, im)
        return {'status': 'evaluated'}, cs, seeds
    def parent(i, a, *, detectors, parent):
        row, _, _ = parent(i, a, detectors)
        row['reconciliation'] = old
        return row
    def folder(f, a, role, *, evaluator):
        return {'photos': [evaluator(im, .01)], 'evaluated_photo_count': 1, 'reconciliation_metrics': {'candidate_count': 1}}
    mod = types.ModuleType('autodocscanner.tools.pr912_validation')
    mod.evaluate_photo = parent; mod._parent_capture = capture; mod.evaluate_folder = folder
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    r = evaluate_folder('unused', {}, 'development')
    assert r['reconciliation_metrics']['candidate_count'] == r['spatial_verification_metrics']['candidate_count'] == 1
