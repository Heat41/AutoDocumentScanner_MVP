"""PR9.13: spatial document evidence, not semantic identity or selection.

Use PR9.11 seed points and PR9.12 evidence on unchanged hypotheses. All
coordinates remain in memory. Thresholds are provisional geometric rules.
"""
from __future__ import annotations
from itertools import combinations
import cv2
import numpy as np

SIZE = 256
GRID = 4


def _normalize(image, corners):
    q = np.asarray(corners, np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError('invalid quadrilateral')
    d = q-q.mean(axis=0)
    q = q[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]
    if not cv2.isContourConvex(q) or cv2.contourArea(q) <= 0:
        raise ValueError('invalid quadrilateral')
    dst = np.float32([[0, 0], [SIZE-1, 0], [SIZE-1, SIZE-1], [0, SIZE-1]])
    t = cv2.getPerspectiveTransform(q, dst)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    patch = cv2.warpPerspective(lab, t, (SIZE, SIZE)).astype(np.float32)
    valid = cv2.warpPerspective(np.ones(image.shape[:2], np.float32), t, (SIZE, SIZE)) >= .999
    return q, t, patch, valid


def _footprint(q, transform, seeds):
    # Each site is counted once even when threshold/polarity views overlap.
    # Use only enclosed groups, and only observed points inside the polygon.
    sites = np.zeros((SIZE//4, SIZE//4), bool)
    union = np.zeros((SIZE, SIZE), np.uint8)
    enclosed = 0
    for seed in seeds:
        p = np.asarray(seed['points'], np.float32)
        if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
            raise ValueError('invalid seed points')
        inside = np.array([cv2.pointPolygonTest(q, tuple(pt), False) >= 0 for pt in p])
        if inside.mean() < .90:
            continue
        mapped = cv2.perspectiveTransform(p[inside][None], transform)[0]
        if not np.isfinite(mapped).all():
            raise ValueError('invalid projected seed')
        mapped = np.clip(mapped, 0, SIZE-1)
        x, y = (mapped//4).astype(int).T
        sites[y, x] = True
        enclosed += 1
        if len(mapped) >= 3:
            hull = cv2.convexHull(np.rint(mapped).astype(np.int32))
            cv2.fillConvexPoly(union, hull, 1)
    # A tile with isolated points alone does not establish spatial support.
    block = SIZE//4//GRID
    density = sites.reshape(GRID, block, GRID, block).sum(axis=(1, 3))
    occupied = density >= 3
    rows = int(occupied.any(axis=1).sum())
    cols = int(occupied.any(axis=0).sum())
    quadrants = int(occupied.reshape(2, 2, 2, 2).any(axis=(1, 3)).sum())
    y, x = np.where(sites)
    spans = sorted([float(np.ptp(x)/(SIZE//4-1)), float(np.ptp(y)/(SIZE//4-1))]) if len(x) else [0., 0.]
    coverage = float(union.mean())
    # Generic 2D spread requirement: at least 3 bands along both axes,
    # three quadrants and >1/8 of surface covered by local group footprints.
    # Not a KTP template; a sparse/occluded document may be unresolved.
    spread = bool(occupied.sum() >= 6 and rows >= 3 and cols >= 3 and quadrants >= 3
                  and spans[0] >= .4 and coverage >= .125)
    return dict(enclosed_groups=enclosed, unique_quantized_sites=int(sites.sum()),
                occupied_tiles=int(occupied.sum()), occupied_rows=rows, occupied_columns=cols,
                occupied_quadrants=quadrants, axis_spans=spans, hull_union_fraction=coverage,
                spatial_spread_supported=spread)


def _surface(patch, valid):
    """Neighboring robust Lab summaries, advisory only, not segmentation.

    No inference of surface beyond the photo. Variable ink/portrait/shadow
    can break agreement, so do not use this cue as a hard rejection.
    """
    n = SIZE//GRID
    tiles = {}
    for y in range(GRID):
        for x in range(GRID):
            s = np.s_[y*n:(y+1)*n, x*n:(x+1)*n]
            v = valid[s]
            if v.mean() < .8:
                continue
            p = patch[s][v]
            med = np.median(p, axis=0)
            noise = float(np.median(np.linalg.norm(p-med, axis=1)))
            tiles[y, x] = (med, noise)
    measured = agreeing = 0
    for (y, x), (a, na) in tiles.items():
        for key in ((y+1, x), (y, x+1)):
            if key not in tiles:
                continue
            b, nb = tiles[key]
            measured += 1
            agreeing += int(np.linalg.norm(a-b) <= 12+na+nb)
    return dict(observed_tiles=len(tiles), measured_neighbor_pairs=measured,
                agreeing_neighbor_pairs=agreeing,
                surface_neighbor_agreement=agreeing/measured if measured else None)


def inspect_spatial(image, corners, seeds, pattern):
    q, t, patch, valid = _normalize(image, corners)
    footprint = _footprint(q, t, seeds)
    return dict(**footprint, **_surface(patch, valid),
                spatial_structure_supported=bool(pattern and footprint['spatial_spread_supported']))


def verify_spatial(image, candidates, seeds, prior):
    """Measure spatial evidence and group-level disagreement; never select."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 32:
        raise ValueError('invalid image')
    if prior.get('auto_select', 'missing') is not None or prior.get('auto_crop_allowed') is not False or prior.get('requires_manual_confirmation') is not True:
        raise ValueError('unsafe parent contract')
    if len(candidates) != len(prior['candidates']):
        raise ValueError('candidate count changed')
    rows = []
    for i, (c, old) in enumerate(zip(candidates, prior['candidates'], strict=True)):
        if old['candidate_id'] != i+1 or old['family'] != c['family']:
            raise ValueError('candidate alignment changed')
        footprint = inspect_spatial(image, c['corners'], seeds, c['structure']['distributed_stroke_pattern'])
        spatial = footprint['spatial_structure_supported']
        boundary = bool(old['surface_boundary_supported'])
        rows.append(dict(candidate_id=i+1, family=c['family'], group_id=old['group_id'],
                         **footprint,
                         full_boundary_supported=boundary,
                         spatial_joint_supported=spatial and bool(old['joint_supported']),
                         spatial_partial_supported=spatial and bool(old['partial_joint_supported']),
                         spatial_without_full_boundary=spatial and not boundary,
                         boundary_without_spatial=boundary and not spatial))
    groups = []
    seen = set()
    for g in prior['groups']:
        ids = g['members']
        if not ids or any(not isinstance(i, int) or i < 0 or i >= len(rows) or i in seen or rows[i]['group_id'] != g['group_id'] for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('invalid group membership')
        seen.update(ids)
        members = [rows[i] for i in ids]
        groups.append(dict(group_id=g['group_id'], members=list(ids),
                           spatial_members=sum(c['spatial_structure_supported'] for c in members),
                           boundary_members=sum(c['full_boundary_supported'] for c in members),
                           joint_members=sum(c['spatial_joint_supported'] for c in members)))
    if len(seen) != len(rows) or len({g['group_id'] for g in groups}) != len(groups):
        raise ValueError('incomplete groups')
    conflicts = []
    for a, b in combinations(groups, 2):
        if a['spatial_members'] and b['spatial_members']:
            kind = 'competing_spatial_groups'
        elif ((a['spatial_members'] and b['boundary_members'] and not b['spatial_members']) or
              (b['spatial_members'] and a['boundary_members'] and not a['spatial_members'])):
            kind = 'spatial_boundary_disagreement'
        else:
            continue
        conflicts.append(dict(a=a['group_id'], b=b['group_id'], kind=kind))
    return dict(candidates=rows, groups=groups, conflicts=conflicts, auto_select=None,
                auto_crop_allowed=False, requires_manual_confirmation=True)
