"""Positive-only reconciliation audit, local coordinates never exported."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from autodocscanner.core.pr912_reconciliation import reconcile


def _collect(original, independent, competing):
    """Exact order: original, recovery, region variants, PR9.10, PR9.11.

    All original hypotheses survive; use provenance, not annotation, to link.
    """
    rows=[]
    def add(corners,family,structure):
        rows.append(dict(corners=corners,family=family,structure=structure))
    for c in original['candidates']:add(c['corners'],c['family'],c['structure'])
    for c in original['candidates']:
        a=c['adaptive_recovery']
        if a['hypothesis'] is not None:add(a['hypothesis'],'adaptive_recovery',a['recovered_structure'])
    for c in original['candidates']:
        for h in c['region_guided']['hypotheses']:add(h['corners'],'region_guided',h['structure'])
    for result in (independent,competing):
        for c in result['candidates']:add(c['corners'],c['family'],c['structure'])
    return rows


def _parent_capture(image,annotation,detectors):
    from autodocscanner.core.pr99_region_guided import audit_regions
    from autodocscanner.core.pr910_structure_discovery import discover_structure_regions
    from autodocscanner.core.pr911_competing_structure import discover_competing_regions
    from autodocscanner.tools.pr99_validation import evaluate_photo as p99
    from autodocscanner.tools.pr910_validation import evaluate_photo as p910
    from autodocscanner.tools.pr911_validation import evaluate_photo as p911
    captured={}
    def capture(key,fn):
        def call(im):
            raw=fn(im);captured[key]=raw;return raw
        return call
    def prior99(im,ref,*,detectors):
        return p99(im,ref,detectors=detectors,audit=capture('original',audit_regions))
    def prior910(im,ref,*,detectors):
        return p910(im,ref,detectors=detectors,discover=capture('independent',discover_structure_regions),parent=prior99)
    row=p911(image,annotation,detectors=detectors,discover=capture('competing',discover_competing_regions),parent=prior910)
    if row['status']!='evaluated':return row,[],[]
    return row,_collect(captured['original'],captured['independent'],captured['competing']),captured['competing']['seeds']


def _earlier_errors(row):
    return ([c['error'] for c in row['candidates']]+
            [c['adaptive_recovery']['error'] for c in row['candidates'] if c['adaptive_recovery']['error'] is not None]+
            [h['error'] for c in row['candidates'] for h in c['region_guided']['hypotheses']]+
            [c['error'] for c in row['structure_discovery']['candidates']]+
            [c['error'] for c in row['competing_structure']['candidates']])


def evaluate_photo(image,annotation,*,detectors=None,parent=_parent_capture,audit=reconcile):
    from autodocscanner.tools.pr9_holdout_validation import _quad,_errors
    row,raw,seeds=parent(image,annotation,detectors)
    if row['status']!='evaluated':return row
    evidence=audit(image.copy(),raw,seeds)
    if evidence.get('auto_select','missing') is not None or evidence.get('auto_crop_allowed') is not False or evidence.get('requires_manual_confirmation') is not True:
        raise ValueError('unsafe reconciliation contract')
    ref=_quad(annotation,image.shape);previous=_earlier_errors(row)
    if len(previous)!=len(raw) or len(raw)!=len(evidence['candidates']):raise ValueError('candidate count changed')
    for i,(c,e,old) in enumerate(zip(raw,evidence['candidates'],previous,strict=True)):
        q=_quad(c['corners'],image.shape)
        if q is None:raise ValueError('invalid candidate')
        mean,worst=_errors(q,ref,image.shape)
        if e['candidate_id']!=i+1 or e['family']!=c['family'] or not np.allclose([mean,worst],[old['mean'],old['max']],rtol=1e-7,atol=1e-9):
            raise ValueError('candidate alignment changed')
        e['error']={'mean':mean,'max':worst}
    # Positive reference control computed only after candidate measurements.
    from autodocscanner.core.pr94_region_boundary import inspect_structure
    from autodocscanner.core.pr912_reconciliation import inspect_candidate
    row['reconciliation']={**evidence,'reference_control':inspect_candidate(image,ref,seeds,inspect_structure(image,ref))}
    return row


def metrics(rows):
    pos=[r for r in rows if r['status']=='evaluated'];flat=[c for r in pos for c in r['reconciliation']['candidates']]
    good=lambda c:c['error']['mean']<=.02
    def stats(v):
        return dict(mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.percentile(v,90)),max=float(max(v))) if v else dict.fromkeys(('mean','median','p90','max'))
    tests=('structure_supported','surface_boundary_supported','region_structure_supported','partial_joint_supported','joint_supported','joint_without_crossing')
    measures={}
    for k in tests:
        supported=[c for c in flat if c[k]]
        groups=[[r['reconciliation']['candidates'][i] for i in g['members'] if r['reconciliation']['candidates'][i][k]] for r in pos for g in r['reconciliation']['groups']]
        groups=[g for g in groups if g]
        measures[k]=dict(supported_candidates=len(supported),accurate_supported=sum(map(good,supported)),wrong_supported=sum(not good(c) for c in supported),severe_supported=sum(c['error']['mean']>.1 for c in supported),accurate_unsupported=sum(good(c) and not c[k] for c in flat),
                         photos_with_accurate_supported=sum(any(c[k] and good(c) for c in r['reconciliation']['candidates']) for r in pos),
                         supported_groups=len(groups),groups_with_accurate_member=sum(any(map(good,g)) for g in groups),groups_with_wrong_member=sum(any(not good(c) for c in g) for g in groups),
                         supported_error=stats([c['error']['mean'] for c in supported]),supported_worst_corner_error=stats([c['error']['max'] for c in supported]))
    best=[min(c['error']['mean'] for c in r['reconciliation']['candidates']) for r in pos if r['reconciliation']['candidates']]
    return dict(candidate_count=len(flat),accurate_candidates=sum(map(good,flat)),geometry_group_count=sum(len(r['reconciliation']['groups']) for r in pos),
                oracle_photos_within_2pct=sum(e<=.02 for e in best),oracle_error=stats(best),evidence=measures,
                side_state_counts={key:dict(Counter(s['state'] for c in flat if pred(c) for s in c['sides'])) for key,pred in [('all',lambda c:True),('accurate',good),('wrong',lambda c:not good(c))]},
                crossing_seed_candidates=sum(bool(c['crossing_seed_ids']) for c in flat),
                photos_with_joint_conflicts=sum(bool(r['reconciliation']['joint_conflicts']) for r in pos),
                conflict_types=dict(Counter(c['kind'] for r in pos for c in r['reconciliation']['joint_conflicts'])),
                reference_centered_sides=sum(r['reconciliation']['reference_control']['centered_sides'] for r in pos),reference_total_sides=4*len(pos),
                reference_joint_supported=sum(r['reconciliation']['reference_control']['joint_supported'] for r in pos))


def evaluate_folder(folder,annotations,role,*,evaluator=evaluate_photo):
    from autodocscanner.tools.pr99_validation import summarize_region
    from autodocscanner.tools.pr910_validation import structure_metrics
    from autodocscanner.tools.pr911_validation import metrics as competing_metrics
    from autodocscanner.tools.pr9_holdout_validation import EXTENSIONS
    if not isinstance(annotations,dict):raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
        try:im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error):im=None
        rows.append(evaluator(im,annotations.get(path.name)))
    out=summarize_region(rows,role);out['structure_discovery_metrics']=structure_metrics(rows);out['competing_structure_metrics']=competing_metrics(rows);out['reconciliation_metrics']=metrics(rows)
    out['mode']='positive_only_reconciliation_audit';out['note']='All PR9.11 and earlier hypotheses retained. Evidence and geometry groups are advisory, not independent votes or identity proof. Oracle and reference controls are retrospective. No automatic selection.'
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description='PR9.12 local positive-only reconciliation audit')
    p.add_argument('folder',type=Path);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--dataset-role',choices=('development','holdout'),required=True);a=p.parse_args(argv)
    try:r=evaluate_folder(a.folder,json.loads(a.annotations.read_text(encoding='utf-8-sig')),a.dataset_role)
    except (OSError,ValueError,TypeError,cv2.error):p.error('Dataset atau kontrak evaluasi tidak valid')
    print(json.dumps(r,indent=2,allow_nan=False));return 0 if r['evaluated_photo_count'] else 2


if __name__=='__main__':raise SystemExit(main())
