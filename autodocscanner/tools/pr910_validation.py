"""Positive-only independent structure discovery audit; no source data exported."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from autodocscanner.core.pr910_structure_discovery import discover_structure_regions


def evaluate_photo(image,annotation,*,detectors=None,discover=discover_structure_regions,parent=None):
    if parent is None:
        from autodocscanner.tools.pr99_validation import evaluate_photo as parent
    from autodocscanner.tools.pr9_holdout_validation import _quad,_errors
    row=parent(image,annotation,detectors=detectors)
    if row['status']!='evaluated':return row
    raw=discover(image.copy())
    if raw.get('auto_select','missing') is not None or raw.get('auto_crop_allowed') is not False:raise ValueError('unsafe structure discovery contract')
    ref=_quad(annotation,image.shape);candidates=[]
    for c in raw['candidates']:
        q=_quad(c['corners'],image.shape)
        if q is None:raise ValueError('invalid structure hypothesis')
        mean,worst=_errors(q,ref,image.shape)
        candidates.append({**{k:c[k] for k in ('family','geometry','seed_ids','supporting_expansions','region_iou','enclosed_seed_fraction','surface_threshold_lab','structure')},'error':{'mean':mean,'max':worst}})
    row['structure_discovery']={**{k:raw[k] for k in ('status','seed_count','raw_group_count','views','seeds')},'candidates':candidates}
    return row


def structure_metrics(rows):
    pos=[r for r in rows if r['status']=='evaluated'];new=[c for r in pos for c in r['structure_discovery']['candidates']]
    def previous(r):
        return [c['error']['mean'] for c in r['candidates']]+[c['adaptive_recovery']['error']['mean'] for c in r['candidates'] if c['adaptive_recovery']['error'] is not None]+[h['error']['mean'] for c in r['candidates'] for h in c['region_guided']['hypotheses']]
    def added(r):return [c['error']['mean'] for c in r['structure_discovery']['candidates']]
    def best(v):return min(v) if v else None
    def stats(v):
        v=[x for x in v if x is not None]
        return {'mean':float(np.mean(v)),'median':float(np.median(v)),'p90':float(np.percentile(v,90)),'max':float(max(v))} if v else dict.fromkeys(('mean','median','p90','max'))
    before=[best(previous(r)) for r in pos];ind=[best(added(r)) for r in pos];after=[best(previous(r)+added(r)) for r in pos]
    good=lambda x:x is not None and x<=.02
    support=[c for c in new if c['structure']['distributed_stroke_pattern']]
    return {'seed_count':sum(r['structure_discovery']['seed_count'] for r in pos),'photos_with_seeds':sum(r['structure_discovery']['seed_count']>0 for r in pos),
            'discovery_status_counts':dict(Counter(r['structure_discovery']['status'] for r in pos)),
            'expansion_status_counts':dict(Counter(a['status'] for r in pos for s in r['structure_discovery']['seeds'] for a in s['attempts'])),
            'independent_candidates':len(new),'photos_with_independent_candidates':sum(x is not None for x in ind),
            'accurate_independent_candidates':sum(good(c['error']['mean']) for c in new),'severe_independent_candidates':sum(c['error']['mean']>.10 for c in new),
            'independent_oracle_photos_within_2pct':sum(map(good,ind)),'independent_oracle_error':stats(ind),
            'previous_oracle_photos_within_2pct':sum(map(good,before)),'augmented_oracle_photos_within_2pct':sum(map(good,after)),
            'previous_oracle_error':stats(before),'augmented_oracle_error':stats(after),
            'additional_accurate_photos':sum(not good(b) and good(a) for b,a in zip(before,after)),
            'photos_with_improved_oracle':sum(a is not None and (b is None or a<b-1e-6) for b,a in zip(before,after)),
            'structure_supported_candidates':len(support),'accurate_structure_supported':sum(good(c['error']['mean']) for c in support),
            'wrong_structure_supported':sum(not good(c['error']['mean']) for c in support),'severe_structure_supported':sum(c['error']['mean']>.10 for c in support),
            'independent_worst_corner_error':stats([c['error']['max'] for c in new])}


def evaluate_folder(folder,annotations,role,*,evaluator=evaluate_photo):
    from autodocscanner.tools.pr99_validation import summarize_region
    from autodocscanner.tools.pr9_holdout_validation import EXTENSIONS
    if not isinstance(annotations,dict):raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
        try:im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error):im=None
        rows.append(evaluator(im,annotations.get(path.name)))
    out=summarize_region(rows,role);out['structure_discovery_metrics']=structure_metrics(rows)
    out['mode']='positive_only_independent_structure_discovery_audit'
    out['note']='All earlier candidates retained. Independent discovery uses no earlier corners. Structure and seed inclusion are not identity proof. Oracle metrics are retrospective.'
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description='PR9.10 positive-only structure discovery audit')
    p.add_argument('folder',type=Path);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--dataset-role',choices=('development','holdout'),required=True);a=p.parse_args(argv)
    try:r=evaluate_folder(a.folder,json.loads(a.annotations.read_text(encoding='utf-8-sig')),a.dataset_role)
    except (OSError,ValueError,TypeError,cv2.error):p.error('Dataset atau anotasi tidak valid')
    print(json.dumps(r,indent=2,allow_nan=False));return 0 if r['evaluated_photo_count'] else 2


if __name__=='__main__':raise SystemExit(main())
