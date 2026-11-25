"""Positive-only region-guided audit; no source pixels or corners exported."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from autodocscanner.core.pr99_region_guided import audit_regions


def evaluate_photo(image,annotation,*,detectors=None,audit=audit_regions):
    from autodocscanner.tools.pr98_validation import evaluate_photo as parent
    from autodocscanner.tools.pr9_holdout_validation import _quad,_errors
    captured=[]
    def run(im):
        r=audit(im);captured.append(r);return r
    row=parent(image,annotation,detectors=detectors,audit=run)
    if row['status']!='evaluated':return row
    ref=_quad(annotation,image.shape)
    for c,raw in zip(row['candidates'],captured[0]['candidates'],strict=True):
        region=raw['region_guided'];hypotheses=[]
        for h in region['hypotheses']:
            p=_quad(h['corners'],image.shape)
            if p is None:raise ValueError('invalid region hypothesis')
            mean,worst=_errors(p,ref,image.shape)
            hypotheses.append({'view':h['view'],'region_iou':h['region_iou'],'structure':h['structure'],
                               'structure_positive_before_and_after':bool(c['structure']['distributed_stroke_pattern'] and h['structure']['distributed_stroke_pattern']),
                               'error':{'mean':mean,'max':worst},'mean_error_change':mean-c['error']['mean']})
        c['region_guided']={'status':region['status'],'views':region['views'],'hypotheses':hypotheses}
    return row


def region_metrics(rows):
    """Rows count correlated variants separately; photo coverage is primary."""
    pos=[r for r in rows if r['status']=='evaluated']
    flat=[c for r in pos for c in r['candidates']]
    pairs=[(c,h) for c in flat for h in c['region_guided']['hypotheses']]
    def previous(r):
        return [c['error']['mean'] for c in r['candidates']]+[c['adaptive_recovery']['error']['mean'] for c in r['candidates'] if c['adaptive_recovery']['error'] is not None]
    def added(r):return [h['error']['mean'] for c in r['candidates'] for h in c['region_guided']['hypotheses']]
    def best(values):return min(values) if values else None
    def stats(values):
        a=[x for x in values if x is not None]
        return {'mean':float(np.mean(a)),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'max':float(max(a))} if a else {'mean':None,'median':None,'p90':None,'max':None}
    before=[best(previous(r)) for r in pos];after=[best(previous(r)+added(r)) for r in pos]
    support=[h for c,h in pairs if h['structure']['distributed_stroke_pattern']]
    return {'original_candidates':len(flat),'region_hypotheses':len(pairs),'photos_with_region_hypotheses':sum(bool(added(r)) for r in pos),
            'view_status_counts':dict(Counter(v['status'] for c in flat for v in c['region_guided']['views'])),
            'improved':sum(h['mean_error_change']< -1e-6 for _,h in pairs),'regressed':sum(h['mean_error_change']>1e-6 for _,h in pairs),
            'unchanged':sum(abs(h['mean_error_change'])<=1e-6 for _,h in pairs),
            'newly_accurate':sum(c['error']['mean']>.02 and h['error']['mean']<=.02 for c,h in pairs),
            'lost_accuracy':sum(c['error']['mean']<=.02 and h['error']['mean']>.02 for c,h in pairs),
            'paired_before_error':stats([c['error']['mean'] for c,h in pairs]),'paired_after_error':stats([h['error']['mean'] for c,h in pairs]),
            'paired_after_worst_corner_error':stats([h['error']['max'] for c,h in pairs]),
            'previous_oracle_photos_within_2pct':sum(e is not None and e<=.02 for e in before),
            'augmented_oracle_photos_within_2pct':sum(e is not None and e<=.02 for e in after),
            'previous_oracle_error':stats(before),'augmented_oracle_error':stats(after),
            'photos_with_improved_oracle':sum(b is not None and a is not None and a<b-1e-6 for b,a in zip(before,after)),
            'region_with_structure':len(support),'accurate_region_with_structure':sum(h['error']['mean']<=.02 for h in support),
            'wrong_region_with_structure':sum(h['error']['mean']>.02 for h in support),
            'severe_region_with_structure':sum(h['error']['mean']>.10 for h in support),
            'photos_with_accurate_region_structure':sum(any(h['structure']['distributed_stroke_pattern'] and h['error']['mean']<=.02 for c in r['candidates'] for h in c['region_guided']['hypotheses']) for r in pos)}


def summarize_region(rows,role):
    from autodocscanner.tools.pr98_validation import summarize_recovery
    out=summarize_recovery(rows,role);out['region_guided_metrics']=region_metrics(rows)
    out['mode']='positive_only_region_guided_audit'
    out['note']='Original and PR9.8 hypotheses retained. Region views share pixels and are not independent. Oracle metrics are retrospective. Structure and region consistency do not establish KTP identity.'
    return out


def evaluate_folder(folder,annotations,role,*,evaluator=evaluate_photo):
    from autodocscanner.tools.pr9_holdout_validation import EXTENSIONS
    if not isinstance(annotations,dict):raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
        try:im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error):im=None
        rows.append(evaluator(im,annotations.get(path.name)))
    return summarize_region(rows,role)


def main(argv=None):
    p=argparse.ArgumentParser(description='PR9.9 positive-only region-guided audit')
    p.add_argument('folder',type=Path);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--dataset-role',choices=('development','holdout'),required=True);a=p.parse_args(argv)
    try:r=evaluate_folder(a.folder,json.loads(a.annotations.read_text(encoding='utf-8-sig')),a.dataset_role)
    except (OSError,ValueError,TypeError,cv2.error):p.error('Dataset atau anotasi tidak valid')
    print(json.dumps(r,indent=2,allow_nan=False));return 0 if r['evaluated_photo_count'] else 2


if __name__=='__main__':raise SystemExit(main())
