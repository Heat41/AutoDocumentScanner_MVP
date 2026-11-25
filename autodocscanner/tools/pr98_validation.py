"""Positive-only paired localization audit; no photo or coordinates exported."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from autodocscanner.core.pr98_adaptive_localization import audit_recovery, recover_tracks
from autodocscanner.tools.pr96_validation import evaluate_photo as evaluate_parent, summarize
from autodocscanner.tools.pr9_holdout_validation import _quad,_errors,_stats,EXTENSIONS


def evaluate_photo(image,annotation,*,detectors=None,audit=audit_recovery):
    captured=[]
    def run(im):
        r=audit(im);captured.append(r);return r
    row=evaluate_parent(image,annotation,detectors=detectors,audit=run,negative=False)
    if row['status']!='evaluated':return row
    ref=_quad(annotation,image.shape)
    # Reference is only a retrospective measurement, never a recovery seed.
    control=recover_tracks(image,ref)
    row['adaptive_reference_control']={'sides':control['sides'],'centered_sides':sum(s['state']=='centered' for s in control['sides'])}
    for c,raw in zip(row['candidates'],captured[0]['candidates'],strict=True):
        rec=raw['adaptive_recovery'];p=_quad(rec['hypothesis'],image.shape)
        error=None
        if p is not None:
            mean,worst=_errors(p,ref,image.shape);error={'mean':mean,'max':worst}
        elif rec['hypothesis'] is not None:raise ValueError('invalid recovery geometry')
        c['adaptive_recovery']={**{k:rec[k] for k in ('status','sides','recovered_sides','boundary_supported','recovered_structure')},
                                'error':error,'mean_error_change':None if error is None else error['mean']-c['error']['mean']}
    return row


def summarize_recovery(rows,role):
    out=summarize(rows,role);positives=[r for r in rows if r['status']=='evaluated']
    flat=[c for r in positives for c in r['candidates']];pairs=[c for c in flat if c['adaptive_recovery']['error'] is not None]
    def good(e):return e['mean']<=.02
    def augmented(r):return [c['error'] for c in r['candidates']]+[c['adaptive_recovery']['error'] for c in r['candidates'] if c['adaptive_recovery']['error'] is not None]
    def supported(r):
        return [c['error'] for c in r['candidates'] if c['adaptive_recovery']['boundary_supported'] and c['structure']['distributed_stroke_pattern']]+[c['adaptive_recovery']['error'] for c in r['candidates'] if c['adaptive_recovery']['error'] is not None and c['adaptive_recovery']['recovered_structure']['distributed_stroke_pattern']]
    best=[min(e['mean'] for e in augmented(r)) for r in positives if augmented(r)]
    support=[e for r in positives for e in supported(r)]
    out['adaptive_recovery_metrics']={'original_candidates':len(flat),'recovered_hypotheses':len(pairs),
        'status_counts':dict(Counter(c['adaptive_recovery']['status'] for c in flat)),
        'improved':sum(c['adaptive_recovery']['mean_error_change']< -1e-6 for c in pairs),
        'regressed':sum(c['adaptive_recovery']['mean_error_change']>1e-6 for c in pairs),
        'unchanged':sum(abs(c['adaptive_recovery']['mean_error_change'])<=1e-6 for c in pairs),
        'paired_before_error':_stats([c['error']['mean'] for c in pairs]),
        'paired_after_error':_stats([c['adaptive_recovery']['error']['mean'] for c in pairs]),
        'paired_after_worst_corner_error':_stats([c['adaptive_recovery']['error']['max'] for c in pairs]),
        'newly_accurate':sum(not good(c['error']) and good(c['adaptive_recovery']['error']) for c in pairs),
        'lost_accuracy':sum(good(c['error']) and not good(c['adaptive_recovery']['error']) for c in pairs),
        'original_oracle_photos_within_2pct':sum(any(good(c['error']) for c in r['candidates']) for r in positives),
        'augmented_oracle_photos_within_2pct':sum(any(good(e) for e in augmented(r)) for r in positives),
        'augmented_oracle_error':_stats(best),
        'supported_with_structure':len(support),'accurate_supported_with_structure':sum(good(e) for e in support),
        'wrong_supported_with_structure':sum(not good(e) for e in support),'severe_supported_with_structure':sum(e['mean']>.10 for e in support),
        'photos_with_accurate_supported':sum(any(good(e) for e in supported(r)) for r in positives),
        'reference_centered_sides':sum(r['adaptive_reference_control']['centered_sides'] for r in positives),
        'reference_total_sides':4*len(positives)}
    out['mode']='positive_only_adaptive_recovery_audit'
    out['note']='Originals retained; recovered hypotheses are not selections. Oracle and reference metrics are retrospective. Evidence does not establish KTP identity.'
    return out


def evaluate_folder(folder,annotations,role,*,evaluator=evaluate_photo):
    if not isinstance(annotations,dict):raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
        try:im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error):im=None
        rows.append(evaluator(im,annotations.get(path.name)))
    return summarize_recovery(rows,role)


def main(argv=None):
    p=argparse.ArgumentParser(description='PR9.8 positive-only adaptive recovery audit')
    p.add_argument('folder',type=Path);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--dataset-role',choices=('development','holdout'),required=True);a=p.parse_args(argv)
    try:r=evaluate_folder(a.folder,json.loads(a.annotations.read_text(encoding='utf-8-sig')),a.dataset_role)
    except (OSError,ValueError,TypeError,cv2.error):p.error('Dataset atau anotasi tidak valid')
    print(json.dumps(r,indent=2,allow_nan=False));return 0 if r['evaluated_photo_count'] else 2


if __name__=='__main__':raise SystemExit(main())
