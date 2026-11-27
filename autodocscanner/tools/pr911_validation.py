"""Local positive-only PR9.11 audit. Never exports pixels, points or filenames."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from autodocscanner.core.pr911_competing_structure import discover_competing_regions


def evaluate_photo(image, annotation, *, detectors=None, discover=discover_competing_regions, parent=None):
    if parent is None:
        from autodocscanner.tools.pr910_validation import evaluate_photo as parent
    from autodocscanner.tools.pr9_holdout_validation import _quad, _errors
    row=parent(image,annotation,detectors=detectors)
    if row['status']!='evaluated':return row
    ref=_quad(annotation,image.shape)
    raw=discover(image.copy())
    if raw.get('auto_select','missing') is not None or raw.get('auto_crop_allowed') is not False or raw.get('requires_manual_confirmation') is not True:
        raise ValueError('unsafe competing region contract')
    seeds=[]
    for seed in raw['seeds']:
        points=np.asarray(seed['points'],np.float32)
        inside=float(np.mean([cv2.pointPolygonTest(ref.astype(np.float32),tuple(p),False)>=0 for p in points]))
        hull=cv2.convexHull(points);area=float(cv2.contourArea(hull));overlap=float(cv2.intersectConvexConvex(hull,ref.astype(np.float32))[0])
        seeds.append({**{k:seed[k] for k in ('seed_id','stroke_count','aligned_count','area_fraction','ambiguous_fraction','view_index','attempts')},
                      'reference_point_fraction':inside,'reference_hull_coverage':min(1.,overlap/max(1.,cv2.contourArea(ref.astype(np.float32)))),'seed_hull_inside_fraction':min(1.,overlap/max(1.,area))})
    candidates=[]
    for c in raw['candidates']:
        q=_quad(c['corners'],image.shape)
        if q is None:raise ValueError('invalid region hypothesis')
        mean,worst=_errors(q,ref,image.shape)
        candidates.append({**{k:c[k] for k in ('family','geometry','seed_ids','supporting_expansions','region_iou','enclosed_seed_fraction','mode','threshold','structure')},'error':{'mean':mean,'max':worst}})
    row['competing_structure']={**{k:raw[k] for k in ('status','seed_count','raw_group_count','views')},'seeds':seeds,'candidates':candidates}
    return row


def metrics(rows):
    pos=[r for r in rows if r['status']=='evaluated'];new=[c for r in pos for c in r['competing_structure']['candidates']]
    seeds=[s for r in pos for s in r['competing_structure']['seeds']]
    def previous(r):
        return [c['error']['mean'] for c in r['candidates']]+[c['adaptive_recovery']['error']['mean'] for c in r['candidates'] if c['adaptive_recovery']['error'] is not None]+[h['error']['mean'] for c in r['candidates'] for h in c['region_guided']['hypotheses']]+[c['error']['mean'] for c in r['structure_discovery']['candidates']]
    def added(r):return [c['error']['mean'] for c in r['competing_structure']['candidates']]
    def best(v):return min(v) if v else None
    def good(v):return v is not None and v<=.02
    def stats(v):
        v=[x for x in v if x is not None]
        return dict(mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.percentile(v,90)),max=float(max(v))) if v else dict.fromkeys(('mean','median','p90','max'))
    before=[best(previous(r)) for r in pos];ind=[best(added(r)) for r in pos];after=[best(previous(r)+added(r)) for r in pos]
    support=[c for c in new if c['structure']['distributed_stroke_pattern']]
    return dict(seed_count=len(seeds),photos_with_seeds=sum(bool(r['competing_structure']['seeds']) for r in pos),
        seeds_mostly_inside_reference=sum(s['reference_point_fraction']>=.9 for s in seeds),
        photos_with_mostly_inside_seed=sum(any(s['reference_point_fraction']>=.9 for s in r['competing_structure']['seeds']) for r in pos),
        seed_reference_point_fraction=stats([s['reference_point_fraction'] for s in seeds]),
        seed_reference_hull_coverage=stats([s['reference_hull_coverage'] for s in seeds]),
        overloaded_tiles=sum(v['overloaded_tiles'] for r in pos for v in r['competing_structure']['views']),
        total_tiles=sum(v['tiles'] for r in pos for v in r['competing_structure']['views']),
        expansion_status_counts=dict(Counter(a['status'] for s in seeds for a in s['attempts'])),
        independent_candidates=len(new),photos_with_independent_candidates=sum(x is not None for x in ind),
        accurate_independent_candidates=sum(good(c['error']['mean']) for c in new),severe_independent_candidates=sum(c['error']['mean']>.1 for c in new),
        independent_oracle_photos_within_2pct=sum(map(good,ind)),independent_oracle_error=stats(ind),
        previous_oracle_photos_within_2pct=sum(map(good,before)),augmented_oracle_photos_within_2pct=sum(map(good,after)),
        previous_oracle_error=stats(before),augmented_oracle_error=stats(after),additional_accurate_photos=sum(not good(b) and good(a) for b,a in zip(before,after)),
        photos_with_improved_oracle=sum(a is not None and (b is None or a<b-1e-6) for b,a in zip(before,after)),
        structure_supported_candidates=len(support),accurate_structure_supported=sum(good(c['error']['mean']) for c in support),wrong_structure_supported=sum(not good(c['error']['mean']) for c in support),severe_structure_supported=sum(c['error']['mean']>.1 for c in support),
        independent_worst_corner_error=stats([c['error']['max'] for c in new]))


def evaluate_folder(folder,annotations,role,*,evaluator=evaluate_photo):
    from autodocscanner.tools.pr99_validation import summarize_region
    from autodocscanner.tools.pr910_validation import structure_metrics
    from autodocscanner.tools.pr9_holdout_validation import EXTENSIONS
    if not isinstance(annotations,dict):raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
        try:im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error):im=None
        rows.append(evaluator(im,annotations.get(path.name)))
    out=summarize_region(rows,role);out['structure_discovery_metrics']=structure_metrics(rows);out['competing_structure_metrics']=metrics(rows)
    out['mode']='positive_only_competing_structure_audit';out['note']='Earlier hypotheses retained. Seed reference metrics are evaluation only; structure and region evidence do not establish KTP identity. Oracle metrics are retrospective.'
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description='PR9.11 positive-only competing structure audit')
    p.add_argument('folder',type=Path);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--dataset-role',choices=('development','holdout'),required=True);a=p.parse_args(argv)
    try:r=evaluate_folder(a.folder,json.loads(a.annotations.read_text(encoding='utf-8-sig')),a.dataset_role)
    except (OSError,ValueError,TypeError,cv2.error):p.error('Dataset atau anotasi tidak valid')
    print(json.dumps(r,indent=2,allow_nan=False));return 0 if r['evaluated_photo_count'] else 2


if __name__=='__main__':raise SystemExit(main())
