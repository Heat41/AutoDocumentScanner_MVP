"""Local independent PR9.3 evaluation; existing human annotation format.

Compare candidate availability with Stable (historical baseline) and Adaptive.
Ground truth is used for retrospective metrics only; no selected/cropped image,
coordinates, filenames, or personal information are emitted or uploaded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from autodocscanner.core.pr93_independent_boundary import discover_boundaries, _image_ok
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry
from autodocscanner.tools.pr9_holdout_validation import SOURCES, EXTENSIONS, _quad, _errors, _stats, _default_detectors


def evaluate_photo(image, annotation, *, detectors=None, discover=None):
    if not _image_ok(image):
        return {'status':'invalid_image'}
    reference=_quad(annotation,image.shape)
    if reference is None:
        return {'status':'invalid_annotation'}
    # Independent discovery runs FIRST, without access to baseline geometry.
    result=(discover_boundaries if discover is None else discover)(image.copy())
    if (result.get('auto_crop_allowed') is not False or result.get('auto_select','missing') is not None):
        raise ValueError('unsafe discovery contract')
    def measure(points):
        p=_quad(points,image.shape)
        if p is None:
            return None
        mean,worst=_errors(p,reference,image.shape)
        return {'mean':mean,'max':worst}
    independent=[]
    for c in result['candidates']:
        error=measure(c['corners'])
        if error is None:
            raise ValueError('invalid independent geometry')
        independent.append({'error':error,'geometry':c['geometry'],
            'line_coverage_min':c['line_coverage_min'],
            'boundary_supported':c['boundary_supported'],
            'nested_in_supported_region':c['nested_in_supported_region'],
            'side_evidence':c['side_evidence']})
    if detectors is None:
        detectors=_default_detectors()
    baselines={}
    for source in SOURCES:
        try:
            corners,_=detectors[source].detect(image.copy())
        except (KeyError,AttributeError,cv2.error,ValueError,TypeError,ArithmeticError):
            baselines[source]={'status':'detector_error','error':None};continue
        if corners is None:
            baselines[source]={'status':'missing_candidate','error':None};continue
        geom,_=inspect_geometry(corners,image.shape)
        baselines[source]={'status':geom['status'],'error':measure(corners)}
    return {'status':'evaluated','baselines':baselines,'independent_status':result['status'],
            'line_count':result['line_count'],'competing_supported_geometry':result['competing_supported_geometry'],
            'independent_candidates':independent}


def summarize(rows, role):
    if role not in ('development','holdout'):
        raise ValueError('invalid dataset role')
    evaluated=[r for r in rows if r['status']=='evaluated']
    def good(e):
        return e is not None and e['mean']<=.02
    metrics={}
    for source in SOURCES:
        errors=[r['baselines'][source]['error'] for r in evaluated]
        measured=[e for e in errors if e is not None]
        metrics[source]={'measured':len(measured),'within_2pct':sum(good(e) for e in errors),
            'mean_corner_error':_stats([e['mean'] for e in measured]),
            'max_corner_error':_stats([e['max'] for e in measured]),
            'severe_over_10pct':sum(e['mean']>.10 for e in measured)}
    def best(r,supported=False):
        es=[c['error']['mean'] for c in r['independent_candidates'] if not supported or c['boundary_supported']]
        return min(es) if es else None
    all_best=[best(r) for r in evaluated]
    supported_best=[best(r,True) for r in evaluated]
    all_candidates=[c for r in evaluated for c in r['independent_candidates']]
    supported=[c for c in all_candidates if c['boundary_supported']]
    def baseline_good(r):
        return any(good(r['baselines'][s]['error']) for s in SOURCES)
    metrics['independent_lines']={
        'photos_with_proposals':sum(v is not None for v in all_best),
        'photos_with_supported_proposals':sum(v is not None for v in supported_best),
        'candidate_count':len(all_candidates),'supported_count':len(supported),
        'supported_over_2pct':sum(not good(c['error']) for c in supported),
        'supported_severe_over_10pct':sum(c['error']['mean']>.10 for c in supported),
        'nested_supported_count':sum(c['nested_in_supported_region'] for c in supported),
        'oracle_photos_within_2pct':sum(v is not None and v<=.02 for v in all_best),
        'supported_oracle_photos_within_2pct':sum(v is not None and v<=.02 for v in supported_best),
        'oracle_error':_stats([v for v in all_best if v is not None]),
        'supported_oracle_error':_stats([v for v in supported_best if v is not None]),
        'additional_photos_with_candidate_within_2pct':sum(not baseline_good(r) and v is not None and v<=.02 for r,v in zip(evaluated,all_best)),
        'ambiguous_supported_photos':sum(r['competing_supported_geometry'] for r in evaluated)}
    return {'dataset_role':role,'photo_count':len(rows),'evaluated_photo_count':len(evaluated),
        'source_metrics':metrics,'auto_select':None,'auto_crop_allowed':False,
        'error_unit':'fraction_of_original_image_diagonal',
        'note':'Oracle means best candidate AFTER annotation, not automatic selection. Dataset role is operator-declared.',
        'photos':[{'case_id':f'D{i:04d}',**r} for i,r in enumerate(rows,1)]}


def evaluate_folder(folder, annotations, role):
    if not isinstance(annotations,dict):
        raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            image=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error):
            image=None
        rows.append(evaluate_photo(image,annotations.get(path.name)))
    return summarize(rows,role)


def main(argv=None):
    parser=argparse.ArgumentParser(description='PR9.3 independent physical-boundary validation (local only)')
    parser.add_argument('folder',type=Path)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--dataset-role',choices=('development','holdout'),required=True)
    args=parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error('Folder foto tidak ditemukan')
    if not args.annotations.is_file():
        parser.error('File anotasi tidak ditemukan')
    try:
        annotations=json.loads(args.annotations.read_text(encoding='utf-8-sig'))
        result=evaluate_folder(args.folder,annotations,args.dataset_role)
    except (OSError,ValueError,TypeError,cv2.error):
        parser.error('Dataset, anotasi atau kontrak evaluasi tidak valid')
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return 0 if result['evaluated_photo_count'] else 2


if __name__=='__main__':
    raise SystemExit(main())
