"""Local PR9.2 evaluation; only original Stable/Adaptive and their hypotheses.

Reuses the PR9.1 annotation format. Ground truth is used after proposals,
never to recover boundaries. Output contains no filenames, points or pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from autodocscanner.core.pr92_boundary_recovery import recover_boundary
from autodocscanner.tools.pr9_holdout_validation import (
    SOURCES, EXTENSIONS, _quad, _errors, _stats, _default_detectors,
)


def evaluate_photo(image, annotation, *, detectors=None):
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
            or image.shape[2]!=3 or min(image.shape[:2])<64):
        return {'status':'invalid_image'}
    reference=_quad(annotation,image.shape)
    if reference is None:
        return {'status':'invalid_annotation'}
    if detectors is None:
        detectors=_default_detectors()
    rows={}
    for source in SOURCES:
        row={'status':'missing_candidate','geometry':None,'sides':[],
             'original_error':None,'recovered_error':None,'recovered_sides':[]}
        try:
            corners,_=detectors[source].detect(image.copy())
        except (KeyError,AttributeError,cv2.error,ValueError,TypeError,ArithmeticError):
            row['status']='detector_error'; rows[source]=row; continue
        if corners is not None:
            result=recover_boundary(image,corners)
            row.update({k:result[k] for k in ('status','geometry','sides','recovered_sides')})
            # Off-frame geometry is reported separately; only in-frame
            # valid candidates enter error statistics.
            p=_quad(corners,image.shape) if result['geometry']['outside_vertex_count']==0 else None
            if p is not None:
                mean,worst=_errors(p,reference,image.shape)
                row['original_error']={'mean':mean,'max':worst}
            if result['hypothesis'] is not None:
                p=_quad(result['hypothesis'],image.shape)
                if p is None:
                    raise ValueError('invalid recovery contract')
                mean,worst=_errors(p,reference,image.shape)
                row['recovered_error']={'mean':mean,'max':worst}
        rows[source]=row
    return {'status':'evaluated','candidates':rows}


def summarize(rows,role):
    evaluated=[r for r in rows if r['status']=='evaluated']
    source_metrics={}
    for source in SOURCES:
        candidates=[r['candidates'][source] for r in evaluated]
        measured=[r for r in candidates if r['original_error'] is not None]
        paired=[r for r in measured if r['recovered_error'] is not None]
        change=[r['recovered_error']['mean']-r['original_error']['mean'] for r in paired]
        counts={}
        for r in candidates:
            counts[r['status']]=counts.get(r['status'],0)+1
        source_metrics[source]={
            'status_counts':counts,'original_measured':len(measured),
            'original_error':_stats([r['original_error']['mean'] for r in measured]),
            'paired_recovery_count':len(paired),
            'paired_original_error':_stats([r['original_error']['mean'] for r in paired]),
            'paired_recovered_error':_stats([r['recovered_error']['mean'] for r in paired]),
            'paired_change':_stats(change),
            'improved':sum(x < -1e-9 for x in change),
            'regressed':sum(x > 1e-9 for x in change),
            'unchanged':sum(abs(x)<=1e-9 for x in change),
            'centered_but_over_2pct':sum(r['status']=='persistent_centered_transitions' and
                                      r['original_error']['mean']>.02 for r in measured),
        }
    return {'dataset_role':role,'photo_count':len(rows),'evaluated_photo_count':len(evaluated),
            'source_metrics':source_metrics,'auto_crop_allowed':False,'auto_select':None,
            'error_unit':'fraction_of_original_image_diagonal',
            'note':'Paired recovery statistics are retrospective, not automatic selection. No holdout independence guarantee.',
            'photos':[{'case_id':f'D{i:04d}',**r} for i,r in enumerate(rows,1)]}


def evaluate_folder(folder,annotations,role):
    if role not in ('development','holdout') or not isinstance(annotations,dict):
        raise ValueError('invalid annotation format or dataset role')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            image=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR)
        except (OSError,cv2.error,ValueError):
            image=None
        rows.append(evaluate_photo(image,annotations.get(path.name)))
    return summarize(rows,role)


def main(argv=None):
    parser=argparse.ArgumentParser(description='PR9.2 local geometry and boundary recovery evaluation')
    parser.add_argument('folder',type=Path)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--dataset-role',choices=('development','holdout'),required=True)
    args=parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error('Folder foto tidak ditemukan atau bukan direktori')
    if not args.annotations.is_file():
        parser.error('File anotasi tidak ditemukan')
    try:
        annotations=json.loads(args.annotations.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):
        parser.error('File anotasi tidak dapat dibaca sebagai JSON')
    if not isinstance(annotations,dict):
        parser.error('Anotasi harus berupa objek JSON')
    try:
        out=evaluate_folder(args.folder,annotations,args.dataset_role)
    except (OSError,ValueError,TypeError,cv2.error):
        parser.error('Evaluasi gagal: periksa dataset atau kontrak modul')
    print(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False))
    return 0 if out['evaluated_photo_count'] else 2


if __name__=='__main__':
    raise SystemExit(main())
