"""Local PR9.6 paired evidence comparison on unchanged PR9.4 proposals.

Explicit null annotations denote negative scenes; omitted entries do not.
Only anonymous measurements are exported. No identity or crop decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from autodocscanner.core.pr96_region_verification import verify_regions, inspect_region_boundary, prepare_lab
from autodocscanner.core.pr95_region_evidence import _side
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry
from autodocscanner.tools.pr9_holdout_validation import _quad
from autodocscanner.tools.pr95_validation import evaluate_photo as evaluate_parent, summarize as summarize_parent
from autodocscanner.tools.pr9_holdout_validation import EXTENSIONS, _stats

FIELDS=('region_sides','region_centered_sides','region_boundary_supported','region_and_structure','either_complete_with_structure','both_complete_with_structure')
EVIDENCE=('boundary_supported','combined_evidence','multiscale_boundary_supported',
          'boundary_and_structure','partial_boundary_and_structure',*FIELDS[2:])


def evaluate_photo(image, annotation, *, detectors=None, audit=verify_regions, negative=False):
    captured=[]
    def discover(im):
        result=audit(im);captured.append(result);return result
    row=evaluate_parent(image,annotation,detectors=detectors,audit=discover,negative=negative)
    if row['status'] not in ('evaluated','negative_scene'): return row
    result=captured[0]
    for c, original in zip(row['candidates'],result['candidates'],strict=True):
        c.update({k:original[k] for k in FIELDS})
    row['groups']=result['groups']
    row['competing_evidence_groups']=result['competing_evidence_groups']
    if not negative:
        # Retrospective control, never used by discovery or candidate evidence.
        q=_quad(annotation,image.shape);lab=prepare_lab(image)
        geom,_=inspect_geometry(q,image.shape)
        shortest=float(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1).min())
        unit=max(1.5,min(.007*min(image.shape[:2]),.025*shortest))
        old=[_side(lab,q[i],q[(i+1)%4],q.mean(axis=0),unit)['state'] for i in range(4)]
        new=[s['state'] for s in inspect_region_boundary(lab,q)]
        row['reference_control']={'geometry':geom['status'],'pr95_side_states':old,'pr96_side_states':new,
                                  'pr95_centered_sides':old.count('centered'),'pr96_centered_sides':new.count('centered')}
    return row


def summarize(rows,role):
    out=summarize_parent(rows,role)
    positives=[r for r in rows if r['status']=='evaluated']
    negatives=[r for r in rows if r['status']=='negative_scene']
    flat=[c for r in positives for c in r['candidates']]
    def good(c): return c['error']['mean']<=.02
    metrics={}
    for key in EVIDENCE:
        supported=[c for c in flat if c[key]]
        group_count=0; accurate_groups=0; severe_groups=0
        for r in positives:
            for group in r['groups']:
                members=[r['candidates'][i] for i in group['members'] if r['candidates'][i][key]]
                if members:
                    group_count+=1
                    accurate_groups+=any(good(c) for c in members)
                    severe_groups+=any(c['error']['mean']>.10 for c in members)
        metrics[key]={'supported_candidates':len(supported),'accurate_supported':sum(good(c) for c in supported),
                      'wrong_supported':sum(not good(c) for c in supported),
                      'severe_supported':sum(c['error']['mean']>.10 for c in supported),
                      'accurate_unsupported':sum(good(c) and not c[key] for c in flat),
                      'photos_with_accurate_supported':sum(any(c[key] and good(c) for c in r['candidates']) for r in positives),
                      'supported_groups':group_count,'groups_with_accurate_supported_member':accurate_groups,
                      'groups_with_severe_supported_member':severe_groups,
                      'supported_error':_stats([c['error']['mean'] for c in supported]),
                      'negative_photos_with_evidence':sum(any(c[key] for c in r['candidates']) for r in negatives)}
    out['paired_evidence_metrics']=metrics
    out['geometry_groups']={'positive_count':sum(len(r['groups']) for r in positives),
                            'negative_count':sum(len(r['groups']) for r in negatives),
                            'competing_evidence_photos':sum(r['competing_evidence_groups'] for r in positives),
                            'competing_negative_photos':sum(r['competing_evidence_groups'] for r in negatives)}
    states={}
    for c in flat:
        for s in c['multiscale_sides']:
            key=s['state'];states[key]=states.get(key,0)+1
    out['side_state_counts']=states
    def region_states(cs):
        counts={}
        for c in cs:
            for side in c['region_sides']:
                key=side['state'];counts[key]=counts.get(key,0)+1
        return counts
    out['region_side_state_counts']={'all':region_states(flat),'accurate':region_states([c for c in flat if good(c)]),'inaccurate':region_states([c for c in flat if not good(c)])}
    controls=[r['reference_control'] for r in positives]
    out['reference_control_metrics']={'photo_count':len(controls),**{method:{
        'centered_sides':sum(c[method+'_centered_sides'] for c in controls),
        'photos_with_four_centered_sides':sum(c[method+'_centered_sides']==4 for c in controls),
        'in_frame_photos_with_four_centered_sides':sum(c['geometry']=='in_frame' and c[method+'_centered_sides']==4 for c in controls)} for method in ('pr95','pr96')}}
    out['note']='Unchanged PR9.4 proposals. PR9.5 and PR9.6 evidence are advisory; union/intersection are diagnostic comparisons, not selection. Reference control is retrospective only. No KTP identity claim.'
    return out


def evaluate_folder(folder,annotations,role,negative_folder=None):
    if not isinstance(annotations,dict): raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS: continue
        try: im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error): im=None
        rows.append(evaluate_photo(im,annotations.get(path.name),negative=path.name in annotations and annotations[path.name] is None))
    if negative_folder is not None:
        if Path(folder).resolve()==Path(negative_folder).resolve(): raise ValueError('negative folder must differ')
        for path in sorted(Path(negative_folder).iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS: continue
            try: im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            except (OSError,ValueError,cv2.error): im=None
            rows.append(evaluate_photo(im,None,negative=True))
    return summarize(rows,role)


def main(argv=None):
    parser=argparse.ArgumentParser(description='PR9.6 regional boundary evidence audit, local only')
    parser.add_argument('folder',type=Path);parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--dataset-role',choices=('development','holdout'),required=True)
    parser.add_argument('--negative-folder',type=Path,help='Folder containing NO target KTP (operator declared)')
    args=parser.parse_args(argv)
    if not args.folder.is_dir(): parser.error('Folder foto tidak ditemukan')
    if not args.annotations.is_file(): parser.error('File anotasi tidak ditemukan')
    if args.negative_folder is not None and not args.negative_folder.is_dir(): parser.error('Folder negatif tidak ditemukan')
    try: result=evaluate_folder(args.folder,json.loads(args.annotations.read_text(encoding='utf-8-sig')),args.dataset_role,args.negative_folder)
    except (OSError,ValueError,TypeError,cv2.error): parser.error('Dataset, anotasi atau kontrak evaluasi tidak valid')
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return 0 if result['evaluated_photo_count'] or result['negative_scene_metrics']['photo_count'] else 2


if __name__=='__main__':
    raise SystemExit(main())
